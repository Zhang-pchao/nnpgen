"""Recover interrupted DFT manifest entries after scheduler outages.

Recovery is conservative: explicit failure markers are never retried, active
jobs are preserved, and manifest writes require the explicit ``--apply`` flag.
PBS and Slurm are supported through small, configurable status queries rather
than cluster-specific host names or paths.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set

from ..contracts import utc_now
from ..utils import write_json
from .archive import RemoteTarget, entry_key, entry_target_name, scan_target, parse_remote_spec


def parse_backend_spec(spec: str) -> tuple:
    raw = str(spec or "").strip()
    if "=" not in raw:
        raise ValueError("Backend must use target=pbs or target=slurm: {0}".format(spec))
    name, backend = raw.split("=", 1)
    backend = backend.strip().lower()
    if backend not in {"pbs", "slurm"}:
        raise ValueError("Unsupported scheduler backend: {0}".format(backend))
    if not name.strip():
        raise ValueError("Backend target name must not be empty")
    return name.strip(), backend


def parse_active_jobs(backend: str, text: str) -> Dict[str, Set[str]]:
    """Parse job ids and names from normalized PBS/Slurm command output."""

    active_ids: Set[str] = set()
    active_names: Set[str] = set()
    backend = str(backend).lower()
    for line in str(text or "").splitlines():
        parts = line.split("|") if "|" in line else line.split()
        if not parts or parts[0].lower() in {"job", "job_id", "id"}:
            continue
        if backend == "slurm" and len(parts) >= 3:
            state = parts[2].upper()
            if state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}:
                active_ids.add(parts[0])
                active_names.add(parts[1])
        elif backend == "pbs" and len(parts) >= 2:
            state = (parts[-2] if len(parts) >= 6 else parts[-1]).upper()
            if state in {"Q", "R", "E", "H", "W", "T"}:
                active_ids.add(parts[0])
                active_names.add(parts[1])
    return {"ids": active_ids, "names": active_names}


def query_active_jobs(target: RemoteTarget, backend: str) -> Dict[str, Set[str]]:
    """Query active jobs for a target using the scheduler's native CLI."""

    user = '$(id -un)'
    if backend == "slurm":
        command = "squeue -h -u {0} -o '%A|%j|%T' 2>/dev/null || true".format(user)
    elif backend == "pbs":
        command = "qstat -u {0} 2>/dev/null || true".format(user)
    else:
        raise ValueError("Unsupported scheduler backend: {0}".format(backend))
    result = subprocess.run(
        ["ssh", target.host, command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return parse_active_jobs(backend, result.stdout)


def _entry_job_values(entry: Mapping[str, Any]) -> Set[str]:
    values: Set[str] = set()
    for key in ("job_id", "pbs_job_id", "slurm_job_id", "pbs_job_name", "slurm_job_name", "job_name"):
        value = str(entry.get(key, "") or "").strip()
        if value:
            values.add(value)
    return values


def recover_manifest_entries(
    manifest: Mapping[str, Any],
    states: Mapping[str, Mapping[Any, Mapping[str, Any]]],
    active_jobs: Mapping[str, Mapping[str, Set[str]]],
    *,
    unavailable_targets: Optional[Set[str]] = None,
    allow_lost_targets: Optional[Set[str]] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a copy of *manifest* with safe recovery decisions applied."""

    output = copy.deepcopy(dict(manifest))
    entries = output.get("entries")
    if not isinstance(entries, list):
        entries = output.get("records")
    if not isinstance(entries, list):
        raise ValueError("Manifest must contain an entries or records list")

    unavailable = set(unavailable_targets or set())
    allowed_lost = set(allow_lost_targets or set())
    stamp = timestamp or utc_now()
    recovered = []
    preserved_active = []
    normalized_terminal = []
    unresolved = []

    target_names = set(states) | set(active_jobs) | unavailable
    target_map = {name: name for name in target_names}
    for entry in entries:
        if not isinstance(entry, dict) or str(entry.get("status", "")).lower() not in {"submitted", "running"}:
            continue
        system, frame = entry_key(entry)
        target_name = entry_target_name(entry, target_map)
        if not target_name:
            unresolved.append({"system": system, "frame": frame, "reason": "unknown_target"})
            continue
        if target_name in unavailable and target_name not in allowed_lost:
            unresolved.append({"system": system, "frame": frame, "target": target_name, "reason": "target_unavailable"})
            continue

        row = states.get(target_name, {}).get((system, frame))
        if row and row.get("result") == "success":
            entry["status"] = "finished"
            normalized_terminal.append({"system": system, "frame": frame, "target": target_name, "status": "finished"})
            continue
        if row and row.get("result") == "failed":
            entry["status"] = "failed"
            normalized_terminal.append({"system": system, "frame": frame, "target": target_name, "status": "failed"})
            continue

        active = active_jobs.get(target_name, {"ids": set(), "names": set()})
        values = _entry_job_values(entry)
        if values & (set(active.get("ids", set())) | set(active.get("names", set()))):
            preserved_active.append({"system": system, "frame": frame, "target": target_name})
            continue

        previous = str(entry.get("status", ""))
        entry["status"] = "selected"
        entry["job_id"] = None
        entry["recovered_after_outage_at"] = stamp
        entry["recovered_previous_status"] = previous
        entry["recovery_reason"] = "scheduler_job_missing_without_terminal_marker"
        recovered.append({"system": system, "frame": frame, "target": target_name, "previous_status": previous})

    summary = {
        "recovered_for_resubmit": len(recovered),
        "preserved_active": len(preserved_active),
        "normalized_terminal": len(normalized_terminal),
        "unresolved": len(unresolved),
    }
    output["recovery_summary"] = summary
    output["recovery_plan"] = {
        "recovered": recovered,
        "preserved_active": preserved_active,
        "normalized_terminal": normalized_terminal,
        "unresolved": unresolved,
    }
    return output


def add_recovery_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest to inspect and optionally update.")
    parser.add_argument("--remote", dest="remote_specs", action="append", required=True, metavar="NAME=HOST:/ROOT", help="Named remote frame root; repeat for multiple targets.")
    parser.add_argument("--backend", dest="backend_specs", action="append", default=[], metavar="NAME=PBS|SLURM", help="Scheduler backend for a target; default is PBS.")
    parser.add_argument("--allow-lost-target", action="append", default=[], metavar="NAME", help="Permit recovery when this target cannot be scanned.")
    parser.add_argument("--apply", action="store_true", help="Write recovery decisions back to the manifest.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")


def run_recovery(args: argparse.Namespace) -> Dict[str, Any]:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = [parse_remote_spec(spec) for spec in args.remote_specs]
    backends = dict(parse_backend_spec(spec) for spec in args.backend_specs)
    states: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    active: Dict[str, Dict[str, Set[str]]] = {}
    unavailable: Set[str] = set()
    for target in targets:
        backend = backends.get(target.name, "pbs")
        try:
            states[target.name] = scan_target(target)
            active[target.name] = query_active_jobs(target, backend)
        except (OSError, subprocess.CalledProcessError) as exc:
            states[target.name] = {}
            active[target.name] = {"ids": set(), "names": set()}
            unavailable.add(target.name)
            print("warning: target={0} unavailable: {1}".format(target.name, exc))

    planned = recover_manifest_entries(
        manifest,
        states,
        active,
        unavailable_targets=unavailable,
        allow_lost_targets=set(args.allow_lost_target),
    )
    planned["manifest"] = str(manifest_path)
    planned["applied"] = bool(args.apply)
    if args.apply:
        planned["controller_status"] = "recovered_after_outage"
        planned["updated_at"] = utc_now()
        write_json(manifest_path, planned)
    report_path = Path(args.report).expanduser().resolve() if args.report else manifest_path.with_name(manifest_path.stem + "_recovery_report.json")
    write_json(report_path, planned)
    print(json.dumps(planned["recovery_summary"], indent=2, sort_keys=True))
    return planned


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plan or apply safe recovery of interrupted DFT entries.")
    add_recovery_arguments(parser)
    run_recovery(parser.parse_args(argv))


if __name__ == "__main__":
    main()
