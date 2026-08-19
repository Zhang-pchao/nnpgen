"""Archive terminal remote DFT frames into a portable local work root.

Remote targets are supplied explicitly as ``name=host:/path`` values.  The
module deliberately has no cluster-specific defaults: manifests can refer to
one or more named targets, while the command line supplies the actual host and
root for a particular campaign.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..dataset.validation import require_valid_dp_dataset
from ..utils import ensure_dir, write_json


REMOTE_SCAN_SCRIPT = r'''
root="$1"
find "$root" -mindepth 2 -maxdepth 2 -type d -print | sort | while IFS= read -r frame_dir; do
  system=$(basename "$(dirname "$frame_dir")")
  frame=$(basename "$frame_dir")
  status=""
  for info in task_info.json frame_info.json; do
    if test -f "$frame_dir/$info"; then
      status=$(awk -F'"' '/"status"[[:space:]]*:/{print tolower($4); exit}' "$frame_dir/$info" 2>/dev/null)
      test -n "$status" && break
    fi
  done
  finished=0; failed=0; poscar=0; outcar=0; ediff=0; footer=0; task_info=0; frame_info=0
  test -f "$frame_dir/tag_finished" && finished=1
  test -f "$frame_dir/tag_failed" && failed=1
  test -f "$frame_dir/POSCAR" && poscar=1
  test -f "$frame_dir/OUTCAR" && outcar=1
  test -f "$frame_dir/task_info.json" && task_info=1
  test -f "$frame_dir/frame_info.json" && frame_info=1
  if test "$outcar" = 1; then
    grep -Fq 'aborting loop because EDIFF is reached' "$frame_dir/OUTCAR" && ediff=1
    grep -Fq 'General timing and accounting informations for this job:' "$frame_dir/OUTCAR" && footer=1
  fi
  result=pending
  if test "$finished" = 1 && test "$failed" = 0 && test "$poscar" = 1 && test "$outcar" = 1 && test "$ediff" = 1 && test "$footer" = 1; then
    result=success
  elif test "$failed" = 1 || test "$status" = failed; then
    result=failed
  fi
  rel=${frame_dir#"$root"/}
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$system" "$frame" "$result" "$status" "$finished" "$failed" "$poscar" "$outcar" "$ediff" "$footer" "$task_info" "$frame_info" "$rel"
done
'''


@dataclass(frozen=True)
class RemoteTarget:
    """A named remote frame root used by a manifest."""

    name: str
    host: str
    root: str


def parse_remote_spec(spec: str) -> RemoteTarget:
    """Parse ``name=host:/absolute/root`` without embedding site details."""

    raw = str(spec or "").strip()
    if "=" not in raw:
        raise ValueError("Remote target must use name=host:/root: {0}".format(spec))
    name, endpoint = raw.split("=", 1)
    name = name.strip()
    endpoint = endpoint.strip()
    if not name or ":" not in endpoint:
        raise ValueError("Remote target must use name=host:/root: {0}".format(spec))
    host, root = endpoint.split(":", 1)
    host = host.strip()
    root = root.strip()
    if not host or not root or not root.startswith("/") or root == "/":
        raise ValueError("Remote target requires a host and absolute root: {0}".format(spec))
    return RemoteTarget(name=name, host=host, root=root.rstrip("/") or "/")


def parse_scan_rows(raw: str, target_name: str = "") -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Parse the tabular output produced by :data:`REMOTE_SCAN_SCRIPT`."""

    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for line in str(raw or "").splitlines():
        fields = line.split("\t")
        if len(fields) != 13:
            continue
        system, frame, result, status, finished, failed, poscar, outcar, ediff, footer, task_info, frame_info, relative = fields
        rows[(system, frame)] = {
            "system": system,
            "frame": frame,
            "result": result,
            "status": status,
            "tag_finished": finished == "1",
            "tag_failed": failed == "1",
            "poscar": poscar == "1",
            "outcar": outcar == "1",
            "ediff_reached": ediff == "1",
            "normal_footer": footer == "1",
            "task_info": task_info == "1",
            "frame_info": frame_info == "1",
            "relative": relative,
            "target": target_name,
        }
    return rows


def scan_target(target: RemoteTarget) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Read frame markers from one target without modifying remote files."""

    command = "bash -s -- {0}".format(shlex.quote(target.root))
    result = subprocess.run(
        [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2",
            target.host,
            command,
        ],
        input=REMOTE_SCAN_SCRIPT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=True,
    )
    return parse_scan_rows(result.stdout, target.name)


def _manifest_records(manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    records = manifest.get("entries")
    if not isinstance(records, list):
        records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Manifest must contain an entries or records list")
    return [x for x in records if isinstance(x, Mapping)]


def _entry_value(entry: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(entry.get(key, "") or "").strip()
        if value:
            return value
    return ""


def entry_key(entry: Mapping[str, Any]) -> Tuple[str, str]:
    system = _entry_value(entry, "system_name", "system")
    frame = _entry_value(entry, "frame_name", "frame")
    if not system or not frame:
        raise ValueError("Manifest entry requires system_name/system and frame_name/frame")
    return system, frame


def entry_target_name(entry: Mapping[str, Any], targets: Mapping[str, RemoteTarget]) -> str:
    explicit = _entry_value(entry, "remote_target", "target", "target_label", "target_server_label", "host_label")
    if explicit in targets:
        return explicit
    if len(targets) == 1:
        return next(iter(targets))
    return explicit


def _copy_frames(target: RemoteTarget, rows: Iterable[Mapping[str, Any]], destination: Path) -> int:
    rows = list(rows)
    files: List[str] = []
    for row in rows:
        relative = str(row["relative"])
        files.extend([relative + "/POSCAR", relative + "/OUTCAR"])
        if row.get("tag_finished"):
            files.append(relative + "/tag_finished")
        if row.get("task_info"):
            files.append(relative + "/task_info.json")
        if row.get("frame_info"):
            files.append(relative + "/frame_info.json")
    if not files:
        return 0
    ensure_dir(destination)
    subprocess.run(
        ["rsync", "-a", "--timeout=60", "--files-from=-", "{0}:{1}/".format(target.host, target.root), str(destination) + "/"],
        input="\n".join(files) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return len(rows)


def archive_manifest(
    manifest: Mapping[str, Any],
    targets: Sequence[RemoteTarget],
    output_root: Path,
    *,
    work_root: Optional[Path] = None,
    require_tag_finished: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Scan and stage convergence-qualified frames, returning an audit report."""

    target_map = {target.name: target for target in targets}
    if len(target_map) != len(list(targets)):
        raise ValueError("Remote target names must be unique")
    records = _manifest_records(manifest)
    scans: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}
    scan_errors: Dict[str, str] = {}
    for target in targets:
        try:
            scans[target.name] = scan_target(target)
        except (OSError, subprocess.SubprocessError) as exc:
            scan_errors[target.name] = str(exc)
            scans[target.name] = {}

    duplicate_successes = []
    for key in {entry_key(entry) for entry in records}:
        successful_targets = sorted(
            name for name, rows in scans.items() if rows.get(key, {}).get("result") == "success"
        )
        if len(successful_targets) > 1:
            duplicate_successes.append({"task_key": "/".join(key), "targets": successful_targets})
    if duplicate_successes:
        raise ValueError("Converged task keys exist on multiple targets: {0}".format(duplicate_successes[:10]))

    by_system: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for entry in records:
        system, _ = entry_key(entry)
        by_system[system].append(entry)

    report: Dict[str, Any] = {
        "schema_version": "nnpgen.archive.v2",
        "manifest_kind": manifest.get("kind", ""),
        "record_count": len(records),
        "targets": [{"name": x.name, "host": x.host, "root": x.root} for x in targets],
        "scan_errors": scan_errors,
        "duplicate_successes": duplicate_successes,
        "systems": {},
        "dry_run": bool(dry_run),
    }
    staging_base = Path(work_root or (Path(output_root) / ".archive_work"))
    if dry_run:
        staging_root = staging_base
    else:
        ensure_dir(staging_base)
        staging_root = Path(tempfile.mkdtemp(prefix="archive-", dir=str(staging_base)))
    report["staging_root"] = str(staging_root)
    for system, entries in sorted(by_system.items()):
        successes: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        pending: List[Dict[str, Any]] = []
        for entry in entries:
            _, frame = entry_key(entry)
            target_name = entry_target_name(entry, target_map)
            if target_name not in target_map:
                pending.append({"frame": frame, "target": target_name, "reason": "unknown_target"})
                continue
            if target_name in scan_errors:
                pending.append({"frame": frame, "target": target_name, "reason": "scan_failed"})
                continue
            row = scans[target_name].get((system, frame))
            if row is None:
                pending.append({"frame": frame, "target": target_name, "reason": "missing_remote_frame"})
                continue
            if row["result"] == "failed":
                failures.append({"frame": frame, "target": target_name, "remote_status": row})
                continue
            marked_success = (
                row["result"] == "success"
                and row["poscar"]
                and row["outcar"]
                and row.get("ediff_reached", False)
                and row.get("normal_footer", False)
                and not row["tag_failed"]
            )
            if require_tag_finished and not row["tag_finished"]:
                marked_success = False
            if marked_success:
                successes.append(dict(row))
            else:
                pending.append({"frame": frame, "target": target_name, "reason": "not_terminal", "remote_status": row})

        rec: Dict[str, Any] = {
            "system": system,
            "expected_frames": len(entries),
            "successful_frames": len(successes),
            "failed_frames": len(failures),
            "pending_frames": len(pending),
            "failures": failures,
            "pending": pending,
            "status": "partial" if pending or failures else "ready",
        }
        if successes and not dry_run:
            staged = 0
            by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in successes:
                by_target[str(row["target"])].append(row)
            for target_name, target_rows in by_target.items():
                staged += _copy_frames(target_map[target_name], target_rows, staging_root)
            rec["staged_frames"] = staged
            rec["staging_root"] = str(staging_root)
            rec["status"] = "staged_partial" if pending or failures else "staged"
        report["systems"][system] = rec

    report["summary"] = {
        "ready_systems": sum(1 for x in report["systems"].values() if x["status"] in {"ready", "staged"}),
        "staged_systems": sum(1 for x in report["systems"].values() if x["status"] == "staged"),
        "successful_frames": sum(int(x["successful_frames"]) for x in report["systems"].values()),
        "staged_frames": sum(int(x.get("staged_frames", 0)) for x in report["systems"].values()),
        "failed_frames": sum(int(x["failed_frames"]) for x in report["systems"].values()),
        "pending_frames": sum(int(x["pending_frames"]) for x in report["systems"].values()),
    }
    return report


def add_archive_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest containing system/frame records.")
    parser.add_argument("--remote", dest="remote_specs", action="append", required=True, metavar="NAME=HOST:/ROOT", help="Named remote frame root; repeat for multiple targets.")
    parser.add_argument("--output-root", type=Path, required=True, help="Local archive output/report root.")
    parser.add_argument("--work-root", type=Path, default=None, help="Local staging root; defaults to OUTPUT_ROOT/.archive_work.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without copying frames.")
    parser.add_argument("--convert", action="store_true", help="Run the local VASP-to-DP-data converter after staging.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Validated DP dataset destination; defaults to OUTPUT_ROOT/dpdata.")
    parser.add_argument("--converter-module", default="nnpgen.dataset.vasp_sp2dpdata", help="Python module used with --convert.")
    parser.add_argument("--require-finished-task-info", action="store_true", help="Pass task_info.json completion validation to the converter.")
    parser.add_argument("--report-name", default="archive_report.json", help="Report filename under output root.")


def run_archive(args: argparse.Namespace) -> Dict[str, Any]:
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    targets = [parse_remote_spec(spec) for spec in args.remote_specs]
    report = archive_manifest(
        manifest,
        targets,
        output_root,
        work_root=args.work_root,
        require_tag_finished=True,
        dry_run=bool(args.dry_run),
    )
    report["manifest"] = str(manifest_path)
    report["output_root"] = str(output_root)
    ensure_dir(output_root)
    report_path = output_root / str(args.report_name)
    write_json(report_path, report)
    if bool(args.convert) and not bool(args.dry_run) and report["summary"]["staged_frames"]:
        work_root = Path(report["staging_root"]).expanduser().resolve()
        dataset_root = Path(args.dataset_root or (output_root / "dpdata")).expanduser().resolve()
        try:
            if dataset_root == output_root:
                raise ValueError("dataset-root must differ from output-root")
            report["conversion"] = rebuild_dp_dataset(
                work_root,
                dataset_root,
                converter_module=str(args.converter_module),
                require_finished_task_info=bool(args.require_finished_task_info),
            )
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            report["conversion"] = {"status": "failed", "error": str(exc)[-4000:]}
    elif bool(args.convert):
        report["conversion"] = {"status": "skipped", "reason": "no_staged_systems_or_dry_run"}
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return report


def rebuild_dp_dataset(
    work_root: Path,
    dataset_root: Path,
    *,
    converter_module: str = "nnpgen.dataset.vasp_sp2dpdata",
    require_finished_task_info: bool = False,
) -> Dict[str, Any]:
    """Convert all staged frames, validate them, then atomically install."""

    work_root = Path(work_root).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    ensure_dir(dataset_root.parent)
    rebuild_root = Path(tempfile.mkdtemp(prefix=".{0}.rebuild-".format(dataset_root.name), dir=str(dataset_root.parent)))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = dataset_root.with_name(dataset_root.name + ".backup-" + stamp)
    command = [
        sys.executable,
        "-m",
        converter_module,
        "--input-root",
        str(work_root),
        "--output-dir",
        str(rebuild_root),
        "--require-tag-finished",
    ]
    if require_finished_task_info:
        command.append("--require-finished-task-info")
    backup_created = False
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        require_valid_dp_dataset(rebuild_root)
        if dataset_root.exists():
            if backup_root.exists():
                raise ValueError("Backup path already exists: {0}".format(backup_root))
            os.replace(dataset_root, backup_root)
            backup_created = True
        try:
            os.replace(rebuild_root, dataset_root)
            validation = require_valid_dp_dataset(dataset_root)
        except Exception:
            if dataset_root.exists():
                os.replace(dataset_root, rebuild_root)
            if backup_created:
                os.replace(backup_root, dataset_root)
            raise
        return {
            "status": "installed",
            "dataset_root": str(dataset_root),
            "backup_root": str(backup_root) if backup_created else "",
            "validation": validation,
            "converter_stdout": result.stdout[-2000:],
        }
    finally:
        if rebuild_root.exists():
            shutil.rmtree(rebuild_root)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Archive terminal remote DFT frames into a portable work root.")
    add_archive_arguments(parser)
    args = parser.parse_args(argv)
    run_archive(args)


if __name__ == "__main__":
    main()
