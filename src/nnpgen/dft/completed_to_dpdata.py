#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ..config import DEFAULT_REMOTE_HOST, PROJECT_ROOT, VASP15_RUN_ROOT


DEFAULT_CONTROLLER_RUN_ROOT_11 = str(PROJECT_ROOT / "run")
DEFAULT_OUTPUT_ROOT_15 = str(VASP15_RUN_ROOT / "step3_smoke_seq_dp")
DEFAULT_REMOTE_ENV_CMD = ""


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def _ssh(host: str, command: str) -> str:
    return _run(["ssh", host, command]).stdout


def _scp_to_remote(local_path: Path, host: str, remote_path: str) -> None:
    _run(["scp", str(local_path), "{0}:{1}".format(host, remote_path)])


def _scp_from_remote(host: str, remote_path: str, local_path: Path) -> None:
    _run(["scp", "{0}:{1}".format(host, remote_path), str(local_path)])


def _remote_exists(host: str, remote_path: str) -> bool:
    out = _ssh(host, "[ -e {0} ] && echo 1 || echo 0".format(shlex.quote(remote_path))).strip()
    return out == "1"


def _safe_int(v) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _is_completed_status_counts(counts: Dict[str, int]) -> bool:
    total = _safe_int(counts.get("total", 0))
    finished = _safe_int(counts.get("finished", 0))
    if total <= 0 or finished != total:
        return False
    for k in ["running", "submitted", "planned", "prepared", "selected", "transferred", "failed"]:
        if _safe_int(counts.get(k, 0)) != 0:
            return False
    return True


def _load_stage2_states(controller_run_root_11: Path) -> List[Dict[str, object]]:
    pattern = str(controller_run_root_11 / "stage2_*" / "controller" / "controller_state.json")
    rows: List[Dict[str, object]] = []
    for state_path_str in sorted(glob.glob(pattern)):
        state_path = Path(state_path_str)
        stage2_name = state_path.parent.parent.name
        try:
            data = json.loads(state_path.read_text())
        except Exception:
            continue

        counts = data.get("status_counts", {}) or {}
        remote_root = str(data.get("run_root_remote", "")).strip()

        rows.append(
            {
                "stage2_name": stage2_name,
                "state_path": str(state_path),
                "remote_root": remote_root,
                "counts": counts,
                "complete": _is_completed_status_counts(counts),
                "last_cycle_at": data.get("last_cycle_at", ""),
            }
        )
    return rows


def add_stage2_completed_to_dpdata_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host-15", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--controller-run-root-11", default=DEFAULT_CONTROLLER_RUN_ROOT_11)
    parser.add_argument("--output-root-15", default=DEFAULT_OUTPUT_ROOT_15)
    parser.add_argument("--remote-env-cmd", default=DEFAULT_REMOTE_ENV_CMD)
    parser.add_argument(
        "--converter-local",
        default=str(PROJECT_ROOT / "nnpgen" / "vasp_sp2dpdata.py"),
        help="Path on server11 to the converter script that will be copied to server15",
    )
    parser.add_argument("--max-frames-per-stage2", type=int, default=0, help="0 means all frames")
    parser.add_argument("--limit-stage2", type=int, default=0, help="0 means no limit")
    parser.add_argument("--include-regex", default="", help="Only include stage2 names matching this regex")
    parser.add_argument("--exclude-regex", default=r"(_old|_dual)", help="Exclude stage2 names matching this regex")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output dir already has set.000/coord.npy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-path", default="", help="Optional JSON summary path on server11")


def run_stage2_completed_to_dpdata(args: argparse.Namespace) -> Dict[str, object]:
    controller_run_root_11 = Path(args.controller_run_root_11)
    converter_local = Path(args.converter_local)

    if not controller_run_root_11.is_dir():
        raise ValueError("controller-run-root-11 not found: {0}".format(controller_run_root_11))
    if not converter_local.is_file():
        raise ValueError("converter-local script not found: {0}".format(converter_local))

    rows = _load_stage2_states(controller_run_root_11)

    include_re = re.compile(args.include_regex) if str(args.include_regex).strip() else None
    exclude_re = re.compile(args.exclude_regex) if str(args.exclude_regex).strip() else None

    selected: List[Dict[str, object]] = []
    for row in rows:
        name = str(row["stage2_name"])
        remote_root = str(row.get("remote_root", "")).strip()
        if not row.get("complete", False):
            continue
        if not remote_root:
            continue
        if include_re and not include_re.search(name):
            continue
        if exclude_re and exclude_re.search(name):
            continue
        selected.append(row)

    if int(args.limit_stage2) > 0:
        selected = selected[: int(args.limit_stage2)]

    remote_tool_dir = str(Path(args.output_root_15) / "_nnpgen_tools")
    remote_tool_path = remote_tool_dir + "/vasp_sp2dpdata.py"

    results: List[Dict[str, object]] = []

    if selected and not args.dry_run:
        _ssh(args.host_15, "mkdir -p {0}".format(shlex.quote(remote_tool_dir)))
        _scp_to_remote(converter_local, args.host_15, remote_tool_path)

    for row in selected:
        name = str(row["stage2_name"])
        remote_root = str(row["remote_root"])
        out_dir = str(Path(args.output_root_15) / (name + "_dpdata"))

        record: Dict[str, object] = {
            "stage2_name": name,
            "remote_root": remote_root,
            "output_dir_15": out_dir,
            "status": "pending",
            "counts": row.get("counts", {}),
        }

        if args.skip_existing and _remote_exists(args.host_15, out_dir + "/set.000/coord.npy"):
            record["status"] = "skipped_existing"
            results.append(record)
            continue

        cmd = "python {tool} --input-root {inp} --output-dir {out} --require-finished-task-info --require-tag-finished".format(
            tool=shlex.quote(remote_tool_path),
            inp=shlex.quote(remote_root),
            out=shlex.quote(out_dir),
        )
        if int(args.max_frames_per_stage2) > 0:
            cmd += " --max-frames {0}".format(int(args.max_frames_per_stage2))

        wrapped = "set -euo pipefail; mkdir -p {outroot}; rm -rf {outdir}; {env}; {cmd}".format(
            outroot=shlex.quote(args.output_root_15),
            outdir=shlex.quote(out_dir),
            env=args.remote_env_cmd,
            cmd=cmd,
        )

        if args.dry_run:
            record["status"] = "dry_run"
            record["run_cmd_15"] = wrapped
            results.append(record)
            continue

        try:
            _ssh(args.host_15, "bash -lc {0}".format(shlex.quote(wrapped)))
            remote_summary = out_dir + "/convert_summary.json"
            fd, tmp_path = tempfile.mkstemp(prefix="nnpgen_stage2_dp_", suffix=".json")
            os.close(fd)
            try:
                _scp_from_remote(args.host_15, remote_summary, Path(tmp_path))
                js = json.loads(Path(tmp_path).read_text())
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            record["frames"] = _safe_int(js.get("frames_converted", 0))
            record["group_count"] = _safe_int(js.get("group_count", 1))
            record["status"] = "converted"
        except subprocess.CalledProcessError as exc:
            record["status"] = "failed"
            record["error"] = exc.stderr[-2000:] if exc.stderr else ""
        results.append(record)

    summary = {
        "created_at": _now(),
        "host_15": args.host_15,
        "controller_run_root_11": str(controller_run_root_11),
        "output_root_15": args.output_root_15,
        "selected_stage2": len(selected),
        "converted": len([r for r in results if r.get("status") == "converted"]),
        "skipped_existing": len([r for r in results if r.get("status") == "skipped_existing"]),
        "failed": len([r for r in results if r.get("status") == "failed"]),
        "results": results,
    }

    summary_path = str(args.summary_path).strip()
    if not summary_path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        summary_path = str(PROJECT_ROOT / "summarize" / ("stage2_completed_to_dpdata_" + stamp + ".json"))

    out_path = Path(summary_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("[Saved] {0}".format(out_path))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert completed Stage-2 DFT groups to DP data on server 15")
    add_stage2_completed_to_dpdata_arguments(parser)
    args = parser.parse_args()
    run_stage2_completed_to_dpdata(args)


if __name__ == "__main__":
    main()
