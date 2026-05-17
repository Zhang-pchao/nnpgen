import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

from ..config import PROJECT_ROOT
from .submit import _collect_status_json_paths, _now, _parse_job_id, run_status
from ..utils import abs_path, read_json, write_json


def _assert_under_project(path: Path) -> None:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path is outside project root: {resolved}")


def _collect_frame_info_paths(run_root: Path) -> List[Path]:
    return sorted(run_root.glob("system_*/frame_*/frame_info.json"))


def _map_to_stage1_status(raw_status: str) -> str:
    s = str(raw_status or "").strip().lower()
    if s in {"prepared", "planned"}:
        return "planned"
    if s in {"submitted", "running", "finished", "failed"}:
        return s
    return "planned"


def _stage1_counts(run_root: Path) -> Dict[str, int]:
    paths = _collect_status_json_paths(run_root)
    counts = {
        "total": 0,
        "planned": 0,
        "submitted": 0,
        "running": 0,
        "finished": 0,
        "failed": 0,
    }
    for p in paths:
        info = read_json(p)
        status = _map_to_stage1_status(info.get("status", "planned"))
        counts["total"] += 1
        counts[status] += 1
    return counts


def _refresh_run_status(run_root: Path) -> Dict[str, int]:
    args = argparse.Namespace(run_root=str(run_root))
    run_status(args)
    return _stage1_counts(run_root)


def _submit_one_frame(info_path: Path, info: Dict, dry_run: bool) -> Tuple[str, str]:
    paths = info.get("paths", {})
    frame_dir = Path(paths.get("frame_dir", info_path.parent)).resolve()
    run_sbatch = Path(paths.get("run_sbatch", frame_dir / "run.sbatch")).resolve()
    post_sbatch = Path(paths.get("post_sbatch", frame_dir / "post.sbatch")).resolve()
    lightweight = bool(info.get("lightweight_poscar_only", False))

    if not run_sbatch.exists():
        raise FileNotFoundError(f"Missing run.sbatch: {run_sbatch}")
    if lightweight and not post_sbatch.exists():
        raise FileNotFoundError(f"Missing post.sbatch for lightweight flow: {post_sbatch}")

    if dry_run:
        return "DRY_MD", "DRY_POST" if lightweight else ""

    md_result = subprocess.run(
        ["sbatch", str(run_sbatch)],
        cwd=str(frame_dir),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    md_job_id = _parse_job_id(md_result.stdout)

    post_job_id = ""
    if lightweight:
        post_result = subprocess.run(
            ["sbatch", f"--dependency=afterok:{md_job_id}", str(post_sbatch)],
            cwd=str(frame_dir),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        post_job_id = _parse_job_id(post_result.stdout)

    return md_job_id, post_job_id


def _write_submission_update(info_path: Path, info: Dict, md_job_id: str, post_job_id: str) -> None:
    info["status"] = "submitted"
    info["job_id"] = md_job_id
    if post_job_id:
        info["post_job_id"] = post_job_id
        info["post_status"] = "submitted"
    info["updated_at"] = _now()
    write_json(info_path, info)


def add_stage1_control_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", required=True, help="11-side run root for Stage-1")
    parser.add_argument("--max-active-md", type=int, default=1, help="Bounded active MD frames (serial=1)")
    parser.add_argument("--max-new-submissions", type=int, default=1, help="Upper bound of new submissions per cycle")
    parser.add_argument("--retry-failed", action="store_true", help="Allow failed frames to be resubmitted")
    parser.add_argument("--continuous", action="store_true", help="Keep filling slots until all frames finished")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval for continuous mode")
    parser.add_argument("--dry-run", action="store_true")


def run_stage1_submit_controlled(args: argparse.Namespace) -> Dict[str, int]:
    run_root = abs_path(args.run_root)
    _assert_under_project(run_root)
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")
    if args.max_active_md <= 0:
        raise ValueError("--max-active-md must be > 0")
    if args.max_new_submissions <= 0:
        raise ValueError("--max-new-submissions must be > 0")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be > 0")

    cycle = 0
    while True:
        cycle += 1
        counts = _refresh_run_status(run_root)
        print(
            f"stage1_cycle={cycle} run_root={run_root} total={counts['total']} "
            f"planned={counts['planned']} submitted={counts['submitted']} "
            f"running={counts['running']} finished={counts['finished']} failed={counts['failed']}"
        )

        if counts["total"] == 0:
            return counts
        if counts["finished"] == counts["total"]:
            return counts

        active = counts["submitted"] + counts["running"]
        slots = max(0, int(args.max_active_md) - active)
        to_submit = min(slots, int(args.max_new_submissions))

        submitted_now = 0
        if to_submit > 0:
            frame_infos = _collect_frame_info_paths(run_root)
            for info_path in frame_infos:
                info = read_json(info_path)
                status = _map_to_stage1_status(info.get("status", "planned"))
                if status == "planned":
                    pass
                elif status == "failed" and args.retry_failed:
                    pass
                else:
                    continue

                md_job_id, post_job_id = _submit_one_frame(info_path, info, dry_run=args.dry_run)
                _write_submission_update(info_path, info, md_job_id, post_job_id)
                submitted_now += 1
                print(
                    f"submitted_frame={info_path.parent.name} system={info.get('system_name','')} "
                    f"md_job_id={md_job_id} post_job_id={post_job_id or '-'}"
                )
                if submitted_now >= to_submit:
                    break

        if not args.continuous:
            return _stage1_counts(run_root)

        latest = _stage1_counts(run_root)
        if latest["finished"] == latest["total"]:
            return latest
        if latest["planned"] == 0 and latest["submitted"] == 0 and latest["running"] == 0:
            # No more work can progress automatically (typically failed-only and retry disabled).
            return latest

        time.sleep(args.poll_seconds)


def add_stage1_progress_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", required=True, help="11-side run root for Stage-1")
    parser.add_argument("--refresh", action="store_true", help="Refresh status via Slurm query before counting")


def run_stage1_progress(args: argparse.Namespace) -> Dict[str, int]:
    run_root = abs_path(args.run_root)
    _assert_under_project(run_root)
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    if args.refresh:
        counts = _refresh_run_status(run_root)
    else:
        counts = _stage1_counts(run_root)

    print(
        f"run_root={run_root} total={counts['total']} planned={counts['planned']} "
        f"submitted={counts['submitted']} running={counts['running']} "
        f"finished={counts['finished']} failed={counts['failed']}"
    )
    return counts
