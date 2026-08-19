"""Read-only PBS and Slurm monitoring for explicitly configured run roots."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..scheduler import parse_qstat_full, parse_scontrol_jobs, summarize_scheduler_jobs
from ..utils import write_json


@dataclass(frozen=True)
class SchedulerTarget:
    name: str
    backend: str
    host: str
    run_root: str


def parse_target_spec(spec: str) -> SchedulerTarget:
    """Parse ``NAME=BACKEND@HOST:/ABSOLUTE/RUN_ROOT``."""

    try:
        name, endpoint = str(spec).split("=", 1)
        backend, remote = endpoint.split("@", 1)
        host, run_root = remote.split(":", 1)
    except ValueError as exc:
        raise ValueError("Target must use NAME=BACKEND@HOST:/RUN_ROOT: {0}".format(spec)) from exc
    name, backend, host, run_root = (value.strip() for value in (name, backend, host, run_root))
    backend = backend.lower()
    if not name or backend not in {"pbs", "slurm"} or not host or not run_root.startswith("/"):
        raise ValueError("Target must use NAME=BACKEND@HOST:/RUN_ROOT: {0}".format(spec))
    return SchedulerTarget(name, backend, host, run_root.rstrip("/") or "/")


def query_scheduler(target: SchedulerTarget, timeout: int = 60) -> List[Dict[str, str]]:
    """Query one scheduler without changing jobs or remote files."""

    ssh = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=2",
        target.host,
    ]
    if target.backend == "slurm":
        ping = subprocess.run(
            ssh + ["scontrol ping"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=True,
        )
        if "UP" not in ping.stdout.upper():
            raise RuntimeError("Slurm control plane is unavailable")
        command = "scontrol show job -o"
        parser = parse_scontrol_jobs
    else:
        command = "qstat -f"
        parser = parse_qstat_full
    result = subprocess.run(
        ssh + [command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=True,
    )
    return parser(result.stdout)


def build_scheduler_summary(
    targets: Sequence[SchedulerTarget],
    query: Callable[[SchedulerTarget], List[Dict[str, str]]] = query_scheduler,
) -> Dict[str, Any]:
    """Build a compact cross-scheduler snapshot; failures remain explicit."""

    rows: List[Dict[str, Any]] = []
    aggregate = {key: 0 for key in ("physical_jobs", "unique_tasks", "duplicate_jobs", "running", "queued")}
    unavailable = 0
    for target in targets:
        base = {
            "name": target.name,
            "backend": target.backend,
            "host": target.host,
            "run_root": target.run_root,
        }
        try:
            metrics = summarize_scheduler_jobs(query(target), target.run_root, target.backend)
            row = {**base, "status": "available", **metrics}
            for key in aggregate:
                aggregate[key] += int(metrics[key])
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            unavailable += 1
            row = {**base, "status": "unavailable", "error": str(exc)[-500:]}
        rows.append(row)
    status = "unavailable" if targets and unavailable == len(targets) else ("partial" if unavailable else "available")
    return {
        "schema_version": "nnpgen.scheduler_summary.v1",
        "status": status,
        "targets": rows,
        "available_targets": len(targets) - unavailable,
        "unavailable_targets": unavailable,
        "aggregate_available": aggregate,
    }


def add_scheduler_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target",
        dest="target_specs",
        action="append",
        required=True,
        metavar="NAME=BACKEND@HOST:/RUN_ROOT",
        help="Scheduler and run root to inspect; repeat for multiple targets.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")


def run_scheduler_summary(args: argparse.Namespace) -> Dict[str, Any]:
    summary = build_scheduler_summary([parse_target_spec(spec) for spec in args.target_specs])
    if args.output:
        output = args.output.expanduser().resolve()
        write_json(output, summary)
        summary["output"] = str(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize PBS and Slurm jobs by configured run root.")
    add_scheduler_arguments(parser)
    run_scheduler_summary(parser.parse_args(argv))


if __name__ == "__main__":
    main()
