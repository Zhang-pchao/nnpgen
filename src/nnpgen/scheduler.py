"""Scheduler output parsers shared by PBS and SLURM workflows."""

from __future__ import annotations

import re
from typing import Dict, List, Optional


def parse_sbatch_job_id(stdout: str) -> str:
    match = re.search(r"Submitted batch job\s+(\d+)", stdout.strip())
    if match:
        return match.group(1)
    match = re.search(r"(\d+)\s*$", stdout.strip())
    if match:
        return match.group(1)
    raise ValueError(f"Could not parse sbatch job id from: {stdout!r}")


def parse_qsub_job_id(stdout: str) -> str:
    text = stdout.strip()
    if not text:
        raise ValueError("Empty qsub output")
    return text.split()[0].split(".")[0]


def map_pbs_state(job_state: str, exit_status: Optional[int] = None) -> str:
    state = (job_state or "").strip().upper()
    if state in {"Q", "H", "W"}:
        return "submitted"
    if state in {"R", "E"}:
        return "running"
    if state in {"C", "F"}:
        return "finished" if exit_status in (None, 0) else "failed"
    return "unknown"


def map_slurm_state(state: str) -> str:
    normalized = (state or "").strip().upper()
    if normalized in {"PENDING", "CONFIGURING", "COMPLETING"}:
        return "submitted"
    if normalized in {"RUNNING", "RESIZING"}:
        return "running"
    if normalized in {"COMPLETED"}:
        return "finished"
    if normalized in {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}:
        return "failed"
    return "unknown"


def parse_squeue_table(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("jobid"):
            continue
        parts = line.split(None, 4)
        if len(parts) >= 4:
            rows.append({"job_id": parts[0], "partition": parts[1], "name": parts[2], "state": parts[3]})
    return rows


def parse_qstat_table(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.lower().startswith("job"):
            continue
        parts = stripped.split()
        if len(parts) >= 5 and re.match(r"^\d+", parts[0]):
            rows.append({"job_id": parts[0].split(".")[0], "name": parts[3], "state": parts[4]})
    return rows
