"""Pure scheduler parsing and capacity-planning helpers."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


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
    if state in {"Q", "H", "W", "T"}:
        return "submitted"
    if state in {"R", "E"}:
        return "running"
    if state in {"C", "F"}:
        return "finished" if exit_status in (None, 0) else "failed"
    return "unknown"


def map_slurm_state(state: str) -> str:
    normalized = re.split(r"[\s+]", (state or "").strip().upper(), maxsplit=1)[0]
    if normalized in {"PENDING", "CONFIGURING"}:
        return "submitted"
    if normalized in {"RUNNING", "RESIZING", "COMPLETING", "SUSPENDED"}:
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


def parse_qstat_full(text: str) -> List[Dict[str, str]]:
    """Parse ``qstat -f`` records, including each job's working directory."""

    jobs: List[Dict[str, str]] = []
    for block in re.split(r"(?:^|\n)Job Id: ", str(text or ""))[1:]:
        lines = block.splitlines()
        if not lines:
            continue
        fields: Dict[str, str] = {}
        current = ""
        for line in lines[1:]:
            stripped = line.strip()
            if " = " in line:
                current, value = stripped.split(" = ", 1)
                fields[current] = value
            elif current and (line.startswith("\t") or line.startswith("        ")):
                fields[current] += stripped
        variables = fields.get("Variable_List", "")
        match = re.search(r"(?:^|,)PBS_O_WORKDIR=([^,]+)", variables)
        jobs.append(
            {
                "job_id": lines[0].strip(),
                "name": fields.get("Job_Name", ""),
                "state": fields.get("job_state", "").upper(),
                "queue": fields.get("queue", ""),
                "workdir": match.group(1).strip() if match else "",
            }
        )
    return jobs


def parse_scontrol_jobs(text: str) -> List[Dict[str, str]]:
    """Parse one-line ``scontrol show job -o`` output."""

    jobs: List[Dict[str, str]] = []
    for line in str(text or "").splitlines():
        fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]*)", line))
        if not fields.get("JobId"):
            continue
        jobs.append(
            {
                "job_id": fields["JobId"],
                "name": fields.get("JobName", ""),
                "state": fields.get("JobState", "").upper(),
                "queue": fields.get("Partition", ""),
                "workdir": fields.get("WorkDir", ""),
            }
        )
    return jobs


def task_key_from_workdir(workdir: str, run_root: str) -> str:
    """Return the first two path components below a configured run root."""

    root = str(run_root or "").rstrip("/")
    path = str(workdir or "").rstrip("/")
    if not root or not path.startswith(root + "/"):
        return ""
    parts = path[len(root) + 1 :].split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 and all(parts[:2]) else ""


def summarize_scheduler_jobs(
    jobs: Iterable[Mapping[str, str]],
    run_root: str,
    backend: str,
) -> Dict[str, Any]:
    """Summarize physical jobs and unique manifest-style task keys."""

    mapper = map_pbs_state if backend.lower() == "pbs" else map_slurm_state
    matched: List[Dict[str, str]] = []
    keys: Counter[str] = Counter()
    states: Counter[str] = Counter()
    for job in jobs:
        task_key = task_key_from_workdir(job.get("workdir", ""), run_root)
        if not task_key:
            continue
        row = dict(job)
        row["task_key"] = task_key
        matched.append(row)
        keys[task_key] += 1
        states[mapper(job.get("state", ""))] += 1
    return {
        "physical_jobs": len(matched),
        "unique_tasks": len(keys),
        "duplicate_jobs": sum(count - 1 for count in keys.values() if count > 1),
        "running": states["running"],
        "queued": states["submitted"],
        "finished": states["finished"],
        "failed": states["failed"],
        "unknown": states["unknown"],
        "task_keys": sorted(keys),
    }


def plan_pbs_slots(
    nodes: Sequence[Mapping[str, Any]],
    *,
    queue_reserve: Optional[Mapping[str, int]] = None,
    avoid_nodes: Optional[Set[str]] = None,
    max_active: int = 0,
    active_jobs: int = 0,
    queued_by_queue: Optional[Mapping[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Plan free-node and bounded queued PBS slots from caller-supplied facts.

    Node records require ``node``, ``queue``, ``state``, and ``cores``.  The
    caller owns site-specific node discovery; this function only applies the
    generic reservation and load-balancing policy.
    """

    reserves = {str(key): max(0, int(value)) for key, value in (queue_reserve or {}).items()}
    avoided = set(avoid_nodes or set())
    usable: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    free: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for node in nodes:
        name = str(node.get("node", ""))
        queue = str(node.get("queue", ""))
        state = str(node.get("state", "")).lower().replace("*", "")
        if not name or not queue or name in avoided or state in {"down", "offline", "offl"}:
            continue
        usable[queue].append(node)
        if state == "free" and int(node.get("running_tasks", 0) or 0) <= 1:
            free[queue].append(node)

    slots: List[Dict[str, Any]] = []
    for queue in sorted(free):
        candidates = sorted(free[queue], key=lambda row: str(row.get("node", "")))
        for node in candidates[reserves.get(queue, 0) :]:
            slots.append(
                {
                    "queue": queue,
                    "node": str(node["node"]),
                    "cores": int(node.get("cores", 0) or 0),
                    "queued_slot": False,
                }
            )

    capacity = max(0, int(max_active) - int(active_jobs)) if int(max_active) > 0 else len(slots)
    slots = slots[:capacity]
    remaining = capacity - len(slots)
    queue_order = [queue for queue in sorted(usable) if reserves.get(queue, 0) == 0 and usable[queue]]
    queued = Counter({str(key): int(value) for key, value in (queued_by_queue or {}).items()})
    assigned: Counter[str] = Counter()
    for _ in range(remaining):
        if not queue_order:
            break
        queue = min(
            queue_order,
            key=lambda name: ((queued[name] + assigned[name]) / len(usable[name]), name),
        )
        cores = max(int(node.get("cores", 0) or 0) for node in usable[queue])
        slots.append({"queue": queue, "node": "", "cores": cores, "queued_slot": True})
        assigned[queue] += 1
    return slots
