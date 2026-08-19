import subprocess

from nnpgen.monitor.schedulers import SchedulerTarget, build_scheduler_summary, parse_target_spec


def test_parse_scheduler_target_is_explicit_and_portable():
    target = parse_target_spec("primary=slurm@login.example:/work/dft")
    assert target == SchedulerTarget("primary", "slurm", "login.example", "/work/dft")


def test_cross_scheduler_summary_keeps_unavailable_distinct_from_zero():
    targets = [
        SchedulerTarget("pbs", "pbs", "pbs.example", "/work/dft"),
        SchedulerTarget("slurm", "slurm", "slurm.example", "/work/dft"),
    ]

    def query(target):
        if target.backend == "slurm":
            raise subprocess.CalledProcessError(1, ["ssh"])
        return [{"job_id": "1", "state": "R", "workdir": "/work/dft/system_001/frame_000001"}]

    summary = build_scheduler_summary(targets, query=query)
    assert summary["status"] == "partial"
    assert summary["available_targets"] == 1
    assert summary["unavailable_targets"] == 1
    assert summary["aggregate_available"]["running"] == 1
    assert summary["targets"][1]["status"] == "unavailable"
    assert "running" not in summary["targets"][1]
