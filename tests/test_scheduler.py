from nnpgen.scheduler import (
    map_pbs_state,
    map_slurm_state,
    parse_qstat_full,
    parse_qstat_table,
    parse_qsub_job_id,
    parse_sbatch_job_id,
    parse_scontrol_jobs,
    parse_squeue_table,
    plan_pbs_slots,
    summarize_scheduler_jobs,
    task_key_from_workdir,
)


def test_job_id_parsers():
    assert parse_sbatch_job_id("Submitted batch job 12345\n") == "12345"
    assert parse_sbatch_job_id("12345") == "12345"
    assert parse_qsub_job_id("98765.server\n") == "98765"


def test_state_mapping():
    assert map_pbs_state("Q") == "submitted"
    assert map_pbs_state("R") == "running"
    assert map_pbs_state("C", 0) == "finished"
    assert map_pbs_state("C", 1) == "failed"
    assert map_slurm_state("PENDING") == "submitted"
    assert map_slurm_state("RUNNING") == "running"
    assert map_slurm_state("COMPLETED") == "finished"
    assert map_slurm_state("FAILED") == "failed"
    assert map_slurm_state("TIMEOUT") == "failed"
    assert map_slurm_state("CANCELLED+") == "failed"


def test_scheduler_table_parsers():
    squeue = "JOBID PARTITION NAME ST TIME\n123 standard vasp RUNNING 1:00\n"
    qstat = "Job id Name User Time Use S Queue\n----- ---- ---- -------- - -----\n456.server user job R normal\n"

    assert parse_squeue_table(squeue)[0]["job_id"] == "123"
    assert parse_qstat_table(qstat)[0]["job_id"] == "456"


def test_full_scheduler_parsers_and_task_keys():
    qstat = """Job Id: 456.server
    Job_Name = vasp
    job_state = R
    queue = normal
    Variable_List = A=1,PBS_O_WORKDIR=/work/run/system_001/frame_000010,
        B=2
"""
    scontrol = "JobId=789 JobName=vasp JobState=PENDING Partition=compute WorkDir=/work/run/system_002/frame_000020\n"

    assert parse_qstat_full(qstat)[0]["workdir"].endswith("frame_000010")
    assert parse_scontrol_jobs(scontrol)[0]["state"] == "PENDING"
    assert task_key_from_workdir("/work/run/system_001/frame_000010", "/work/run") == "system_001/frame_000010"


def test_scheduler_summary_counts_physical_and_duplicate_jobs():
    jobs = [
        {"job_id": "1", "state": "R", "workdir": "/run/system_001/frame_000001"},
        {"job_id": "2", "state": "Q", "workdir": "/run/system_001/frame_000001"},
        {"job_id": "3", "state": "Q", "workdir": "/other/system_002/frame_000002"},
    ]

    summary = summarize_scheduler_jobs(jobs, "/run", "pbs")
    assert summary["physical_jobs"] == 2
    assert summary["unique_tasks"] == 1
    assert summary["duplicate_jobs"] == 1
    assert summary["running"] == 1
    assert summary["queued"] == 1


def test_pbs_slot_plan_applies_reserves_avoidance_and_active_limit():
    nodes = [
        {"node": "a1", "queue": "a", "state": "free", "cores": 16, "running_tasks": 0},
        {"node": "a2", "queue": "a", "state": "free", "cores": 16, "running_tasks": 0},
        {"node": "b1", "queue": "b", "state": "busy", "cores": 24, "running_tasks": 24},
        {"node": "b2", "queue": "b", "state": "free", "cores": 24, "running_tasks": 0},
    ]

    slots = plan_pbs_slots(
        nodes,
        queue_reserve={"a": 1},
        avoid_nodes={"b2"},
        max_active=4,
        active_jobs=1,
        queued_by_queue={"b": 0},
    )
    assert slots[0] == {"queue": "a", "node": "a2", "cores": 16, "queued_slot": False}
    assert slots[1:] == [
        {"queue": "b", "node": "", "cores": 24, "queued_slot": True},
        {"queue": "b", "node": "", "cores": 24, "queued_slot": True},
    ]
