from nnpgen.scheduler import (
    map_pbs_state,
    map_slurm_state,
    parse_qstat_table,
    parse_qsub_job_id,
    parse_sbatch_job_id,
    parse_squeue_table,
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


def test_scheduler_table_parsers():
    squeue = "JOBID PARTITION NAME ST TIME\n123 standard vasp RUNNING 1:00\n"
    qstat = "Job id Name User Time Use S Queue\n----- ---- ---- -------- - -----\n456.server user job R normal\n"

    assert parse_squeue_table(squeue)[0]["job_id"] == "123"
    assert parse_qstat_table(qstat)[0]["job_id"] == "456"
