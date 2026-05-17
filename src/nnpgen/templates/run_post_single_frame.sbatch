#!/bin/bash
#SBATCH -N {nodes}
#SBATCH -n {ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH -t {walltime}
#SBATCH --job-name={job_name}

set -euo pipefail

cd {project_root}

module load conda
if [ -n "{conda_env}" ]; then
  conda activate {conda_env}
fi

export OMP_NUM_THREADS={omp_num_threads}
export PYTHONPATH={project_root}:${{PYTHONPATH:-}}

{python_exe} -m nnpgen.md.postprocess \
  --frame-dir {frame_dir} \
  --frame-info {frame_info_path} \
  --frame-record {frame_record_path} \
  --pooled-poscar-path {pooled_poscar_path} \
  --lightweight-poscar-only {lightweight_poscar_only} \
  --cleanup-frame-dir {cleanup_frame_dir} \
  --delete-initial {delete_initial} \
  --delete-slurm-logs {delete_slurm_logs}
