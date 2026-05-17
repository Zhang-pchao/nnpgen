#!/bin/bash
#SBATCH -N {nodes}
#SBATCH -n {ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH -t {walltime}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --job-name={job_name}

set -euo pipefail

cd {frame_dir}

MPS_MAP="${{NNPGEN_MPS_MAP:-}}"
if [ -n "$MPS_MAP" ] && [ -f "$MPS_MAP" ]; then
  source "$MPS_MAP"
fi

module load conda
if [ -n "{conda_env}" ]; then
  conda activate {conda_env}
fi

export OMP_NUM_THREADS={omp_num_threads}
export PYTHONPATH={project_root}:${{PYTHONPATH:-}}

{python_exe} -m nnpgen.md.mace \
  --input {input_xyz} \
  --output-xyz {output_xyz} \
  --final-xyz {final_xyz} \
  --final-poscar {final_poscar} \
  --steps {steps} \
  --temperature {temperature} \
  --timestep-fs {timestep_fs} \
  --friction {friction} \
  --interval {interval} \
  --device {device} \
  --model-path {model_path}
