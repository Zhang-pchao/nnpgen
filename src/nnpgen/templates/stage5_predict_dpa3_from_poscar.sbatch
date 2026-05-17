#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH -t 24:00:00
#SBATCH --gpus-per-node=3090:1
#SBATCH -J s5_dpa3_predict
#SBATCH -o stage5_predict.%j.out
#SBATCH -e stage5_predict.%j.err

set -euo pipefail

# Required inputs (override via --export=ALL,... in sbatch)
INPUT_ROOT="${INPUT_ROOT:?INPUT_ROOT is required}"
MODEL="${MODEL:?MODEL is required}"
OUTPUT_DP_DIR="${OUTPUT_DP_DIR:?OUTPUT_DP_DIR is required}"

# Optional inputs
BATCH_SIZE="${BATCH_SIZE:-8}"
DP2XYZ="${DP2XYZ:-1}"
CONDA_ENV="${CONDA_ENV:-${NNPGEN_CONDA_ENV:-}}"

module load conda
if [ -n "$CONDA_ENV" ]; then
  conda activate "$CONDA_ENV"
fi

export OMP_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=1
export DP_INTER_OP_PARALLELISM_THREADS=1

python -m nnpgen.train.predict_poscar \
  --input-root "$INPUT_ROOT" \
  --output-dir "$OUTPUT_DP_DIR" \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE"

if [ "$DP2XYZ" = "1" ]; then
  python -m nnpgen.dataset.dpdata_to_extxyz \
    --dataset-dir "$OUTPUT_DP_DIR" \
    --output-xyz "$OUTPUT_DP_DIR/DEEP2XYZ.xyz"
fi
