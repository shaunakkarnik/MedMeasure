#!/bin/bash
#SBATCH --job-name=medmeasure-v2-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=L40S
#SBATCH --mem-per-gpu=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-11%3
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
module load python/3.11.9
cd "$SLURM_SUBMIT_DIR"
WORK="${MEDMEASURE_WORK:-$(dirname "$SLURM_SUBMIT_DIR")}"
export HF_HOME="$WORK/hf-cache"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Four disjoint shards per model; run at most three tasks simultaneously.
text_checkpoint="${1:?Pass the selected text checkpoint}"
joint_checkpoint="${2:?Pass the selected joint checkpoint}"
output="${3:-$WORK/runs/v2-official-$SLURM_ARRAY_JOB_ID}"
models=(frozen text joint)
model="${models[$((SLURM_ARRAY_TASK_ID / 4))]}"
index="$((SLURM_ARRAY_TASK_ID % 4))"
options=()
if [[ "$model" == text ]]; then
  options=(--checkpoint "$text_checkpoint")
elif [[ "$model" == joint ]]; then
  options=(--checkpoint "$joint_checkpoint")
fi

srun .venv-v1/bin/python scripts/version2_test.py evaluate \
  --data-root "$WORK/data/medvision" --manifests "$WORK/audit" \
  --output "$output/$model/shard-$index" \
  --shard-index "$index" --shard-count 4 --local-files-only "${options[@]}"
