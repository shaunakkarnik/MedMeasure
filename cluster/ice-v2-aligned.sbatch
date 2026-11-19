#!/bin/bash
#SBATCH --job-name=medmeasure-aligned
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A40|A100-80GB|H100|H200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
module load python/3.11.9
cd "$SLURM_SUBMIT_DIR"
WORK="${MEDMEASURE_WORK:-$(dirname "$SLURM_SUBMIT_DIR")}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MPLBACKEND=Agg
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mode="${1:-smoke}"
options=()
if [[ "$mode" == smoke ]]; then
  model_index="$SLURM_ARRAY_TASK_ID"
  index=0
  options=(--smoke)
elif [[ "$mode" == test ]]; then
  # Submit test mode with --array=0-11%3: four shards per model.
  model_index="$((SLURM_ARRAY_TASK_ID / 4))"
  index="$((SLURM_ARRAY_TASK_ID % 4))"
else
  echo "Pass smoke or test." >&2
  exit 1
fi
models=(original-frozen v2-original-text v2-original-joint)
names=(frozen text joint)
model="${models[$model_index]}"
name="${names[$model_index]}"
output="${2:-$WORK/runs/v2-aligned-$mode-$SLURM_ARRAY_JOB_ID}"
srun .venv-original/bin/python scripts/version2_original.py evaluate \
  --upstream "$WORK/MedVision-original" --data-root "$WORK/data/medvision" \
  --audit "$WORK/audit" --model "$WORK/models/$model" \
  --output "$output/$name/shard-$index" --shard-index "$index" --shard-count 4 \
  --content-order image_first "${options[@]}"
