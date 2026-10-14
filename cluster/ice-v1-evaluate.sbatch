#!/bin/bash
#SBATCH --job-name=medmeasure-v1-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=HX00
#SBATCH --mem-per-gpu=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load python/3.11.9
export WORK=/storage/ice1/4/3/skarnik35/medmeasure
cd "$WORK/MedMeasure"
export HF_HOME="$WORK/hf-cache"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Default is a ten-row smoke run; pass 0 for the complete official test.
LIMIT="${1:-10}"
CHECKPOINT="${MEDMEASURE_CHECKPOINT:-$WORK/runs/v1-cached-6251584/best.pt}"
echo "Checkpoint: $CHECKPOINT"
echo "Evaluation row limit: $LIMIT (0 means full official test)"
git rev-parse HEAD
srun .venv-v1/bin/python scripts/version1.py evaluate \
  --data-root "$WORK/data/medvision" --manifests "$WORK/audit" \
  --checkpoint "$CHECKPOINT" --limit "$LIMIT" --local-files-only \
  --output "$WORK/runs/v1-eval-$SLURM_JOB_ID"
