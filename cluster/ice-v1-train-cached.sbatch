#!/bin/bash
#SBATCH --job-name=medmeasure-v1-cached
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
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Time one complete epoch before selecting a longer training schedule.
srun .venv-v1/bin/python scripts/version1.py train \
  --data-root "$WORK/data/medvision" --manifests "$WORK/audit" \
  --feature-cache "$WORK/v1-features" \
  --output "$WORK/runs/v1-cached-$SLURM_JOB_ID" --epochs 1
