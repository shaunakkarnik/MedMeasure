#!/bin/bash
#SBATCH --job-name=medmeasure-v1-cache
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

# Rerunning this job verifies completed entries and extracts only missing rows.
srun .venv-v1/bin/python scripts/version1.py cache-features \
  --data-root "$WORK/data/medvision" --manifests "$WORK/audit" \
  --output "$WORK/v1-features" --local-files-only
