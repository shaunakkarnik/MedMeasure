#!/bin/bash
#SBATCH --job-name=medmeasure-v1-overfit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=HX00
#SBATCH --mem-per-gpu=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load python/3.11.9
export WORK=/storage/ice1/4/3/skarnik35/medmeasure
cd "$WORK/MedMeasure"
export HF_HOME="$WORK/hf-cache"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

srun nvidia-smi
srun .venv-v1/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()'
git rev-parse HEAD
srun .venv-v1/bin/python scripts/version1.py train \
  --data-root "$WORK/data/medvision" --manifests "$WORK/audit" \
  --output "$WORK/runs/v1-overfit-$SLURM_JOB_ID" \
  --overfit-rows 4 --epochs 100 --local-files-only
