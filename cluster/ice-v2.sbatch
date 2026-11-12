#!/bin/bash
#SBATCH --job-name=medmeasure-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=HX00
#SBATCH --mem-per-gpu=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --exclude=atl1-1-03-013-3-0

set -euo pipefail
module load python/3.11.9
WORK="${MEDMEASURE_WORK:-$(dirname "$SLURM_SUBMIT_DIR")}"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$WORK/hf-cache"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

command="${1:?Pass train or evaluate followed by its options}"
shift
git rev-parse HEAD
srun .venv-v1/bin/python scripts/version2.py "$command" \
  --data-root "$WORK/data/medvision" --manifests "$WORK/audit" \
  --output "$WORK/runs/v2-$SLURM_JOB_ID" --local-files-only "$@"
