#!/bin/bash
#SBATCH --job-name=medmeasure-original-setup
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load python/3.11.9
cd "$SLURM_SUBMIT_DIR"
WORK="${MEDMEASURE_WORK:-$(dirname "$SLURM_SUBMIT_DIR")}"
text_checkpoint="${1:-$WORK/runs/v2-7087274/best.pt}"
joint_checkpoint="${2:-$WORK/runs/v2-7087275/best.pt}"
export HF_HOME="$WORK/hf-cache"
export PIP_CACHE_DIR="$WORK/pip-cache-original"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export WANDB_MODE=disabled
export DS_BUILD_OPS=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
UPSTREAM="$WORK/MedVision-original"

if [[ ! -d "$UPSTREAM/.git" ]]; then
  git clone https://github.com/YongchengYAO/MedVision.git "$UPSTREAM"
fi
git -C "$UPSTREAM" checkout --detach 45fa8e52684a65da6e1c0ef6ef422d5d4c3e94bd
if [[ -n "$(git -C "$UPSTREAM" status --porcelain)" ]]; then
  echo "Pinned upstream checkout has local modifications." >&2
  exit 1
fi
if [[ ! -x .venv-original/bin/python ]]; then
  python -m venv .venv-original
fi
PYTHON="$PWD/.venv-original/bin/python"
"$PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
# Use the same pinned environment recipe as the saved original baseline.
"$PYTHON" -m pip install --no-deps -r "$UPSTREAM/requirements/requirements_eval_medvision-v0.txt"
# Wheels are built from a temporary copy to preserve the checked source tree.
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
cp -R "$UPSTREAM/." "$BUILD/MedVision"
"$PYTHON" -m pip install --no-deps "$BUILD/MedVision"
"$PYTHON" -m pip install --no-deps "$BUILD/MedVision/src/medvision_bm/medvision_lmms_eval"
"$PYTHON" scripts/prepare_original_data.py --work "$WORK"

# Use the already-tested training environment for checkpoint deserialization.
.venv-v1/bin/python scripts/export_version2.py \
  --checkpoint "$text_checkpoint" --output "$WORK/models/v2-original-text"
.venv-v1/bin/python scripts/export_version2.py \
  --checkpoint "$joint_checkpoint" --output "$WORK/models/v2-original-joint"
"$PYTHON" -c 'import torch, transformers, vllm, datasets, lmms_eval; print(torch.__version__, transformers.__version__, vllm.__version__, datasets.__version__)'
echo "Benchmark setup and both exports completed."
