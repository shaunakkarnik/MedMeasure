#!/usr/bin/env bash
# EXAMPLE ONLY: confirm partition, account, GPU syntax, memory and wall time locally.
# Do not submit until the cluster administrator's requirements are known.
#SBATCH --job-name=medmeasure-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=medmeasure-%j.log
set -euo pipefail
: "${MEDMEASURE_ROOT:?Set absolute repository path}"
: "${MEDMEASURE_ENV:?Set absolute Python environment path}"
source "${MEDMEASURE_ENV}/bin/activate"
bash "${MEDMEASURE_ROOT}/cluster/run_baseline.sh"
