#!/bin/bash
#SBATCH --job-name=medmeasure-gpu-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=HX00
#SBATCH --mem-per-gpu=96G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load python/3.11.9
cd /storage/ice1/4/3/skarnik35/medmeasure/MedMeasure
export PYTHONUNBUFFERED=1

# Inspect GPU visibility and exercise CUDA in the same Slurm step/process.
srun .venv-v1/bin/python - <<'PY'
import os
import socket
import subprocess
import torch

print('Node:', socket.gethostname())
print('Job:', os.environ.get('SLURM_JOB_ID'))
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('Torch:', torch.__version__, 'Runtime:', torch.version.cuda)
subprocess.run(['nvidia-smi'], check=True)
print('Visible CUDA devices:', torch.cuda.device_count())
torch.cuda.init()
print('Selected GPU:', torch.cuda.get_device_name(0))
# Test host-to-device transfer, allocation, bfloat16 compute and return transfer.
values = torch.ones((32, 32), dtype=torch.bfloat16).to('cuda:0')
product = values @ values
torch.cuda.synchronize()
assert torch.equal(product.cpu(), torch.full((32, 32), 32, dtype=torch.bfloat16))
print('PASS: CUDA context, transfers and bfloat16 computation work.')
PY
