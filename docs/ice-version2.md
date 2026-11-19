# V2 training on Georgia Tech ICE

Both objectives use the same KiTS23 training examples and vision adapters.
See [V2 model and losses](version2.md).

## Prepare a fresh ICE environment

Place the checkout at `$WORK/MedMeasure`. Place prepared scans under
`$WORK/data/medvision/Datasets/KiTS23` and annotation plans under
`$WORK/data/medvision/metadata`. Restore a portable archive using the
[prepared-data guide](local-to-colab.md). A format-1 Colab backup must have
its checksum verified before extracting `./Datasets` and `./metadata`
into this layout. It records its original Colab path.

From the checkout, create the environment and build the split manifests.
Use your own scratch directory for `WORK`.

```bash
WORK="$(cd .. && pwd)"
module load python/3.11.9
python -m venv .venv-v1
.venv-v1/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv-v1/bin/python -m pip install -r requirements-v1.txt matplotlib==3.9.2

.venv-v1/bin/python scripts/prepare_data.py \
  --plan "$WORK/data/medvision/metadata/benchmark_plan_biometry_v1.4.0.json.gz" \
  --legacy-plan "$WORK/data/medvision/metadata/benchmark_plan_biometry_v1.0.0.json.gz" \
  --legacy-plan "$WORK/data/medvision/metadata/benchmark_plan_biometry_v1.1.1.json.gz" \
  --output "$WORK/audit"
```

The manifest builder requires a new output directory. The expected row counts
are 6,897 train, 1,263 validation, and 3,179 test. Download model weights once
from a node with network access before submitting offline GPU jobs.

```bash
export HF_HOME="$WORK/hf-cache"
.venv-v1/bin/python -u - <<'PYTHON'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

pin = json.loads(Path("configs/kits23-pilot.json").read_text())
print(snapshot_download(pin["model_repo"], revision=pin["model_revision"], max_workers=1))
PYTHON
```

## Train or resume

Run from the checkout. Launchers use the checkout's parent as the scratch root,
or `MEDMEASURE_WORK` if set to an absolute path.

```bash
mkdir -p logs
sbatch cluster/ice-v2.sbatch train --objective text --epochs 3
sbatch cluster/ice-v2.sbatch train --objective joint --epochs 3
```

For a quick execution check, add `--limit 16 --epochs 1` in place of
`--epochs 3`. Each job writes validation predictions, epoch summaries,
`best.pt`, and `last.pt` under `$WORK/runs/v2-JOBID/`.

To extend a run, pass `--resume "$WORK/runs/v2-JOBID/last.pt"` with the same
objective and settings and set `--epochs` to the total epoch budget.
An interrupted epoch repeats. Use the [final evaluation runbook](version2-aligned.md)
to export and score the selected checkpoints.
