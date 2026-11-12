# Version 2 on Georgia Tech ICE

V2 compares text-only fine-tuning with text plus auxiliary mask supervision.
Both arms start from the pinned MedVision-V0 checkpoint and train rank-8
LoRA adapters on the QKV projections of vision blocks 24–31, with the same
sizing examples, row order, and adapter settings. The joint arm adds the V1
decoder and BCE plus Dice, initially weighted by 1.0. Evaluation generates
text from the adapted VLM with the decoder removed.

The pilot uses the V1 CT rendering and the pinned baseline system and sizing
prompt templates with physical pixel spacing and tagged axis lengths. Generation
allows 4096 tokens. The parser accepts a numeric pair with optional parentheses. The local scorer uses the pinned
float16 references and the baseline's MAE, MRE, and within-10% definitions.
The rendering, decoding implementation, and local parser differ from the historical upstream run.
Evaluate frozen MedVision under the V2 protocol for a matched starting point.

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

## Copy the V2 files to ICE

The commands assume the existing ICE V1 checkout and its `.venv-v1`,
`data/medvision`, `audit`, and `hf-cache` under the existing scratch root.
From your Mac, run this inside the local MedMeasure repository, substituting
the same SSH destination you used to log in to ICE.

```bash
tar -czf /tmp/medmeasure-v2.tar.gz \
  configs/version2-kits23.json configs/version2-prompts.json medmeasure/version2.py scripts/version2.py \
  cluster/ice-v2.sbatch docs/ice-version2.md
scp /tmp/medmeasure-v2.tar.gz YOUR_ICE_SSH_DESTINATION:~/medmeasure-v2.tar.gz
```

From your ICE SSH session, set `WORK` to the full path of your own
MedMeasure scratch directory. The launcher uses the parent of the checkout
by default. Set and export `MEDMEASURE_WORK` before submitting if your data
and cache live elsewhere.

```bash
WORK=/path/to/your/ICE/scratch/medmeasure
cd "$WORK/MedMeasure"
git switch -c codex/version2
tar -xzf ~/medmeasure-v2.tar.gz
mkdir -p logs
sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S --time=00:30:00 cluster/ice-v2.sbatch train --objective joint --epochs 1 --limit 16
sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S --time=00:30:00 cluster/ice-v2.sbatch train --objective text --epochs 1 --limit 16
```

Use `squeue -u "$USER"` to monitor and `tail -f logs/medmeasure-v2-JOBID.out`
to read progress. A completed smoke run writes `epoch-1.json`, validation
predictions, `best.pt`, and `last.pt` under `$WORK/runs/v2-JOBID/`.
The small runs check execution, including backward passes, and their scores
are unsuitable for comparing the objectives.

## Train and evaluate

Once both smoke runs complete, start both arms from the original checkpoint.
One full epoch per arm gives an initial matched comparison and measures runtime
before committing more GPU hours.

```bash
sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S cluster/ice-v2.sbatch train --objective text --epochs 1
sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S cluster/ice-v2.sbatch train --objective joint --epochs 1
```

To extend a completed run, resubmit with the same objective and pass
`--resume "$WORK/runs/v2-JOBID/last.pt" --epochs 3`. The epoch count is the
total budget. An interrupted epoch repeats. Checkpoint selection uses
validation within-10% accuracy, then answer success rate and MRE.

When the training budget and settings are fixed, run the following with the
full training job IDs. All three commands use the official test selection.

```bash
sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S cluster/ice-v2.sbatch evaluate
sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S cluster/ice-v2.sbatch evaluate --checkpoint "$WORK/runs/v2-TEXT_JOBID/best.pt"
sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S cluster/ice-v2.sbatch evaluate --checkpoint "$WORK/runs/v2-JOINT_JOBID/best.pt"
```

Compare the all-row within-10% rate and answer success rate from
`summary.json`. The per-row `predictions.jsonl` supports paired MAE and MRE
on examples both arms successfully answered. The V1 feature cache contains
frozen features, so V2 runs the backbone on the restored scans.
