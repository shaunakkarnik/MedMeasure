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

## Copy the V2 files to ICE

The commands assume the existing ICE V1 checkout and its `.venv-v1`,
`data/medvision`, `audit`, and `hf-cache` under the existing scratch root.
From your Mac, run this inside the local MedMeasure repository, substituting
the same SSH destination you used to log in to ICE.

```bash
tar -czf /tmp/medmeasure-v2.tar.gz \
  configs/version2-kits23.json configs/version2-prompts.json medmeasure/version2.py scripts/version2.py \
  cluster/ice-v2.sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S docs/ice-version2.md
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
sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S --time=00:30:00 cluster/ice-v2.sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S train --objective joint --epochs 1 --limit 16
sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S --time=00:30:00 cluster/ice-v2.sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S train --objective text --epochs 1 --limit 16
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
sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S cluster/ice-v2.sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S train --objective text --epochs 1
sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S cluster/ice-v2.sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S train --objective joint --epochs 1
```

To extend a completed run, resubmit with the same objective and pass
`--resume "$WORK/runs/v2-JOBID/last.pt" --epochs 3`. The epoch count is the
total budget. An interrupted epoch repeats. Checkpoint selection uses
validation within-10% accuracy, then answer success rate and MRE.

When the training budget and settings are fixed, run the following with the
full training job IDs. All three commands use the official test selection.

```bash
sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S cluster/ice-v2.sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S evaluate
sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S cluster/ice-v2.sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S evaluate --checkpoint "$WORK/runs/v2-TEXT_JOBID/best.pt"
sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S cluster/ice-v2.sbatch --partition=ice-gpu --gres=gpu:l40s:1 --constraint=L40S evaluate --checkpoint "$WORK/runs/v2-JOINT_JOBID/best.pt"
```

Compare the all-row within-10% rate and answer success rate from
`summary.json`. The per-row `predictions.jsonl` supports paired MAE and MRE
on examples both arms successfully answered. The V1 feature cache contains
frozen features, so V2 runs the backbone on the restored scans.
