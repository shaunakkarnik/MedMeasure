# KiTS23 cluster runbook

These are **prepared instructions, not executed cluster jobs**. Confirm the scheduler,
GPU allocation, account/partition, wall-time policy, storage quota, internet access
and approved environment method before running. The GPU runner currently uses one
CUDA GPU and batch size one. It makes no claim that a particular VRAM size suffices.
Do not perform preprocessing or inference on a login node without local permission.

## 1. Environment and paths

Use Python 3.11. CPU geometry checks can use a separate environment:

```bash
python3.11 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
```

Set paths appropriate to the server (examples only):

```bash
export MEDMEASURE_ROOT=/absolute/path/to/MedMeasure
export MEDMEASURE_DATA=/absolute/persistent/storage/medvision
export MEDMEASURE_MODEL=/absolute/persistent/storage/MedVision-V0-7B
export MEDMEASURE_UPSTREAM=/absolute/path/to/MedVision
```

Keep source, metadata, weights and results on durable storage. Scratch may hold
working images if its deletion policy is understood. Downloads may require the
full dataset even for a ten-row smoke test. Reserve space after checking source
archive sizes; no total storage estimate has been measured here.

For the GPU environment, follow the pinned upstream
[MedVision-V0 launcher dependency setup](https://github.com/YongchengYAO/MedVision/blob/45fa8e52684a65da6e1c0ef6ef422d5d4c3e94bd/script/benchmark-TL/eval__MedVision-V0-7B__TL.sh).
Install `medvision_bm` from the commit below and its **vendored** lmms-eval, not a
separately updated lmms-eval release. Match the upstream pinned requirements,
including `datasets==3.6.0` (the scripted dataset is incompatible with datasets 4+).
The school may require a module/container rather than a virtual environment; CUDA
installation is intentionally not automated before those requirements are known.

```bash
git clone https://github.com/YongchengYAO/MedVision.git "$MEDMEASURE_UPSTREAM"
git -C "$MEDMEASURE_UPSTREAM" checkout 45fa8e52684a65da6e1c0ef6ef422d5d4c3e94bd
```

## 2. Stage data explicitly, without a GPU

The metadata-only path needs `huggingface-hub` (the upstream environment pins
0.35.3). Run from the repository root:

```bash
python scripts/stage_data.py --data-dir "$MEDMEASURE_DATA"
python scripts/prepare_data.py \
  --plan "$MEDMEASURE_DATA/metadata/benchmark_plan_biometry_v1.4.0.json.gz" \
  --legacy-plan "$MEDMEASURE_DATA/metadata/benchmark_plan_biometry_v1.0.0.json.gz" \
  --legacy-plan "$MEDMEASURE_DATA/metadata/benchmark_plan_biometry_v1.1.1.json.gz" \
  --output runs/kits23-data-audit
```

Use a new output directory if the local audit already exists. Manifests contain
paths relative to `Datasets/KiTS23`, so they can move between machines. Case IDs,
row ordering and hashes are independent of the storage root.

When real-data storage and CPU execution are approved, run in the prepared upstream
environment on an appropriate CPU/transfer node:

```bash
python scripts/stage_data.py --data-dir "$MEDMEASURE_DATA" --with-images
```

This invokes upstream downloads, preprocessing and package installation, and builds
HF train/test caches. The script constrains the upstream loader's nested MedVision
snapshot downloads to the pinned revision; otherwise those calls default to main.
It may require configured HF authentication. It does not download the VLM or use a GPU.
Preprocessing is potentially lengthy. Do not run concurrent staging jobs in one root.

```bash
python scripts/check_geometry.py \
  --manifest runs/kits23-data-audit/train.jsonl \
  --dataset-dir "$MEDMEASURE_DATA/Datasets/KiTS23" \
  --output runs/kits23-geometry-smoke --limit 20
```

The checker samples across reference sizes. Inspect every generated overlay. Use
`--limit 0` for all manifest rows when the smoke check passes. Missing files,
alignment problems, failed fits and mismatched axes are reported as failures with
a nonzero exit code. A directory of passing synthetic results is not this gate.

## 3. Stage the model, then prepare a run

The default checkpoint is pinned to `9d7de00824e37730dd2f207d33099244bc5f2a75`
in the pilot configuration. This was resolved from public model metadata without
downloading weights. An override must be a full commit SHA, never `main`. This command downloads weights into an empty directory but does not run them:

```bash
export MEDMEASURE_MODEL_REVISION=9d7de00824e37730dd2f207d33099244bc5f2a75
python scripts/stage_model.py --revision "$MEDMEASURE_MODEL_REVISION" --output "$MEDMEASURE_MODEL"
python scripts/prepare_baseline.py \
  --upstream "$MEDMEASURE_UPSTREAM" \
  --data-dir "$MEDMEASURE_DATA" --model-path "$MEDMEASURE_MODEL" \
  --model-revision "$MEDMEASURE_MODEL_REVISION" \
  --output runs/kits23-smoke --limit 10
```

Preparation is safe on a CPU machine and does not inspect/load model weights.
Generate the run on the server so its absolute paths are valid there. Commas in the
model path are unsupported by the upstream model-argument format.

For the scored run, prepare a **different directory** and add:

```text
--sample-indices runs/kits23-data-audit/test_indices.json
```

This evaluates all 3,179 official test rows, overriding the smoke limit. For the
separate historical-disjoint diagnostic, use `test_legacy_disjoint_indices.json`.
Those 825 rows are an additional diagnostic, not a replacement official test set.

## Optional: prepare disjoint jobs for time-limited allocations

```bash
python scripts/plan_evaluation.py \
  --manifest runs/kits23-data-audit/test.jsonl \
  --output runs/kits23-evaluation-shards --shard-size 100
```

This creates 32 index files covering all 3,179 rows exactly once. The diagnostic
manifest creates 9 files covering 825 rows. These files were prepared locally; no
jobs were submitted. Treat 100 rows as an initial job size, not a runtime estimate.
Tune it after a timed smoke run. Case slices can cross job boundaries; the fixed
train/validation/test assignment does not change.

Prepare one fresh baseline run per shard using `--sample-indices` with its index
file. Do not mix official and diagnostic logs: the diagnostic overlaps the official
set and must be summarized separately. You can also derive the diagnostic from a
complete official run later; running it separately is optional.

## 4. Execute only after a GPU allocation is available

No command in this section has been executed. From an allocated GPU shell with the
prepared environment active:

```bash
export MEDMEASURE_RUN_DIR="$MEDMEASURE_ROOT/runs/kits23-smoke"
bash cluster/run_baseline.sh
```

`cluster/slurm.example.sh` is an optional template if the school uses Slurm. Adjust
its account/partition/GPU/time/memory directives and environment activation first.
No scheduler is assumed by the actual runner, and no automatic submission exists.

The preflight checks selected image/mask/landmark existence, annotation and code
hashes, declared checkpoint provenance, and HF dataset row identity/order. It
records the selected rows, package versions and GPU details. The generated run
uses offline HF mode and disables dataset code refresh; a missing cache should
fail instead of silently fetching a different revision.

The task definition inherits official prompts and scoring, with a pinned dataset
revision and a distinct task name. The direct lmms-eval CLI is intentional: the
higher-level upstream launcher only accepts index **ranges**, whereas our diagnostic
subset requires an arbitrary index list. Resolution (512×512), system prompt and
4,096-token output budget follow the upstream MedVision-V0 T/L launcher.

Runs are not automatically resumable. The launcher refuses an existing results
directory. After interruption, preserve the partial logs, prepare a fresh directory
with explicitly selected remaining indices, and reconcile identities before scoring.
Do not concatenate overlapping sample logs or silently score an incomplete run.

## 5. Summarize complete outputs

Locate the one task's samples JSONL under the run's results directory:

```bash
python scripts/summarize_baseline.py \
  --samples /absolute/path/to/task_samples.jsonl \
  --selected-rows runs/kits23-smoke/selected_rows.json \
  --output runs/kits23-smoke/summary.json
```

This consumes the upstream parser's per-row metrics rather than introducing a
second answer parser. It refuses missing/duplicate/unexpected sample identities.
MAE/MRE are successful finite rows only; below-10% rates are reported with both
all-row and successful-row denominators. MRE values are fractions, MAE is mm.
The ten-row smoke result is an execution check, not a representative baseline.

For a sharded evaluation, pass every disjoint sample log explicitly and the complete
manifest as the denominator:

```bash
python scripts/summarize_baseline.py \
  --selected-rows runs/kits23-data-audit/test.jsonl \
  --samples /absolute/path/to/shard-000-samples.jsonl \
  --samples /absolute/path/to/shard-001-samples.jsonl \
  --output runs/kits23-complete-summary.json
```

The two logs above illustrate the syntax; all 32 shard logs are needed for the
current 100-row plan. Missing rows or duplicate rows cause an error. Results are
pooled per example, rather than averaging shard means with unequal sample counts.
