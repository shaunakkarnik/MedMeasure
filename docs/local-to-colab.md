# Prepare KiTS23 locally, run MedVision-V0 inference in Colab

Use a local **CPU-only** environment to download/preprocess scans, validate reference
geometry and create a portable archive. Upload the archive to Drive before starting
a GPU runtime. Colab restores it to local disk and rebuilds only the small test-row
metadata cache using the unchanged pinned loader; it never falls back to scan
staging in this workflow.

This evaluates the existing KiTS23 v1.4.0 axial sizing baseline (3,179 test rows),
plus the 825-row/43-case diagnostic from the same predictions. Detection is not yet
implemented. The full split has potential checkpoint training exposure; the diagnostic
is not proven unseen across all training tasks.

## 1. Commit the pipeline, then prepare locally

Commit and push the new scripts, notebook, configuration and documentation before
running, so collaborators and remote inference use the same code. Record the full
commit SHA with `git rev-parse HEAD`. Use a new run name for this route rather than
changing the code/configuration of an existing sizing run.

The following commands work from the repository on macOS or Linux. They use `uv`
(available on the project owner's Mac) to create an isolated Python 3.11 environment;
if you don't have it, follow the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
No Docker, GPU, torch, vLLM, or model weights are needed for local data preparation.

```bash
cd /Users/shaunakkarnik/Desktop/CodingProjects/MedMeasure
uv venv --python 3.11 --seed .venv-data
uv pip install --python .venv-data/bin/python -r requirements-local-data.txt
```

Substitute your checkout path if different. Check available disk space before the
large transfer:

```bash
df -h .
```

The upstream loader fetches the full KiTS23-Lite scan archive and performs its
standard RAS+ conversion. Peak storage includes the downloaded scan zip, extracted
scans, and later a portable tar alongside the staged data. Total size, runtime and
macOS performance have not been measured with real scans here. More CPU workers
can use substantially more RAM; start with two and lower to one if needed.

Start preparation:

```bash
.venv-data/bin/python scripts/prepare_local_data.py --workers 2
```

Defaults:

- Processed data: `data/medvision/`.
- Audit, 20-example geometry overlays and package versions: `runs/local-data-preparation/`.
- Portable bundle: `data/exports/kits23-f4040ed7d2d2b45e09c1a996ad2969f2018051d3-portable-v2.tar`.
- Completed sidecar: the same filename followed by `.json`.

These directories are ignored by Git. To use a larger/external disk, set
`--data-dir`, `--output` and `--archive` explicitly. The archive must be outside the
staged data directory. Paths without commas are required by the evaluator later.

The command stages pinned source/annotations/scans, builds upstream train/test
metadata, prepares case-separated manifests, checks 20 geometry examples and
packages the data. It prints per-phase times and uses `--no-deps` for the pinned
dataset package because the CPU requirements already cover KiTS23's imports; it
avoids installing unrelated dataset integrations. It does not run the VLM.

**Review every local geometry PNG before using the bundle for evaluation.** A failed
geometry check stops packaging. A passing numeric check still requires visual review.
If a previous audit/geometry directory is incomplete or mismatched, preserve it and
use a new `--output` directory. Repeated runs reuse available staging files; this is
not a guarantee of row-level or mid-conversion resumption after interruption.

## 2. Upload to Drive once

Create `MyDrive/MedMeasure/cache/` in your Google Drive. Upload both the `.tar` and
`.tar.json` files from `data/exports/`. Check Drive quota and wait for both uploads
to finish before allocating a GPU.

This new **portable format 2** includes processed scans/masks, official plans,
landmarks, pinned source/loader, upstream completion markers and local validation
provenance. It excludes machine-specific Hugging Face caches and tokens. Absolute
Arrow filenames are regenerated at the destination rather than patched in-place.
Model weights are not included and still need staging on Colab.

Old `kits23-<revision>-v140.tar` backups are format 1 and remain usable only at their
original data root with the original notebook. They cannot substitute for this
portable-v2 bundle.

## 3A. Browser notebook route

Open [the local-data inference notebook in Colab](https://colab.research.google.com/github/shaunakkarnik/MedMeasure/blob/main/notebooks/medvision_v0_kits23_local_data.ipynb),
then **File → Save a copy in Drive**. Choose **A100 GPU** and **High-RAM** if available.
Use a new run name, for example `kits23-medvision-v0-local-data-v140`.

Set `CODE_COMMIT` to your full committed pipeline SHA. Leave `CACHE_CODE_COMMIT`
blank for a new run. Run cells individually in order:

1. Stream output helper and configuration/checkout.
2. Mount Drive and require both portable bundle files.
3. Install the isolated pinned GPU environment.
4. Restore processed scans and rebuild destination test metadata offline.
5. Prepare manifests/shards from restored metadata.
6. Download the pinned model weights.
7. Run geometry check, review overlays and set `GEOMETRY_REVIEWED=True` for smoke inference.
8. Run one chunk (`MAX_CHUNKS=1`), then set `MAX_CHUNKS=0` to finish pending chunks.
9. Score the complete run and optionally export results to GitHub.

No download/preprocess-scans cell exists in this notebook. Missing/incomplete bundles,
wrong revisions, corrupted archives, unexpected download attempts, incorrect absolute
paths or incomplete row coverage fail rather than silently triggering scan staging.
Transfer/extraction, environment installation, model downloads/loading and metadata
checks still take GPU-session time. The runtime is not literally GPU-only at every
moment, but the lengthy scan download/reorientation happens before allocation.

## 3B. Colab CLI route

The CLI uses the same inference notebook setup and the same baseline scripts. The
bootstrap below stops after environment/data/model/geometry setup. Smoke and full
inference are separate explicit commands. Install the CLI using Google's
[official instructions](https://github.com/googlecolab/google-colab-cli#installation)
and complete Google authentication when prompted.

From your local repo, after committing the pipeline, create a small Git bundle of
committed code and a non-secret configuration file:

```bash
git bundle create /tmp/medmeasure.bundle HEAD
python3 - <<'PY'
import json, subprocess
from pathlib import Path
config = {
    'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'run_name': 'kits23-medvision-v0-local-cli-v140',
    'shard_size': 100,
    'max_chunks': 1,
}
Path('/tmp/medmeasure-config.json').write_text(json.dumps(config, indent=2)+'\n')
PY
```

The Git bundle preserves the real commit identity and lets a private repository run
without sending GitHub credentials to Colab. It contains committed history/files;
scans and archives under ignored `data/` are excluded. Don't include tokens in the
configuration file.

Only after the processed Drive upload finishes:

```bash
colab new -s medmeasure --gpu A100 --high-mem
colab drivemount -s medmeasure
colab upload -s medmeasure /tmp/medmeasure.bundle /content/medmeasure.bundle
colab upload -s medmeasure /tmp/medmeasure-config.json /content/medmeasure-config.json
colab exec -s medmeasure -f scripts/colab_inference_setup.py
```

Follow authentication prompts. This sets up a named runtime. The bootstrap retains
`script`, `COMMON`, and the output paths in the remote kernel for subsequent steps.
To review overlays, open the printed Drive geometry folder (or use `colab url -s
medmeasure --open` to inspect the runtime in the browser). Review every overlay,
then explicitly run the ten-example smoke test:

```bash
colab exec -s medmeasure <<'PY'
script('colab_baseline.py', 'smoke', *COMMON)
PY
```

If it succeeds, run one chunk, then the remainder and scoring:

```bash
colab exec -s medmeasure <<'PY'
script('colab_baseline.py', 'run', *COMMON, '--max-chunks', 1)
PY
```

```bash
colab exec -s medmeasure <<'PY'
script('colab_baseline.py', 'run', *COMMON, '--max-chunks', 0)
script('colab_baseline.py', 'score', *COMMON)
PY
```

The chunk wrapper enforces smoke completion. Stop and investigate any error before
running later steps. Export an execution log and retrieve results before releasing
the VM:

```bash
colab exec -s medmeasure <<'PY'
shutil.make_archive('/content/medmeasure-results', 'zip', PERSIST)
PY
colab download -s medmeasure /content/medmeasure-results.zip /tmp/medmeasure-results.zip
colab log -s medmeasure -o /tmp/medmeasure-execution.ipynb
colab stop -s medmeasure
```

The results ZIP includes audit/shards, geometry, smoke/attempt evidence and scores;
it does not include scans or model weights. Results also remain in Drive. Inspect
and commit compact finalized results locally rather than trying to push using the
Git-bundle remote. The existing browser notebook has a separate GitHub export route.

If the CLI/kernel restarts, rerun setup with the same configuration before the
remaining steps. Completed chunks are checked and reused. If local restore was
interrupted, preserve/remove the partial runtime data directory before retrying;
it is never overwritten automatically. Cloud setup and CLI behavior have not been
executed live here.

## What has actually been checked

- CPU unit tests cover geometry, chunk/scoring integrity, logging, legacy cache
  restrictions, portable relocation and the generated local-loader evaluator task.
- An integration check used the unchanged pinned MedVision loader and real official
  annotation plans with **placeholder scan/mask files**, not real CT data. It rebuilt
  and validated all 3,179 rows offline, then exported/restored to a different absolute
  root and repeated those checks. Network download calls were blocked during cache
  rebuilding. This checks metadata/path portability, not real-scan parity.
- Notebook cells/scripts compile; the inference notebook has no scan-staging call.
- Real local scan download/reorientation, real-data geometry, Drive throughput/quota,
  Colab dependency compatibility, CLI lifecycle and GPU inference remain unverified.

For lower-level recovery and output definitions, see the [Colab runbook](colab-runbook.md).
