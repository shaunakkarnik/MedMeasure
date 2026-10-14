# Version 1 KiTS23 pilot

The frozen-backbone specialist is implemented in `medmeasure/version1.py`, with
training and evaluation in `scripts/version1.py`. It predicts a tumor mask from
the image and question, then computes physical-space ellipse axes. It does not
generate a textual measurement answer.

## Architecture and input contract

- Load the existing pinned MedVision-V0 checkpoint and processor from
  `configs/kits23-pilot.json`, using Transformers 4.51.3.
- The released config also contains a redundant nested `text_config` from
  Transformers 4.56.1. The loader checks matching architecture fields and removes
  that nested dict in memory for 4.51.3; cached config files and weights stay intact.
- Capture vision block outputs at zero-based layers 7, 15, 23, and 31. Reverse
  attention-window ordering and merge-block packing to recover the patch grid.
- Run the image/question prompt through the language model. Use its final
  nonpadding token after the assistant generation prefix as the target vector.
  No answer, mask, landmark, target box, or reference measurement enters this path.
- Project/fuse the four feature maps to 128 channels. A projected target query
  attends to the spatial features; its mask embedding scores per-pixel features.
  Bilinear upsampling returns logits on the native slice grid.
- Freeze the entire backbone in evaluation mode. Train feature/query projections
  and the mask decoder with equally weighted BCE and soft Dice losses.

Defaults are in `configs/version1-kits23.json`. The pilot uses one image per
forward and gradient update. The CT soft-tissue window is exactly the upstream
KiTS23 normalization, HU [-160, 240]. Arrays retain their native row/column order.
There are no crops, pads, reorientations, scale bars, or target overlays in model
inputs. The processor resizes the whole slice; the default 200,704-pixel budget
makes a 512×512 slice 448×448. Measurements use original full-precision header
spacing after restoring logits to the native grid and thresholding at 0.5.

The specialist question and overlay-free rendering differ from the textual
baseline's official prompt/rendering. Disclose this along with additional mask
supervision; use the same specialist inputs for future architecture ablations.

## Setup and first run

For a fresh Colab runtime, open
[`notebooks/medmeasure_v1_training.ipynb`](../notebooks/medmeasure_v1_training.ipynb)
from GitHub after committing/pushing it with the implementation. Save a copy in
Drive, choose an A100 GPU if available, and set `CODE_COMMIT` to the full commit
SHA. The notebook clones that exact code, restores the prepared archive from
Drive, creates case-separated manifests, checks geometry, stages the pinned
backbone, and runs the four-row diagnostic before full training. Run cells
individually and inspect both geometry and overfit overlays. For private GitHub
repositories, set `PRIVATE_REPO=True` and enable a read-only `GITHUB_TOKEN` in
Colab Secrets. Keep `CODE_COMMIT` and `RUN_NAME` unchanged on resume; checkpoints
and source provenance persist under `MyDrive/MedMeasure/<RUN_NAME>/`.

Use a separate Python 3.11 environment with `requirements-v1.txt`. The commands
below assume the repository root, the existing processed data at `data/medvision`,
and audited manifests at `runs/local-data-preparation-fixed/audit`. In Colab,
restore the portable bundle first using the [local-data guide](local-to-colab.md),
and substitute the restored data/manifests paths. These commands do not stage scans.

```bash
python3.11 -m venv .venv-v1
.venv-v1/bin/python -m pip install -r requirements-v1.txt

# First GPU diagnostic: cache four frozen feature sets and overfit the decoder.
.venv-v1/bin/python scripts/version1.py train \
  --data-root data/medvision \
  --manifests runs/local-data-preparation-fixed/audit \
  --output runs/v1-overfit --overfit-rows 4 --epochs 100
```

Inspect the final training overlays (green target, magenta prediction), falling
`overfit_loss`, and `train_dice_online`. Overfit mode selects checkpoints on the
small training subset and skips validation; it is a diagnostic, not a trained
benchmark system. Frozen maps are cached in CPU memory only in this mode.

After the real-backbone overfit and alignment gate passes, start a fresh broader
training run:

```bash
.venv-v1/bin/python scripts/version1.py train \
  --data-root data/medvision \
  --manifests runs/local-data-preparation-fixed/audit \
  --output runs/v1-train --epochs 10

# A ten-row smoke run; its summary explicitly identifies partial coverage.
.venv-v1/bin/python scripts/version1.py evaluate \
  --data-root data/medvision \
  --manifests runs/local-data-preparation-fixed/audit \
  --checkpoint runs/v1-train/best.pt \
  --output runs/v1-smoke --limit 10

# Full official test and historical-disjoint diagnostic from the same predictions.
.venv-v1/bin/python scripts/version1.py evaluate \
  --data-root data/medvision \
  --manifests runs/local-data-preparation-fixed/audit \
  --checkpoint runs/v1-train/best.pt \
  --output runs/v1-test
```

`--local-files-only` requires a previously staged checkpoint/processor. The default
device is CUDA with a bfloat16 backbone and float32 decoder. GPU memory/runtime
requirements for the released 7B backbone have not been measured. CPU float32 is
available explicitly, but local checks use tiny random models, not the 7B weights.

## Persistent frozen-feature cache

For the frozen first run, extract training/validation features once, then reuse
them every epoch. This is optional; uncached training and test evaluation retain
their existing paths. The Colab notebook has not been changed to use this cache.

```bash
# Run on a GPU after geometry and the four-row overfit diagnostic pass.
.venv-v1/bin/python scripts/version1.py cache-features \
  --data-root data/medvision \
  --manifests runs/local-data-preparation-fixed/audit \
  --output runs/v1-features --local-files-only

# Decoder-only training: no backbone weights/processor are loaded.
.venv-v1/bin/python scripts/version1.py train \
  --data-root data/medvision \
  --manifests runs/local-data-preparation-fixed/audit \
  --feature-cache runs/v1-features --output runs/v1-cached --epochs 1
```

Use `--epochs 10` for a fresh ten-epoch run once timing is established, or resume
from completed epochs with `--resume`, keeping `--feature-cache` unchanged.
Training still uses the CUDA decoder by default and preserves case-separated
validation and checkpoint selection. The entire feature cache need not fit RAM:
one row is loaded at a time.

Extraction saves the four restored vision maps and question-conditioned query
**without changing their tensor dtype or values**. It also saves rendered RGB,
native target masks and spacing to avoid repeatedly decompressing CT/mask volumes
during training. Targets are stored only after the image/question forward pass;
they do not enter the backbone. Only training and validation rows are extracted;
test rows and evaluation are not cached.

Each completed row is saved atomically. Rerun the same extraction command after
interruption to verify/reuse saved rows and continue missing ones. Use
`--max-rows 100` to extract at most 100 **new** rows per invocation (zero means all
remaining rows). Partial extraction exits successfully with an explicit incomplete
message, but training refuses the cache until `complete.json` exists. Run only
one extraction job per cache directory; no concurrent writer support is provided.
Corrupt rows are rejected, not silently regenerated.

The first row prints a projected total cache size; completion records actual
bytes. Check disk **quota** before the full pass: filesystem free-space checks do
not know your ICE user quota. Cache storage is additional to scans and weights.
Four 1280-channel 32×32 bfloat16 maps alone are 10 MiB per row, about 80 GiB over
8,160 rows; native images/masks and serialization add more. This is a dimensional
estimate, not measured released-backbone cache usage. Float32 features double the
map storage. Cache extraction still needs the full backbone on the GPU.

Cache identity checks model/dataset pins, exact train/validation rows, source
image/mask checksums, feature layers, question, processor pixel settings, backbone
dtype, extraction source hashes, and Torch/Transformers versions. Source volumes
are hashed once at the start of each invocation; they remain required for that
integrity check. Every loaded cache entry has its checksum verified. A mismatch
requires a new cache directory. Decoder settings such as learning rate can change
for a fresh run without re-extraction; existing resume checks still apply.

Do not reuse these caches for backbone adaptation, varying image augmentation,
or a different image/question preprocessing route. Disk caching introduces no
feature quantization, but released-model/GPU equivalence and throughput still
need verification. Training run/checkpoint metadata records the cache identity.

## Splits, checkpoints, and outputs

Before loading weights, the CLI verifies the annotation plan checksum, exact
manifest rows against the pinned official plan, patient separation, and complete
ordered official test membership. Training/validation rows must originate from
official training cases. Evaluation additionally rejects cases used for decoder
training or checkpoint selection. Ground-truth masks are loaded after prediction
for test scoring and overlays.

Training saves `run.json`, `epochs.jsonl`, `last.pt`, and `best.pt`. Checkpoints hold
the decoder, optimizer, completed epoch, random states, configuration, case splits,
manifest hashes, and model/dataset pins; they exclude frozen backbone weights.
Normal training selects `best.pt` by held-out validation loss. Resume from
`--resume runs/v1-train/last.pt --epochs 20 --output runs/v1-train-resumed`, keeping
the matching `best.pt` beside the input checkpoint. Resumption is at the last
completed epoch; an interrupted epoch is repeated. No new output directory is
overwritten. Resumed runs preserve the previous best checkpoint.

Evaluation writes native binary mask PNGs, overlays, flushed per-row
`predictions.jsonl`, `selected_rows.jsonl`, `run.json`, and `summary.json`. It uses
the existing benchmark-compatible physical-space ellipse fit, choosing the
largest valid **predicted** component without consulting target overlap. Empty
masks, unfit components, and nonfinite logits count as failures. Box coordinates
are native `[row_min, col_min, row_max_exclusive, col_max_exclusive]` bounds of the
whole predicted mask; this is not a detection-task implementation.

Scores include per-row mean major/minor-axis MAE and MRE, strict MRE <10% rates
with all-row and success-only denominators, success/failure counts, Dice, and IoU.
MAE/MRE use the same float16 reference axes as the baseline HF loader;
full-precision planner references and oracle-style errors are also saved.
Full-test scoring requires complete coverage before deriving the historical-disjoint
diagnostic. Smoke runs do not produce that diagnostic. Infrastructure errors abort
the run, retaining partial prediction logs; an absent summary is not a full score.

The released backbone's possible exposure to 104 v1.4.0 test cases remains a
limitation. The 43-case subset is absent from audited historical sizing training
plans, not proven unseen across all tasks. Fine-tuned textual/segmentation
comparators, broader training, encoder adaptation, and Version 2 remain separate
research steps.

## Checks completed during implementation

```bash
.venv-v1/bin/python -m unittest discover -s tests -p test_version1.py -v
```

Seven offline tests cover non-square/multi-window spatial restoration, target-vector
parity with the original Hugging Face multimodal forward, question conditioning,
decoder gradients with a frozen backbone, synthetic small-mask overfitting,
checkpoint serialization, training/resume/evaluation, official split rejection,
and failure/coverage denominators. Persistent-cache checks cover lossless
bfloat16 roundtrip, interrupted extraction/resume, test-row exclusion, stale
question/source rejection, corruption checks, exact two-epoch decoder-weight
parity, and cached train/resume with backbone and scan loading blocked.
The workflow check uses actual tiny randomly
initialized Qwen2.5-VL modules and a processor stub, not released weights.

Three real training slices spanning reference sizes matched pinned upstream
normalization exactly, and ground-truth masks reproduced both planner axes with
zero measured error. See `runs/version1-native-grid-check/summary.json`; one overlay
was visually inspected. These are data/geometry diagnostics. Released-checkpoint
processor inference, real-backbone small-batch overfitting, GPU training, and test
performance remain unverified.

Adapter reference: [Hugging Face Qwen2.5-VL implementation, v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py).
