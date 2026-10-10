# Version 1 architecture review — 2026-10-10

Verdict: coherent frozen-backbone research baseline, with no blocking wiring defect
found in the reviewed paths. Accurate small-lesion measurement and useful language
conditioning remain unproven. No model or training code changed during this review.

## Sound parts

- Intermediate features bypass the vision-to-language merger. The adapter reverses
  both window ordering and merge-block packing before spatial decoding, consistent
  with the [pinned Transformers implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py).
- Image/question conditioning uses a causal prompt hidden state available at
  inference. Ground-truth masks and reference measurements do not enter that path.
- Backbone evaluation mode and disabled gradients are enforced. Decoder projections,
  attention and mask embeddings receive gradients. GroupNorm suits batch size one.
- Native-grid restoration, full-precision physical spacing, predicted-component
  selection, patient separation and failure denominators are handled explicitly.

## Main risks and recommendations

1. **Coarse output is the largest architectural risk.** In
   `medmeasure/version1.py`, all four feature maps share a patch grid: fusion is
   across depth, not across spatial scales. With a 448-pixel input and the standard
   Qwen2.5-VL 14-pixel patches, the decoder produces 32×32 logits and interpolates
   them to 512×512. It has no learned spatial refinement after upsampling.
   Interpolation permits sub-grid boundaries but restricts their shape. Before
   broad training, inspect small-target overfits and axis errors. If insufficient,
   compare learned feature upsampling/refinement before the final mask projection
   against increased input resolution. A native-image detail branch is another
   option, but changes the frozen-representation experiment more substantially.

2. **Checkpoint validation does not track the primary outcome.**
   `scripts/version1.py` selects the best checkpoint using BCE + Dice alone.
   Reasonable segmentation loss does not ensure accurate ellipse axes, especially
   when disconnected false positives change the selected component. Retain the
   simple training loss, but log held-out measurement MRE, below-10% fraction and
   failures, preferably with size-stratified results. Decide the selection rule
   using validation before test evaluation.

3. **The language branch is mechanically valid but not established as useful.**
   The frozen final prompt token has no learned segmentation-token objective, and
   every pilot example asks the same kidney-tumor question. The decoder can learn
   a fixed task while ignoring target semantics. This is acceptable for the pilot;
   it cannot establish general target selection. Compare against a learned fixed
   query on identical data, and evaluate differing targets before making broader
   language-conditioning claims.

## Validation performed

All six existing Version 1 tests passed again in `/tmp/medmeasure-v1-env`, including
tiny actual Qwen module parity, spatial restoration, gradients and workflow checks.
The existing synthetic overfit test places the target in a feature channel and
predicts at that feature resolution; it verifies optimization wiring, not real
feature quality or coarse-to-native boundary recovery.

A CPU diagnostic optimized free mask logits on three previously checked **training**
slices, bypassing all backbone/decoder features. Initialization was the area-resized
target, clamped to [0.01, 0.99] and converted to logits. Adam used learning rate 0.1
for 250 steps with the existing native-grid BCE + Dice objective and bilinear
interpolation (`align_corners=False`). Threshold was 0.5. MRE below uses the mean
relative major/minor-axis error against full-precision planner references.

| Training slice | Reference axes (mm) | Grid | Dice | Axis MRE |
| --- | --- | --- | --- | --- |
| case_00103, 476 | 2.14 / 2.14 | 32×32 | 0.389 | 145.29% |
| case_00103, 476 | 2.14 / 2.14 | 128×128 | 0.842 | 23.64% |
| case_00442, 716 | 38.39 / 35.74 | 32×32 | 0.962 | 1.04% |
| case_00442, 716 | 38.39 / 35.74 | 128×128 | 0.995 | 0.19% |
| case_00430, 24 | 181.66 / 133.28 | 32×32 | 0.996 | 0.014% |
| case_00430, 24 | 181.66 / 133.28 | 128×128 | 1.000 rounded | 0.054% |

This is a label-assisted representational diagnostic, not model inference, a
mathematical capacity bound, or a benchmark result. Finite optimization and the
segmentation objective also affect the results. It supports investigating small
targets; it does not prove a particular decoder change will solve them.

Released processor/backbone execution, the checkpoint's actual runtime grid, GPU
memory and real-feature overfitting remain unverified. The 32×32 calculation uses
the documented resize and the installed pinned Qwen implementation's default patch
size. No weights were downloaded and no GPU jobs were run.
