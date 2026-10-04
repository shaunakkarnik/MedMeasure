# MedMeasure: Project Context and Plan

## Goal and motivation

MedMeasure investigates whether explicit spatial prediction can improve fine-grained quantitative analysis of medical images. The project has two intended outcomes:

1. A **specialized segmentation-based measurement system** built on MedVision-V0.
2. A **VLM with improved quantitative answer generation**, using segmentation supervision or predicted spatial features.

The primary reference is [MedVision: Benchmarking Quantitative Medical Image Analysis](https://arxiv.org/abs/2511.18676v3). Its benchmark covers anatomical and abnormality detection, tumor/lesion sizing, and anatomical angle/distance measurement. MedVision-V0 is a Qwen2.5-VL-based model trained on MedVision with supervised and reinforcement fine-tuning. It generates textual coordinates and measurements, but still struggles with small targets and tumor/lesion localization and sizing.

The paper's error analysis identifies limited visual perception as a remaining bottleneck: eliminating arithmetic error does not substantially resolve measurement error. This motivates a spatial prediction pathway, although it does not establish whether the limitation lies in the encoder, feature compression, or how the language model uses visual information.

Our architectural precedent is [Qwen3-VL-Seg](https://arxiv.org/abs/2605.07141), which combines visual features and language-conditioned representations in a mask decoder. We will adapt these principles to MedVision-V0 rather than switch backbones initially. All improvements described below are research hypotheses, not established results.

## Version 1: Specialized segmentation-based measurement

**Purpose:** Test whether a mask decoder using MedVision-V0's representations can produce accurate medical segmentation and improve the resulting measurements.

**Inputs:** A medical image, a question identifying the target, and physical spacing metadata for measurement.

**Proposed architecture:**

```text
Image → MedVision-V0 vision encoder → intermediate spatial features ─┐
                                                                    ├→ mask decoder → target mask
Image + question → MedVision-V0 language model → target hidden state ┘

Target mask + physical spacing → geometric measurement → box / ellipse axes
```

- Extract features from multiple intermediate layers of the existing vision encoder. Project and fuse them into spatial features for dense prediction.
- Extract a language-conditioned hidden state identifying the requested target. This is a vector representation, not the final generated measurement text. The exact extraction position remains an implementation choice; it must not depend on ground-truth answer content unavailable at inference.
- Use the target representation to condition a lightweight mask decoder, for example through a projected query attending to the spatial features, followed by mask prediction and upsampling.
- Derive bounding boxes and lesion measurements from predicted masks using the benchmark's geometric conventions. MedVision's lesion-size labels are major/minor axes of ellipses fitted in physical coordinates; these are not interchangeable with arbitrary maximum-diameter measurements.

Begin with a frozen backbone and train the decoder and feature/query projections with a segmentation loss such as BCE plus Dice. Then evaluate vision-encoder adaptation through LoRA or selective unfreezing. The frozen version tests a practical starting point, not an assumption that adaptation is unnecessary.

This version is a **VLM-based specialist measurement system**. Its final numbers come from geometric computation rather than language-model generation. It can be compared on the same measurement tasks, but a gain does not demonstrate improved VLM reasoning. MedVision's BiomedParse comparison provides a precedent for this type of specialist evaluation.

## Version 2: Improved VLM-generated measurements

**Purpose:** Improve measurements generated through the language model, addressing MedVision's emphasis on end-to-end image-and-text-to-text quantitative analysis.

Retain the VLM's textual measurement output as the primary evaluated answer. Investigate two routes in sequence:

1. **Auxiliary segmentation supervision.** Jointly train segmentation and textual measurement objectives, updating shared visual features or adapters used by the language model. The mask decoder can be omitted at inference. A decoder attached to an entirely frozen backbone cannot, by itself, improve the unchanged text-generation pathway.
2. **Explicit spatial feedback.** Encode predicted masks or mask-guided regional features into representations the VLM consumes before generating its answer. This requires an explicit feedback mechanism; attaching an output head does not create one. Training must account for imperfect predicted masks rather than rely exclusively on ground-truth masks.

Version 1 provides a diagnostic: accurate mask-derived measurements alongside inaccurate generated answers would indicate that useful spatial evidence exists but is not successfully used by answer generation. Transfer to the language model remains a separate hypothesis.

## Project sequence

1. **Establish data and baseline.** Begin with a focused lesion-sizing development subset that supports the small-target research goal. Evaluate unchanged MedVision-V0 on the same annotation version and evaluation examples used for the proposed system to establish a matched baseline. Audit case splits and potential training exposure. Verify that ground-truth masks passed through the measurement pipeline reproduce reference measurements.
2. **Implement Version 1.** Build the intended decoder with both intermediate visual features and VLM conditioning from the outset. Verify spatial alignment and overfit a small batch before broader training.
3. **Evaluate and refine Version 1.** Measure segmentation and downstream measurement quality. Explore encoder adaptation if frozen features are insufficient.
4. **Run explanatory ablations.** Compare intermediate-feature fusion against final-layer features, VLM conditioning against simpler target conditioning, and frozen against adapted encoders. These explain the combined model's results; they are not prerequisites for implementing it.
5. **Develop Version 2.** Start with auxiliary segmentation supervision, then investigate explicit spatial feedback. Evaluate generated measurements separately from geometric outputs.
6. **Expand scope.** Extend to additional lesion datasets and detection tasks. For anatomical angles/distances, investigate landmark heatmaps or another explicit point representation; a lesion mask alone does not solve these tasks.

## Evaluation and implementation requirements

- **Primary outcomes:** Measurement MAE, MRE, the benchmark's below-10%-error rate, and success/failure rate. Report segmentation Dice/IoU as supporting diagnostics; better overlap does not necessarily imply better measurements.
- **Comparisons:** Include released MedVision-V0 and a matched additional-fine-tuning baseline. For Version 1, include a segmentation comparator trained on the same masks where feasible. Disclose additional mask supervision and geometric post-processing.
- **Controlled ablations:** Hold data, resolution, and training budget as consistent as possible. When removing VLM conditioning, retain target information through a category embedding or text-only representation. A fixed-target pilot may use a task-specific decoder.
- **Data integrity:** Pin dataset, checkpoint, preprocessing, and evaluator versions. Preserve official test cases; derive validation partitions at patient/case level and audit cross-dataset overlap. Adjacent slices are not independent patients.
- **Geometry integrity:** Track orientation, resizing, cropping, padding, and per-axis spacing. Restore visual-token spatial ordering before reshaping intermediate features. Validate image/mask/measurement overlays.
- **Honest inference:** Use predicted masks and any predicted localization inputs at test time. Label ground-truth-mask or box experiments as diagnostics. Include empty masks and failed fits in failure reporting under the evaluation protocol.
- **Interpretation:** Report specialist-system gains and VLM-generated-answer gains separately. Neither benchmark improvement alone establishes clinical readiness.

The initial milestone is a working Version 1 pilot with reproducible measurement evaluation. Broader tumor/lesion evaluation is needed to establish whether improvements generalize beyond the development subset.

## Resources

- [MedVision project and benchmark](https://medvision-vlm.github.io/)
- [MedVision code](https://github.com/YongchengYAO/MedVision)
- [MedVision-V0 checkpoint](https://huggingface.co/YongchengYAO/MedVision-V0-7B)
- [MedVision dataset](https://huggingface.co/datasets/YongchengYAO/MedVision)
