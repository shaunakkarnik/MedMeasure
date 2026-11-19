# Version 2

V2 tests whether auxiliary tumor-mask supervision improves MedVision-V0's
generated axis measurements. Both arms start from the same pinned checkpoint
and train rank-8 LoRA adapters on the QKV projections in vision blocks 24–31.
The original backbone weights stay frozen.

| Arm | Objective | Trainable parameters |
| --- | --- | --- |
| Text | Answer-token cross-entropy | Vision LoRA adapters |
| Joint | Answer-token cross-entropy plus mask BCE and soft Dice | Vision LoRA adapters and V1 mask decoder |

The joint decoder uses restored feature maps from vision blocks 7, 15, 23, and
31 and a query from the last prompt token. Causal attention keeps that query
independent of the reference answer. Inference generates text from the adapted
VLM using the image, sizing question, and physical pixel spacing.

- [Model, losses, and generation](../medmeasure/version2.py)
- [Training and checkpoint entry point](../scripts/version2.py)
- [Experiment settings](../configs/version2-kits23.json) and [prompt templates](../configs/version2-prompts.json)
- [ICE training setup](ice-version2.md) and [final evaluation runbook](version2-aligned.md)

After each epoch, validation ranks checkpoints by all-row within-10% accuracy,
then answer success rate, then lower mean relative error. Both arms selected
epoch 3. Resume uses `last.pt` and validates the source and experiment contract.
Keep training snapshot `14e92dd` available for the existing checkpoints.

Frozen MedVision and both V2 checkpoints use the same image-first benchmark.
V1 contributes its saved mask-derived measurements on the same test slices.
See the [results index](../results/README.md) for available evidence and
[current work](../CONTINUITY.md) for run status.
