# MedMeasure

MedMeasure studies tumor measurement with MedVision-V0 using KiTS23 axial CT
slices, clinical questions, and physical pixel spacing.

| Experiment | Trained parameters | Measurement output |
| --- | --- | --- |
| Released MedVision-V0 baseline | Released checkpoint | Generated text |
| Version 1 | Mask decoder and its projections | Ellipse geometry applied to a predicted mask |
| Version 2, text objective | Vision LoRA adapters | Generated text |
| Version 2, joint objective | Vision LoRA adapters and mask decoder | Generated text |

V2 compares answer supervision against answer plus mask supervision. Both arms
adapt the QKV projections in vision blocks 24–31 using rank-8 LoRA. Their original
backbone weights stay frozen. The joint arm uses the V1 mask decoder during
training, and evaluation generates answers through the adapted VLM.

## Start here

- [V2 model and training](docs/version2.md) describes the two objectives and
  links to ICE setup.
- [Final V2 evaluation](docs/version2-aligned.md) covers setup, sharding, and
  the four-method comparison. **The full evaluation is running and results are pending.**
- [V1 guide](docs/version1.md) covers the frozen decoder and feature cache.
- [Baseline and data audit](docs/baseline-and-data.md) describes the official
  baseline, dataset pins, and split exposure.
- [Project plan](docs/project-plan.md) records the research goals and methodology.
- [Current work](CONTINUITY.md) records pending experiments and validation limits.

## Repository structure

| Path | Purpose |
| --- | --- |
| `medmeasure/` | Model components, data contracts, geometry, and visualization helpers |
| `scripts/version1.py` | V1 feature extraction, decoder training, and mask evaluation |
| `scripts/version2.py` | V2 text/joint training and generated-answer evaluation |
| `scripts/version2_original.py`, `scripts/compare_final_results.py` | Sharded image-first evaluation, merge, and paired comparison |
| `scripts/prepare_data.py` | Case-separated manifests and legacy split audit |
| `scripts/stage_data.py`, `scripts/stage_model.py` | Pinned dataset and model downloads |
| `scripts/prepare_baseline.py`, `scripts/preflight_baseline.py`, `scripts/execute_baseline.py` | Official baseline preparation, checks, and inference |
| `scripts/summarize_baseline.py`, `scripts/compare_version1_baseline.py` | Saved-output scoring and paired V1 comparison |
| `scripts/check_geometry.py` | Native-grid image/mask alignment and measurement checks |
| `configs/` | Dataset/model revisions, experiment settings, and prompt templates |
| `cluster/` | Slurm launchers and baseline allocation examples |
| `tests/` | Offline data, geometry, cache, and tiny-model checks |
| `notebooks/` | Prediction viewer and retained Colab workflows |
| `results/` | Saved baseline and V1 predictions, summaries, masks, and provenance |
| `docs/` | Methodology, reports, and execution guides |

The remaining data scripts package and restore prepared scans or rebuild
destination metadata. Their workflow is documented in the
[prepared-data guide](docs/local-to-colab.md). The Colab scripts and notebooks
support the earlier baseline and V1 workflows.

## Environment and validation

V1 and V2 use Python 3.11 and `requirements-v1.txt`, including
Transformers 4.51.3 and Torch 2.6.0. Follow the ICE guide for CUDA installation.
The baseline evaluator has a separate upstream environment described in the
[baseline cluster runbook](docs/cluster-runbook.md). Visualization dependencies
and usage are in the [comparison viewer guide](docs/baseline-v1-visualization.md).

Run offline checks from the repository root in the appropriate environment:

```bash
python -m unittest discover -s tests -v
```

Some checks skip when optional dependencies or local data are unavailable.
GPU training requires the separately staged scans and pinned model weights.

## Data and results

Prepared scans, model weights, feature caches, and training checkpoints live
outside Git. Use the paths in the ICE guide for a new installation.
`results/` retains the row-level evidence needed to rescore the reported V1
and baseline runs, along with V2 validation and smoke summaries.

Training and validation are separated by case within the official training
split. The released backbone may have seen 104 official test cases through its
historical sizing training split. The final comparison reports the 43-case
historical-disjoint diagnostic alongside full-test results.

## Earlier execution workflows

The [baseline Colab notebook](notebooks/medvision_v0_kits23_baseline.ipynb),
[V1 Colab training notebook](notebooks/medmeasure_v1_training.ipynb), and
[prepared-data inference notebook](notebooks/medvision_v0_kits23_local_data.ipynb)
are retained with their [Colab runbook](docs/colab-runbook.md) and
[collaborator handoff](docs/colab-collaborator-handoff.md).
