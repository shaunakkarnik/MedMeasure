# Phase 1: KiTS23 data and baseline

## Agreed scope

Use **KiTS23 Task01 axial kidney tumors (label 2)** as the development pilot, with
**MedVision annotation v1.4.0** (the newest published version inspected on 2026-10-04).
Evaluate unchanged MedVision-V0 on those same annotations. The broader research
scope remains tumor/lesion sizing and detection; KiTS23 is an implementation pilot,
not the entire eventual evaluation. The main evaluation uses **official v1.4.0
annotations and case membership**, superseding the proposed custom original-split
protocol. Immediate scope is sizing on the full v1.4.0 test set, followed by the
43-case diagnostic extracted from that run. Historical v1.0.0 evaluation is deferred;
retain its metadata and pins for later, keeping version scores separate. Detection
is a possible extension, not part of the implemented baseline. Its unchanged official
plans require a cross-task case-overlap audit before joint training.

The scripts and generated manifests below already use the chosen official v1.4.0
sizing split. The scorer supports diagnostic selection with `--subset-of`, validating
exact full-run log coverage before selecting diagnostic rows. Historical execution
and detection evaluation are not yet implemented.
Checkpoint exposure to current test cases remains a limitation, even when the
official protocol is followed. A model trained on v1.4.0 training cases needs a
separate training run excluding original test cases for a held-out v1.0.0 comparison.

The immediate compute preference is Google Colab; see the [notebook and runbook](colab-runbook.md).
No cluster jobs, GPU inference, real image preprocessing, or model downloads have
been executed. Colab installation and end-to-end GPU compatibility are unverified.

## What is implemented

- `scripts/stage_data.py`: explicit network staging of pinned dataset code/annotations;
  optionally downloads and preprocesses images on a CPU node.
- `scripts/prepare_data.py`: verifies plan checksums, emits case-separated train,
  validation and test manifests, reports size distributions and cross-version overlap.
- `scripts/check_geometry.py`: checks native image/mask grids and physical-space ellipse
  axes against full-precision references; saves overlays and file hashes.
- `scripts/stage_model.py`: downloads a specified checkpoint revision without loading it.
- `scripts/prepare_baseline.py`: prepares a run directory without inference.
- `scripts/preflight_baseline.py`: checks staged inputs, source hashes and loader ordering,
  including the actual HF dictionary-of-lists measurement schema and float16 spacing.
- `scripts/plan_evaluation.py`: prepares disjoint index lists for later time-limited jobs.
- `scripts/execute_baseline.py`: runs the upstream model/evaluator with a local task
  definition that pins the HF dataset revision. Invoked only by an explicit GPU run.
- `scripts/summarize_baseline.py`: summarizes upstream parsed scores and size groups,
  checks log completeness, and reports both failure-inclusive and success-only rates.
- `cluster/`: scheduler-neutral launcher and an **unsubmitted Slurm example**.

See [cluster runbook](cluster-runbook.md) for commands. Revision pins and checksums
are in [the pilot configuration](../configs/kits23-pilot.json). The older
[task inventory](../configs/baseline-task-inventory.csv) is retained as reference;
loaded metadata counts have now been checked for this pilot.

## Actual metadata audit

Downloaded only the approximately 33 MB annotation archive and public source files.
The following counts come from real v1.4.0 plans, not synthetic data. No scans were
needed for this audit. The checkpoint revision is now pinned from public metadata;
no weights were downloaded. Full generated manifests and the detailed audit are under
`runs/kits23-data-audit/` (ignored by Git; reproducible using the runbook).

| Partition | Eligible slice-target rows | Cases with eligible rows | Rows with major axis <10 mm |
| --- | ---: | ---: | ---: |
| Training | 6,897 | 274 | 257 |
| Validation | 1,263 | 67 | 55 |
| Official test | 3,179 | 147 | 123 |
| Test absent from supplied historical training plans | 825 | 43 | 36 |

Validation uses 20% of the 342 official training **cases**, selected by deterministic
SHA-256 ordering with seed 42; 68 cases go to validation, but one has no eligible
single-component axial rows. All slices of a case remain together. Original patient
linkage beyond the dataset case ID is not available in this metadata audit.

There are 53 official test cases with at least one eligible cross-section below
10 mm. These thresholds describe **2D cross-sections**, not clinical categories or
3D tumor size. A small cross-section can be the edge of a large tumor. Most rows
are above 20 mm, so an overall score alone will not answer the small-target question.
Actual pixel occupancy after model preprocessing remains to be audited with images.

**104 of 147 new test cases occur in the union of v1.0.0/v1.1.1 training plans.**
The existing audit preserves the official updated test partition. A separate diagnostic manifest
excludes those cases; it is not called proven unseen, since exact checkpoint
training exposure and cross-dataset overlap still need auditing. Applying the same
new test set to both models controls examples, but does not remove historical
training exposure. Do not use official test cases to tune models or size thresholds.

## Geometry conventions and validation

### Alternative original-split count diagnostic (not the selected protocol)

Repartitioning the existing v1.4.0 manifests using the audited v1.0.0 case
membership retains 342 training/validation cases and 147 test cases. There are
104 cases moving in each direction. Of the 8,160 current training/validation
slice-target rows, 2,519 move to test and 2,354 move back from the current test set,
leaving **7,995 rows (2.02% fewer)** and 3,344 test rows. Training/validation rows
with reference major axis below 10 mm change from 312 to 291 (6.73% fewer).

These are metadata-only counts computed from `audit.json` and the existing
train/validation/test JSONL manifests. Moving case IDs belong to their expected
partitions and restored training sample IDs are unique. This alternative was not
implemented and is superseded by the official v1.4.0 protocol. Detection-row counts
and cross-task training exposure have not been audited.

v1.4.0 fits contours in physical coordinates and stores continuous major/minor
lengths. Rounded landmark endpoints are for display and must not be used to
recompute reference lengths. The adapter reproduces upstream component connectivity,
contour selection, physical-size floor and invalid-fit guards. It rejects a
single-target row with multiple components rather than silently picking a lesion.

The checker requires image/mask shapes and affines to agree with the plan, checks
spacing/orientation, hashes inspected files, and compares both axes with absolute
1e-4 mm plus relative 1e-5 tolerance. This is for full-precision **plan** values;
the HF loader quantizes reference fields to float16 for benchmark evaluation.
The diagnostic overlay uses a fixed CT window on the original grid. It does not
validate the model processor's rendered input, resizing or token-grid alignment.

Locally verified:

- All 11,339 eligible train/test metadata rows match the pinned upstream flattening
  routine in ordering and reference major-axis values.
- Both axes and fit counts match the pinned upstream fitter for 100 synthetic
  ellipses with varied rotation and anisotropic spacing.
- Nine CPU tests cover anisotropic spacing, empty/tiny masks, multiple components,
  invalid spacing, HF measurement fields, exact shard coverage, strict scoring
  thresholds, and duplicate-result rejection.
- Synthetic NIfTI image/mask files pass the CLI checker; an overlay was inspected.
- Repeated metadata preparation gives identical audits; altered affine metadata and
  incomplete score logs are rejected. Synthetic scoring preserves failure denominators.
- Generated shell scripts pass syntax checks; Python files compile without running GPU code.

**Not yet verified:** geometry parity on real KiTS23 masks, real-image overlays,
HF cache staging on the cluster, CUDA dependency compatibility, memory requirements,
model inference or end-to-end scoring on real outputs. Synthetic tests do not
establish real-data parity or baseline performance.

## Completion gates

1. Stage real images/masks and inspect geometry overlays spanning target sizes.
2. Resolve or explicitly characterize checkpoint training/test overlap.
3. Validate a ten-row MedVision-V0 smoke run in the chosen GPU environment.
4. Evaluate the frozen official v1.4.0 test manifest, reporting the historical-disjoint
   diagnostic separately and disclosing potential checkpoint exposure. Keep official
   test cases out of further training. Use validated subset-log selection to derive
   the 43-case diagnostic without a second inference run. Historical v1.0.0 evaluation
   is deferred. Label each report with its annotation version and case selection.
   Save raw answers, exact selected rows, package
   versions, checkpoint revision and GPU information.
5. Report MAE/MRE, strict below-10%-mean-relative-error rate, failure rates and size
   strata. The scorer exposes all-row and successful-row denominators explicitly.
   Preserve upstream outputs as well; nonfinite outputs are counted as failures
   in our report. Patient-level uncertainty and per-axis reporting remain follow-up
   analyses rather than implemented claims.

## Sources

- [Pinned benchmark source](https://github.com/YongchengYAO/MedVision/tree/45fa8e52684a65da6e1c0ef6ef422d5d4c3e94bd)
- [Pinned dataset source](https://huggingface.co/datasets/YongchengYAO/MedVision/tree/f4040ed7d2d2b45e09c1a996ad2969f2018051d3)
- [v1.4.0 annotation and split changes](https://github.com/YongchengYAO/MedVision/blob/45fa8e52684a65da6e1c0ef6ef422d5d4c3e94bd/docs/dataset-release/release-v1.4.0.md)
- [KiTS23 source dataset](https://kits-challenge.org/kits23/)
