# Continuity

## Current focus and decisions

- Phase 1: KiTS23 Task01 axial kidney-tumor sizing is the development pilot; broader tumor/lesion evaluation remains the goal.
- Immediate baseline: official pinned MedVision v1.4.0 **sizing annotations and case split**, then derive the 43-case diagnostic from that full run. Reserve validation within official training and keep official test cases out of further training. Evaluate unchanged MedVision-V0 and the proposed system on identical references/examples. Historical v1.0.0 evaluation is explicitly deferred; retain metadata/pins for later. Detection remains a possible extension, not implemented or committed for the immediate baseline; audit its own official plans and cross-task case overlap first. Exact revisions: [pilot config](configs/kits23-pilot.json).
- Immediate compute preference: Google Colab for the unchanged MedVision-V0 baseline; defer school cluster setup. [Notebook](notebooks/medvision_v0_kits23_baseline.ipynb) and [Colab runbook](docs/colab-runbook.md) are prepared; installation and GPU compatibility remain unverified. **Do not submit cluster jobs** until the user arranges access.

## State and evidence

- Implemented staging, case-separated manifests, geometry checks/overlays, baseline preparation/preflight, evaluation shards, and scoring. Commands: [cluster runbook](docs/cluster-runbook.md).
- Existing manifests use the chosen official v1.4.0 sizing split: 8,160 training/validation rows and 3,179 test rows. Of 147 test cases, 104 occur in the v1.0.0 training plan: disclose potential checkpoint exposure. The scorer now validates full-run coverage before selecting the 43-case/825-row diagnostic with `--subset-of`; no second inference run is needed. Historical and detection evaluation are not implemented.
- Colab wrapper supports checked chunk reuse, fresh retries for interrupted chunks, a required smoke run, and full/diagnostic scoring. Notebook includes isolated Python 3.11 setup, pinned downloads, geometry review, Drive outputs and optional GitHub result export. Model download resume requires matching revision provenance.
- Nine CPU tests pass. Geometry matched upstream on 100 synthetic ellipses; metadata ordering/reference checks covered all 11,339 rows. Synthetic NIfTI alignment and overlays checked. Details: [phase-one report](docs/baseline-and-data.md).
- No real scans or model weights downloaded; no GPU inference run. Real-mask geometry parity, cluster compatibility, and baseline scores remain unverified. Generated manifests/shards are under ignored `runs/` and can be regenerated.
- Colab additions: seven baseline/recovery/scoring CPU tests pass; notebook code cells and changed scripts compile. Full test discovery in the current local Python could not import geometry tests because OpenCV is missing. Prior nine-test validation above used the earlier CPU environment; Colab setup, downloads, GitHub export and GPU inference have not been executed.

## Next steps

1. Audit checkpoint exposure and case overlap across training tasks. Historical evaluation is later work; models trained on v1.4.0 cases need separate training excluding v1.0.0 test cases for held-out historical comparisons.
2. Arrange storage, stage real images/masks, and verify reference measurements and overlays across target sizes. Audit visual target size; small 2D cross-sections do not necessarily mean small tumors.
3. Commit/push the prepared Colab workflow, set its `CODE_COMMIT`, validate installation/staging/geometry and a small GPU smoke run, then finish the frozen evaluation and both scores. Bring compact results back via the notebook's optional GitHub branch export.
