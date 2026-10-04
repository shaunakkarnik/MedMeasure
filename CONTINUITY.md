# Continuity

## Current focus and decisions

- Phase 1: KiTS23 Task01 axial kidney-tumor sizing is the development pilot; broader tumor/lesion evaluation remains the goal.
- Use pinned MedVision v1.4.0 annotations and evaluate unchanged MedVision-V0 on the same examples. Historical reproduction is deferred. Exact revisions: [pilot config](configs/kits23-pilot.json).
- Target a school Linux/CUDA cluster. **Do not execute GPU jobs or submit cluster jobs** until the user arranges access. Colab is out of scope for now.

## State and evidence

- Implemented staging, case-separated manifests, geometry checks/overlays, baseline preparation/preflight, evaluation shards, and scoring. Commands: [cluster runbook](docs/cluster-runbook.md).
- Real annotation metadata audited: 8,160 training/validation rows and 3,179 test rows. Of 147 test cases, 104 occur in older training plans; a separate 43-case/825-row diagnostic is prepared, but is not proven unseen by the checkpoint.
- Nine CPU tests pass. Geometry matched upstream on 100 synthetic ellipses; metadata ordering/reference checks covered all 11,339 rows. Synthetic NIfTI alignment and overlays checked. Details: [phase-one report](docs/baseline-and-data.md).
- No real scans or model weights downloaded; no GPU inference run. Real-mask geometry parity, cluster compatibility, and baseline scores remain unverified. Generated manifests/shards are under ignored `runs/` and can be regenerated.

## Next steps

1. Arrange storage, stage real images/masks, and verify reference measurements and overlays across target sizes. Audit visual target size; small 2D cross-sections do not necessarily mean small tumors.
2. Clarify checkpoint training exposure and report the official and historical-disjoint evaluations separately.
3. Once cluster details and execution authorization are available, validate the environment, run a smoke test, then score the frozen evaluation manifests.
