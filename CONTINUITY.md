# Continuity

- **Final V2 evaluation completed.** ICE array `7162509` merged and produced
  the four-method comparison for 3,179 test slices and the 825-slice
  historical-disjoint subset. [Saved ICE summary](results/v2-aligned-test-7162509/summary.json).
  The reported evaluator, adapter, V1 predictions, and test manifest hashes
  match local files. V2 row-level predictions remain on ICE.
- Selected epoch-3 checkpoints are `runs/v2-7087274/best.pt` (text) and
  `runs/v2-7087275/best.pt` (joint). CPU setup job `7140637` prepared the
  pinned benchmark environment and model exports. Training checkpoint contracts
  depend on the runtime sources from `14e92dd`. The final evaluation records
  the execution script and adapter hashes for all shards.
- On ICE, the checkout lives at `$WORK/MedMeasure` beside `data/medvision`,
  `audit`, `hf-cache`, `models`, and `runs`. The merged comparison is at
  `$WORK/runs/final-comparison-7162509/`. [Evaluation runbook](docs/version2-aligned.md).
- The [results index](results/README.md) links the historical baseline, saved
  V1 predictions and paired analysis, and earlier V2 runs to their evaluators.
  [Project plan](docs/project-plan.md).
- Handoff checks passed for the image-first adapter, Slurm syntax, and the
  comparison command with saved baseline data as a fixture. The full local
  suite needs the project dependencies. The live array and final outputs have
  not been verified locally.
