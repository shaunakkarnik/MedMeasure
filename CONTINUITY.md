# Continuity

- **Final V2 evaluation results are pending.** The full ICE array is reported
  running as of November 19. The shared image-first smoke array `7161524`
  produced 10/10 scorable answers for each of frozen, text, and joint.
  [Smoke summaries](results/v2-aligned-smoke-7161524/).
- Selected epoch-3 checkpoints are `runs/v2-7087274/best.pt` (text) and
  `runs/v2-7087275/best.pt` (joint). CPU setup job `7140637` prepared the
  pinned benchmark environment and model exports. Training checkpoint contracts
  depend on the runtime sources from `14e92dd`. The final evaluation records
  the execution script and adapter hashes for all shards.
- On ICE, the checkout lives at `$WORK/MedMeasure` beside `data/medvision`,
  `audit`, `hf-cache`, `models`, and `runs`. After the array finishes, merge
  the four shards per model and run the four-method comparison on full test and
  the 43-case historical-disjoint subset. [Evaluation runbook](docs/version2-aligned.md).
- The [results index](results/README.md) links the historical baseline, saved
  V1 predictions and paired analysis, and earlier V2 runs to their evaluators.
  [Project plan](docs/project-plan.md).
- Handoff checks passed for the image-first adapter, Slurm syntax, and the
  comparison command with saved baseline data as a fixture. The full local
  suite needs the project dependencies. The live array and final outputs have
  not been verified locally.
