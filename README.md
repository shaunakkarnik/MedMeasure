# MedMeasure
Medical VLM for quantitative measurement tasks

See the [project context and plan](docs/project-plan.md) for the research goals, proposed architectures, and implementation sequence.

See [CONTINUITY.md](CONTINUITY.md) for current decisions, progress, and next steps;
the [phase-one report](docs/baseline-and-data.md) for detailed checks; and the
[cluster runbook](docs/cluster-runbook.md) for execution instructions.

For Google Colab, use the [baseline notebook](notebooks/medvision_v0_kits23_baseline.ipynb)
and [Colab runbook](docs/colab-runbook.md). The notebook saves restartable evaluation
chunks to Drive and can push completed results to a GitHub branch.

New collaborators can follow the [Colab handoff guide](docs/colab-collaborator-handoff.md)
for exact setup, backup, evaluation, and result-sharing steps.

To move scan download/preprocessing off GPU time, use the
[local CPU → Drive → Colab pipeline](docs/local-to-colab.md) and its
[inference notebook](notebooks/medvision_v0_kits23_local_data.ipynb).
The guide includes browser and Colab CLI routes; the GPU setup requires a portable
prepared-data bundle and never falls back to scan staging.
