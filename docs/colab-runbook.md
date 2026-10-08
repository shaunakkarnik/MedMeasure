# Colab baseline

Use [the notebook](../notebooks/medvision_v0_kits23_baseline.ipynb) for the existing
KiTS23 pilot: unchanged MedVision-V0, all 3,179 official v1.4.0 axial sizing test rows,
and the 825-row/43-case legacy-disjoint diagnostic from the same predictions.
The full split has potential released-checkpoint training exposure. Legacy-disjoint
means absent from the audited legacy sizing training plans, not proven unseen in
all model training tasks. This is research baseline evaluation, not clinical use.

## Start

1. Commit and push the notebook, scripts and configuration to GitHub.
2. Open the notebook through Colab's GitHub notebook picker, or use
   `https://colab.research.google.com/github/shaunakkarnik/MedMeasure/blob/main/notebooks/medvision_v0_kits23_baseline.ipynb`
   once it is on `main` (substitute your branch otherwise).
3. Select a GPU runtime; prefer an A100 if available. No VRAM requirement has been
   measured. Keep the upstream model precision and evaluation parameters unchanged.
4. Set `CODE_COMMIT` to the full commit containing the notebook and scripts. Keep
   it, `RUN_NAME`, and `SHARD_SIZE` fixed throughout a run. The notebook fetches that
   commit and clones the supporting code; opening a notebook alone does not clone it.
5. Optional Colab Secrets: `HF_TOKEN` for HF access and `GITHUB_TOKEN` for private
   cloning/export. Enable notebook access. A GitHub fine-grained token for export
   needs this repository's Contents read/write permission. Set the Git author name
   and email for the optional export. Tokens never enter remotes or notebook source.
6. Run setup/staging, inspect geometry overlays, then run the smoke cell. Start
   with `MAX_CHUNKS=1`; set it to zero and rerun the evaluation cell to finish all
   pending chunks. Score only when every chunk is complete.

The isolated Python 3.11 environment follows the pinned upstream launcher's frozen
requirements. The source distributions' dependency declarations conflict with the
launcher pins, so installation uses `--no-deps` and preserves the launcher's versions.
Colab's preinstalled notebook environment stays separate. Large installation and
real-data transfers are explicit notebook steps. Installation, GPU memory use,
real-mask parity and end-to-end Colab inference remain unverified until executed.
Do not treat notebook syntax/CPU checks as evidence that GPU evaluation works.

## Persistence and recovery

Outputs live in `MyDrive/MedMeasure/<RUN_NAME>/`: audit, shard plan, geometry checks,
and evaluation attempts. Data, HF caches, weights, source and Python environment
live on the runtime's local disk. The upstream loader may download/preprocess the
full KiTS23 dataset and build training/test caches even for a short smoke run; no
storage or staging-time estimate has been measured. Watch free disk space.
A new Colab VM needs setup and staging again. To reuse data/weights across sessions,
you may store archives on Drive and restore locally, but this notebook does not
claim that the CT dataset fits your Drive quota or Colab disk allocation.

Completed chunks have a completion marker plus a sample-log checksum. Before reuse,
the wrapper verifies indices, identities and exact coverage. Evaluation provenance
includes the code commit, pinned config, manifest hash and shard plan. A changed
configuration is refused: restore it or start a new run name.

Interrupted chunks are preserved and retried from their beginning in a fresh
`attempt-NNN` directory. Partial attempts are never pooled into final scores.
This is chunk-level recovery, not token-level or row-level generation resumption.
If geometry/audit/shard generation is interrupted before its summary exists,
preserve the partial directory and regenerate that step into a clean directory.
Model staging supports `--resume` only with matching repo/revision provenance.

The scorer requires exact coverage of the official full manifest. Its `--subset-of`
option validates the complete logs first, then selects the diagnostic manifest;
missing/duplicate/unexpected full-run rows still fail. MAE/MRE are successful-finite
row means, and failure/below-10%-error rates explicitly state their denominators.
Existing summaries are not overwritten; remove them explicitly only when rescoring
is intended. Smoke results are execution checks, not representative baseline scores.

## Bring results back through Git

The optional export cells create a separate runtime checkout based on latest
`origin/main`, stage `results/<RUN_NAME>/`, and show the staged file summary before
committing/pushing `results/<RUN_NAME>` as a new branch. Export includes full and
diagnostic summaries, complete upstream result logs, per-chunk provenance, selected
rows, package/GPU information, audit and shard plan. It excludes scans, weights,
geometry PNGs, smoke inference and incomplete attempts, which remain in Drive.
Inspect export size before pushing; use Drive for unusually large output files.

If push fails after commit, rerun only the push command shown in the notebook.
The export refuses an existing local checkout or existing result directory to avoid
silently overwriting another export. Use a distinct run name for a new experiment.

Locally, preserve current changes, then:

```bash
git fetch origin
git merge origin/results/kits23-medvision-v0-v140
```

Substitute your run name. Alternatively, check out the results branch directly.
Ask Codex to analyze `results/<RUN_NAME>/`. Colab's **Save a copy in GitHub** saves
the notebook itself; the export cells handle generated files.

For lower-level commands see the [cluster runbook](cluster-runbook.md); its scheduler
instructions are irrelevant to Colab. No cluster jobs are submitted by this workflow.
