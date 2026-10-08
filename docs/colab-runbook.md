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

## Cell output and the current-runtime logging fix

The updated notebook defines `stream_command` before setup and routes subprocess
stdout/stderr through the notebook's Python output stream. Byte-based forwarding
shows progress messages without waiting for a newline; `PYTHONUNBUFFERED=1` and
UTF-8 encoding propagate to Python child processes. Nonzero exits still raise an
error, and interrupting a future streamed command terminates its process group.
Git query stdout remains captured intentionally for commit checks and file retrieval.

For a notebook already running the old helper, wait for its active download/staging
cell to finish. Copy the entire contents of
[colab_logging_hotfix.py](../notebooks/colab_logging_hotfix.py) into a new code cell
and run it before backup/geometry/inference. It redefines `run`, so the existing
`script` helper automatically uses live output for subsequent commands. No Git
checkout, dependency reinstall, `CODE_COMMIT` change or runtime restart is needed.
Save the edited notebook copy in Drive. The hotfix cannot recover previous output
or attach to a child process that was already started. Do not interrupt an active
download solely to apply it.

## Persistence and recovery

Outputs live in `MyDrive/MedMeasure/<RUN_NAME>/`: audit, shard plan, geometry checks,
and evaluation attempts. Data, HF caches, weights, source and Python environment
live on the runtime's local disk. The upstream loader may download/preprocess the
full KiTS23 dataset and build training/test caches even for a short smoke run; no
storage or staging-time estimate has been measured. Watch free disk space.
A new Colab VM needs environment setup again. The notebook now supports a shared
processed-data archive under `MyDrive/MedMeasure/cache/kits23-<dataset-revision>-v140.tar`.
Its completed JSON sidecar records dataset provenance, original runtime path, size
and archive SHA-256. Both files are required; partial or corrupt backups are refused.
Before backing up, run the size-estimate cell and check Drive's actual free quota.
`SAVE_DATA_CACHE=False` skips saving when storage is insufficient.

Backup includes the staged data root: processed scans/masks, completion markers,
source, metadata and HF loader caches. Authentication-token files and locks are
excluded. Model weights live outside this root and are not backed up. The tar is
uncompressed because scans are already compressed, and it writes directly to Drive
without needing another full-size local copy. Do not modify/stage data during backup.
An interrupted backup is preserved as `.partial`; preserve or remove it explicitly
before retrying. If interrupted after archive rename but before the JSON sidecar,
the archive is incomplete and must be preserved/removed before a fresh backup.

New runtimes verify the checksum, restore into the same empty `/content/medmeasure-data`
directory (HF caches may contain absolute paths), then run normal staging/preflight.
Processed files and upstream completion markers are reused; environment installation
and task-cache checks still run. Restore reads the archive for validation and extraction
and can take time. It requires enough local disk space. Real-data backup, Drive quota,
and end-to-end restored loader behavior remain unverified; synthetic round trips pass.

When resuming an older sizing run with the updated notebook, keep its original
`CODE_COMMIT` and set `CACHE_CODE_COMMIT` to the new full commit containing the helper.
This fetches the standalone cache script without changing the inference checkout or
its recorded provenance. New runs may leave `CACHE_CODE_COMMIT` blank to use `CODE_COMMIT`.

### Back up from an already-running notebook

Editing or pushing the local notebook does not change Colab's running notebook or
VM. Add the following cells to the existing notebook after staging finishes. Keep
its current `CODE_COMMIT`; no checkout, environment reinstall or restart is needed.
First commit/push the new helper and notebook changes locally.

Fetch just the helper and inspect the required storage (uses existing notebook variables):

```python
git('fetch', 'origin', 'main', cwd=REPO,
    authenticated=bool(secret('GITHUB_TOKEN')))
CACHE_HELPER = Path('/content/cache_colab_data.py')
CACHE_HELPER.write_text(git('show', 'FETCH_HEAD:scripts/cache_colab_data.py', cwd=REPO)+'\n')
DATA_ARCHIVE = Path('/content/drive/MyDrive/MedMeasure/cache') / f"kits23-{lock['dataset_revision']}-v140.tar"
run(PYTHON, CACHE_HELPER, 'inspect', '--data-dir', DATA,
    '--archive', DATA_ARCHIVE, '--config', REPO/'configs/kits23-pilot.json')
```

After checking available Drive quota, run a separate cell:

```python
run(PYTHON, CACHE_HELPER, 'backup', '--data-dir', DATA,
    '--archive', DATA_ARCHIVE, '--config', REPO/'configs/kits23-pilot.json')
```

Wait for `Backup complete` and confirm both archive and JSON sidecar appear in Drive.
The existing data is only read; sizing and subsequent KiTS23 detection can still use
the current runtime paths. Save the edited Colab notebook copy in Drive too.

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
