# Run the MedVision-V0 baseline in Colab

This guide is for a collaborator who already has access to the MedMeasure GitHub
repository. Use your own Google account and Drive. You do not need access to the
original operator's Drive to start.

For the newer workflow that prepares scans locally before GPU allocation, follow
[local CPU → Drive → Colab](local-to-colab.md). The instructions below retain the
original all-in-Colab route and its pinned notebook version.

The notebook evaluates unchanged MedVision-V0 on **3,179 official KiTS23 v1.4.0
axial tumor-sizing test examples**, then derives the **825-example/43-case
legacy-disjoint diagnostic** from the same predictions. Detection is not implemented
yet. The main split has potential checkpoint training exposure; the diagnostic is
not proven unseen across all training tasks.

The original Colab session disconnected before the new backup cells were applied.
No completed processed-data backup has been reported. Expect a fresh download and
preprocessing unless a surviving runtime or complete archive is recovered. A
disconnection alone does not establish that the original VM was deleted.

## 1. Open the fixed notebook

Open [the pinned notebook in Colab](https://colab.research.google.com/github/shaunakkarnik/MedMeasure/blob/82337dc06cb645d7ed5e0bc66476dae8155124bb/notebooks/medvision_v0_kits23_baseline.ipynb).
Choose **File → Save a copy in Drive** and work in that copy.

This version includes live subprocess output, processed-data backup/restore, and
restartable evaluation chunks. No manual logging or backup patches are needed.

Select **Runtime → Change runtime type → A100 GPU**, enable **High-RAM**, and save.
These are recommended initial settings; the actual memory requirements and complete
Colab GPU workflow have not been validated. If A100 is unavailable, coordinate with
the project owner before spending time on another GPU configuration.

## 2. Configure your run

Run the **Stream command output** helper cell first. In the next configuration cell,
set:

```python
CODE_COMMIT = "82337dc06cb645d7ed5e0bc66476dae8155124bb"
CACHE_CODE_COMMIT = ""
RUN_NAME = "kits23-medvision-v0-v140-collaborator"
SHARD_SIZE = 100
MAX_CHUNKS = 1
SAVE_DATA_CACHE = True
```

Choose a distinct `RUN_NAME` for your experiment. Keep its code commit, run name,
and shard size unchanged when resuming. A blank `CACHE_CODE_COMMIT` uses the same
commit as `CODE_COMMIT`.

For optional GitHub result export, set `GIT_NAME` to your Git commit author name
(usually your real name, not your GitHub username) and `GIT_EMAIL` to your author
email. You can check your local settings with:

```bash
git config user.name
git config user.email
```

In Colab's **Secrets** sidebar (key icon), add your own `GITHUB_TOKEN` if the
repository is private or you plan to push results. Enable notebook access. For
export, a fine-grained token needs this repository's **Contents: read/write**
permission, and your GitHub account must have repository write access. Add
`HF_TOKEN` if Hugging Face authentication is needed. Do not paste credentials into
notebook cells or share another person's token.

## 3. Run setup and staging

Run code cells individually, in order. **Do not use Run all**: the notebook includes
manual storage/geometry review and optional publishing cells. Stop at an unexpected
error and share the failing cell and its output with the project owner.

Run these sections:

1. Configuration/repository setup.
2. **Persistent outputs and runtime paths**: authorize your Drive.
3. **Install the pinned evaluation environment**.
4. **Restore processed data from Drive, if available**: this skips restoration if
   there is no completed archive in your Drive.
5. **Stage pinned metadata and prepare the frozen evaluation**.
6. **Download and preprocess scans; stage checkpoint**.

Scans and weights download directly onto Colab; they do not need to be on your
computer. The upstream loader may download/preprocess the full KiTS23 dataset and
build train/test caches even though inference uses only the selected test examples.
Watch runtime disk space. A GPU allocation remains occupied during this CPU-heavy
staging step.

## 4. Save the processed data before evaluation

Under **Save processed data for future runtimes**:

1. Run the size-estimate cell.
2. Check your available [Google Drive storage](https://drive.google.com/drive/quota),
   leaving space beyond the estimate.
3. Run the backup cell after staging has finished, without another process modifying
   the data directory.
4. Wait for **Backup complete** and confirm that `MyDrive/MedMeasure/cache/` contains
   both `kits23-<dataset-revision>-v140.tar` and its matching `.tar.json` sidecar.

Backup copies the already-staged scans, masks, metadata, completion markers, and
loader caches. It does not redownload or rerun preprocessing, and it leaves the
original runtime data available for inference. Model weights are separate and are
not backed up. If Drive storage is insufficient, set `SAVE_DATA_CACHE=False` to skip
backup, but the runtime-local data will not survive VM deletion.

Future runtimes using this notebook restore the archive onto local disk before
staging. Restoration takes time and requires sufficient local storage; environment
installation, loader checks, and model-weight downloads still run. Real-data restore
and upstream cache reuse remain unverified until the first successful restored run.

## 5. Check geometry and run the smoke test

Run the first cell under **Geometry check and visual review**. It checks 20 examples
and displays four overlays. Review all PNGs in:

```text
MyDrive/MedMeasure/<RUN_NAME>/geometry-smoke/
```

Check that the tumor contour and measurement axes align with the visible target.
If the checks pass and the overlays look correct, set this in the next cell:

```python
GEOMETRY_REVIEWED = True
```

Run that cell to execute the ten-example smoke test. If geometry or inference fails,
resolve the error before starting the full evaluation. Smoke results are execution
checks, not representative baseline scores.

## 6. Run chunks and score

Run the cell under **Run evaluation chunks** with `MAX_CHUNKS=1` to evaluate one
100-example chunk first. After it succeeds, replace that cell with:

```python
MAX_CHUNKS = 0
script('colab_baseline.py', 'run', *COMMON, '--max-chunks', MAX_CHUNKS)
```

Run it to finish all pending chunks. Completed chunks are verified and skipped;
interrupted chunks are retried from the beginning in fresh attempt directories.

After all chunks complete, run **Score complete results**. The scorer refuses
missing, duplicate, unexpected, or modified logs. It produces full-test and
legacy-disjoint diagnostic summaries using the same predictions. Existing summary
files are not overwritten automatically.

Results persist in:

```text
MyDrive/MedMeasure/<RUN_NAME>/evaluation/
```

## 7. Share results through GitHub

Run **Optional: prepare results for GitHub**. Inspect its staged-file summary and
export size, then run the following commit/push cell. This creates the branch
`results/<RUN_NAME>` with predictions, metrics, manifests, and environment details.
It excludes scans, weights, geometry PNGs, smoke results, and incomplete attempts.

Share the branch name with the project owner. To retrieve it locally, preserve any
local changes, then run:

```bash
git fetch origin
git merge origin/results/kits23-medvision-v0-v140-collaborator
```

Substitute your actual run name. Alternatively check out the results branch directly.
If push fails after the commit succeeds, retry only the push command, as described
at the bottom of the notebook. **Save a copy in GitHub** saves the notebook itself;
the export cells publish generated result files.

## Resume and share a cache

If the runtime is replaced, reopen your saved notebook and keep the same
`CODE_COMMIT`, `RUN_NAME`, and `SHARD_SIZE`. Rerun setup, restore/staging, and the
remaining cells. Drive-backed completed chunks will be reused. If the original VM
only disconnected and reconnects successfully, its files may still be available.

To let another collaborator reuse your processed data, share both cache files.
The notebook expects them in that person's own `MyDrive/MedMeasure/cache/`; they can
copy both there if they have sufficient quota. Merely sharing a folder does not
automatically put it at that path. Share results or cache folders as needed rather
than your entire Drive.

For partial backups, recovery details, and lower-level commands, see the
[Colab runbook](colab-runbook.md). For research scope and evaluation caveats, see the
[project plan](project-plan.md) and [phase-one report](baseline-and-data.md).
