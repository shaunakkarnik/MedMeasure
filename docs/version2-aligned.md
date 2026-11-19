# Final V2 evaluation on ICE

The comparison uses released MedVision-V0, V2 text and joint epoch-3 checkpoints,
and V1's saved geometric measurements on the 3,179 official KiTS23 test slices.
The three text models use the pinned upstream vLLM benchmark with image-first prompts.
See [current work](../CONTINUITY.md) for run status and the
[results index](../results/README.md) for saved evidence.

## Setup and smoke test

Run from the checkout with the [ICE data, audit, cache, and training environment](ice-version2.md)
in place. The launchers use the parent of the checkout as `$WORK`, or an
absolute `MEDMEASURE_WORK` override. For a fresh benchmark environment, supply
both selected checkpoints to the CPU setup job:

```bash
WORK="${MEDMEASURE_WORK:-$(cd .. && pwd)}"
mkdir -p logs
sbatch cluster/ice-v2-original-prepare.sbatch \
  "$WORK/runs/v2-7087274/best.pt" "$WORK/runs/v2-7087275/best.pt"
```

The setup builds `.venv-original`, prepares benchmark metadata, and writes the
three model directories under `$WORK/models`. It needs the prepared-data
archive and checksum sidecar in `$WORK` if staging metadata must be recovered.
Allow about 34 GB for the two merged model exports. Use the existing setup and
exports when continuing the current run.

After setup succeeds, a smoke array runs ten examples per model:

```bash
sbatch --time=00:30:00 cluster/ice-v2-aligned.sbatch smoke
```

## Full evaluation

After the smoke tasks complete successfully and answers are inspected:

```bash
sbatch --array=0-11%3 cluster/ice-v2-aligned.sbatch test
```

Four disjoint interleaved shards cover all 3,179 test examples for each
model. Tasks 0–3 are frozen, 4–7 text, and 8–11 joint. The array permits up
to three simultaneous tasks. Each requests one compatible NVIDIA GPU,
96 GB host memory, and eight hours.

Logs retain generation progress. Receipts bind the content order,
input-adapter hash, execution-script hash, model/checkpoint provenance,
shard membership, and manifest hash. Completed shards can be retried safely
under the same output root. An incomplete shard starts a new attempt.

For example, retry task 5 using the original array's output root:

```bash
sbatch --array=5 cluster/ice-v2-aligned.sbatch test \
  "$WORK/runs/v2-aligned-test-ARRAY_JOB_ID"
```

## Merge and compare

Once all 12 tasks complete, set the array job ID and run:

```bash
WORK="${MEDMEASURE_WORK:-$(cd .. && pwd)}"
JOB_ID=ARRAY_JOB_ID
RUN="$WORK/runs/v2-aligned-test-$JOB_ID"

for model in frozen text joint; do
  .venv-original/bin/python scripts/version2_original.py merge \
    --audit "$WORK/audit" \
    --output "$RUN/$model"
done

.venv-original/bin/python scripts/compare_final_results.py \
  --frozen-run "$RUN/frozen" \
  --text-run "$RUN/text" \
  --joint-run "$RUN/joint" \
  --output "$WORK/runs/final-comparison-$JOB_ID"
```

The merge rejects missing, duplicate or mismatched shards and verifies
original question strings and float16 targets against the old baseline logs.
The final comparison requires a frozen run with matching image-first
content order and adapter/execution code. It reports all-row threshold
accuracy and failure counts plus common-success MAE/MRE across all four
methods, including V1's saved geometric predictions.

The comparison also reports MAE and MRE on each method's successful rows.
V1 uses its trained 448-pixel processor budget. The text models use the
benchmark processor defaults. Report the full test and historical-disjoint
subset together.
