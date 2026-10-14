# Version 1 first full KiTS23 result

ICE evaluation job **6252813** reports complete official test coverage. Training
was ten total epochs with a frozen backbone; the best validation loss was at
epoch 8. The expected checkpoint is `runs/v1-cached-6251584/best.pt`, selected by
validation loss before the full test. Serialized checkpoint/run provenance has
not yet been retrieved. The source for the specialist numbers is the user's pasted
summary, saved in [summary.json](../results/v1-eval-6252813/summary.json), with
[receipt/validation limits](../results/v1-eval-6252813/receipt.json) and
[machine-readable comparison](../results/v1-eval-6252813/comparison.json).

Baseline summaries come from merged PR #1 at commit
`1866d3c3c1b1ad61d021e8f1e117004dad98e752`, under
`results/kits23-medvision-v0-v140-rik/`. Earlier local rescoring of its 32 shard
logs reproduced the full baseline summary. Baseline provenance declares the same
model/dataset revision and annotation version as the specialist pilot, but the
specialist's per-row identities/run hashes still need retrieval and comparison.

## Primary comparison: all-row below-10%-error rate

Failures are included in the denominator and do not qualify as below 10% error.

| Split | Rows | Baseline | Version 1 | Difference |
| --- | ---: | ---: | ---: | ---: |
| Full official test | 3,179 | 25.39% (807) | 43.91% (1,396) | +18.53 percentage points |
| Historical-disjoint diagnostic | 825 | 11.64% (96) | 38.30% (316) | +26.67 percentage points |

This supports an improvement in the fraction of reported measurement errors below
10% under these respective systems. It is a descriptive comparison, without
patient-level confidence intervals or paired significance analysis.

## Measurement errors and failures

| Split / metric | Baseline | Version 1 |
| --- | ---: | ---: |
| Full: finite success rate | 100.00% | 95.38% |
| Full: success-only MAE | 11.59 mm | 6.62 mm |
| Full: success-only MRE | 44.06% | 24.51% |
| Diagnostic: finite success rate | 100.00% | 93.70% |
| Diagnostic: success-only MAE | 12.22 mm | 7.07 mm |
| Diagnostic: success-only MRE | 55.68% | 30.79% |

**Success-only errors have different denominators.** Version 1 omits its 147 full-
test failures from MAE/MRE while the baseline has 3,179 finite rows. Do not claim
these MAE/MRE changes as a paired error reduction until baseline errors are
recomputed on the same successful subset. The all-row rate above avoids this
failure-exclusion problem.

Version 1 full-test failures: **104 empty masks + 43 failed ellipse fits**. The
historical-disjoint subset has **36 empty masks + 16 failed fits**. Full-test
all-row Dice is **0.6617**, IoU **0.5761**; diagnostic Dice is **0.5986**, IoU
**0.5141**. The textual baseline has no directly comparable predicted-mask Dice.

## Interpretation and next work

- Training online Dice 0.9066 and noisy validation-loss gains suggest a
  generalization gap; the full-test result does not erase that concern. Validation
  Dice was not recorded in the supplied epoch metrics, so do not equate training
  and test Dice with a directly matched overfit estimate.
- This is a specialist with additional mask supervision, geometric measurement,
  and different prompt/rendering from the textual baseline. It does not establish
  improved VLM-generated reasoning or isolate the benefit of conditioning.
- Released-backbone exposure remains: 104 official test cases appear in audited
  legacy sizing training. The 43-case diagnostic is absent from those audited
  sizing plans, not proven unseen across all tasks.
- Retrieve `run.json`, per-row predictions, selected rows, overlays and training
  checkpoints/logs. Audit matching identities/pins, compare paired successful-row
  errors, stratify by patient/reference size and inspect empty-mask/failed-fit cases.
- Retain this run as the first result. Future dropout/learning-rate changes should
  be selected on validation cases, with test results treated as descriptive after
  this test exposure. Validation scoring is not yet implemented in the current CLI.

## Paired outputs now available

The transferred prediction/selected-row files contain all 3,179 unique rows and
match the baseline manifest SHA256 and every scored reference axis.
[Paired output report](../results/v1-baseline-paired-6252813/README.md),
[CSV](../results/v1-baseline-paired-6252813/paired.csv), and
[summary](../results/v1-baseline-paired-6252813/summary.json) pair all 32 baseline
shards with specialist predictions. On the same 3,032 successful rows, baseline
MAE is 11.6272 mm vs 6.6233 mm; MRE is 39.5123% vs 24.5121%. Version 1 has lower
MRE on 1,958 and higher MRE on 1,074 rows. This does not include its 147 failures
in those error means; all-row accuracy continues to include them. The baseline
produces text/axis outputs rather than masks; no baseline segmentation is inferred.
Specialist run.json is now present: its manifest hash and model pin match, and
its recorded runtime source hashes match the local inference files. The checkpoint
SHA256 is recorded, but checkpoint bytes have not been copied for independent
verification. [Side-by-side notebook](../notebooks/baseline_vs_version1.ipynb) now
visualizes reported baseline axes/reconstructed ellipses and V1 mask-derived fits;
see the [coordinate audit and usage guide](baseline-v1-visualization.md).
