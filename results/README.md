# Results index

All full-test runs use the 3,179 official KiTS23 v1.4.0 axial sizing slices.
The historical-disjoint diagnostic covers 825 slices from 43 cases absent from
the audited historical sizing training plans. See the [split audit](../docs/baseline-and-data.md)
for how those cases were selected.

| Run | Model or checkpoint | Split and evaluator | Saved evidence |
| --- | --- | --- | --- |
| Final four-method comparison `7162509` | Released MedVision-V0, V2 Text and Joint epoch-3 exports, and saved V1 measurements | Full test and historical-disjoint subset. The three text models use the pinned upstream vLLM evaluator with image-first prompts. | [Final summary](v2-aligned-test-7162509/summary.json) and [runbook](../docs/version2-aligned.md). The paired predictions are stored on ICE. |
| Image-first smoke test `7161524` | Same three text models as the final comparison | First 10 official test slices. Checks that each model produces scorable answers. | [Frozen](v2-aligned-smoke-7161524/frozen.json), [Text](v2-aligned-smoke-7161524/text.json), [Joint](v2-aligned-smoke-7161524/joint.json) |
| Earlier V2 full test `7118618` | Frozen MedVision-V0 and the selected V2 Text and Joint epoch-3 checkpoints | Full test and historical-disjoint subset, using the [local Transformers evaluator](../scripts/version2_test.py). | [Frozen](v2-official-7118618/frozen-summary.json), [Text](v2-official-7118618/text-summary.json), [Joint](v2-official-7118618/joint-summary.json). The checkout contains summaries transcribed from ICE. |
| V2 checkpoint validation | Frozen MedVision-V0 and Text and Joint epochs 1–3. Selected epoch-3 checkpoints are `runs/v2-7087274/best.pt` (Text) and `runs/v2-7087275/best.pt` (Joint). | 1,263 held-out validation slices, scored by the [V2 training script](../scripts/version2.py). | [All validation summaries](v2-validation-epoch1/), including epochs 2 and 3. |
| V1 full test `6252813` | Frozen MedVision-V0 with the trained mask decoder. The expected decoder checkpoint is `runs/v1-cached-6251584/best.pt`, selected by validation loss at epoch 8. | Full test and historical-disjoint subset. The [V1 evaluator](../scripts/version1.py) measures ellipse axes from predicted masks. | [V1 report](../docs/version1-results.md), [run metadata](v1-eval-6252813/run.json), and [paired results against the historical baseline](v1-baseline-paired-6252813/README.md). |
| Historical released-model baseline | Released MedVision-V0 at the revision recorded in the [evaluation metadata](kits23-medvision-v0-v140-rik/evaluation.json). | Full test and historical-disjoint subset, using the original pinned upstream benchmark. Its sample logs provide the prompt and target references for the final image-first runner. | [Full test](kits23-medvision-v0-v140-rik/test-summary.json), [diagnostic subset](kits23-medvision-v0-v140-rik/test_legacy_disjoint-summary.json), and [sample logs](kits23-medvision-v0-v140-rik/chunks/). |

Use the final four-method comparison to assess the effect of adding mask
supervision under the shared image-first protocol. Its threshold accuracy uses
all slices, counting failed predictions as misses. It reports MAE and mean
relative error on the slices where all four methods produced scorable answers,
as well as errors over each method's own successful predictions.
