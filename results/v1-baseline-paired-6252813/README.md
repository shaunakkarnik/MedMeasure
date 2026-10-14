# Paired Version 1 / baseline outputs

All 3,179 case/slice/label identities matched uniquely. The selected-row manifest
SHA256 exactly matches baseline provenance; every scored float16 major/minor
reference matches. Baseline source: merged commit `1866d3c3c1b1ad61d021e8f1e117004dad98e752`,
32 sample logs. Version 1: transferred `results/v1-eval-6252813/` files.

- [Per-row CSV](paired.csv): reference axes, baseline final answer text/upstream
  errors, Version 1 axes/errors/status/Dice, and local mask/available-overlay paths.
- [Per-row JSONL](paired.jsonl): same information plus full baseline response text.
- [Summary](summary.json): full/diagnostic/size groups and paired error comparisons.

On the **same 3,032 rows with successful measurements from both systems**, baseline
MAE is **11.63 mm** vs Version 1 **6.62 mm**;
MRE is **39.51%** vs **24.51%**.
Version 1 has lower MRE on 1,958 rows and higher MRE
on 1,074. The 147 Version 1 failures remain excluded
from these paired error means, but included as misses in all-row below-10% rates.

All-row below-10%-error outcomes: both 472; baseline only 335; Version 1 only 924;
neither 1,448. The net gain is 589 rows. Adjacent slices are correlated; these are
descriptive row counts, not independent-patient significance claims.

## Visual review examples

The baseline predicts text and axis endpoints/lengths, not a segmentation mask.
Its final answer text can be compared with the reference axes and Version 1
measurements beside the native-grid overlay. Its textual endpoints are not drawn
on these overlays: baseline prompt rendering differs, and its coordinate transform
has not been independently verified for this comparison. Do not treat its output
as a predicted mask.

These are the first twenty rows with saved overlays, all from case_00145; they
are a systematic preview, not a representative sample of all 147 test cases.

| Sample | Reference major / minor (mm) | Baseline answer text | Version 1 major / minor (mm) | Overlay |
| --- | --- | --- | --- | --- |
| case_00145 slice 1 | 23.75 / 15.53 | (68.874, 53.819) | empty_mask | [View](../v1-eval-6252813/case_00145_1.png) |
| case_00145 slice 2 | 28.33 / 23.31 | (15.232, 15.232) | 26.00 / 24.33 | [View](../v1-eval-6252813/case_00145_2.png) |
| case_00145 slice 3 | 33.62 / 30.16 | (33.250, 25.000) | 29.18 / 26.51 | [View](../v1-eval-6252813/case_00145_3.png) |
| case_00145 slice 4 | 38.12 / 32.94 | (37.240, 31.048) | 38.08 / 33.45 | [View](../v1-eval-6252813/case_00145_4.png) |
| case_00145 slice 5 | 57.00 / 43.34 | (40.252, 31.522) | 55.13 / 39.01 | [View](../v1-eval-6252813/case_00145_5.png) |
| case_00145 slice 6 | 60.53 / 49.00 | (53.759, 37.679) | 59.49 / 45.19 | [View](../v1-eval-6252813/case_00145_6.png) |
| case_00145 slice 7 | 65.19 / 55.81 | (43.778, 31.004) | 63.68 / 50.94 | [View](../v1-eval-6252813/case_00145_7.png) |
| case_00145 slice 8 | 69.69 / 60.59 | (39.118, 31.046) | 67.34 / 56.46 | [View](../v1-eval-6252813/case_00145_8.png) |
| case_00145 slice 9 | 75.00 / 62.09 | (55.951, 41.194) | 73.23 / 59.80 | [View](../v1-eval-6252813/case_00145_9.png) |
| case_00145 slice 10 | 78.25 / 62.88 | (40.115, 10.460) | 75.34 / 62.44 | [View](../v1-eval-6252813/case_00145_10.png) |
| case_00145 slice 12 | 82.62 / 64.69 | (54.179, 43.939) | 81.51 / 71.72 | [View](../v1-eval-6252813/case_00145_12.png) |
| case_00145 slice 13 | 84.12 / 68.31 | (72.054, 62.130) | 76.73 / 70.77 | [View](../v1-eval-6252813/case_00145_13.png) |
| case_00145 slice 14 | 85.81 / 68.44 | (76.223, 67.414) | 78.85 / 68.55 | [View](../v1-eval-6252813/case_00145_14.png) |
| case_00145 slice 15 | 85.75 / 67.88 | (77.553, 68.029) | 87.60 / 78.72 | [View](../v1-eval-6252813/case_00145_15.png) |
| case_00145 slice 16 | 86.38 / 67.75 | (82.938, 76.913) | 88.23 / 80.25 | [View](../v1-eval-6252813/case_00145_16.png) |
| case_00145 slice 17 | 88.00 / 63.72 | (80.868, 75.876) | 84.19 / 67.79 | [View](../v1-eval-6252813/case_00145_17.png) |
| case_00145 slice 18 | 86.50 / 63.16 | (78.144, 71.879) | 84.82 / 68.12 | [View](../v1-eval-6252813/case_00145_18.png) |
| case_00145 slice 19 | 83.12 / 60.62 | (65.584, 56.312) | 83.16 / 64.69 | [View](../v1-eval-6252813/case_00145_19.png) |
| case_00145 slice 20 | 81.75 / 58.53 | (44.636, 35.111) | 81.54 / 58.84 | [View](../v1-eval-6252813/case_00145_20.png) |
| case_00145 slice 21 | 81.69 / 56.50 | (44.878, 33.268) | 76.02 / 56.15 | [View](../v1-eval-6252813/case_00145_21.png) |
