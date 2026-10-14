# Side-by-side baseline / Version 1 sizing viewer

Open [the notebook](../notebooks/baseline_vs_version1.ipynb) in Jupyter or VS Code,
select a Python kernel, and run cells in order. No GPU or weights are needed.
Dependencies are `requirements-cpu.txt` plus IPython; optional `ipywidgets` supplies
the case/slice dropdowns. The notebook contains an installation command and a
manual `show_case()` fallback.

Defaults use `data/medvision/Datasets/KiTS23`, the transferred evaluation directory
`results/v1-eval-6252813`, and `results/v1-baseline-paired-6252813/paired.jsonl`.
These paths are editable. Move the notebook together with the
`medmeasure/comparison_viz.py` helper if using another checkout. Regenerate the
paired file with `scripts/compare_version1_baseline.py` if needed.

The left panel draws baseline **reported normalized endpoints** as solid orange
axes, with a dashed reconstruction from their lengths, midpoint-average center,
and major-axis direction. Text endpoints can be inconsistent, so the viewer reports
center disagreement, nonperpendicularity, out-of-frame coordinates and geometric
lengths separately from the scored final answer. Final lengths alone do not supply
location/orientation; six unparseable responses show no fabricated baseline ellipse.

The right panel draws the saved Version 1 mask in magenta and the same physical
ellipse fit used for scoring. Empty masks/failed fits retain their failure status.
Optional green contours and matching reference ellipses are ground-truth review
aids, not model inputs. Both panels share native CT orientation and view limits.
Target-centered zoom can clip distant wrong predictions and is labelled accordingly.

Controls select any case/slice, toggle reference overlays, and toggle zoom. Other
cells list successes/failures, print the original baseline reasoning, and export
the last displayed slice. Test examples are descriptive comparisons; tune future
training changes on validation cases.

## Checks performed

- Verified pinned baseline renderer SHA256 against the pilot config. Axial loading
  is direct `volume[:, :, slice_idx]` followed by `Image.fromarray`; full-frame
  resizing has no transpose/flip/crop/padding. **Endpoint coordinates use a
  separate lower-left origin**: native column = x × width; row = (1 − y) × height.
  Verified the pinned [sizing training target conversion](https://github.com/YongchengYAO/MedVision/blob/45fa8e52684a65da6e1c0ef6ef422d5d4c3e94bd/src/medvision_bm/sft/sft_utils.py#L1336).
  The original viewer incorrectly assumed a top-left endpoint origin; corrected
  on October 14. Earlier baseline overlays/previews were vertically mirrored.
  Saved final sizing predictions and benchmark scores are unaffected.
- All 16 notebook cells (7 code) validated and executed in a temporary Python 3.11
  Jupyter kernel; widget case/slice/zoom/reference callbacks exercised programmatically.
- Three real test-slice figures rendered, including a saved empty mask and another
  case. Recomputed mask ellipse axes match saved Version 1 measurements. Two figures
  visually inspected for readable layout and consistent orientation.
- Analytical anisotropic ellipse sampling, known normalized endpoint projection,
  missing endpoints, and a real unparseable-response figure checked. All 3,179
  responses scanned: 3,173 unambiguous endpoint sets, six unavailable.
- The notebook does not rerun models or recalculate benchmark scores. The actual
  browser frontend widget appearance was not inspected; kernel callbacks work.

Preview figures and an executed check copy are under ignored
`runs/baseline-v1-visual-review/`. No training source or feature-cache contract changed.

October 14 correction checks: inverted the upstream formula on an asymmetric
100×200 grid, checked physical lengths with unequal pixel spacing, and regenerated
three real previews. Case 00565 slice 10 baseline center is ~9 pixels from the
target centroid (previously ~211); case 00145 slice 10 is ~15 (previously ~150).
Case 00145 slice 1 remains poorly localized after the correct conversion; this is
a model prediction error, not a reason to select a coordinate convention per row.

Crash follow-up: the October 14 macOS diagnostic report shows a native Matplotlib
`_macosx` FigureCanvas/AppKit abort during an earlier command-line check. The
notebook now explicitly selects inline rendering before importing pyplot and
reloads the helper when rerun. All seven code cells passed again in a fresh kernel
without an external backend override; inline image output and PNG export verified.
Restart the kernel and run all cells when opening the updated notebook.
