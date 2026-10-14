"""Visualize baseline CoT axes and mask-derived Version 1 ellipses on native CT."""
from functools import lru_cache
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from medmeasure.geometry import fit_mask
from medmeasure.version1_data import load_image, load_target


NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
POINT = re.compile(r'[\(\[]\s*(' + NUMBER + r')\s*,\s*(' + NUMBER + r')\s*[\)\]]')


def baseline_axes(response, shape, spacing):
    """Parse explicit step answers only; never invent location from final lengths."""
    axes = []
    for step in (1, 2):
        answers = re.findall(fr'<step-{step}-answer>(.*?)</step-{step}-answer>', response, re.DOTALL)
        if len(answers) != 1:
            return None
        points = POINT.findall(answers[0])
        if len(points) != 2:
            return None
        # Upstream sizing training targets use a lower-left origin:
        # x=column/width, y=1-row/height. Convert to the displayed native grid.
        normalized = np.asarray(points, dtype=float)
        normalized[:, 1] = 1 - normalized[:, 1]
        axes.append(normalized * [shape[1], shape[0]])
    axes = np.asarray(axes)
    if not np.isfinite(axes).all():
        return None
    physical = axes * [spacing[1], spacing[0]]
    vectors = physical[:, 1] - physical[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 0):
        return None
    centers = physical.mean(axis=1)
    deviation = np.degrees(np.arcsin(np.clip(abs(np.dot(*vectors)) / np.prod(lengths), 0, 1)))
    normalized = axes / [shape[1], shape[0]]
    return {'points_xy': axes, 'major_mm': float(lengths[0]), 'minor_mm': float(lengths[1]),
            'center_gap_mm': float(np.linalg.norm(centers[0] - centers[1])),
            'perpendicular_deviation_deg': float(deviation),
            'out_of_frame': bool(np.any((normalized < 0) | (normalized > 1))),
            # Inconsistent axes do not define one exact ellipse: retain the
            # actual segments and label the midpoint-average reconstruction.
            'ellipse': {'center_xy_mm': centers.mean(axis=0), 'axes_mm': lengths,
                        'angle_deg': float(np.degrees(np.arctan2(vectors[0, 1], vectors[0, 0])))}}


def ellipse_points(fit, spacing, count=180):
    """Sample in physical x/y, then divide per-axis spacing (anisotropy-safe)."""
    angle = np.radians(fit['angle_deg'])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    theta = np.linspace(0, 2 * np.pi, count)
    radii = np.asarray(fit['axes_mm']) / 2
    physical = np.stack([radii[0] * np.cos(theta), radii[1] * np.sin(theta)], axis=1)
    physical = physical @ rotation.T + np.asarray(fit['center_xy_mm'])
    return physical / [spacing[1], spacing[0]]


def draw_fit(ax, fit, spacing, color, label):
    curve = ellipse_points(fit, spacing)
    ax.plot(curve[:, 0], curve[:, 1], '--', color=color, linewidth=1.5, label=label)


class ComparisonViewer:
    def __init__(self, dataset_dir, run_dir, paired_file):
        self.dataset_dir, self.run_dir = Path(dataset_dir), Path(run_dir)
        self.pairs = [json.loads(line) for line in Path(paired_file).read_text().splitlines() if line.strip()]
        selected = [json.loads(line) for line in (self.run_dir / 'selected_rows.jsonl').read_text().splitlines() if line.strip()]
        self.rows = {row['sample_id']: row for row in selected}
        self.by_id = {row['sample_id']: row for row in self.pairs}
        if len(self.rows) != len(selected) or len(self.by_id) != len(self.pairs) or set(self.rows) != set(self.by_id):
            raise ValueError('Paired outputs and selected rows must match uniquely')
        for pair in self.pairs:
            row = self.rows[pair['sample_id']]
            expected = {axis: float(np.float16(row[axis])) for axis in ('major_mm', 'minor_mm')}
            actual = {axis: pair['reference_' + axis] for axis in expected}
            if actual != expected:
                raise ValueError('Paired reference does not match the selected row')

    @lru_cache(maxsize=8)
    def scene(self, sample_id):
        row, pair = self.rows[sample_id], self.by_id[sample_id]
        image, spacing, volume = load_image(row, self.dataset_dir)
        target = load_target(row, self.dataset_dir, volume)
        mask_path = self.run_dir / 'masks' / f"{row['case_id']}_{row['slice_idx']}.png"
        with Image.open(mask_path) as saved:
            prediction = np.asarray(saved.convert('L')) > 0
        if prediction.shape != target.shape:
            raise ValueError('Saved prediction mask differs from native grid')
        fits, _ = fit_mask(prediction, spacing)
        fit = fits[0] if fits else None
        if pair['version1_status'] == 'success':
            if fit is None or not np.allclose(
                    [fit['major_mm'], fit['minor_mm']],
                    [pair['version1_major_mm'], pair['version1_minor_mm']], atol=1e-4, rtol=1e-5):
                raise ValueError('Recomputed ellipse differs from scored Version 1 measurement')
        elif fit is not None:
            raise ValueError('Saved failed prediction unexpectedly has a valid ellipse')
        reference_fits, _ = fit_mask(target, spacing)
        reference_fit = min(reference_fits, key=lambda f: abs(f['major_mm'] - row['major_mm'])
                            + abs(f['minor_mm'] - row['minor_mm'])) if reference_fits else None
        reference_matches = reference_fit is not None and np.allclose(
            [reference_fit['major_mm'], reference_fit['minor_mm']],
            [row['major_mm'], row['minor_mm']], atol=1e-4, rtol=1e-5)
        return np.asarray(image), spacing, target, prediction, fit, reference_fit if reference_matches else None

    def figure(self, sample_id, show_reference=True, zoom=False):
        pair = self.by_id[sample_id]
        image, spacing, target, prediction, fit, reference_fit = self.scene(sample_id)
        parsed = baseline_axes(pair['baseline_response'], target.shape, spacing)
        fig, axes = plt.subplots(1, 2, figsize=(13, 7))
        for ax in axes:
            ax.imshow(image, origin='upper')
            if show_reference and target.any():
                ax.contour(target, levels=[.5], colors=['lime'], linewidths=.9)
                if reference_fit:
                    draw_fit(ax, reference_fit, spacing, 'lime', 'Reference ellipse')
            ax.set_xlim(-.5, image.shape[1] - .5)
            ax.set_ylim(image.shape[0] - .5, -.5)
            ax.set_aspect('equal')
            ax.axis('off')
        if parsed:
            for i, segment in enumerate(parsed['points_xy']):
                axes[0].plot(segment[:, 0], segment[:, 1], 'o-', color='orange', markersize=3,
                             linewidth=1.5, label='Baseline reported axes' if i == 0 else None)
            draw_fit(axes[0], parsed['ellipse'], spacing, 'orange', 'Endpoint reconstruction')
        else:
            axes[0].text(.03, .97, 'No unambiguous endpoints: no ellipse drawn', transform=axes[0].transAxes,
                         color='white', va='top', fontsize=9, bbox={'facecolor': 'black', 'alpha': .7})
        if prediction.any():
            axes[1].contour(prediction, levels=[.5], colors=['magenta'], linewidths=1)
        if fit:
            draw_fit(axes[1], fit, spacing, 'magenta', 'Predicted-mask ellipse')
        if zoom and target.any():
            rr, cc = np.where(target)
            pad = max(30, int(max(np.ptp(rr), np.ptp(cc)) * .75))
            for ax in axes:
                ax.set_xlim(max(-.5, cc.min() - pad), min(image.shape[1] - .5, cc.max() + pad))
                ax.set_ylim(min(image.shape[0] - .5, rr.max() + pad), max(-.5, rr.min() - pad))
        axes[0].set_title('Baseline: text-reported axes (orange)', fontsize=12)
        axes[1].set_title('Version 1: predicted mask and fitted ellipse (magenta)', fontsize=12)
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='lower right', fontsize=8)
        major, minor = pair['version1_major_mm'], pair['version1_minor_mm']
        v1_text = f'{major:.2f}, {minor:.2f} mm' if major is not None else pair['version1_status']
        baseline_mre = f"{pair['baseline_mre'] * 100:.1f}%" if pair['baseline_mre'] is not None else 'failed'
        v1_mre = f"{pair['version1_mre'] * 100:.1f}%" if pair['version1_mre'] is not None else 'failed'
        fig.suptitle(f"{pair['case_id']} | axial slice {pair['slice_idx']} | {pair['size_bin']}"
                     + (' | target-centered zoom (may clip distant predictions)' if zoom else ''), fontsize=13)
        fig.text(.5, .075, f"Reference: {pair['reference_major_mm']:.2f}, {pair['reference_minor_mm']:.2f} mm"
                 f"   |   Baseline final answer: {pair['baseline_answer_text']} mm"
                 f"   |   Version 1: {v1_text}", ha='center', fontsize=10)
        details = f'Baseline scored MRE: {baseline_mre} | Version 1 scored MRE: {v1_mre} | V1 Dice: {pair["version1_dice"]:.3f}'
        if parsed:
            details += (f"\nBaseline endpoint lengths: {parsed['major_mm']:.2f}, {parsed['minor_mm']:.2f} mm"
                        f" | centers apart: {parsed['center_gap_mm']:.2f} mm"
                        f" | deviation from perpendicular: {parsed['perpendicular_deviation_deg']:.1f}°")
            if parsed['out_of_frame']:
                details += ' | endpoints outside image'
        fig.text(.5, .025, details, ha='center', fontsize=9)
        fig.subplots_adjust(left=.01, right=.99, top=.89, bottom=.13, wspace=.05)
        return fig, parsed

    def show(self, sample_id, show_reference=True, zoom=False):
        from IPython.display import display
        self.current_sample_id = sample_id
        figure, parsed = self.figure(sample_id, show_reference, zoom)
        display(figure)
        plt.close(figure)
        return parsed

    def browse(self):
        """Optional Jupyter widgets; manual show() works without ipywidgets."""
        import ipywidgets as widgets
        from IPython.display import display, clear_output

        case = widgets.Dropdown(options=sorted({r['case_id'] for r in self.pairs}), description='Case:')
        sample = widgets.Dropdown(description='Slice:')
        reference = widgets.Checkbox(value=True, description='Show reference')
        zoom = widgets.Checkbox(value=False, description='Zoom to target')
        output = widgets.Output()

        def render(change=None):
            if sample.value is None:
                return
            with output:
                clear_output(wait=True)
                self.show(sample.value, reference.value, zoom.value)

        def update_slices(change=None):
            chosen = [r for r in self.pairs if r['case_id'] == case.value]
            sample.options = [(f"{r['slice_idx']} | {r['version1_status']} | Dice {r['version1_dice']:.2f}",
                               r['sample_id']) for r in chosen]
            render()

        case.observe(update_slices, names='value')
        for control in (sample, reference, zoom):
            control.observe(render, names='value')
        update_slices()
        display(widgets.VBox([widgets.HBox([case, sample]), widgets.HBox([reference, zoom]), output]))
        return case, sample, reference, zoom
