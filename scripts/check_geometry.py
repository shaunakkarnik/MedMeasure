"""CPU-only image/mask validation and reference-measurement parity with overlays."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medmeasure.data import sha256, write_json, write_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--dataset-dir', type=Path, required=True, help='Directory containing KiTS23 Images/ and Masks/')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=20, help='0 checks all rows; otherwise sample across reference sizes')
    args = p.parse_args()
    if args.limit < 0:
        p.error('limit cannot be negative')
    import nibabel as nib
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from medmeasure.geometry import fit_mask

    records = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if not records:
        p.error('Empty manifest')
    if args.limit and len(records) > args.limit:
        records.sort(key=lambda r: r['major_mm'])
        records = [records[i] for i in np.linspace(0, len(records)-1, args.limit, dtype=int)]
    args.output.mkdir(parents=True, exist_ok=False)
    results, hashes = [], {}
    for row in records:
        result = {'sample_id': row['sample_id'], 'status': 'failed'}
        try:
            image_path = args.dataset_dir / row['image_file']
            mask_path = args.dataset_dir / row['mask_file']
            image, mask = nib.load(image_path), nib.load(mask_path)
            for path in [image_path, mask_path]:
                if str(path) not in hashes:
                    hashes[str(path)] = sha256(path)
            if image.shape != mask.shape or not np.allclose(image.affine, mask.affine, atol=1e-5):
                raise ValueError('Image/mask grid mismatch')
            if list(image.shape) != row['image_size_3d'] or not np.allclose(image.affine, row['affine'], atol=1e-4):
                raise ValueError('Image differs from annotation grid')
            if list(nib.aff2axcodes(image.affine)) != row['orientation']:
                raise ValueError('Orientation differs from annotation')
            if row['slice_dim'] != 2 or row['label'] != 2:
                raise ValueError('This pilot only supports axial tumor label 2')
            spacing = image.header.get_zooms()[:2]
            if not np.allclose(spacing, row['pixel_size'], rtol=1e-5):
                raise ValueError('Spacing differs from annotation')
            idx = row['slice_idx']
            image2d = np.asarray(image.dataobj[:, :, idx])
            mask2d = np.asarray(mask.dataobj[:, :, idx]) == row['label']
            fits, components = fit_mask(mask2d, spacing)
            result.update(components=components, fits=len(fits), mask_pixels=int(mask2d.sum()),
                          mask_fraction=float(mask2d.mean()))
            if components != 1 or len(fits) != 1:
                raise ValueError('Expected exactly one component and one valid fit')
            fit = fits[0]
            errors = {axis: fit[axis]-row[axis] for axis in ['major_mm', 'minor_mm']}
            # Full-precision planner references, not float16 loader values.
            passed = all(np.isclose(fit[a], row[a], atol=1e-4, rtol=1e-5) for a in errors)
            result.update(status='passed' if passed else 'mismatch', errors_mm=errors,
                          predicted={a: fit[a] for a in errors}, reference={a: row[a] for a in errors})
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image2d, cmap='gray', vmin=-160, vmax=240)
            ax.contour(mask2d, levels=[.5], colors=['lime'], linewidths=.6)
            # Physical x/y axes back to native array columns/rows, with no image reorientation.
            angle = np.deg2rad(fit['angle_deg'])
            center = np.array(fit['center_xy_mm'])
            for length, direction, color in [(fit['axes_mm'][0], angle, 'cyan'),
                                               (fit['axes_mm'][1], angle+np.pi/2, 'yellow')]:
                delta = length/2*np.array([np.cos(direction), np.sin(direction)])
                endpoints = np.array([center-delta, center+delta]) / np.array(spacing)[::-1]
                ax.plot(endpoints[:,0], endpoints[:,1], color=color, linewidth=1)
            ax.set_title(f"{row['case_id']} slice {idx}: {result['status']}\n{fit['major_mm']:.3f} × {fit['minor_mm']:.3f} mm")
            ax.axis('off')
            fig.savefig(args.output / f"{row['case_id']}_{idx}.png", dpi=120, bbox_inches='tight')
            plt.close(fig)
        except (OSError, ValueError, IndexError) as error:
            result['reason'] = str(error)
        results.append(result)
    write_rows(args.output / 'results.jsonl', results)
    failed = sum(r['status'] != 'passed' for r in results)
    write_json(args.output / 'summary.json', {
        'manifest_sha256': sha256(args.manifest), 'checked': len(results), 'failed': failed,
        'files_sha256': hashes, 'absolute_tolerance_mm': 1e-4, 'relative_tolerance': 1e-5,
        'visual_review_required': True, 'numpy': np.__version__,
        'note': 'Diagnostic native-grid CT window; not the VLM rendered-input preprocessing.',
    })
    print(f'Checked {len(results)} rows; {failed} failed. Inspect overlays before accepting parity.')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
