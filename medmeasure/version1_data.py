"""Native-grid KiTS23 inputs, split checks, and specialist measurement metrics."""
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from medmeasure.data import read_plan, rows, sha256
from medmeasure.geometry import fit_mask, matches_annotation_metadata


def read_manifest(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    identities = [r['sample_id'] for r in records]
    if not records or len(set(identities)) != len(identities):
        raise ValueError('Manifest must contain nonempty, unique sample identities')
    return records


def validate_manifests(manifests, plan_path, pin):
    """Check exact official references, then reject patient overlap between splits."""
    if sha256(plan_path) != pin['sha256']['benchmark_plan_biometry_v1.4.0.json.gz']:
        raise ValueError('Annotation plan differs from pinned v1.4.0 plan')
    task = read_plan(plan_path)
    official = {split: {r['sample_id']: r for r in rows(task, split)} for split in ('train', 'test')}
    cases = {}
    for name, records in manifests.items():
        source = official['test' if name == 'test' else 'train']
        for record in records:
            if source.get(record['sample_id']) != record:
                raise ValueError(f"{name} row differs from the pinned official plan: {record['sample_id']}")
        cases[name] = {r['case_id'] for r in records}
    names = list(cases)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if cases[name] & cases[other]:
                raise ValueError(f'Case overlap between {name} and {other}')
    return {name: sorted(ids) for name, ids in cases.items()}


def load_image(row, dataset_dir):
    """Same KiTS23 soft-tissue HU window; array axes are never transposed."""
    image = nib.load(Path(dataset_dir) / row['image_file'])
    if row['slice_dim'] != 2 or row['label'] != 2:
        raise ValueError('Only axial KiTS23 tumor label 2 is supported')
    if list(image.shape) != row['image_size_3d']:
        raise ValueError('Image shape differs from annotation grid')
    if (not matches_annotation_metadata(image.affine, row['affine'])
            or list(nib.aff2axcodes(image.affine)) != row['orientation']):
        raise ValueError('Image affine/orientation differs from annotation grid')
    spacing = np.asarray(image.header.get_zooms()[:2], dtype=float)
    if not matches_annotation_metadata(spacing, row['pixel_size']):
        raise ValueError('Image spacing differs from annotation metadata')
    native = np.asarray(image.dataobj[:, :, row['slice_idx']], dtype=np.float32)
    rendered = ((np.clip(native, -160, 240) + 160) / 400 * 255).astype(np.uint8)
    return Image.fromarray(rendered).convert('RGB'), spacing, image


def load_target(row, dataset_dir, image):
    mask = nib.load(Path(dataset_dir) / row['mask_file'])
    if mask.shape != image.shape or not np.allclose(mask.affine, image.affine, atol=1e-5, rtol=0):
        raise ValueError('Image and mask grids differ')
    return np.asarray(mask.dataobj[:, :, row['slice_idx']]) == row['label']


def processor_inputs(processor, image, question, device):
    # The interface has no reference values, masks, landmarks, or target boxes.
    messages = [{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors='pt')
    return {name: value.to(device) for name, value in inputs.items()}


def measure_prediction(mask, spacing):
    """Use the largest valid predicted component, independent of target overlap."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {'status': 'empty_mask', 'components': 0, 'valid_fits': 0,
                'box_rc_exclusive': None, 'measurement': None}
    rr, cc = np.where(mask)
    box = [int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1]
    fits, components = fit_mask(mask, spacing)
    result = {'status': 'success' if fits else 'failed_fit', 'components': components,
              'valid_fits': len(fits), 'box_rc_exclusive': box, 'measurement': None}
    if fits:
        result['measurement'] = {axis: float(fits[0][axis]) for axis in ('major_mm', 'minor_mm')}
    return result


def score_prediction(row, prediction, predicted_mask, target):
    intersection = int((predicted_mask & target).sum())
    total = int(predicted_mask.sum() + target.sum())
    union = int((predicted_mask | target).sum())
    result = {'sample_id': row['sample_id'], 'case_id': row['case_id'], **prediction,
              'dice': 2 * intersection / total if total else 1.,
              'iou': intersection / union if union else 1., 'mae_mm': None, 'mre': None,
              'reference': {axis: float(np.float16(row[axis])) for axis in ('major_mm', 'minor_mm')},
              'planner_reference': {axis: row[axis] for axis in ('major_mm', 'minor_mm')}}
    if prediction['status'] == 'success':
        errors = [abs(prediction['measurement'][axis] - result['reference'][axis])
                  for axis in ('major_mm', 'minor_mm')]
        result['mae_mm'] = float(np.mean(errors))
        result['mre'] = float(np.mean([error / result['reference'][axis] for error, axis in
                                     zip(errors, ('major_mm', 'minor_mm'))]))
        result['planner_mae_mm'] = float(np.mean([
            abs(prediction['measurement'][axis] - row[axis]) for axis in ('major_mm', 'minor_mm')]))
    return result


def summarize_predictions(results, expected_rows):
    expected = {r['sample_id'] for r in expected_rows}
    actual = [r['sample_id'] for r in results]
    if not expected or len(expected) != len(expected_rows) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Refusing a score with duplicate, missing, or unexpected rows')
    valid = [r for r in results if r['status'] == 'success']
    below = sum(r['mre'] < .1 for r in valid)
    return {'total': len(results), 'finite_successes': len(valid), 'failures': len(results) - len(valid),
            'status_counts': dict(Counter(r['status'] for r in results)),
            'mae_mm_success_only': float(np.mean([r['mae_mm'] for r in valid])) if valid else None,
            'mre_fraction_success_only': float(np.mean([r['mre'] for r in valid])) if valid else None,
            'success_fraction': len(valid) / len(results),
            'below_10_percent_fraction_all_rows': below / len(results),
            'below_10_percent_fraction_success_only': below / len(valid) if valid else None,
            'dice_mean_all_rows': float(np.mean([r['dice'] for r in results])),
            'iou_mean_all_rows': float(np.mean([r['iou'] for r in results]))}


def save_overlay(path, image, prediction, target):
    import cv2
    pixels = np.asarray(image).copy()
    for mask, color in ((target, (0, 255, 0)), (prediction, (255, 0, 255))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pixels, contours, -1, color, 1)
    Image.fromarray(pixels).save(path)
