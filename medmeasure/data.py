"""Read versioned KiTS23 plans without importing the self-installing HF loader."""
import gzip
import hashlib
import json
import math
from pathlib import Path

DATASET_REVISION = 'f4040ed7d2d2b45e09c1a996ad2969f2018051d3'
ANNOTATION_VERSION = '1.4.0'
TASK = 'KiTS23_TumorLesionSize_Task01_Axial-CoT'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_plan(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        plan = json.load(stream)
    task = plan['tasks'][0]
    if plan['dataset_info']['dataset'] != 'KiTS23' or task['target_label'] != 2:
        raise ValueError('Expected KiTS23 Task01, tumor label 2')
    return task


def case_ids(task, split):
    ids = [case['case_ID'] for case in task[f'{split}_cases']]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate case IDs in {split}')
    return set(ids)


def rows(task, split):
    """Mirror the pinned loader's axial single-component order; keep loader indices."""
    result = []
    for case in task[f'{split}_cases']:
        info = case['image_file_info']
        for item in case.get('slice_profiles_z', []):
            if item['n_total_clusters'] > 1:
                continue
            profiles = item['slice_profile']
            if len(profiles) != 1:
                raise ValueError(f"Expected one measurable component: {case['case_ID']}")
            axes = {axis['metric_key']: axis['metric_value'] for axis in profiles[0]}
            major, minor = axes['L-1-2'], axes['L-3-4']
            if not all(math.isfinite(x) and x > 0 for x in [major, minor]) or major < minor:
                raise ValueError('Invalid reference axes')
            if any(axis['metric_unit'] != 'mm' for axis in profiles[0]):
                raise ValueError('Expected millimeter measurements')
            result.append({
                'sample_id': f"KiTS23:{case['case_ID']}:2:{item['slice_idx']}:2",
                'case_id': case['case_ID'], 'official_split': split,
                'loader_index': len(result), 'slice_dim': 2, 'slice_idx': item['slice_idx'],
                'label': 2, 'image_file': case['image_file'], 'mask_file': case['mask_file'],
                'landmark_file': case['landmark_file'], 'image_size_3d': info['array_size'],
                'pixel_size': info['voxel_size'][:2], 'voxel_size': info['voxel_size'],
                'affine': info['affine'], 'orientation': info['orientation'],
                'major_mm': major, 'minor_mm': minor,
            })
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_rows(path, records):
    with open(path, 'w') as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
