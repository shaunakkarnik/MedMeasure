"""CPU-only contracts shared by baseline preflight and local validation."""
from pathlib import Path
import math


def validate_loaded_row(actual, expected):
    """Validate the pinned HF Sequence-of-struct representation and float16 labels."""
    import numpy as np

    for field in ['image_file', 'mask_file']:
        if Path(actual[field]).name != Path(expected[field]).name:
            raise ValueError(f'Loader {field} differs from manifest')
    for field in ['slice_idx', 'slice_dim', 'label']:
        if actual[field] != expected[field]:
            raise ValueError(f'Loader {field} differs from manifest')
    profile = actual['biometric_profile']
    # datasets.Sequence(dict) produces a dict of lists, as consumed by
    # upstream doc_to_target_TumorLesionSize. Reject extra targets explicitly.
    for name, field in [('major_mm', 'metric_value_major_axis'),
                        ('minor_mm', 'metric_value_minor_axis')]:
        values = profile[field]
        if len(values) != 1:
            raise ValueError('Expected exactly one target measurement')
        target = float(np.float16(expected[name]))
        if not math.isfinite(values[0]) or values[0] != target:
            raise ValueError(f'Loader {field} differs from float16 reference')
    if profile['metric_unit'] != ['mm']:
        raise ValueError('Expected millimeter target units')
    if not np.array_equal(np.asarray(actual['pixel_size']),
                          np.asarray(expected['pixel_size'], dtype=np.float16).astype(float)):
        raise ValueError('Loader spacing differs from float16 reference')
