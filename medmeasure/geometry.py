"""KiTS23 v1.4.0 physical-space ellipse geometry.

Matches __fit_ellipses in medvision_ds/utils/benchmark_planner.py at dataset
revision f4040ed7d2d2b45e09c1a996ad2969f2018051d3. Display endpoints are not used
for measurement: the reference stores continuous OpenCV ellipse axes.
"""
import cv2
import numpy as np
from scipy.ndimage import label


def matches_annotation_metadata(actual, recorded):
    """Compare headers at the planner's three-decimal metadata precision.

    The small absolute tolerance covers float32 JSON spacing values. Ellipse
    measurements still use full-precision headers and their own strict check.
    """
    return np.allclose(np.round(np.asarray(actual, dtype=float), 3), recorded,
                       atol=1e-6, rtol=0)


def fit_mask(mask, spacing):
    spacing = np.asarray(spacing, dtype=float)
    if mask.ndim != 2 or spacing.shape != (2,) or not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError('Expected 2D mask and positive finite row/column spacing')
    components, count = label(mask)
    sizes = np.bincount(components.ravel())[1:]
    fits = []
    for index in np.argsort(-sizes):
        roi = (components == index + 1).astype(np.uint8)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = contours[0].squeeze()
        if contour.ndim != 2 or len(contour) < 5:
            continue
        real = np.column_stack((contour[:, 0]*spacing[1], contour[:, 1]*spacing[0]))
        center, axes, angle = cv2.fitEllipse(real.astype(np.float32))
        if not np.all(np.isfinite([*center, *axes, angle])):
            continue
        major, minor = max(axes), min(axes)
        rr, cc = np.where(roi)
        diagonal = np.hypot((rr.max()-rr.min()+1)*spacing[0], (cc.max()-cc.min()+1)*spacing[1])
        if minor < min(spacing) or major > 1.5*diagonal or major < max(2., 2*max(spacing)):
            continue
        fits.append({'major_mm': major, 'minor_mm': minor, 'center_xy_mm': center,
                     'axes_mm': axes, 'angle_deg': angle, 'pixels': int(sizes[index])})
    return fits, int(count)
