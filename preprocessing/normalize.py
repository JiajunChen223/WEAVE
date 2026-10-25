"""Fixed source normalization; statistics are never fitted per crop."""
import numpy as np


def normalize(image, divisor, offset=0.0):
    if not np.isfinite(divisor) or divisor <= 0 or not np.isfinite(offset):
        raise ValueError("A positive fixed divisor and finite offset are required")
    raw = np.asarray(image)
    if raw.ndim != 3 or np.iscomplexobj(raw) or not np.isfinite(raw).all():
        raise ValueError("A finite real three-dimensional image is required")
    value = (raw.astype(np.float64) - offset) / divisor
    if value.min() < -1e-7 or value.max() > 1.0 + 1e-7:
        raise ValueError("Fixed source normalization must produce values in [0, 1]")
    return value.astype(np.float32)
