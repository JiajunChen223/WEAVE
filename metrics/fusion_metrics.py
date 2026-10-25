"""Four fusion metrics with mean band-wise PSNR and Wang SSIM."""
import math
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion

COLUMNS = ("MPSNR_dB", "MSSIM_Wang", "SAM_degrees", "ERGAS")
EPS = 1e-8
MSE_FLOOR = 1e-12

def scene_scores(reference, prediction, *, scale, region_mask=None):
    """Score matching HWC arrays with an optional fixed reference mask."""
    if type(scale) is not int or scale < 1:
        raise ValueError("scale must be the explicit positive HR/LR ratio")
    if np.iscomplexobj(reference) or np.iscomplexobj(prediction):
        raise ValueError("real-valued HWC arrays required")
    ref = np.asarray(reference, dtype=np.float64)
    raw = np.asarray(prediction, dtype=np.float64)
    if ref.ndim != 3 or ref.shape != raw.shape or min(ref.shape[:2]) < 11:
        raise ValueError("matching HWC grids with at least an 11x11 window required")
    if not np.isfinite(ref).all() or not np.isfinite(raw).all():
        raise ValueError("nonfinite predictions are errors, never repaired")
    if np.any(ref < 0) or np.any(ref > 1):
        raise ValueError("Reference values must be in [0,1]")
    if region_mask is None:
        region = np.ones(ref.shape[:2], dtype=bool)
    else:
        region = np.asarray(region_mask)
        if region.dtype != np.bool_ or region.shape != ref.shape[:2]:
            raise ValueError("fixed boolean HxW reference region required")
    valid = region & (np.linalg.norm(ref, axis=-1) > EPS)
    if not valid.any():
        raise ValueError("empty reference support")
    centers = binary_erosion(valid, structure=np.ones((11, 11), dtype=bool), border_value=0)
    if not centers.any():
        raise ValueError("no complete valid SSIM window")
    est = np.clip(raw, 0.0, 1.0)
    diff = est[valid] - ref[valid]
    mse_band = np.mean(diff * diff, axis=0)
    mpsnr = float(np.mean(10.0 * np.log10(1.0 / np.maximum(mse_band, MSE_FLOOR))))

    # sigma=1.5, radius5 => truncate=5/1.5. Population covariance matches Wang.
    kernel = dict(sigma=(1.5, 1.5, 0.0), truncate=5.0 / 1.5, mode="reflect")
    mu_x = gaussian_filter(ref, **kernel)
    mu_y = gaussian_filter(est, **kernel)
    var_x = np.maximum(gaussian_filter(ref * ref, **kernel) - mu_x * mu_x, 0.0)
    var_y = np.maximum(gaussian_filter(est * est, **kernel) - mu_y * mu_y, 0.0)
    cov = gaussian_filter(ref * est, **kernel) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    local = ((2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)) / np.maximum(
        (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2), 1e-18)
    mssim = float(local[centers].mean())

    x, y = ref[valid], est[valid]
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    nonzero_prediction = ny > EPS
    angle = np.full(nx.shape, 90.0, dtype=np.float64)
    # Preserve nonzero spectral norms for dark pixels.
    denominator = nx[nonzero_prediction] * ny[nonzero_prediction]
    cosine = np.sum(x[nonzero_prediction] * y[nonzero_prediction], axis=1) / denominator
    angle[nonzero_prediction] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    sam = float(angle.mean())
    mu = ref[valid].mean(axis=0)
    ergas = float(100.0 / scale * math.sqrt(np.mean(
        mse_band / np.maximum(np.abs(mu), EPS)**2)))
    return {
        "metrics": dict(zip(COLUMNS, (mpsnr, mssim, sam, ergas))),
        "diagnostics": {
            "valid_pixels": int(valid.sum()), "valid_SSIM_centers": int(centers.sum()),
            "zero_prediction_pixels": int((~nonzero_prediction).sum()),
            "clipped_fraction": float(np.mean((raw[valid] < 0) | (raw[valid] > 1))),
            "floored_MSE_bands": int((mse_band < MSE_FLOOR).sum()),
            "near_zero_reference_mean_bands": int((np.abs(mu) < EPS).sum()),
            "raw_min": float(raw[valid].min()), "raw_max": float(raw[valid].max()),
        },
    }


def macro_average(rows):
    """Average already-computed complete-scene scores with equal scene weight."""
    rows = list(rows)
    if not rows:
        raise ValueError("A nonempty collection of scene scores is required")
    values = np.asarray([[r["metrics"][k] for k in COLUMNS] for r in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Finite metric values required")
    return dict(zip(COLUMNS, values.mean(axis=0).tolist()))
