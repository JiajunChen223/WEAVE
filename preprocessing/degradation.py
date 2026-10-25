"""Fixed circular blur, point sampling and spectral mixing."""
import numpy as np


def spatial_degrade(image, geometry):
    image = np.asarray(image)
    if image.ndim != 3 or not np.isfinite(image).all() or np.iscomplexobj(image):
        raise ValueError("Finite real HWC image required")
    h, w = image.shape[:2]
    s, k = geometry.scale, geometry.kernel_size
    if h % s or w % s or min(h, w) < k:
        raise ValueError("Image must fit the PSF and be divisible by the spatial scale")
    axis = np.arange(k, dtype=np.float64) - (k - 1) / 2
    taps = np.exp(-(axis**2) / (2 * geometry.sigma**2))
    if k % 2 == 0:
        taps[k // 2 - 1 : k // 2 + 1] = 1.0
    taps /= taps.sum()
    psf = np.zeros((h, w), dtype=np.float64)
    psf[:k, :k] = np.outer(taps, taps)
    psf = np.roll(psf, (-(k // 2), -(k // 2)), axis=(0, 1))
    blurred = np.fft.ifft2(
        np.fft.fft2(image, axes=(0, 1)) * np.fft.fft2(psf)[..., None], axes=(0, 1)
    ).real
    py, px = geometry.phase
    return blurred[py::s, px::s].astype(np.float32)


def make_pair(reference, response, geometry):
    """Simulated pairs: augment/crop the reference before calling this function."""
    return {
        "target": reference,
        "lr_hsi": spatial_degrade(reference, geometry),
        "hr_msi": np.einsum("mc,hwc->hwm", response, reference).astype(np.float32),
    }
