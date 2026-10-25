"""Explicit crop coordinates and paired geometric augmentation."""
import numpy as np


def legal_origins(mask, crop_size, stride=1):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("A boolean HW mask is required")
    if type(crop_size) is not int or crop_size <= 0 or type(stride) is not int or stride <= 0:
        raise ValueError("Positive integer crop size and stride required")
    h, w = mask.shape
    if crop_size > min(h, w):
        raise ValueError("Crop does not fit the image")
    summed = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    yy, xx = np.meshgrid(np.arange(0, h-crop_size+1, stride),
                         np.arange(0, w-crop_size+1, stride), indexing="ij")
    count = (summed[yy+crop_size, xx+crop_size] - summed[yy, xx+crop_size]
             - summed[yy+crop_size, xx] + summed[yy, xx])
    return np.column_stack((yy[count == crop_size**2], xx[count == crop_size**2]))


def augment(image, code):
    if type(code) is not int or not 0 <= code < 8:
        raise ValueError("Augmentation code must be an integer from 0 to 7")
    value = np.rot90(image, code % 4, axes=(0, 1))
    if code >= 4:
        value = value[:, ::-1]
    return np.ascontiguousarray(value)
