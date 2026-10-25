"""Baseline training losses."""
import torch
from torch import nn
from torch.nn import functional as F


LEARNERS = ("cssnet", "psrt", "dcinn", "le_mamba", "ddif", "otias", "hsc_sampling", "emr_diff", "bhsrnet")


def feature_variance(image, features, window):
    batch, channels, h, w = image.shape
    if features.shape[0] != batch or features.shape[-2:] != (h, w):
        raise ValueError("CSSNet feature labels must align with the source image")
    indices = torch.argsort(features.argmax(1).reshape(batch, -1), dim=1)
    selected = torch.gather(image.reshape(batch, channels, -1), 2,
                            indices[:, None, :].expand(-1, channels, -1))
    usable = h * w // window * window
    if usable < window:
        raise ValueError("Not enough pixels for a complete CSSNet feature group")
    groups = selected[..., :usable].reshape(batch, channels, window, usable // window)
    return torch.log(groups.std(dim=2).sum().clamp_min(torch.finfo(image.dtype).tiny))


def bhsr_loss(stages, auxiliary, target, completed_updates):
    if not isinstance(stages, (list, tuple)) or not stages:
        raise ValueError("BHSR-Net must return nonempty stage estimates")
    weight = 0.05 / len(stages)
    loss = target.new_zeros(())
    for i, value in enumerate(stages):
        loss = loss + F.l1_loss(value, target) * weight
        if i > 0:
            weight *= 2
    gray_a, gray_b = auxiliary.mean(1, keepdim=True), target.mean(1, keepdim=True)
    kx = target.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]
    ky = target.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])[None, None]
    gradient = ((F.conv2d(gray_a, kx, padding=1) - F.conv2d(gray_b, kx, padding=1)).abs()
                + (F.conv2d(gray_a, ky, padding=1) - F.conv2d(gray_b, ky, padding=1)).abs()).mean()
    return loss + (0.01 if completed_updates < 2000 else 0.05) * gradient


class NativeTrainingAdapter(nn.Module):
    """Map upstream components to a scalar training_loss(batch, update).

    CSSNet returns (image, HSI features, MSI features); BHSR-Net returns
    (stage images, auxiliary image). CSSNet and LE-Mamba require their SSIM
    criterion. DDIF and EMR-Diff supply training_loss(lr, ms, target).
    """

    def __init__(self, method, native, criterion=None):
        super().__init__()
        if method not in LEARNERS:
            raise ValueError("Unknown learning baseline")
        if not isinstance(native, nn.Module):
            raise TypeError("A native torch.nn.Module is required")
        if method in ("cssnet", "le_mamba") and not callable(criterion):
            raise TypeError("This method requires its upstream SSIM criterion")
        if method in ("ddif", "emr_diff") and not callable(getattr(native, "training_loss", None)):
            raise TypeError("The native method-specific training_loss is required")
        self.method, self.native, self.criterion = method, native, criterion

    def training_loss(self, batch, completed_updates):
        if type(completed_updates) is not int or not 0 <= completed_updates < 10000:
            raise ValueError("Successful-update index must be in [0, 9999]")
        lr, ms, target = (batch[k] for k in ("lr_hsi", "hr_msi", "target"))
        if self.method in ("ddif", "emr_diff"):
            return self.native.training_loss(lr, ms, target)
        output = self.native(lr, ms)
        if self.method == "hsc_sampling" and isinstance(output, (tuple, list)):
            output = output[0]
        if self.method == "bhsrnet":
            stages, auxiliary = output
            return bhsr_loss(stages, auxiliary, target, completed_updates)
        if self.method == "cssnet":
            image, hs_features, ms_features = output
            return (F.l1_loss(image, target) + 0.01 * feature_variance(lr, hs_features, 9)
                    + 0.01 * feature_variance(ms, ms_features, 16)
                    + 0.1 * self.criterion(target, image))
        if self.method == "le_mamba":
            loss = self.criterion(output, target)
            return loss[0] if isinstance(loss, (tuple, list)) else loss
        if not isinstance(output, torch.Tensor) or output.shape != target.shape:
            raise ValueError("Native training image must match the reference")
        return F.l1_loss(output, target)
