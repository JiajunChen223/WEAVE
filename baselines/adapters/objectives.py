"""Training objectives for native diffusion components."""
import torch
from torch import nn
from torch.nn import functional as F


class DDIFTraining(nn.Module):
    def __init__(self, diffusion, condition_builder):
        super().__init__()
        self.diffusion, self.condition_builder = diffusion, condition_builder

    def training_loss(self, lr_hsi, hr_msi, target):
        base, condition = self.condition_builder(lr_hsi, hr_msi)
        self.diffusion.loss_func.active_mask = None
        loss, _ = self.diffusion(target-base, cond=condition)
        return loss


class EMRTraining(nn.Module):
    def __init__(self, net, diffusion, condition_builder, msi_channels):
        super().__init__()
        self.net, self.diffusion = net, diffusion
        self.condition_builder, self.msi_channels = condition_builder, msi_channels

    def training_loss(self, lr_hsi, hr_msi, target):
        up, condition = self.condition_builder(lr_hsi, hr_msi)
        target = torch.cat([target, target[:, :self.msi_channels]], 1)
        time = torch.randint(5, (condition.shape[0],), device=condition.device)
        noisy = self.diffusion.forward_addnoise(target, condition, time,
                                                torch.randn_like(condition), hr_msi)
        residual, features = self.net(noisy, hr_msi, up, time)
        loss = F.l1_loss(residual+condition, target)
        for idx in (2, 4, 6):
            feature = features[idx]
            stride = target.shape[-1] // feature.shape[-1]
            if stride < 1:
                raise ValueError("Invalid decoder feature resolution")
            reference = target[..., ::stride, ::stride]
            lower_condition = condition[..., ::stride, ::stride]
            if reference.shape != feature.shape or lower_condition.shape != feature.shape:
                raise ValueError("Native decoder features must match point-sampled targets")
            loss = loss + F.l1_loss(feature+lower_condition, reference)
        return loss
