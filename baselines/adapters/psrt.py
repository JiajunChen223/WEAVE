"""PSRT training component factory."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from torch import nn
from torch.nn import functional as F
from baselines.source import verify_source


class ObservationBinding(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native

    def forward(self, lr_hsi, hr_msi):
        if any(n % 8 for n in hr_msi.shape[-2:]):
            raise ValueError("PSRT training image dimensions must be divisible by eight")
        enlarged = F.interpolate(lr_hsi, size=hr_msi.shape[-2:], mode="bicubic", align_corners=False)
        return self.native(hr_msi, enlarged)


def build(*, config, source_dir, device):
    verify_source("psrt", source_dir)
    path = Path(source_dir) / "model/model_SR.py"
    text = path.read_text(encoding="utf-8")
    replacements = {
        "self.in_channels = 31": "self.in_channels = args.n_bands",
        "img_size=self.img_size, in_chans=34, embed_dim=self.embed":
            "img_size=self.img_size, in_chans=args.n_bands+args.n_bands_rgb, embed_dim=self.embed",
    }
    for before, after in replacements.items():
        if text.count(before) != 1:
            raise ValueError("Pinned PSRT channel binding no longer matches")
        text = text.replace(before, after, 1)
    name = "_weave_support_psrt_native"
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(text, str(path), "exec"), module.__dict__)
        args = SimpleNamespace(n_bands=config["hsi_channels"], n_bands_rgb=config["msi_channels"])
        native = ObservationBinding(module.PSRTnet(args)).to(device)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return {"native": native}
