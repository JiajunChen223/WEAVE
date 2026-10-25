"""Spatial observation settings, independent of a reconstruction model."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Geometry:
    scale: int
    kernel_size: int
    sigma: float
    phase: tuple[int, int]

    def __post_init__(self):
        if type(self.scale) is not int or self.scale < 1:
            raise ValueError("Positive integer scale required")
        if type(self.kernel_size) is not int or self.kernel_size < 1:
            raise ValueError("Positive integer kernel size required")
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Positive finite sigma required")
        if len(self.phase) != 2 or any(type(p) is not int or not 0 <= p < self.scale for p in self.phase):
            raise ValueError("Sampling phase must be two integers within the scale")
