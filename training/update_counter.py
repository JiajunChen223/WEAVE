"""Successful optimizer-step counter."""
from dataclasses import dataclass


@dataclass
class UpdateCounter:
    target: int = 10000
    completed: int = 0
    skipped: int = 0

    def __post_init__(self):
        if any(type(v) is not int for v in (self.target, self.completed, self.skipped)):
            raise ValueError("Update counts must be integers")
        if self.target < 1 or not 0 <= self.completed <= self.target or self.skipped < 0:
            raise ValueError("Invalid successful-update budget")

    @property
    def done(self):
        return self.completed == self.target

    def record(self, success):
        if type(success) is not bool:
            raise ValueError("Explicit boolean step status required")
        if self.done:
            raise ValueError("Training budget already complete")
        if success:
            self.completed += 1
        else:
            self.skipped += 1
