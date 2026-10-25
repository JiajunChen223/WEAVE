"""Write the current final training state using an atomic replacement."""
import os
import tempfile
from pathlib import Path
import torch


def save_last(folder, state):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".last_", suffix=".tmp", dir=folder)
    os.close(descriptor)
    try:
        torch.save(state, name)
        os.replace(name, folder / "last.pt")
    finally:
        if Path(name).exists():
            Path(name).unlink()
