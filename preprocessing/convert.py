"""Convert one explicitly selected source array into normalized HWC NumPy data."""
import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .normalize import normalize


def convert(source, output, *, key=None, layout="HWC", divisor=1.0, offset=0.0):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Conversion output must differ from its source")
    if output.suffix.lower() != ".npy":
        raise ValueError("Output must have the .npy extension")
    if source.suffix.lower() == ".mat":
        if not key:
            raise ValueError("MAT conversion requires an explicit array key")
        image = loadmat(source)[key]
    elif source.suffix.lower() == ".npz":
        if not key:
            raise ValueError("NPZ conversion requires an explicit array key")
        with np.load(source, allow_pickle=False) as archive:
            image = archive[key]
    elif source.suffix.lower() == ".npy":
        image = np.load(source, allow_pickle=False)
    else:
        raise ValueError("Supported inputs are .mat, .npy and .npz")
    if layout == "CHW":
        image = np.moveaxis(image, 0, -1)
    elif layout != "HWC":
        raise ValueError("Array layout must be HWC or CHW")
    value = normalize(image, divisor, offset)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, value, allow_pickle=False)
    return value.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--key")
    parser.add_argument("--layout", choices=("HWC", "CHW"), default="HWC")
    parser.add_argument("--divisor", type=float, required=True)
    parser.add_argument("--offset", type=float, default=0.0)
    args = parser.parse_args()
    print(convert(args.source, args.output, key=args.key, layout=args.layout,
                  divisor=args.divisor, offset=args.offset))


if __name__ == "__main__":
    main()
