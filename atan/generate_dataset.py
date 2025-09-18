import argparse
import math
import numpy as np
from typing import Tuple

from atanlib import compute_atan_angle


def linspace_inclusive(n: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, num=n, dtype=np.float32)


def generate_grid(arg: int) -> Tuple[np.ndarray, np.ndarray]:
    a_vals = linspace_inclusive(arg)
    b_vals = linspace_inclusive(arg)

    N = arg * arg
    X = np.empty((N, 2), dtype=np.float32)
    Y = np.empty((N, 1), dtype=np.float32)  # angle normalized to [-1, 1]

    idx = 0
    for a in a_vals:
        for b in b_vals:
            angle_deg = compute_atan_angle(float(a), float(b))  # angle in [-180, 180]
            angle_normalized = angle_deg / 180.0  # normalize to [-1, 1]
            X[idx, 0] = a
            X[idx, 1] = b
            Y[idx, 0] = np.float32(angle_normalized)
            idx += 1

    return X, Y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg", type=int, default=100, help="grid resolution per axis (default: 100)")
    parser.add_argument("--out", type=str, default=None, help="output npy path (default: dataset_arg{arg}.npy)")
    args = parser.parse_args()

    X, Y = generate_grid(args.arg)

    out_path = args.out or f"dataset_arg{args.arg}.npy"
    np.save(out_path, {"X": X, "Y": Y})


if __name__ == "__main__":
    main()
