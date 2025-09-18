import argparse
import math
import numpy as np
from typing import Tuple

from .svpwmlib import compute_svpwm_duties_from_dq_theta


def linspace_inclusive(n: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, num=n, dtype=np.float32)


def generate_grid(arg: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_vals = linspace_inclusive(arg)
    q_vals = linspace_inclusive(arg)
    theta_vals = linspace_inclusive(arg)

    N = arg * arg * arg
    X = np.empty((N, 3), dtype=np.float32)
    Y1 = np.empty((N, 3), dtype=np.float32)  # T0, T1, T2
    Y2 = np.empty((N, 3), dtype=np.float32)  # u, v, w

    idx = 0
    for d in d_vals:
        for q in q_vals:
            for t in theta_vals:
                T0, T1, T2, u, v, w = compute_svpwm_duties_from_dq_theta(float(d), float(q), float(t))
                X[idx, 0] = d
                X[idx, 1] = q
                X[idx, 2] = t
                Y1[idx, 0] = np.float32(T0)
                Y1[idx, 1] = np.float32(T1)
                Y1[idx, 2] = np.float32(T2)
                Y2[idx, 0] = np.float32(u)
                Y2[idx, 1] = np.float32(v)
                Y2[idx, 2] = np.float32(w)
                idx += 1

    return X, Y1, Y2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg", type=int, default=100, help="grid resolution per axis (default: 100)")
    parser.add_argument("--out", type=str, default=None, help="output npy path (default: dataset_arg{arg}.npy)")
    args = parser.parse_args()

    X, Y1, Y2 = generate_grid(args.arg)

    out_path = args.out or f"dataset_arg{args.arg}.npy"
    np.save(out_path, {"X": X, "Y1": Y1, "Y2": Y2})


if __name__ == "__main__":
    main()


