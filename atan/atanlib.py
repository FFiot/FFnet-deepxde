import math
from typing import Tuple
import argparse
import numpy as np
import matplotlib.pyplot as plt


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def compute_atan_angle(a: float, b: float) -> float:
    """Compute angle from a,b coordinates using atan2.

    Standard atan2 function mapping (a,b) to angle in degrees.
    Returns: angle in [-180, 180] degrees.
    """
    # Use atan2(b, a) to get angle in radians
    angle_rad = math.atan2(b, a)
    
    # Convert to degrees
    angle_deg = math.degrees(angle_rad)
    
    # Ensure angle is in [-180, 180] range
    angle_deg = clamp(angle_deg, -180.0, 180.0)
    
    return angle_deg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=1.0, help="a in [-1,1] (default: 1.0)")
    parser.add_argument("--b", type=float, default=0.0, help="b in [-1,1] (default: 0.0)")
    parser.add_argument("--samples", type=int, default=360, help="b samples for fixed a")
    parser.add_argument("--save", type=str, default=None, help="output png path")
    parser.add_argument("--show", action="store_true", default=True, help="show figure (default: True)")
    args = parser.parse_args()

    a = float(clamp(args.a, -1.0, 1.0))
    b_vals = np.linspace(-1.0, 1.0, num=max(2, args.samples), dtype=np.float32)
    angles = np.empty((b_vals.shape[0],), dtype=np.float32)
    
    for i, b in enumerate(b_vals):
        angle = compute_atan_angle(a, float(b))
        angles[i] = angle

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot angle vs b
    ax.plot(b_vals, angles, label="angle", linewidth=2)
    ax.set_xlabel("b")
    ax.set_ylabel("angle (degrees)")
    ax.set_title(f"atan2(b, a) vs b (a={a:.3f})")
    ax.set_ylim(-180, 180)
    ax.set_yticks(np.arange(-180, 181, 45))
    ax.grid(True, which='major', linewidth=1.0, alpha=0.7)
    ax.grid(True, which='minor', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.legend()
    
    plt.tight_layout()

    if args.save:
        # Ensure output goes to atan/image directory
        if not args.save.startswith('atan/image/'):
            import os
            filename = os.path.basename(args.save)
            args.save = f"atan/image/{filename}"
        plt.savefig(args.save, dpi=150)
    if args.show:
        plt.show()
