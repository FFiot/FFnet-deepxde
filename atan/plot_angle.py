import argparse
import numpy as np
import matplotlib.pyplot as plt

from atanlib import compute_atan_angle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=1.0, help="a in [-1,1] (default: 1.0)")
    parser.add_argument("--b", type=float, default=0.0, help="b in [-1,1] (default: 0.0)")
    parser.add_argument("--samples", type=int, default=360, help="b samples for fixed a")
    parser.add_argument("--save", type=str, default=None, help="output png path")
    parser.add_argument("--show", action="store_true", default=True, help="show figure (default: True)")
    args = parser.parse_args()

    a = float(max(-1.0, min(1.0, args.a)))
    b_vals = np.linspace(-1.0, 1.0, num=max(2, args.samples), dtype=np.float32)
    angles = np.empty((b_vals.shape[0],), dtype=np.float32)
    
    for i, b in enumerate(b_vals):
        angle_deg = compute_atan_angle(a, float(b))  # angle in [-180, 180]
        angle_normalized = angle_deg / 180.0  # normalize to [-1, 1]
        angles[i] = angle_normalized

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot angle vs b
    ax.plot(b_vals, angles, label="angle", linewidth=2)
    ax.set_xlabel("b")
    ax.set_ylabel("angle (normalized)")
    ax.set_title(f"atan2(b, a) vs b (a={a:.3f})")
    ax.set_ylim(-1, 1)
    ax.set_yticks(np.arange(-1, 1.1, 0.25))
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


if __name__ == "__main__":
    main()
