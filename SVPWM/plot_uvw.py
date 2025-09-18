import argparse
import numpy as np
import matplotlib.pyplot as plt

from .svpwmlib import compute_svpwm_duties_from_dq_theta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=float, default=0.0, help="d in [-1,1] (default: 0.0)")
    parser.add_argument("--q", type=float, default=1.0, help="q in [-1,1] (default: 1.0)")
    parser.add_argument("--samples", type=int, default=360, help="theta samples")
    parser.add_argument("--save", type=str, default=None, help="output png path")
    parser.add_argument("--show", action="store_true", default=True, help="show figure (default: True)")
    args = parser.parse_args()

    d = float(max(-1.0, min(1.0, args.d)))
    q = float(max(-1.0, min(1.0, args.q)))
    thetas = np.linspace(-1.0, 1.0, num=max(2, args.samples), dtype=np.float32)
    T012 = np.empty((thetas.shape[0], 3), dtype=np.float32)
    uvw = np.empty((thetas.shape[0], 3), dtype=np.float32)
    for i, t in enumerate(thetas):
        T0, T1, T2, u, v, w = compute_svpwm_duties_from_dq_theta(d, q, float(t))
        T012[i, 0] = T0
        T012[i, 1] = T1
        T012[i, 2] = T2
        uvw[i, 0] = u
        uvw[i, 1] = v
        uvw[i, 2] = w

    x_deg = (thetas + 1.0) * 0.5 * 360.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot T0, T1, T2
    ax1.plot(x_deg, T012[:, 0], label="T0")
    ax1.plot(x_deg, T012[:, 1], label="T1")
    ax1.plot(x_deg, T012[:, 2], label="T2")
    ax1.set_xlabel("electrical angle (deg)")
    ax1.set_ylabel("T0/T1/T2 [0,1]")
    ax1.set_title(f"SVPWM T0/T1/T2 vs angle (d={d:.3f}, q={q:.3f})")
    ax1.set_xticks(np.arange(0, 361, 60))
    ax1.set_xticks(np.arange(0, 361, 30), minor=True)
    ax1.grid(True, which='major', axis='x', linewidth=1.0, alpha=0.7)
    ax1.grid(True, which='minor', axis='x', linestyle='--', linewidth=0.8, alpha=0.4)
    ax1.legend()
    
    # Plot u, v, w
    ax2.plot(x_deg, uvw[:, 0], label="u")
    ax2.plot(x_deg, uvw[:, 1], label="v")
    ax2.plot(x_deg, uvw[:, 2], label="w")
    ax2.set_xlabel("electrical angle (deg)")
    ax2.set_ylabel("u/v/w [0,1]")
    ax2.set_title(f"SVPWM u/v/w vs angle (d={d:.3f}, q={q:.3f})")
    ax2.set_xticks(np.arange(0, 361, 60))
    ax2.set_xticks(np.arange(0, 361, 30), minor=True)
    ax2.grid(True, which='major', axis='x', linewidth=1.0, alpha=0.7)
    ax2.grid(True, which='minor', axis='x', linestyle='--', linewidth=0.8, alpha=0.4)
    ax2.legend()
    
    plt.tight_layout()

    if args.save:
        # Ensure output goes to SVPWM/image directory
        if not args.save.startswith('SVPWM/image/'):
            import os
            filename = os.path.basename(args.save)
            args.save = f"SVPWM/image/{filename}"
        plt.savefig(args.save, dpi=150)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()


