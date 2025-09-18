import math
from typing import Tuple
import argparse
import numpy as np
import matplotlib.pyplot as plt


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def normalize_angle_unit(theta_unit: float) -> float:
    """Map theta in [-1,1] to [0, 2*pi)."""
    theta = (theta_unit + 1.0) * 0.5 * 2.0 * math.pi
    theta %= 2.0 * math.pi
    return theta


def compute_svpwm_duties_from_dq_theta(d: float, q: float, theta_unit: float) -> Tuple[float, float, float, float, float, float]:
    """Compute T0/T1/T2 and uvw duties from d,q,theta_unit(all in [-1,1]).

    Standard SVPWM with sector-based T1/T2/T0 and center-aligned PWM.
    Returns: (T0, T1, T2, u, v, w) in original [0,1] range.
    """
    # Inverse Park to alpha-beta (normalized units)
    theta = normalize_angle_unit(theta_unit)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    alpha = d * cos_t - q * sin_t
    beta = d * sin_t + q * cos_t

    # Reference vector magnitude and angle
    mag = math.hypot(alpha, beta)
    ang = math.atan2(beta, alpha)
    if ang < 0.0:
        ang += 2.0 * math.pi

    # Sector (0..5)
    sector = int(ang / (math.pi / 3.0))
    sector = min(max(sector, 0), 5)
    ang_s = ang - sector * (math.pi / 3.0)

    # Linear modulation scaling: m = sqrt(3) * mag, cap at 1-eps
    m = math.sqrt(3.0) * mag
    if m > 1.0:
        m = 1.0

    # Switching times
    T1 = m * math.sin((math.pi / 3.0) - ang_s)
    T2 = m * math.sin(ang_s)
    T1 = max(0.0, T1)
    T2 = max(0.0, T2)
    # Center-aligned: T0 split equally at head/tail
    T_sum = T1 + T2
    if T_sum >= 1.0:
        # Numerical guard
        scale = 1.0 / (T_sum + 1e-12)
        T1 *= scale
        T2 *= scale
        T_sum = T1 + T2
    T0 = max(0.0, 1.0 - T_sum)

    # Duty assignment per sector (A->u, B->v, C->w)
    if sector == 0:
        du = T1 + T2 + T0 * 0.5
        dv = T2 + T0 * 0.5
        dw = T0 * 0.5
    elif sector == 1:
        du = T1 + T0 * 0.5
        dv = T1 + T2 + T0 * 0.5
        dw = T0 * 0.5
    elif sector == 2:
        du = T0 * 0.5
        dv = T1 + T2 + T0 * 0.5
        dw = T2 + T0 * 0.5
    elif sector == 3:
        du = T0 * 0.5
        dv = T1 + T0 * 0.5
        dw = T1 + T2 + T0 * 0.5
    elif sector == 4:
        du = T2 + T0 * 0.5
        dv = T0 * 0.5
        dw = T1 + T2 + T0 * 0.5
    else:  # sector == 5
        du = T1 + T2 + T0 * 0.5
        dv = T0 * 0.5
        dw = T1 + T0 * 0.5

    du = clamp(du, 0.0, 1.0)
    dv = clamp(dv, 0.0, 1.0)
    dw = clamp(dw, 0.0, 1.0)

    # Return original values without normalization
    return (T0, T1, T2, du, dv, dw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=float, default=0.0, help="d in [-1,1] (default: 0.0)")
    parser.add_argument("--q", type=float, default=1.0, help="q in [-1,1] (default: 1.0)")
    parser.add_argument("--samples", type=int, default=360, help="theta samples")
    parser.add_argument("--save", type=str, default=None, help="output png path")
    parser.add_argument("--show", action="store_true", default=True, help="show figure (default: True)")
    args = parser.parse_args()

    d = float(clamp(args.d, -1.0, 1.0))
    q = float(clamp(args.q, -1.0, 1.0))
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
