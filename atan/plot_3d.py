import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_dataset(path: str):
    data = np.load(path, allow_pickle=True).item()
    X = data["X"].astype(np.float32)
    Y = data["Y"].astype(np.float32)  # angle
    return X, Y


def plot_3d_surface(X, Y, max_points=10000):
    """Plot 3D surface showing a, b vs angle"""
    # Sample data if too large
    N = X.shape[0]
    if N > max_points:
        sel = np.random.choice(N, size=max_points, replace=False)
        X_sampled = X[sel]
        Y_sampled = Y[sel]
    else:
        X_sampled = X
        Y_sampled = Y
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    scatter = ax.scatter(X_sampled[:, 0], X_sampled[:, 1], Y_sampled[:, 0], 
                        c=Y_sampled[:, 0], cmap='viridis', s=1, alpha=0.6)
    
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('angle (normalized)')
    ax.set_title('atan2(b, a) - 3D Visualization')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='angle (normalized)')
    
    return fig, ax


def generate_synthetic_data(resolution=100):
    """Generate synthetic atan2 data for demonstration"""
    from atanlib import compute_atan_angle
    
    a_vals = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    b_vals = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    
    N = resolution * resolution
    X = np.empty((N, 2), dtype=np.float32)
    Y = np.empty((N, 1), dtype=np.float32)
    
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
    parser.add_argument("--path", type=str, default=None, help="input npy path (optional, will generate synthetic data if not provided)")
    parser.add_argument("--max_points", type=int, default=10000, help="max points for scatter plot")
    parser.add_argument("--resolution", type=int, default=100, help="resolution for synthetic data generation")
    parser.add_argument("--save", action="store_true", help="save figures")
    parser.add_argument("--show", action="store_true", default=True, help="show figures interactively (default: True)")
    args = parser.parse_args()

    if args.path:
        X, Y = load_dataset(args.path)
        print(f"Loaded dataset from {args.path} with {X.shape[0]} samples")
    else:
        X, Y = generate_synthetic_data(args.resolution)
        print(f"Generated synthetic dataset with {X.shape[0]} samples (resolution={args.resolution})")
    
    print(f"Input range: a=[{X[:, 0].min():.3f}, {X[:, 0].max():.3f}], b=[{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]")
    print(f"Output range: angle=[{Y[:, 0].min():.3f}, {Y[:, 0].max():.3f}]")

    print("Generating 3D scatter plot...")
    fig, ax = plot_3d_surface(X, Y, args.max_points)
    if args.save:
        if args.path:
            base = os.path.splitext(os.path.basename(args.path))[0]
        else:
            base = f"synthetic_res{args.resolution}"
        fig.savefig(f"image/{base}_3d_scatter.png", dpi=150, bbox_inches='tight')
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
