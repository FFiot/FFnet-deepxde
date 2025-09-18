import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(path: str):
    data = np.load(path, allow_pickle=True).item()
    X = data["X"].astype(np.float32)
    Y = data["Y"].astype(np.float32)  # angle
    return X, Y


def plot_hist(ax, values, title: str, bins: int = 50, x_range: tuple = (-1.0, 1.0)):
    ax.hist(values, bins=bins, range=x_range, color="#4C78A8", alpha=0.9)
    ax.set_title(title)
    ax.set_xlim(x_range)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="input npy path")
    parser.add_argument("--max_points", type=int, default=10000, help="max points for scatter sampling")
    parser.add_argument("--save", action="store_true", help="save figures next to dataset")
    parser.add_argument("--show", action="store_true", help="show figures interactively")
    args = parser.parse_args()

    X, Y = load_dataset(args.path)

    # Histograms for inputs and outputs
    fig1, axes1 = plt.subplots(1, 3, figsize=(12, 4))
    # Input histograms (range [-1,1])
    plot_hist(axes1[0], X[:, 0], "a histogram", x_range=(-1.0, 1.0))
    plot_hist(axes1[1], X[:, 1], "b histogram", x_range=(-1.0, 1.0))
    # Angle histogram (range [-1,1])
    plot_hist(axes1[2], Y[:, 0], "angle histogram", x_range=(-1.0, 1.0))
    fig1.tight_layout()

    # Scatter sampling for relationship glimpse
    N = X.shape[0]
    sel = np.random.choice(N, size=min(args.max_points, N), replace=False)
    Xs, Ys = X[sel], Y[sel]

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    # a vs angle
    axes2[0].scatter(Xs[:, 0], Ys[:, 0], s=1, alpha=0.5)
    axes2[0].set_xlabel("a")
    axes2[0].set_ylabel("angle (normalized)")
    axes2[0].set_title("a vs angle")
    axes2[0].set_ylim(-1, 1)
    # b vs angle
    axes2[1].scatter(Xs[:, 1], Ys[:, 0], s=1, alpha=0.5)
    axes2[1].set_xlabel("b")
    axes2[1].set_ylabel("angle (normalized)")
    axes2[1].set_title("b vs angle")
    axes2[1].set_ylim(-1, 1)
    fig2.tight_layout()

    if args.save:
        base = os.path.splitext(os.path.basename(args.path))[0]
        fig1.savefig(f"atan/image/{base}_hist.png", dpi=150)
        fig2.savefig(f"atan/image/{base}_scatter.png", dpi=150)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
