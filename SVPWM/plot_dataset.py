import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(path: str):
    data = np.load(path, allow_pickle=True).item()
    X = data["X"].astype(np.float32)
    Y1 = data["Y1"].astype(np.float32)  # T0, T1, T2
    Y2 = data["Y2"].astype(np.float32)  # u, v, w
    return X, Y1, Y2


def plot_hist(ax, values, title: str, bins: int = 50, x_range: tuple = (-1.0, 1.0)):
    ax.hist(values, bins=bins, range=x_range, color="#4C78A8", alpha=0.9)
    ax.set_title(title)
    ax.set_xlim(x_range)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="input npy path")
    parser.add_argument("--max_points", type=int, default=100000, help="max points for scatter sampling")
    parser.add_argument("--save", action="store_true", help="save figures next to dataset")
    parser.add_argument("--show", action="store_true", help="show figures interactively")
    args = parser.parse_args()

    X, Y1, Y2 = load_dataset(args.path)

    # Histograms for inputs and outputs
    fig1, axes1 = plt.subplots(3, 3, figsize=(12, 9))
    # Input histograms (range [-1,1])
    plot_hist(axes1[0, 0], X[:, 0], "d histogram", x_range=(-1.0, 1.0))
    plot_hist(axes1[0, 1], X[:, 1], "q histogram", x_range=(-1.0, 1.0))
    plot_hist(axes1[0, 2], X[:, 2], "theta histogram", x_range=(-1.0, 1.0))
    # T0/T1/T2 histograms (range [0,1])
    plot_hist(axes1[1, 0], Y1[:, 0], "T0 histogram", x_range=(0.0, 1.0))
    plot_hist(axes1[1, 1], Y1[:, 1], "T1 histogram", x_range=(0.0, 1.0))
    plot_hist(axes1[1, 2], Y1[:, 2], "T2 histogram", x_range=(0.0, 1.0))
    # u/v/w histograms (range [0,1])
    plot_hist(axes1[2, 0], Y2[:, 0], "u histogram", x_range=(0.0, 1.0))
    plot_hist(axes1[2, 1], Y2[:, 1], "v histogram", x_range=(0.0, 1.0))
    plot_hist(axes1[2, 2], Y2[:, 2], "w histogram", x_range=(0.0, 1.0))
    fig1.tight_layout()

    # Scatter sampling for relationship glimpse
    N = X.shape[0]
    sel = np.random.choice(N, size=min(args.max_points, N), replace=False)
    Xs, Y1s, Y2s = X[sel], Y1[sel], Y2[sel]

    fig2, axes2 = plt.subplots(3, 3, figsize=(12, 9))
    # X vs T0/T1/T2
    axes2[0, 0].scatter(Xs[:, 0], Y1s[:, 0], s=1, alpha=0.5)
    axes2[0, 0].set_xlabel("d")
    axes2[0, 0].set_ylabel("T0")
    axes2[0, 1].scatter(Xs[:, 1], Y1s[:, 1], s=1, alpha=0.5)
    axes2[0, 1].set_xlabel("q")
    axes2[0, 1].set_ylabel("T1")
    axes2[0, 2].scatter(Xs[:, 2], Y1s[:, 2], s=1, alpha=0.5)
    axes2[0, 2].set_xlabel("theta")
    axes2[0, 2].set_ylabel("T2")
    # X vs u/v/w
    axes2[1, 0].scatter(Xs[:, 0], Y2s[:, 0], s=1, alpha=0.5)
    axes2[1, 0].set_xlabel("d")
    axes2[1, 0].set_ylabel("u")
    axes2[1, 1].scatter(Xs[:, 1], Y2s[:, 1], s=1, alpha=0.5)
    axes2[1, 1].set_xlabel("q")
    axes2[1, 1].set_ylabel("v")
    axes2[1, 2].scatter(Xs[:, 2], Y2s[:, 2], s=1, alpha=0.5)
    axes2[1, 2].set_xlabel("theta")
    axes2[1, 2].set_ylabel("w")
    # T vs u/v/w
    axes2[2, 0].scatter(Y1s[:, 0], Y2s[:, 0], s=1, alpha=0.5)
    axes2[2, 0].set_xlabel("T0")
    axes2[2, 0].set_ylabel("u")
    axes2[2, 1].scatter(Y1s[:, 1], Y2s[:, 1], s=1, alpha=0.5)
    axes2[2, 1].set_xlabel("T1")
    axes2[2, 1].set_ylabel("v")
    axes2[2, 2].scatter(Y1s[:, 2], Y2s[:, 2], s=1, alpha=0.5)
    axes2[2, 2].set_xlabel("T2")
    axes2[2, 2].set_ylabel("w")
    fig2.tight_layout()

    if args.save:
        base = os.path.splitext(os.path.basename(args.path))[0]
        fig1.savefig(f"SVPWM/image/{base}_hist.png", dpi=150)
        fig2.savefig(f"SVPWM/image/{base}_scatter.png", dpi=150)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()


