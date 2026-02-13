import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from function import get_all_functions


def plot_all_3d(dim=2, grid_points=200, save_path="benchmark_surfaces.png"):
    if dim != 2:
        raise ValueError("3D surface plotting only makes sense for dim=2")

    funcs = get_all_functions(dim=dim)

    n = len(funcs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(5 * ncols, 4 * nrows))

    for i, cfg in enumerate(funcs, 1):
        f = cfg["f"]
        lower = cfg["lower"]
        upper = cfg["upper"]
        name = cfg["name"]

        x = np.linspace(lower, upper, grid_points)
        y = np.linspace(lower, upper, grid_points)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for ix in range(X.shape[0]):
            for iy in range(X.shape[1]):
                Z[ix, iy] = f(np.array([X[ix, iy], Y[ix, iy]]))

        ax = fig.add_subplot(nrows, ncols, i, projection="3d")
        ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
        ax.set_title(name)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved all surfaces to: {save_path}")


if __name__ == "__main__":
    plot_all_3d(dim=2, grid_points=150, save_path="benchmark_surfaces.png")
