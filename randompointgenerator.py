import numpy as np


def _broadcast_bounds(lower, upper, dim):
    """
    Convert lower/upper to arrays of shape (dim,).
    Allows scalar or array-like bounds.
    """
    if np.isscalar(lower):
        lower = np.full(dim, lower, dtype=float)
    else:
        lower = np.asarray(lower, dtype=float)
        if lower.shape[0] != dim:
            raise ValueError(f"lower must have length {dim}")

    if np.isscalar(upper):
        upper = np.full(dim, upper, dtype=float)
    else:
        upper = np.asarray(upper, dtype=float)
        if upper.shape[0] != dim:
            raise ValueError(f"upper must have length {dim}")

    if np.any(upper <= lower):
        raise ValueError("All upper bounds must be > lower bounds")

    return lower, upper


def random_points(
    n=1,
    dim=2,
    lower=-5.0,
    upper=5.0,
    seed=None,
):
    """
    Generate n random points uniformly in [lower, upper]^dim.

    Parameters
    ----------
    n : int
        Number of points to generate (default=1)
    dim : int
        Dimension of each point (default=2)
    lower : float or array-like
        Lower bound(s) (default=-5.0)
    upper : float or array-like
        Upper bound(s) (default=5.0)
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    points : np.ndarray, shape (n, dim)
        Randomly sampled points
    """
    rng = np.random.default_rng(seed)

    lower, upper = _broadcast_bounds(lower, upper, dim)

    # Sample uniformly for each dimension
    points = rng.uniform(low=lower, high=upper, size=(n, dim))
    return points


# --------- Quick self-test ---------

if __name__ == "__main__":
    # Default usage
    X = random_points()
    print("Default (1 point, 2D):")
    print(X)

    # 5 points in 3D with custom bounds
    X = random_points(n=5, dim=3, lower=-10, upper=10, seed=42)
    print("\n5 points in 3D, bounds [-10, 10]:")
    print(X)

    # Per-dimension bounds
    lower = [-5, 0, 10]
    upper = [5, 1, 20]
    X = random_points(n=4, dim=3, lower=lower, upper=upper, seed=1)
    print("\nPer-dimension bounds:")
    print(X)
