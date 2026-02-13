import numpy as np

def pairwise_distances(X):
    """
    Compute pairwise Euclidean distances for X of shape (N, D).
    Returns matrix of shape (N, N).
    """
    # Efficient: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    X = np.asarray(X, dtype=float)
    sq = np.sum(X**2, axis=1, keepdims=True)
    D2 = sq + sq.T - 2.0 * (X @ X.T)
    D2 = np.maximum(D2, 0.0)  # numerical safety
    return np.sqrt(D2)


def density_gaussian(X, sigma):
    """
    Gaussian kernel density:
    rho_i = sum_j exp( - d(x_i, x_j)^2 / (2*sigma^2) )
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    D = pairwise_distances(X)
    K = np.exp(-(D**2) / (2.0 * sigma**2))
    rho = np.sum(K, axis=1)
    return rho


def density_cutoff(X, radius):
    """
    Cutoff (sharing radius) density:
    rho_i = number of points within 'radius' of x_i (including itself)
    """
    if radius <= 0:
        raise ValueError("radius must be > 0")

    D = pairwise_distances(X)
    rho = np.sum(D <= radius, axis=1).astype(float)
    return rho


def shared_scores(scores, X, method="gaussian", sigma=1.0, radius=1.0, eps=1e-12):
    """
    Apply anti-crowding (fitness sharing) to 'scores'.

    Parameters
    ----------
    scores : array-like, shape (N,)
        Base scores used for selection (e.g., fitness or rank scores).
        Larger is assumed better.
    X : array-like, shape (N, D)
        Population points.
    method : str
        "gaussian" or "cutoff"
    sigma : float
        Kernel width for gaussian density.
    radius : float
        Neighborhood radius for cutoff density.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    shared : np.ndarray, shape (N,)
        Scores penalized by crowding: score_i / rho_i
    rho : np.ndarray, shape (N,)
        Density values.
    """
    scores = np.asarray(scores, dtype=float)
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D (N, D)")
    if scores.ndim != 1 or scores.shape[0] != X.shape[0]:
        raise ValueError("scores must be 1D with same length as X")

    if method == "gaussian":
        rho = density_gaussian(X, sigma=sigma)
    elif method == "cutoff":
        rho = density_cutoff(X, radius=radius)
    else:
        raise ValueError("method must be 'gaussian' or 'cutoff'")

    shared = scores / (rho + eps)
    return shared, rho


# --------- Quick self-test ---------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Create a population with two clusters
    A = rng.normal(loc=0.0, scale=0.2, size=(10, 2))
    B = rng.normal(loc=5.0, scale=0.2, size=(5, 2))
    X = np.vstack([A, B])

    # Suppose raw scores favor cluster A slightly
    scores = np.concatenate([np.full(10, 10.0), np.full(5, 9.5)])

    shared, rho = shared_scores(scores, X, method="gaussian", sigma=0.5)

    print("Raw scores:", scores)
    print("Density (rho):", rho.round(2))
    print("Shared scores:", shared.round(2))
