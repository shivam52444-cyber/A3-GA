import numpy as np
from anticrowding import shared_scores


def _normalize_probs(p, eps=1e-12):
    p = np.asarray(p, dtype=float)
    p = np.maximum(p, 0.0)
    s = np.sum(p)
    if s <= eps:
        # fallback to uniform if everything is zero or numerical issues
        return np.ones_like(p) / len(p)
    return p / s


def compute_selection_probs(
    X,
    scores,
    baseline_lambda=0.05,
    use_anticrowding=True,
    crowding_method="gaussian",
    sigma=1.0,
    radius=1.0,
):
    """
    Compute selection probabilities for a population.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Population points.
    scores : np.ndarray, shape (N,)
        Base scores (higher is better). Can be fitness or any ranking score.
    baseline_lambda : float in [0, 1]
        Weight of uniform baseline probability. Ensures no one has zero prob.
    use_anticrowding : bool
        Whether to apply fitness sharing (anti-crowding).
    crowding_method : str
        "gaussian" or "cutoff"
    sigma : float
        Kernel width for gaussian crowding.
    radius : float
        Radius for cutoff crowding.

    Returns
    -------
    probs : np.ndarray, shape (N,)
        Selection probabilities summing to 1.
    info : dict
        Extra info (shared_scores, rho) for logging/debugging.
    """
    X = np.asarray(X, dtype=float)
    scores = np.asarray(scores, dtype=float)

    N = X.shape[0]
    if scores.shape[0] != N:
        raise ValueError("scores must have same length as X")

    # Make scores non-negative and stable
    # (If you pass ranks or already-shaped scores, this is fine too)
    min_s = np.min(scores)
    if min_s < 0:
        scores = scores - min_s  # shift to make non-negative

    # Apply anti-crowding (fitness sharing) if requested
    if use_anticrowding:
        shared, rho = shared_scores(
            scores=scores,
            X=X,
            method=crowding_method,
            sigma=sigma,
            radius=radius,
        )
        base = shared
        info = {"rho": rho, "shared_scores": shared}
    else:
        base = scores.copy()
        info = {"rho": None, "shared_scores": None}

    # Normalize base scores to probabilities
    p_fit = _normalize_probs(base)

    # Add baseline probability (entropy floor)
    # p_i = (1 - lambda) * p_fit_i + lambda * (1/N)
    lam = float(baseline_lambda)
    if not (0.0 <= lam <= 1.0):
        raise ValueError("baseline_lambda must be in [0, 1]")

    p = (1.0 - lam) * p_fit + lam * (1.0 / N)
    p = _normalize_probs(p)

    info["p_fit"] = p_fit
    info["p_final"] = p

    return p, info


def sample_parents(
    X,
    probs,
    n_parents,
    replace=True,
    seed=None,
):
    """
    Sample parents from population according to probabilities.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Population.
    probs : np.ndarray, shape (N,)
        Selection probabilities (sum to 1).
    n_parents : int
        Number of parents to sample.
    replace : bool
        Sample with replacement (True is typical for GAs).
    seed : int or None
        RNG seed.

    Returns
    -------
    parents : np.ndarray, shape (n_parents, D)
        Sampled parents.
    indices : np.ndarray, shape (n_parents,)
        Indices of selected parents.
    """
    rng = np.random.default_rng(seed)

    N = X.shape[0]
    probs = _normalize_probs(probs)

    indices = rng.choice(N, size=n_parents, replace=replace, p=probs)
    parents = X[indices].copy()
    return parents, indices


# --------- Quick self-test ---------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Fake population
    X = rng.normal(size=(20, 2))

    # Fake scores (higher is better)
    scores = rng.random(20)

    # Compute probabilities with anti-crowding
    probs, info = compute_selection_probs(
        X,
        scores,
        baseline_lambda=0.1,
        use_anticrowding=True,
        crowding_method="gaussian",
        sigma=0.5,
    )

    print("Sum of probs:", probs.sum())
    print("First 5 probs:", probs[:5])

    # Sample parents
    parents, idx = sample_parents(X, probs, n_parents=10, replace=True, seed=42)
    print("Sampled indices:", idx)
