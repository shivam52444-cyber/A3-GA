import numpy as np

# All mutation operators follow this signature:
# op(x, rng) -> mutated_x
# where x is a 1D numpy array
# rng is a numpy Generator


def _check(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    return x


# --------- Core mutation families ---------

def gaussian_mutation(sigma=1.0, per_dim=True, p=1.0):
    """
    sigma: standard deviation of Gaussian noise
    per_dim: if True, sample noise per dimension, else one scalar noise
    p: probability of mutating each coordinate (mask)
    """
    def op(x, rng):
        x = _check(x)
        d = x.shape[0]

        if per_dim:
            noise = rng.normal(0.0, sigma, size=d)
        else:
            noise = rng.normal(0.0, sigma)
            noise = np.full(d, noise)

        if p < 1.0:
            mask = rng.random(d) < p
            noise = noise * mask

        return x + noise

    return op


def cauchy_mutation(gamma=1.0, per_dim=True, p=1.0):
    """
    gamma: scale parameter of Cauchy distribution
    per_dim: if True, sample noise per dimension, else one scalar noise
    p: probability of mutating each coordinate (mask)
    """
    def op(x, rng):
        x = _check(x)
        d = x.shape[0]

        if per_dim:
            noise = rng.standard_cauchy(size=d) * gamma
        else:
            noise = rng.standard_cauchy() * gamma
            noise = np.full(d, noise)

        if p < 1.0:
            mask = rng.random(d) < p
            noise = noise * mask

        return x + noise

    return op


# --------- Registry builder ---------

def get_all_mutations():
    """
    Returns a list of dicts:
    { "name": str, "op": callable }
    Generates many variants using parameter grids.
    """
    ops = []

    # Parameter grids
    sigmas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    gammas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    ps = [1.0, 0.5, 0.2]  # full, half, sparse mutation

    # Gaussian variants
    for sigma in sigmas:
        for p in ps:
            ops.append({
                "name": f"gauss_sigma{sigma}_p{p}_perdim",
                "op": gaussian_mutation(sigma=sigma, per_dim=True, p=p)
            })
            ops.append({
                "name": f"gauss_sigma{sigma}_p{p}_shared",
                "op": gaussian_mutation(sigma=sigma, per_dim=False, p=p)
            })

    # Cauchy variants
    for gamma in gammas:
        for p in ps:
            ops.append({
                "name": f"cauchy_gamma{gamma}_p{p}_perdim",
                "op": cauchy_mutation(gamma=gamma, per_dim=True, p=p)
            })
            ops.append({
                "name": f"cauchy_gamma{gamma}_p{p}_shared",
                "op": cauchy_mutation(gamma=gamma, per_dim=False, p=p)
            })

    return ops


# --------- Quick self-test ---------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=10)

    ops = get_all_mutations()
    print(f"Total mutation operators: {len(ops)}")

    for item in ops[:5]:
        y = item["op"](x, rng)
        print(item["name"], y[:3])
