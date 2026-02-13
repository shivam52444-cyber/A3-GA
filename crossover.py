import numpy as np

# All crossover operators follow this signature:
# op(x1, x2, rng) -> child
# where x1, x2 are 1D numpy arrays of same shape
# rng is a numpy Generator for reproducibility


# --------- Basic utilities ---------

def _check(x1, x2):
    if x1.shape != x2.shape:
        raise ValueError("Parents must have the same shape")


# --------- Core crossover families ---------

def arithmetic(alpha):
    def op(x1, x2, rng):
        _check(x1, x2)
        return alpha * x1 + (1 - alpha) * x2
    return op


def convex_blend(alpha):
    def op(x1, x2, rng):
        _check(x1, x2)
        return (1 - alpha) * x1 + alpha * x2
    return op


def uniform_mask(p=0.5):
    def op(x1, x2, rng):
        _check(x1, x2)
        mask = rng.random(x1.shape) < p
        child = np.where(mask, x1, x2)
        return child
    return op


def one_point():
    def op(x1, x2, rng):
        _check(x1, x2)
        d = x1.shape[0]
        if d == 1:
            return x1.copy()
        k = rng.integers(1, d)
        return np.concatenate([x1[:k], x2[k:]])
    return op


def two_point():
    def op(x1, x2, rng):
        _check(x1, x2)
        d = x1.shape[0]
        if d <= 2:
            return uniform_mask(0.5)(x1, x2, rng)
        a, b = sorted(rng.integers(0, d, size=2))
        child = x1.copy()
        child[a:b] = x2[a:b]
        return child
    return op


def shuffle_then_one_point():
    def op(x1, x2, rng):
        _check(x1, x2)
        d = x1.shape[0]
        idx = np.arange(d)
        rng.shuffle(idx)
        y1 = x1[idx]
        y2 = x2[idx]
        k = rng.integers(1, d)
        child = np.concatenate([y1[:k], y2[k:]])
        inv = np.argsort(idx)
        return child[inv]
    return op


def blend_alpha(alpha):
    # BLX-alpha style
    def op(x1, x2, rng):
        _check(x1, x2)
        lo = np.minimum(x1, x2)
        hi = np.maximum(x1, x2)
        diff = hi - lo
        low = lo - alpha * diff
        high = hi + alpha * diff
        return rng.uniform(low, high)
    return op


def extrapolate(beta):
    def op(x1, x2, rng):
        _check(x1, x2)
        return x1 + beta * (x1 - x2)
    return op


def midpoint():
    def op(x1, x2, rng):
        _check(x1, x2)
        return 0.5 * (x1 + x2)
    return op


def noisy_midpoint(sigma):
    def op(x1, x2, rng):
        _check(x1, x2)
        child = 0.5 * (x1 + x2)
        noise = rng.normal(0.0, sigma, size=x1.shape)
        return child + noise
    return op


def geometric():
    def op(x1, x2, rng):
        _check(x1, x2)
        # avoid negative or zero by abs + eps
        eps = 1e-8
        return np.sqrt(np.abs(x1 * x2) + eps)
    return op


def linear_combo():
    def op(x1, x2, rng):
        _check(x1, x2)
        a = rng.random()
        b = 1 - a
        return a * x1 + b * x2
    return op


def random_weighted():
    def op(x1, x2, rng):
        _check(x1, x2)
        w = rng.random(size=x1.shape)
        return w * x1 + (1 - w) * x2
    return op


def pick_parent():
    def op(x1, x2, rng):
        _check(x1, x2)
        if rng.random() < 0.5:
            return x1.copy()
        else:
            return x2.copy()
    return op


def swap_blocks(block_size):
    def op(x1, x2, rng):
        _check(x1, x2)
        d = x1.shape[0]
        child = x1.copy()
        for i in range(0, d, block_size):
            if rng.random() < 0.5:
                child[i:i+block_size] = x2[i:i+block_size]
        return child
    return op


def gaussian_mix(sigma):
    def op(x1, x2, rng):
        _check(x1, x2)
        a = rng.normal(0.5, sigma, size=x1.shape)
        return a * x1 + (1 - a) * x2
    return op


# --------- Registry builder ---------

def get_all_crossovers():
    """
    Returns a list of dicts:
    { "name": str, "op": callable }
    At least ~50 operators via parameterized variants.
    """
    ops = []

    # Arithmetic / convex variants
    for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
        ops.append({"name": f"arith_{alpha}", "op": arithmetic(alpha)})
        ops.append({"name": f"convex_{alpha}", "op": convex_blend(alpha)})

    # Uniform mask variants
    for p in [0.2, 0.5, 0.8]:
        ops.append({"name": f"uniform_{p}", "op": uniform_mask(p)})

    # BLX-alpha variants
    for a in [0.1, 0.3, 0.5, 1.0]:
        ops.append({"name": f"blx_{a}", "op": blend_alpha(a)})

    # Extrapolation variants
    for b in [0.25, 0.5, 1.0, 2.0]:
        ops.append({"name": f"extrap_{b}", "op": extrapolate(b)})

    # Noisy midpoint variants
    for s in [0.01, 0.05, 0.1, 0.2]:
        ops.append({"name": f"noisy_mid_{s}", "op": noisy_midpoint(s)})

    # Gaussian mix variants
    for s in [0.05, 0.1, 0.2, 0.5]:
        ops.append({"name": f"gaussmix_{s}", "op": gaussian_mix(s)})

    # Block swap variants
    for bs in [1, 2, 4, 8]:
        ops.append({"name": f"swapblock_{bs}", "op": swap_blocks(bs)})

    # Single operators
    ops.extend([
        {"name": "one_point", "op": one_point()},
        {"name": "two_point", "op": two_point()},
        {"name": "shuffle_one_point", "op": shuffle_then_one_point()},
        {"name": "midpoint", "op": midpoint()},
        {"name": "geometric", "op": geometric()},
        {"name": "linear_combo", "op": linear_combo()},
        {"name": "random_weighted", "op": random_weighted()},
        {"name": "pick_parent", "op": pick_parent()},
    ])

    return ops


# --------- Quick self-test ---------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=10)
    x2 = rng.normal(size=10)

    ops = get_all_crossovers()
    print(f"Total crossover operators: {len(ops)}")

    for i, item in enumerate(ops[:5]):
        child = item["op"](x1, x2, rng)
        print(item["name"], child[:3])
