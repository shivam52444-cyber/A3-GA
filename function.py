import numpy as np

# --------- Helper utilities ---------

def sample_uniform(lower, upper, dim, n=1, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(lower, upper, size=(n, dim))


def _ensure_array(x, dim):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        if x.shape[0] != dim:
            raise ValueError(f"Expected dimension {dim}, got {x.shape[0]}")
    else:
        raise ValueError("x must be a 1D array")
    return x


# --------- Benchmark functions ---------
# Each function returns a dict with:
# - "f": callable f(x)
# - "lower": lower bound
# - "upper": upper bound
# - "dim": dimension
# - "name": function name


def sphere(lower=-5.12, upper=5.12, dim=2):
    def f(x):
        x = _ensure_array(x, dim)
        return np.sum(x ** 2)
    return {"f": f, "lower": lower, "upper": upper, "dim": dim, "name": "Sphere"}


def rosenbrock(lower=-5.0, upper=10.0, dim=2):
    def f(x):
        x = _ensure_array(x, dim)
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    return {"f": f, "lower": lower, "upper": upper, "dim": dim, "name": "Rosenbrock"}


def rastrigin(lower=-5.12, upper=5.12, dim=2):
    def f(x):
        x = _ensure_array(x, dim)
        return 10 * dim + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    return {"f": f, "lower": lower, "upper": upper, "dim": dim, "name": "Rastrigin"}


def ackley(lower=-5.0, upper=5.0, dim=2):
    def f(x):
        x = _ensure_array(x, dim)
        a = 20
        b = 0.2
        c = 2 * np.pi
        term1 = -a * np.exp(-b * np.sqrt(np.mean(x ** 2)))
        term2 = -np.exp(np.mean(np.cos(c * x)))
        return term1 + term2 + a + np.e
    return {"f": f, "lower": lower, "upper": upper, "dim": dim, "name": "Ackley"}


def griewank(lower=-600.0, upper=600.0, dim=2):
    def f(x):
        x = _ensure_array(x, dim)
        sum_term = np.sum(x ** 2) / 4000.0
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1))))
        return sum_term - prod_term + 1
    return {"f": f, "lower": lower, "upper": upper, "dim": dim, "name": "Griewank"}


def schwefel(lower=-500.0, upper=500.0, dim=2):
    def f(x):
        x = _ensure_array(x, dim)
        return 418.9829 * dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    return {"f": f, "lower": lower, "upper": upper, "dim": dim, "name": "Schwefel"}



# --------- Registry ---------

def get_all_functions(dim=2):
    """
    Returns a list of function configs (dicts).
    You can change dim here; bounds stay default unless overridden.
    """
    return [
        sphere(dim=dim),
        rosenbrock(dim=dim),
        rastrigin(dim=dim),
        ackley(dim=dim),
        griewank(dim=dim),
        schwefel(dim=dim),
    ]
