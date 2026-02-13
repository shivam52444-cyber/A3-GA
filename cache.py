import numpy as np


class EvalCache:
    def __init__(self, tol=1e-12):
        """
        tol: quantization tolerance for hashing floating points.
             Points closer than tol are treated as identical.
        """
        self.tol = tol
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def _key(self, x):
        """
        Convert a float vector into a hashable key with quantization.
        """
        x = np.asarray(x, dtype=float)
        if self.tol > 0:
            xq = np.round(x / self.tol) * self.tol
        else:
            xq = x
        return tuple(xq.tolist())

    def evaluate_one(self, x, f):
        """
        Evaluate f(x) using cache if available.
        """
        k = self._key(x)
        if k in self.cache:
            self.hits += 1
            return self.cache[k]
        else:
            self.misses += 1
            val = f(x)
            self.cache[k] = val
            return val

    def evaluate_batch(self, X, f):
        """
        Evaluate a batch of points X (shape: N, D) with caching.
        Returns np.ndarray of shape (N,).
        """
        vals = np.empty(X.shape[0], dtype=float)
        for i, x in enumerate(X):
            vals[i] = self.evaluate_one(x, f)
        return vals

    def stats(self):
        return {
            "cache_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
        }
