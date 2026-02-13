import numpy as np

from randompointgenerator import random_points
from samel import compute_selection_probs, sample_parents
from crossover import get_all_crossovers
from mutation import gaussian_mutation, cauchy_mutation
from cache import EvalCache


class GeneticOptimizer:
    def __init__(
        self,
        f,                 # callable f(x) to MINIMIZE
        dim,
        lower,
        upper,
        max_evals=100_000,
        seed=None,

        # Selection / diversity
        baseline_lambda=0.05,
        use_anticrowding=True,
        crowding_method="gaussian",
        crowding_sigma=1.0,

        # Fractions (initial)
        frac_gen0=0.65,    # 50% generated from parents
        frac_rand0=0.05,   # 20% random points

        # Floors (never go below this)
        frac_gen_min=0.20,
        frac_rand_min=0.02,

        # Decay
        frac_decay=0.995,

        # Mutation schedule
        sigma0=1.0,
        sigma_decay=0.99,
        p_cauchy0=0.1,
        p_cauchy_decay=0.5,

        # Crossover adaptation
        operator_baseline=0.05,
        op_lr=0.1,
        max_reward=1.0,
        penalty_weight=0.25,

        # Restart
        stagnation_window=50,
        stagnation_eps=1e-12,
        elite_fraction=0.1,

        # Cache
        cache_tol=1e-12,
    ):
        self.f = f
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.max_evals = int(max_evals)

        self.rng = np.random.default_rng(seed)

        # Population size rule: N = min(10*d, max_evals/10)
        self.pop_size = int(min(10 * dim, max_evals / 10))
        self.pop_size = max(self.pop_size, 10)  # safety floor

        # Selection / diversity
        self.baseline_lambda = baseline_lambda
        self.use_anticrowding = use_anticrowding
        self.crowding_method = crowding_method
        self.crowding_sigma = crowding_sigma

        # Fractions
        self.frac_gen0 = frac_gen0
        self.frac_rand0 = frac_rand0
        self.frac_gen_min = frac_gen_min
        self.frac_rand_min = frac_rand_min
        self.frac_decay = frac_decay

        # Mutation schedule
        self.sigma0 = sigma0
        self.sigma_decay = sigma_decay
        self.p_cauchy0 = p_cauchy0
        self.p_cauchy_decay = p_cauchy_decay

        # Crossover operators
        self.crossovers = get_all_crossovers()
        self.num_ops = len(self.crossovers)

        # Operator adaptation
        self.operator_baseline = operator_baseline
        self.op_lr = op_lr
        self.max_reward = max_reward
        self.penalty_weight = penalty_weight
        self.op_scores = np.ones(self.num_ops, dtype=float)

        # Restart
        self.stagnation_window = stagnation_window
        self.stagnation_eps = stagnation_eps
        self.elite_fraction = elite_fraction

        # Cache
        self.cache = EvalCache(tol=cache_tol)

        # History
        self.best_history = []
        self.op_prob_history = []

    # ---- Evaluation with cache ----
    def _evaluate(self, X):
        return self.cache.evaluate_batch(X, self.f)

    # ---- Operator probabilities ----
    def _operator_probs(self):
        s = np.maximum(self.op_scores, 0.0)
        if np.sum(s) <= 0:
            p = np.ones(self.num_ops) / self.num_ops
        else:
            p = s / np.sum(s)

        eta = self.operator_baseline
        p = (1.0 - eta) * p + eta * (1.0 / self.num_ops)
        p = p / np.sum(p)
        return p

    # ---- Update operator scores (EMA + forgetting) ----
    def _update_operator_scores(self, k, improvement):
        imp = float(np.clip(improvement, -self.max_reward, self.max_reward))

        if imp > 0:
            reward = imp
        else:
            reward = self.penalty_weight * imp  # negative penalty

        lr = self.op_lr
        self.op_scores[k] = (1.0 - lr) * self.op_scores[k] + lr * max(0.0, reward)

        # Forget others
        for j in range(self.num_ops):
            if j != k:
                self.op_scores[j] *= (1.0 - lr)

        self.op_scores = np.maximum(self.op_scores, 1e-12)

    # ---- Main loop ----
    def run(self, verbose=True):
        # ---- Initialization ----
        X = random_points(
            n=self.pop_size, dim=self.dim, lower=self.lower, upper=self.upper
        )
        fitness = self._evaluate(X)

        best_val = np.min(fitness)
        self.best_history = [best_val]

        # Schedules
        frac_gen_t = self.frac_gen0
        frac_rand_t = self.frac_rand0

        sigma_t = self.sigma0
        p_cauchy_t = self.p_cauchy0

        # Stagnation tracking
        best_window = [best_val]

        gen = 0

        # ---- Main loop: stop by evaluation budget ----
        while self.cache.misses < self.max_evals:
            gen += 1

            # Convert to maximization scores for selection
            scores = -fitness

            probs, _ = compute_selection_probs(
                X=X,
                scores=scores,
                baseline_lambda=self.baseline_lambda,
                use_anticrowding=self.use_anticrowding,
                crowding_method=self.crowding_method,
                sigma=self.crowding_sigma,
            )

            # How many of each type
            n_gen = int(frac_gen_t * self.pop_size)
            n_rand = int(frac_rand_t * self.pop_size)

            # Ensure we don't exceed pop size
            n_gen = min(n_gen, self.pop_size)
            n_rand = min(n_rand, self.pop_size - n_gen)

            # Elites fill the rest
            n_elite = self.pop_size - n_gen - n_rand
            n_elite = max(n_elite, 0)

            # ---- Elitism ----
            elite_k = max(1, int(self.elite_fraction * self.pop_size))
            idx_best = np.argsort(fitness)[:elite_k]
            elites = X[idx_best]
            elites_fit = fitness[idx_best]

            # ---- Sample parents ----
            parents, parent_idx = sample_parents(
                X=X, probs=probs, n_parents=2 * n_gen, replace=True
            )

            # ---- Prepare mutation operators ----
            gauss_mut = gaussian_mutation(sigma=sigma_t, per_dim=True, p=1.0)
            cauchy_mut = cauchy_mutation(gamma=sigma_t, per_dim=True, p=1.0)

            # ---- Crossover probabilities ----
            op_probs = self._operator_probs()
            self.op_prob_history.append(op_probs.copy())

            # ---- Generate children ----
            children = []
            children_ops = []

            for i in range(n_gen):
                p1 = parents[2 * i]
                p2 = parents[2 * i + 1]

                k = self.rng.choice(self.num_ops, p=op_probs)
                op = self.crossovers[k]["op"]

                child = op(p1, p2, self.rng)

                # Mutation schedule
                if self.rng.random() < p_cauchy_t:
                    child = cauchy_mut(child, self.rng)
                else:
                    child = gauss_mut(child, self.rng)

                child = np.clip(child, self.lower, self.upper)

                children.append(child)
                children_ops.append(k)

            children = np.array(children) if len(children) > 0 else np.empty((0, self.dim))

            # ---- Random points ----
            if n_rand > 0:
                rand_pts = random_points(
                    n=n_rand, dim=self.dim, lower=self.lower, upper=self.upper
                )
            else:
                rand_pts = np.empty((0, self.dim))

            # ---- Build new population ----
            new_X = np.vstack([children, rand_pts, elites])

            # If still short (due to rounding), fill with best elites
            if new_X.shape[0] < self.pop_size:
                k = self.pop_size - new_X.shape[0]
                fill = elites[:k] if elites.shape[0] >= k else elites
                new_X = np.vstack([new_X, fill])

            # ---- Evaluate new population ----
            new_fitness = self._evaluate(new_X)

            # ---- Update operator scores ----
            for i, k_op in enumerate(children_ops):
                child_fit = new_fitness[i]
                pidx1 = parent_idx[2 * i]
                pidx2 = parent_idx[2 * i + 1]
                parent_best = min(fitness[pidx1], fitness[pidx2])

                improvement = parent_best - child_fit
                self._update_operator_scores(k_op, improvement)

            # ---- Move to next generation ----
            X = new_X
            fitness = new_fitness

            best_val = np.min(fitness)
            self.best_history.append(best_val)

            # ---- Stagnation check ----
            best_window.append(best_val)
            if len(best_window) > self.stagnation_window:
                best_window.pop(0)
                if max(best_window) - min(best_window) < self.stagnation_eps:
                    if verbose:
                        print(f"[Gen {gen}] Restart triggered (stagnation).")

                    # Keep elites, refill rest with random
                    elite_k = max(1, int(self.elite_fraction * self.pop_size))
                    idx_best = np.argsort(fitness)[:elite_k]
                    elites = X[idx_best]
                    elites_fit = fitness[idx_best]

                    rest = random_points(
                        n=self.pop_size - elite_k,
                        dim=self.dim,
                        lower=self.lower,
                        upper=self.upper,
                    )
                    X = np.vstack([elites, rest])
                    fitness = np.concatenate([elites_fit, self._evaluate(rest)])

                    best_window = [np.min(fitness)]

            # ---- Update schedules ----
            frac_gen_t = max(self.frac_gen_min, frac_gen_t * self.frac_decay)
            frac_rand_t = max(self.frac_rand_min, frac_rand_t * self.frac_decay)

            sigma_t *= self.sigma_decay
            p_cauchy_t *= self.p_cauchy_decay
            p_cauchy_t = np.clip(p_cauchy_t, 0.0, 1.0)

            if verbose and (gen % 10 == 0 or gen == 1):
                print(
                    f"Gen {gen:4d} | Best = {best_val:.6f} | "
                    f"Evals = {self.cache.misses}/{self.max_evals} | "
                    f"frac_gen={frac_gen_t:.3f} | frac_rand={frac_rand_t:.3f} | "
                    f"sigma={sigma_t:.4f} | p_cauchy={p_cauchy_t:.3f}"
                )

            # ---- Termination by budget ----
            if self.cache.misses >= self.max_evals:
                break

        # ---- Return best ----
        idx_best = np.argmin(fitness)
        return (
            X[idx_best],
            fitness[idx_best],
            np.array(self.best_history),
            np.array(self.op_prob_history),
            self.cache.stats(),
        )


# --------- Quick example ---------
if __name__ == "__main__":
    from function import rastrigin

    cfg = rastrigin(dim=10)

    opt = GeneticOptimizer(
        f=cfg["f"],
        dim=10,
        lower=cfg["lower"],
        upper=cfg["upper"],
        max_evals=50_000,
        seed=42,
        crowding_sigma=1.0,
    )

    best_x, best_f, history, op_hist, cache_stats = opt.run(verbose=True)
    print("Best f:", best_f)
    print("Cache stats:", cache_stats)
