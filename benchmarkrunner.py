import sys
import os
import numpy as np
import pandas as pd

from function import rastrigin
from engine import GeneticOptimizer

# Baselines
import cma
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import optuna


# ----------------------------
# Common config
# ------DE----------------------
DIM = 20
LOWER = -512
UPPER = 512
E_MAX = 100
LOG_EVERY = 1000
SEEDS = [0, 1, 2, 3, 4]   # increase later if needed

OUT_FILE = "benchmark_results.csv"

cfg = rastrigin(dim=DIM)
f = cfg["f"]


# ----------------------------
# Logger + budget wrapper
# ----------------------------
class EvalLogger:
    def __init__(self, algo_name, seed):
        self.algo = algo_name
        self.seed = seed
        self.records = []
        self.best_so_far = np.inf
        self.next_log = LOG_EVERY

    def update(self, eval_count, value):
        if value < self.best_so_far:
            self.best_so_far = value

        while eval_count >= self.next_log and self.next_log <= E_MAX:
            self.records.append({
                "algo": self.algo,
                "seed": self.seed,
                "evals": self.next_log,
                "best_f": self.best_so_far,
            })
            self.next_log += LOG_EVERY

    def finalize(self):
        return self.records


class BudgetedFunction:
    def __init__(self, f, logger):
        self.f = f
        self.logger = logger
        self.evals = 0

    def __call__(self, x):
        self.evals += 1
        val = self.f(x)
        self.logger.update(self.evals, val)
        return val


# ----------------------------
# 1) Your GA
# ----------------------------
def run_ga(seed):
    logger = EvalLogger("GA", seed)
    bf = BudgetedFunction(f, logger)

    opt = GeneticOptimizer(
        f=bf,
        dim=DIM,
        lower=LOWER,
        upper=UPPER,
        max_evals=E_MAX,
        seed=seed,
    )

    opt.run(verbose=False)
    return logger.finalize()


# ----------------------------
# 2) CMA-ES
# ----------------------------
def run_cmaes(seed):
    logger = EvalLogger("CMAES", seed)
    bf = BudgetedFunction(f, logger)

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(LOWER, UPPER, size=DIM)

    es = cma.CMAEvolutionStrategy(x0, 2.5, {"seed": seed, "maxfevals": E_MAX})

    while not es.stop():
        xs = es.ask()
        vals = [bf(x) for x in xs]
        es.tell(xs, vals)
        if bf.evals >= E_MAX:
            break

    return logger.finalize()


# ----------------------------
# 3) DE / PSO via pymoo
# ----------------------------
class RastriginProblem(Problem):
    def __init__(self, f_wrap):
        super().__init__(n_var=DIM, n_obj=1, xl=LOWER, xu=UPPER)
        self.f_wrap = f_wrap

    def _evaluate(self, X, out, *args, **kwargs):
        vals = np.array([self.f_wrap(x) for x in X])
        out["F"] = vals.reshape(-1, 1)


def run_pymoo(algo_name, seed):
    logger = EvalLogger(algo_name, seed)
    bf = BudgetedFunction(f, logger)

    problem = RastriginProblem(bf)

    if algo_name == "DE":
        algo = DE(pop_size=100)
    elif algo_name == "PSO":
        algo = PSO(pop_size=100)
    else:
        raise ValueError("Unknown pymoo algo")

    minimize(
        problem,
        algo,
        termination=("n_eval", E_MAX),
        seed=seed,
        verbose=False,
    )

    return logger.finalize()


# ----------------------------
# 4) Optuna (TPE / BOHB-like)
# ----------------------------
def run_optuna(seed):
    logger = EvalLogger("OPTUNA_TPE", seed)
    bf = BudgetedFunction(f, logger)

    def objective(trial):
        x = np.array([trial.suggest_float(f"x{i}", LOWER, UPPER) for i in range(DIM)])
        return bf(x)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=E_MAX)

    return logger.finalize()


# ----------------------------
# 5) Random Search
# ----------------------------
def run_random(seed):
    logger = EvalLogger("RANDOM", seed)
    rng = np.random.default_rng(seed)

    for i in range(E_MAX):
        x = rng.uniform(LOWER, UPPER, size=DIM)
        val = f(x)
        logger.update(i + 1, val)

    return logger.finalize()


# ----------------------------
# Main entry: ONE ALGO AT A TIME
# ----------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python benchmark_runner.py [GA|CMAES|DE|PSO|OPTUNA|RANDOM]")
        sys.exit(1)

    algo = sys.argv[1].upper()

    runners = {
        "GA": run_ga,
        "CMAES": run_cmaes,
        "DE": lambda seed: run_pymoo("DE", seed),
        "PSO": lambda seed: run_pymoo("PSO", seed),
        "OPTUNA": run_optuna,
        "RANDOM": run_random,
    }

    if algo not in runners:
        print("Unknown algorithm:", algo)
        sys.exit(1)

    all_records = []

    for seed in SEEDS:
        print(f"Running {algo} with seed {seed}...")
        recs = runners[algo](seed)
        all_records += recs

    df_new = pd.DataFrame(all_records)

    if os.path.exists(OUT_FILE):
        df_old = pd.read_csv(OUT_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(OUT_FILE, index=False)
    print(f"Saved results to {OUT_FILE}")


if __name__ == "__main__":
    main()
