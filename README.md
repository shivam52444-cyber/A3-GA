Adaptive Genetic Algorithm for Black-Box Optimization

This project implements a research-oriented Genetic Algorithm (GA) framework for continuous black-box optimization and benchmarks it against strong baselines like PSO, DE, CMA-ES, Optuna (TPE) and Random Search under a fixed evaluation budget.

The focus is on:

Evaluation-budgeted optimization (not iteration-based)

Caching of function evaluations

Diversity preservation (random immigrants, anti-crowding)

Adaptive crossover selection

Scheduled mutation (Cauchy → Gaussian)

Fair, reproducible benchmarking

The code is designed to be modular, extensible, and research-friendly.

📁 Project Structure
.
├── engine.py                  # Main GA engine
├── selection.py               # Selection + anti-crowding
├── crossover.py               # Multiple crossover operators
├── mutation.py                # Gaussian & Cauchy mutation
├── randompointgenerator.py    # Random point generator
├── eval_cache.py              # Evaluation caching
├── functions.py               # Benchmark functions (Rastrigin, etc.)
├── plot.py                    # Function visualization (optional)
├── benchmark_runner.py        # Run benchmarks (one algorithm at a time)
├── benchmark_results.csv      # Saved results (generated)
└── README.md                  # This file

🎯 Goals

Build a from-scratch GA suitable for expensive black-box functions

Control everything by number of function evaluations

Compare fairly against strong baselines

Analyze convergence behavior, not just final scores

Create a foundation for:

Hyperparameter optimization

Expensive ML training objectives

Research-style optimizer experiments

🧠 Key Features of the GA

Evaluation budget as main stopping criterion

Evaluation cache to avoid recomputing same points

Population size:

𝑁
=
min
⁡
(
10
⋅
𝑑
,
  
max_evals
/
10
)
N=min(10⋅d,max_evals/10)

Population composition (initially):

~50% generated from parents

~20% random immigrants

Rest = elites

Both generated & random fractions decay over time (with floors)

Mutation schedule:

Mix of Cauchy & Gaussian

Cauchy probability and variance decay over time

Adaptive crossover selection:

Multiple crossover operators

Probabilities updated via reward/penalty with baseline + forgetting

Restart on stagnation

Anti-crowding to preserve diversity

📦 Installation

It’s recommended to use a virtual environment.

pip install numpy pandas matplotlib
pip install cma
pip install pymoo
pip install optuna


(Optional)

pip install scipy

🧪 Benchmark Setup (Current)

Function: Rastrigin

Dimension: 50

Bounds: [-5.12, 5.12]

Evaluation budget: 10,000

Logging: Every 1,000 evaluations

Seeds: 5

Algorithms supported:

GA (this project)

PSO (via pymoo)

DE (via pymoo)

CMA-ES (via cma)

Optuna TPE (BOHB-like baseline)

Random Search

▶️ How to Run Benchmarks (One by One)

To avoid overloading your machine, run one algorithm at a time:

python benchmark_runner.py GA
python benchmark_runner.py CMAES
python benchmark_runner.py DE
python benchmark_runner.py PSO
python benchmark_runner.py OPTUNA
python benchmark_runner.py RANDOM


Each command will:

Run that algorithm for all seeds

Use the same function, bounds, and budget

Log best-so-far every 1000 evaluations

Append results to:

benchmark_results.csv

📊 Output Format

The results file (benchmark_results.csv) has:

algo, seed, evals, best_f


Example:

GA,0,1000,41842.42
GA,0,2000,12000.11
...
PSO,1,1000,51000.55
...


This format is ready for:

Convergence plots

Mean / median curves

Statistical comparison across seeds

🧠 Interpreting Results

Lower best_f is better (global optimum for Rastrigin = 0)

Compare:

Speed of improvement (early vs late)

Final quality at budget limit

Stability across seeds

In high dimensions (e.g., 50D):

CMA-ES is expected to dominate

PSO/DE are strong baselines

Random Search is a sanity check

The GA’s goal is to close the gap via better exploitation + adaptation

🔬 Research Notes

The framework is evaluation-budget driven, not generation-driven

Caching makes it suitable for expensive objectives

The GA is intentionally modular:

You can add new crossover operators

New mutation schedules

New diversity mechanisms

New benchmark functions

This is a research / experimentation codebase, not a library product

🚧 Current Status

The GA:

Works correctly

Beats random search by a large margin

Shows strong early exploration

Currently underperforms PSO/CMA-ES on 50D Rastrigin

Next steps:

Improve exploitation

Reduce late-stage randomness

Add more directional / local refinement behavior

Run ablation studies# A3-GA
