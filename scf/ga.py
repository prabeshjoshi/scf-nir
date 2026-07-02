"""
Genetic-algorithm optimisation of Spectral Contrast Features.

A chromosome is a (P, 5) integer array whose rows are genes
[s1, w1, s2, w2, active]. The GA minimises the penalised fitness of
Equation (2):

    fitness = RMSE_cv( X_train, y | active SCF features )  +  lambda * n_active

Default hyperparameters.
"""

import random
import numpy as np

from .features import make_scf_features
from .plsr import cv_rmse


# ----- Table 2 defaults -------------------------------------------------------
DEFAULTS = dict(
    pop_size=40,          # Population size
    generations=250,      # Maximum generations
    P=8,                  # Maximum number of feature pairs
    max_width=20,         # Maximum bin width (spectral variables)
    cx_prob=0.70,         # Crossover probability (feature-level)
    mut_prob=0.25,        # Per-gene mutation probability
    tournament_k=3,       # Tournament size
    lam=0.05,             # Penalty per active feature (lambda)
    cv=5,                 # Fitness cross-validation folds
)


def _rng(seed):
    return random.Random(seed)


def initialize_population(pop_size, P, n_var, max_width, rng):
    pop = []
    for _ in range(pop_size):
        chrom = np.zeros((P, 5), dtype=int)
        for i in range(P):
            s1 = rng.randrange(0, n_var)
            w1 = rng.randint(1, max_width)
            s2 = rng.randrange(0, n_var)
            w2 = rng.randint(1, max_width)
            active = rng.choice([0, 1])
            chrom[i] = [s1, w1, s2, w2, active]
        pop.append(chrom)
    return pop


def crossover(p1, p2, rng):
    """Single-point / uniform crossover at the feature (gene) level."""
    c1, c2 = p1.copy(), p2.copy()
    for i in range(p1.shape[0]):
        if rng.random() < 0.5:
            c1[i, :], c2[i, :] = p2[i, :].copy(), p1[i, :].copy()
    return c1, c2


def mutate(chrom, n_var, max_width, mut_prob, rng):
    """Per-gene mutation: with probability mut_prob, resample one field."""
    new = chrom.copy()
    for i in range(new.shape[0]):
        if rng.random() < mut_prob:
            field = rng.randrange(0, 5)
            if field == 0:
                new[i, 0] = rng.randrange(0, n_var)
            elif field == 1:
                new[i, 1] = rng.randint(1, min(max_width, n_var - new[i, 0]))
            elif field == 2:
                new[i, 2] = rng.randrange(0, n_var)
            elif field == 3:
                new[i, 3] = rng.randint(1, min(max_width, n_var - new[i, 2]))
            else:
                new[i, 4] = 1 - new[i, 4]
    return new


def tournament_selection(pop, fitnesses, k, rng):
    chosen = rng.sample(range(len(pop)), k)
    best = min(chosen, key=lambda idx: fitnesses[idx])
    return pop[best].copy()


def evaluate_fitness(chrom, X, y, lam, cv, n_components, max_lv, seed):
    """Equation (2): penalised cross-validated RMSE of the active features."""
    F, active_idx = make_scf_features(X, chrom)
    rmse = cv_rmse(F, y, n_components=n_components, cv=cv, seed=seed, max_lv=max_lv)
    return rmse + lam * len(active_idx)


def run_ga(
    X, y,
    pop_size=DEFAULTS["pop_size"],
    generations=DEFAULTS["generations"],
    P=DEFAULTS["P"],
    max_width=DEFAULTS["max_width"],
    cx_prob=DEFAULTS["cx_prob"],
    mut_prob=DEFAULTS["mut_prob"],
    tournament_k=DEFAULTS["tournament_k"],
    lam=DEFAULTS["lam"],
    cv=DEFAULTS["cv"],
    n_components=None,
    max_lv=10,
    seed=0,
    verbose=True,
):
    """Run the GA and return (best_chromosome, best_fitness, history).

    Notes
    -----
    * ``n_components=None`` selects PLSR latent variables by inner CV inside
      the fitness evaluation (paper-faithful, Section 2.4). Passing an integer
      fixes the count, which is markedly faster for a quick demonstration run.
    * ``generations`` defaults to the Table 2 value (250). A stochastic search
      of this length takes a few minutes on the wheat dataset; small run-to-run
      differences in the selected bins and reported metrics are expected and do
      not affect the method's conclusions.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_var = X.shape[1]
    rng = _rng(seed)

    def fit_of(chrom):
        return evaluate_fitness(chrom, X, y, lam, cv, n_components, max_lv, seed)

    pop = initialize_population(pop_size, P, n_var, max_width, rng)
    fitnesses = [fit_of(ind) for ind in pop]
    history = []

    for gen in range(generations):
        order = sorted(range(len(pop)), key=lambda i: fitnesses[i])
        # Elitism: carry the two best chromosomes forward unchanged.
        new_pop = [pop[order[0]].copy(), pop[order[1]].copy()]
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fitnesses, tournament_k, rng)
            p2 = tournament_selection(pop, fitnesses, tournament_k, rng)
            if rng.random() < cx_prob:
                c1, c2 = crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2
            c1 = mutate(c1, n_var, max_width, mut_prob, rng)
            c2 = mutate(c2, n_var, max_width, mut_prob, rng)
            new_pop.extend([c1, c2])
        pop = new_pop[:pop_size]
        fitnesses = [fit_of(ind) for ind in pop]

        best_idx = int(np.argmin(fitnesses))
        history.append(fitnesses[best_idx])
        if verbose and (gen % max(1, generations // 10) == 0 or gen == generations - 1):
            n_active = int(pop[best_idx][:, 4].sum())
            print(f"  gen {gen + 1:>4}/{generations}  "
                  f"fitness={fitnesses[best_idx]:.4f}  active={n_active}")

    best_idx = int(np.argmin(fitnesses))
    return pop[best_idx], fitnesses[best_idx], history
