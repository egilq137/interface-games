"""
Chunk 8a: the pure replicator equation (Eq. 21) - the change per generation.
Model + intuition: notes/replicator-equation.md, notes/replicator-dynamics.md

Turn a payoff matrix + a population mix into each strategy's rate of change. This is the
engine under Fig. 14's flow field. Cost is NOT a term here: feed a cost-adjusted matrix
(cost_layer.cost_adjusted_matrix) and cost flows in through the fitnesses.

Recipe (population_shares x aligned to matrix.strategies, matrix P with P[i][j] = payoff to i vs j):
    strategy_fitness    f    = P @ x            # row i . x  -> expected payoff of i vs the population
    mean_fitness        fbar = x . f            # population average
    replicator_velocity dx   = x * (f - fbar)   # grow if above average, shrink if below

Integrating dx into a trajectory over time (Euler step) is the NEXT sub-chunk.

Tests: sim/tests/test_replicator.py
"""
from __future__ import annotations

import numpy as np

from array_types import PopulationShares, PopulationVelocity, StrategyFitness
from payoff_matrix import PayoffMatrix


def strategy_fitness(population_shares: PopulationShares, matrix: PayoffMatrix) -> StrategyFitness:
    """Expected payoff of each strategy against the current population (frequency-dependent).

    No re-simulation: the matrix already stores each matchup's average, so strategy i's fitness
    is just its row weighted by how likely each opponent is - i.e. the population shares.
    `population_shares` must be aligned to `matrix.strategies` (canonical Truth, CR3, IF3).
    """
    return matrix.values @ np.asarray(population_shares, dtype=float)


def mean_fitness(population_shares: PopulationShares, matrix: PayoffMatrix) -> float:
    """The population-average fitness: each strategy's fitness weighted by its share.

    Extinct strategies (share 0) drop out automatically - no need to special-case them.
    """
    shares = np.asarray(population_shares, dtype=float)
    return float(shares @ strategy_fitness(shares, matrix))


def replicator_velocity(population_shares: PopulationShares, matrix: PayoffMatrix) -> PopulationVelocity:
    """Replicator rate of change dx_i = x_i * (f_i - fbar): grow above average, shrink below.

    Two guaranteed properties (see the tests):
      - components sum to 0, so shares stay on the simplex (conservation);
      - x_i = 0 => dx_i = 0, so an absent strategy can never spontaneously appear.
    """
    shares = np.asarray(population_shares, dtype=float)
    fitness = strategy_fitness(shares, matrix)
    return shares * (fitness - mean_fitness(shares, matrix))
