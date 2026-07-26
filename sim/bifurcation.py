"""
Chunk 8c: find the bifurcation cost band (~4.27%).
Model + intuition: notes/bifurcation-search.md

Detect where the replicator flow's winner flips from Truth to IF3 as perceptual cost rises.
The transition is a thin COEXISTENCE BAND, not a knife-edge: below it Truth wins outright,
above it IF3 wins outright, and inside neither reaches its corner.

Method (three layers):
  1. attractor(matrix)      - run a trajectory, read the corner it settles into (or coexistence).
  2. coarse sweep           - confirm the winner flips exactly once (monotone) and bracket it.
  3. binary search          - pin each band edge inside its bracket. Valid only because (2) held.

Tests: sim/tests/test_bifurcation.py
"""
from __future__ import annotations

import numpy as np

from array_types import PopulationShares
from cost_layer import cost_adjusted_matrix
from payoff_matrix import PayoffMatrix
from stepper import euler_step

CENTER: PopulationShares = np.array([1 / 3, 1 / 3, 1 / 3])
COEXISTENCE = "coexistence"


def attractor(
    matrix: PayoffMatrix,
    initial_shares: PopulationShares = CENTER,
    time_step: float = 0.1,
    generations: int = 2000,
    win_threshold: float = 0.99,
) -> str:
    """Where the replicator flow settles from `initial_shares`: a strategy name, or "coexistence".

    Runs the trajectory `generations` steps, then reads the leading share. A strategy is the
    winner only if its share clears `win_threshold`; otherwise no corner was reached and the
    outcome is coexistence (an interior or edge attractor).
    """
    state = np.asarray(initial_shares, dtype=float)
    state = state / state.sum()
    for _ in range(generations):
        state = euler_step(state, matrix, time_step)
    leader = int(np.argmax(state))
    return matrix.strategies[leader] if state[leader] >= win_threshold else COEXISTENCE


def _edge(predicate, low: float, high: float, tolerance_percent: float) -> float:
    """Binary-search the cost where `predicate` flips False->True (predicate(low)=False, (high)=True)."""
    while high - low > tolerance_percent:
        mid = 0.5 * (low + high)
        if predicate(mid):
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def bifurcation_band(
    base_matrix: PayoffMatrix,
    initial_shares: PopulationShares = CENTER,
    time_step: float = 0.1,
    generations: int = 2000,
    win_threshold: float = 0.99,
    max_percent: float = 12.0,
    sweep_step_percent: float = 1.0,
    tolerance_percent: float = 0.01,
) -> tuple[float, float]:
    """The coexistence band (lower_edge, upper_edge) in cost-percent-of-truth's-payoff.

    Below lower_edge Truth wins; above upper_edge IF3 wins; between them they coexist. Raises
    ValueError if the coarse sweep shows a non-monotone flip (which would make binary search unsafe)
    or if the cost range does not bracket the transition.
    """
    def outcome(percent: float) -> str:
        matrix = cost_adjusted_matrix(base_matrix, percent)
        return attractor(matrix, initial_shares, time_step, generations, win_threshold)

    grid = np.arange(0.0, max_percent + 1e-9, sweep_step_percent)
    outcomes = [outcome(percent) for percent in grid]
    truth_wins = [o == "Truth" for o in outcomes]
    if3_wins = [o == "IF3" for o in outcomes]

    if not truth_wins[0]:
        raise ValueError("Truth does not win at zero cost; check the base matrix or start point.")
    if not if3_wins[-1]:
        raise ValueError(f"IF3 does not win by {max_percent}% cost; widen max_percent.")
    # Monotone structure: Truth-wins must be a prefix, IF3-wins a suffix - no flip-backs.
    if any(truth_wins[i] and not truth_wins[i - 1] for i in range(1, len(truth_wins))):
        raise ValueError("Truth wins non-monotonically across cost; binary search would be unsafe.")
    if any(if3_wins[i - 1] and not if3_wins[i] for i in range(1, len(if3_wins))):
        raise ValueError("IF3 wins non-monotonically across cost; binary search would be unsafe.")

    last_truth = max(p for p, won in zip(grid, truth_wins) if won)
    first_non_truth = min(p for p, won in zip(grid, truth_wins) if not won)
    last_non_if3 = max(p for p, won in zip(grid, if3_wins) if not won)
    first_if3 = min(p for p, won in zip(grid, if3_wins) if won)

    lower_edge = _edge(lambda p: outcome(p) != "Truth", last_truth, first_non_truth, tolerance_percent)
    upper_edge = _edge(lambda p: outcome(p) == "IF3", last_non_if3, first_if3, tolerance_percent)
    return (lower_edge, upper_edge)
