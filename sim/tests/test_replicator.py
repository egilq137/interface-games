"""
Tests for sim/replicator.py (Chunk 8a: the pure replicator equation, Eq. 21).

A worked hand case pins the three quantities (strategy_fitness, mean_fitness,
replicator_velocity); the rest check the two defining properties - conservation
(velocities sum to 0) and sticky extinction (x_i=0 => dx_i=0) - plus corner fixed
points and composition with a real cost-adjusted matrix.

Run:  python -m pytest sim/tests -q
"""
import os
import sys

import numpy as np
import pytest

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SIM_DIR)

import replicator  # noqa: E402
from cost_layer import cost_adjusted_matrix  # noqa: E402
from payoff_matrix import PayoffMatrix, base_payoff_matrix  # noqa: E402

STRATEGIES = ("Truth", "CR3", "IF3")


def matrix_from(values) -> PayoffMatrix:
    return PayoffMatrix(strategies=STRATEGIES, values=np.array(values, dtype=float))


# A deterministic 3x3 matrix worked fully by hand.
#   P = [[2,1,0],[0,2,1],[1,0,2]],  x = [0.5, 0.3, 0.2]
#   f    = P @ x    = [1.3, 0.8, 0.9]
#   fbar = x . f    = 0.65 + 0.24 + 0.18 = 1.07
#   dx   = x*(f-fbar) = [0.115, -0.081, -0.034]  (sums to 0)
HAND_MATRIX = matrix_from([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
HAND_SHARES = np.array([0.5, 0.3, 0.2])


# ==========================================================================
# strategy_fitness
# ==========================================================================

def test_strategy_fitness_is_row_dot_shares():
    """Each strategy's fitness is its matrix row weighted by the population shares."""
    fitness = replicator.strategy_fitness(HAND_SHARES, HAND_MATRIX)
    assert fitness == pytest.approx([1.3, 0.8, 0.9])


def test_strategy_fitness_matches_worked_truth_row():
    """The session example: Truth row [63.4, 65.7, 64.5] vs (0.5,0.3,0.2) -> 64.31."""
    matrix = matrix_from([[63.4, 65.7, 64.5], [0, 0, 0], [0, 0, 0]])
    fitness = replicator.strategy_fitness([0.5, 0.3, 0.2], matrix)
    assert fitness[0] == pytest.approx(64.31)


# ==========================================================================
# mean_fitness
# ==========================================================================

def test_mean_fitness_is_shares_dot_fitness():
    """Population average = each strategy's fitness weighted by its share."""
    assert replicator.mean_fitness(HAND_SHARES, HAND_MATRIX) == pytest.approx(1.07)


def test_mean_fitness_ignores_extinct_strategy():
    """A strategy at 0% share cannot affect the average, however fit it would be."""
    shares = np.array([0.6, 0.4, 0.0])
    huge = matrix_from([[1, 1, 1], [1, 1, 1], [999, 999, 999]])  # IF3 row enormous but extinct
    tame = matrix_from([[1, 1, 1], [1, 1, 1], [0, 0, 0]])
    assert replicator.mean_fitness(shares, huge) == replicator.mean_fitness(shares, tame)


# ==========================================================================
# replicator_velocity
# ==========================================================================

def test_replicator_velocity_hand_case():
    """dx = x*(f - fbar) for the fully worked matrix."""
    velocity = replicator.replicator_velocity(HAND_SHARES, HAND_MATRIX)
    assert velocity == pytest.approx([0.115, -0.081, -0.034])


def test_velocity_components_sum_to_zero():
    """Conservation: winners' gains cancel losers' losses, so shares stay on the simplex."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        shares = rng.dirichlet([1, 1, 1])                 # random point in the simplex
        matrix = matrix_from(rng.uniform(50, 70, size=(3, 3)))
        assert replicator.replicator_velocity(shares, matrix).sum() == pytest.approx(0.0, abs=1e-12)


def test_extinct_strategy_has_zero_velocity():
    """x_i = 0 => dx_i = 0: an absent strategy can never spontaneously appear."""
    shares = np.array([0.7, 0.3, 0.0])
    velocity = replicator.replicator_velocity(shares, HAND_MATRIX)
    assert velocity[2] == 0.0


@pytest.mark.parametrize("corner", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
def test_pure_corners_are_fixed_points(corner):
    """A population of a single strategy does not move (all velocities zero)."""
    velocity = replicator.replicator_velocity(np.array(corner, dtype=float), HAND_MATRIX)
    assert velocity == pytest.approx([0.0, 0.0, 0.0])


def test_fittest_strategy_grows_and_least_fit_shrinks():
    """Sign check: above-average grows (dx>0), below-average shrinks (dx<0)."""
    velocity = replicator.replicator_velocity(HAND_SHARES, HAND_MATRIX)
    fitness = replicator.strategy_fitness(HAND_SHARES, HAND_MATRIX)
    assert velocity[np.argmax(fitness)] > 0
    assert velocity[np.argmin(fitness)] < 0


# ==========================================================================
# Composition with the real (cost-adjusted) payoff matrix
# ==========================================================================

def test_velocity_conserves_on_a_real_cost_adjusted_matrix():
    """End to end: base matrix -> cost layer -> replicator velocity still sums to 0."""
    base = base_payoff_matrix(2_000, np.random.default_rng(1))
    adjusted = cost_adjusted_matrix(base, truth_cost_percent=10)
    velocity = replicator.replicator_velocity([0.4, 0.3, 0.3], adjusted)
    assert velocity.sum() == pytest.approx(0.0, abs=1e-12)
