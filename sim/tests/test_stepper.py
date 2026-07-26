"""
Tests for sim/stepper.py (Chunk 8b: Euler stepping).

Check the single step (hand case, always-legal output, overshoot clamp, fixed points),
then the trajectory (shape, every row on the simplex, sticky extinction, S-curve
convergence to the fittest corner) and composition with a real cost-adjusted matrix.

Run:  python -m pytest sim/tests -q
"""
import os
import sys

import numpy as np
import pytest

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SIM_DIR)

import stepper  # noqa: E402
from cost_layer import cost_adjusted_matrix  # noqa: E402
from payoff_matrix import PayoffMatrix, base_payoff_matrix  # noqa: E402

STRATEGIES = ("Truth", "CR3", "IF3")


def matrix_from(values) -> PayoffMatrix:
    return PayoffMatrix(strategies=STRATEGIES, values=np.array(values, dtype=float))


# Same hand-worked matrix as the replicator tests:
#   x = [0.5,0.3,0.2],  dx = [0.115, -0.081, -0.034]
HAND_MATRIX = matrix_from([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
HAND_SHARES = np.array([0.5, 0.3, 0.2])

# Strategy 0 (Truth) strictly dominates: its fitness is 3 whatever the mix, the others 1 and 0.
DOMINANCE_MATRIX = matrix_from([[3, 3, 3], [1, 1, 1], [0, 0, 0]])


# ==========================================================================
# euler_step
# ==========================================================================

def test_euler_step_hand_case():
    """x + dx*dt for a small step: [0.5,0.3,0.2] + 0.1*[0.115,-0.081,-0.034]."""
    stepped = stepper.euler_step(HAND_SHARES, HAND_MATRIX, time_step=0.1)
    assert stepped == pytest.approx([0.5115, 0.2919, 0.1966])


def test_euler_step_returns_a_valid_simplex_point():
    """Output shares are non-negative and sum to 1."""
    stepped = stepper.euler_step(HAND_SHARES, HAND_MATRIX, time_step=0.5)
    assert np.all(stepped >= 0)
    assert stepped.sum() == pytest.approx(1.0)


def test_euler_step_clamps_overshoot_from_a_too_big_step():
    """A large step drives a share negative pre-clamp; the result is still a legal simplex point."""
    stepped = stepper.euler_step(HAND_SHARES, HAND_MATRIX, time_step=5.0)
    assert np.all(stepped >= 0)
    assert stepped.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("corner", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
def test_euler_step_holds_at_a_pure_corner(corner):
    """A single-strategy population does not move (velocity zero -> stays put)."""
    stepped = stepper.euler_step(np.array(corner, dtype=float), HAND_MATRIX, time_step=1.0)
    assert stepped == pytest.approx(corner)


def test_euler_step_keeps_extinct_strategy_extinct():
    """A strategy at 0% stays at 0% after a step (no spontaneous resurrection)."""
    stepped = stepper.euler_step(np.array([0.7, 0.3, 0.0]), HAND_MATRIX, time_step=0.3)
    assert stepped[2] == 0.0


# ==========================================================================
# trajectory
# ==========================================================================

def test_trajectory_has_expected_shape_and_start():
    """(generations+1, strategies); row 0 is the normalized initial mix."""
    path = stepper.trajectory([2.0, 1.0, 1.0], HAND_MATRIX, time_step=0.1, generations=10)
    assert path.shape == (11, 3)
    assert path[0] == pytest.approx([0.5, 0.25, 0.25])   # normalized from [2,1,1]


def test_trajectory_generations_zero_is_just_the_start():
    """Zero steps -> a single row, the starting state."""
    path = stepper.trajectory(HAND_SHARES, HAND_MATRIX, time_step=0.1, generations=0)
    assert path.shape == (1, 3)
    assert path[0] == pytest.approx(HAND_SHARES)


def test_trajectory_every_row_is_on_the_simplex():
    """Every state along the path is non-negative and sums to 1."""
    path = stepper.trajectory([0.4, 0.3, 0.3], HAND_MATRIX, time_step=0.2, generations=50)
    assert np.all(path >= 0)
    assert path.sum(axis=1) == pytest.approx(np.ones(51))


def test_trajectory_converges_to_the_dominant_corner():
    """With a strictly dominant strategy, the population flows to its corner."""
    path = stepper.trajectory([0.4, 0.3, 0.3], DOMINANCE_MATRIX, time_step=0.1, generations=300)
    assert path[-1] == pytest.approx([1.0, 0.0, 0.0], abs=1e-3)


def test_trajectory_winner_grows_monotonically_the_s_curve():
    """The winning strategy's share only ever increases (the S-curve is monotone)."""
    winner = stepper.trajectory([0.4, 0.3, 0.3], DOMINANCE_MATRIX, time_step=0.1, generations=300)[:, 0]
    assert np.all(np.diff(winner) >= -1e-12)


# ==========================================================================
# Composition with the real (cost-adjusted) payoff matrix
# ==========================================================================

def test_trajectory_on_real_matrix_at_high_cost_shrinks_truth():
    """At 10% cost truth is below average, so from a uniform start its share falls over time."""
    base = base_payoff_matrix(2_000, np.random.default_rng(1))
    adjusted = cost_adjusted_matrix(base, truth_cost_percent=10)
    path = stepper.trajectory([1 / 3, 1 / 3, 1 / 3], adjusted, time_step=0.1, generations=200)
    assert path[-1][0] < path[0][0]                       # Truth share (col 0) declined
    assert np.all(path >= 0) and path.sum(axis=1) == pytest.approx(np.ones(201))
