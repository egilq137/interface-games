"""
Tests for sim/bifurcation.py (Chunk 8c: the bifurcation band).

attractor is checked on synthetic matrices with known outcomes (a dominant strategy;
a neutral matrix -> coexistence) and on the real cost-adjusted matrix (Truth at low cost,
IF3 at high cost). bifurcation_band is checked for ordered edges, reproduction of the
paper's ~4.27% transition, the Truth/IF3 verdicts just outside the band, and its guard
against a too-narrow cost range.

Run:  python -m pytest sim/tests -q
"""
import os
import sys

import numpy as np
import pytest

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SIM_DIR)

import bifurcation  # noqa: E402
from cost_layer import cost_adjusted_matrix  # noqa: E402
from payoff_matrix import PayoffMatrix, base_payoff_matrix  # noqa: E402

STRATEGIES = ("Truth", "CR3", "IF3")


def matrix_from(values) -> PayoffMatrix:
    return PayoffMatrix(strategies=STRATEGIES, values=np.array(values, dtype=float))


DOMINANCE_MATRIX = matrix_from([[3, 3, 3], [1, 1, 1], [0, 0, 0]])   # Truth strictly dominates
NEUTRAL_MATRIX = matrix_from([[1, 1, 1], [1, 1, 1], [1, 1, 1]])     # all equal -> nothing moves


@pytest.fixture(scope="module")
def base_matrix() -> PayoffMatrix:
    """The cost-0 matrix, built once and shared across the (slower) band tests."""
    return base_payoff_matrix(200_000, np.random.default_rng(5))


# ==========================================================================
# attractor
# ==========================================================================

def test_attractor_names_the_dominant_strategy():
    """A strictly dominant strategy is the attractor the flow reaches."""
    assert bifurcation.attractor(DOMINANCE_MATRIX) == "Truth"


def test_attractor_reports_coexistence_when_no_corner_is_reached():
    """No share clears the threshold (nothing moves) -> coexistence, not a winner."""
    assert bifurcation.attractor(NEUTRAL_MATRIX) == bifurcation.COEXISTENCE


def test_attractor_flips_with_cost_on_the_real_matrix(base_matrix):
    """Low cost -> Truth corner; high cost -> IF3 corner."""
    assert bifurcation.attractor(cost_adjusted_matrix(base_matrix, 1)) == "Truth"
    assert bifurcation.attractor(cost_adjusted_matrix(base_matrix, 10)) == "IF3"


# ==========================================================================
# bifurcation_band
# ==========================================================================

def test_band_edges_are_ordered(base_matrix):
    """lower_edge <= upper_edge (Truth loses before IF3 fully takes over)."""
    lower, upper = bifurcation.bifurcation_band(base_matrix)
    assert lower <= upper


def test_band_matches_the_paper_bifurcation(base_matrix):
    """The transition sits at the paper's ~4.27% of truth's payout."""
    lower, upper = bifurcation.bifurcation_band(base_matrix)
    midpoint = 0.5 * (lower + upper)
    assert midpoint == pytest.approx(4.27, abs=0.7)


def test_just_below_band_truth_wins_just_above_if3_wins(base_matrix):
    """Sanity: outside the band the outcomes are the clean corner wins."""
    lower, upper = bifurcation.bifurcation_band(base_matrix)
    assert bifurcation.attractor(cost_adjusted_matrix(base_matrix, lower - 0.5)) == "Truth"
    assert bifurcation.attractor(cost_adjusted_matrix(base_matrix, upper + 0.5)) == "IF3"


def test_band_raises_when_cost_range_too_narrow(base_matrix):
    """If IF3 never wins within max_percent, the search refuses rather than guessing."""
    with pytest.raises(ValueError):
        bifurcation.bifurcation_band(base_matrix, max_percent=2.0)
