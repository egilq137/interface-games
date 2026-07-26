"""
Tests for sim/payoff_matrix.py (Chunk 6: the cost-0 payoff matrix).

The headline test reproduces the paper's Table 3 (Appendix B) from our model. The
rest check the pieces: strategy dispatch, the coin-flip expected payoff (hand case),
and the labeled-matrix bookkeeping.

Run:  python -m pytest sim/tests -q
"""
import os
import sys

import numpy as np
import pytest

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SIM_DIR)

import competition  # noqa: E402
import model  # noqa: E402
import payoff_matrix as pm  # noqa: E402


def fresh_rng(seed=0):
    return np.random.default_rng(seed)


# Paper Table 3 (Appendix B, cost = 0): focal-vs-opponent expected payoffs.
PAPER_TABLE_3 = {
    ("Truth", "Truth"): 63.43, ("Truth", "CR3"): 65.72, ("Truth", "IF3"): 64.46,
    ("CR3", "Truth"): 58.11, ("CR3", "CR3"): 60.15, ("CR3", "IF3"): 59.05,
    ("IF3", "Truth"): 60.83, ("IF3", "CR3"): 63.19, ("IF3", "IF3"): 61.77,
}


# ==========================================================================
# strategy_preferences(name, resource_values)
# ==========================================================================

def test_strategy_preferences_dispatches_to_each_strategy():
    """Each name yields exactly what its own preference function (with frozen boundaries) gives."""
    scene = model.sample_competitions(500, fresh_rng(1))
    assert np.array_equal(pm.strategy_preferences("Truth", scene), competition.truth_preferences(scene))
    assert np.array_equal(pm.strategy_preferences("CR3", scene), competition.cr3_preferences(scene, pm.CR3_BOUNDARIES))
    assert np.array_equal(pm.strategy_preferences("IF3", scene), competition.if3_preferences(scene, pm.IF3_BOUNDARIES))


def test_strategy_preferences_rejects_unknown_name():
    """A typo / unknown strategy is a hard error, not a silent wrong answer."""
    with pytest.raises(ValueError):
        pm.strategy_preferences("Simple", model.sample_competitions(10, fresh_rng()))


# ==========================================================================
# expected_payoff(...)  -- the coin-flip average of the two roles
# ==========================================================================

def test_expected_payoff_coin_flip_hand_case():
    """One deterministic competition worked by hand.

    Scene [20, 50, 90] -> utilities [32.465, 100, 13.535]. Focal = Truth; opponent
    always wants territory 1 (the value-50 one).
      - Focal first : Truth takes territory 1 -> utility(50) = 100.
      - Focal second: opponent takes territory 1, Truth takes best of {0,2} = utility(20).
      coin-flip average = (utility(50) + utility(20)) / 2.
    """
    scene = np.array([[20, 50, 90]])
    focal = competition.truth_preferences(scene)          # [[32.465, 100, 13.535]]
    opponent = np.array([[0.0, 100.0, 0.0]])              # always prefers territory 1
    expected = 0.5 * (float(model.utility(50)) + float(model.utility(20)))
    assert pm.expected_payoff(focal, opponent, scene, fresh_rng()) == pytest.approx(expected, abs=1e-9)


def test_expected_payoff_truth_self_matches_the_oracle():
    """Truth-vs-Truth via expected_payoff must land on 63.43 and agree with the
    independent truth_vs_truth_expected_payoff helper.
    """
    scene = model.sample_competitions(400_000, fresh_rng(2))
    truth = competition.truth_preferences(scene)
    from_matrix = pm.expected_payoff(truth, truth, scene, fresh_rng(3))
    independent = competition.truth_vs_truth_expected_payoff(400_000, fresh_rng(2))
    assert from_matrix == pytest.approx(63.43, abs=0.2)
    assert from_matrix == pytest.approx(independent, abs=0.1)


# ==========================================================================
# PayoffMatrix bookkeeping
# ==========================================================================

def test_payoff_matrix_lookup_by_name():
    """payoff_of resolves names to the right (row, col) entry."""
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    matrix = pm.PayoffMatrix(strategies=("Truth", "CR3", "IF3"), values=values)
    assert matrix.payoff_of("Truth", "IF3") == 3.0     # row 0, col 2
    assert matrix.payoff_of("IF3", "Truth") == 7.0     # row 2, col 0
    assert matrix.payoff_of("CR3", "CR3") == 5.0


def test_base_payoff_matrix_has_expected_shape_and_labels():
    """The matrix is 3x3 in canonical strategy order."""
    matrix = pm.base_payoff_matrix(2_000, fresh_rng(4))
    assert matrix.strategies == ("Truth", "CR3", "IF3")
    assert matrix.values.shape == (3, 3)


# ==========================================================================
# The headline: reproduce the paper's Table 3
# ==========================================================================

def test_base_payoff_matrix_reproduces_table_3():
    """Every cell of our cost-0 matrix matches the paper's Table 3 within tolerance.

    1,000,000 competitions per cell (the paper used 100,000,000) to shrink Monte-Carlo
    variance and cover more of the territory-draw distribution.
    """
    matrix = pm.base_payoff_matrix(1_000_000, fresh_rng(5))
    for (focal, opponent), paper_value in PAPER_TABLE_3.items():
        assert matrix.payoff_of(focal, opponent) == pytest.approx(paper_value, abs=0.5), (
            f"{focal} vs {opponent}: got {matrix.payoff_of(focal, opponent):.2f}, paper {paper_value}"
        )
