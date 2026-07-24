"""
Tests for sim/optimize.py (Chunk 5: the boundary optimizer).

Same philosophy: each function checked on its own against independently-computed
expectations. The full CR3/IF3 anchor (boundaries ~30/70 etc.) is a slow validation
run, reported separately - here we prove the mechanics.

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
import optimize  # noqa: E402


def fresh_rng(seed=0):
    return np.random.default_rng(seed)


# ==========================================================================
# solo_foraging_payoff(preferences, round_utilities, rng)
# ==========================================================================

def test_solo_foraging_payoff_takes_the_most_preferred_territory():
    """Ordinary case: forager takes its highest-preference territory and collects its worth."""
    preferences = np.array([[10.0, 30.0, 20.0]])   # territory 1 is most preferred
    round_utilities = np.array([[5.0, 9.0, 7.0]])
    assert optimize.solo_foraging_payoff(preferences, round_utilities, fresh_rng()) == pytest.approx(9.0)


def test_solo_foraging_truth_equals_best_of_three():
    """Independent oracle: Truth prefers by utility, so its solo payoff = mean of the
    best-of-three utilities, computed by taking the row max.
    """
    foraging_rounds = model.sample_competitions(50_000, fresh_rng(1))
    round_utilities = model.utility(foraging_rounds)
    truth = competition.truth_preferences(foraging_rounds)

    engine = optimize.solo_foraging_payoff(truth, round_utilities, fresh_rng())
    independent = round_utilities.max(axis=1).mean()
    assert engine == pytest.approx(independent)


def test_solo_foraging_all_ties_equals_mean_territory_worth():
    """If every territory is equally preferred, the pick is uniform-random, so the
    expected payoff is just the average territory worth.
    """
    foraging_rounds = model.sample_competitions(200_000, fresh_rng(2))
    round_utilities = model.utility(foraging_rounds)
    flat_preferences = np.zeros_like(round_utilities)   # nothing distinguishes the territories

    engine = optimize.solo_foraging_payoff(flat_preferences, round_utilities, fresh_rng(3))
    assert engine == pytest.approx(round_utilities.mean(), abs=0.1)


# ==========================================================================
# optimize_boundaries(...)
# ==========================================================================

def test_optimize_boundaries_returns_the_argmax():
    """The result must be the candidate with the highest solo payoff, cross-checked
    by scoring each candidate independently.
    """
    foraging_rounds = model.sample_competitions(20_000, fresh_rng(4))
    round_utilities = model.utility(foraging_rounds)
    candidates = np.array([[30.0, 70.0], [10.0, 20.0], [45.0, 55.0], [5.0, 95.0]])

    independent = []
    for boundaries in candidates:
        prefs = competition.cr3_preferences(foraging_rounds, boundaries)
        independent.append(optimize.solo_foraging_payoff(prefs, round_utilities, fresh_rng(0)))
    expected_winner = candidates[int(np.argmax(independent))]

    result = optimize.optimize_boundaries(competition.cr3_preferences, candidates, foraging_rounds, round_utilities)
    assert np.array_equal(result.optimal_boundaries, expected_winner)
    assert result.optimal_payoff == pytest.approx(max(independent))
    assert result.candidate_payoffs == pytest.approx(independent)


def test_optimize_boundaries_is_deterministic():
    """Common random numbers -> identical result across runs."""
    foraging_rounds = model.sample_competitions(10_000, fresh_rng(5))
    round_utilities = model.utility(foraging_rounds)
    candidates = np.array([[30.0, 70.0], [40.0, 60.0]])

    a = optimize.optimize_boundaries(competition.cr3_preferences, candidates, foraging_rounds, round_utilities)
    b = optimize.optimize_boundaries(competition.cr3_preferences, candidates, foraging_rounds, round_utilities)
    assert np.array_equal(a.candidate_payoffs, b.candidate_payoffs)
    assert np.array_equal(a.optimal_boundaries, b.optimal_boundaries)


# ==========================================================================
# candidate generators
# ==========================================================================

def test_cr3_candidate_boundaries_are_all_ordered_pairs_in_range():
    """Every candidate is a strictly-ordered pair inside [1, 99]; count = C(99,2)."""
    candidates = optimize.cr3_candidate_boundaries()
    assert candidates.shape == (99 * 98 // 2, 2)
    assert np.all(candidates[:, 0] < candidates[:, 1])
    assert candidates.min() >= 1 and candidates.max() <= 99


def test_if3_symmetric_candidates_are_symmetric_and_ordered():
    """Every candidate has 4 boundaries symmetric about the peak (sum to 100 in pairs),
    strictly ascending, with inner width < outer width.
    """
    candidates = optimize.if3_symmetric_candidate_boundaries()
    assert candidates.shape[1] == 4
    assert np.allclose(candidates[:, 0] + candidates[:, 3], 100)   # outer pair symmetric about 50
    assert np.allclose(candidates[:, 1] + candidates[:, 2], 100)   # inner pair symmetric about 50
    assert np.all(np.diff(candidates, axis=1) > 0)                 # strictly ascending
    assert candidates.min() >= 1 and candidates.max() <= 99
