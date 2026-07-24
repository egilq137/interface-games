"""
Tests for sim/model.py (Chunk 1: the world).

Philosophy (see the /test-every-function skill): each function is tested on its own,
on the ordinary case AND the edges, against expected values computed INDEPENDENTLY of
the code under test (hand-computed literals, or Python's `math` rather than the module's
numpy formula) - so a test passing means more than "the code agrees with itself".

Run:  python -m pytest sim/tests -q
"""
import math
import os
import sys

import numpy as np
import pytest

# Make `import model` work regardless of where pytest is invoked from.
SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SIM_DIR)

import model  # noqa: E402


# ==========================================================================
# utility(resource_value)
# ==========================================================================

# Reference worths, computed by hand from U(v) = 100*exp(-(v-50)^2/800).
# These are literals, NOT produced by the code, so they are an independent oracle.
HAND_COMPUTED_UTILITY = {
    50: 100.0,          # exp(0)
    70: 60.653066,      # 100*exp(-0.5)
    30: 60.653066,      # 100*exp(-0.5)  (mirror of 70)
    90: 13.533528,      # 100*exp(-2.0)
    10: 13.533528,      # 100*exp(-2.0)  (mirror of 90)
    20: 32.465247,      # 100*exp(-1.125)
    1: 4.9724873,       # 100*exp(-2401/800)
    100: 4.393693,      # 100*exp(-2500/800)
}


@pytest.mark.parametrize("value, expected_worth", HAND_COMPUTED_UTILITY.items())
def test_utility_matches_hand_computed_values(value, expected_worth):
    """Ordinary case: worth equals independently hand-computed Gaussian values."""
    assert model.utility(value) == pytest.approx(expected_worth, abs=1e-5)


def test_utility_peaks_at_fifty():
    """The peak is exactly PEAK_UTILITY, and 50 beats every other value in the domain."""
    assert model.utility(50) == pytest.approx(100.0)
    domain = np.arange(1, 101)
    assert model.utility(domain).max() == pytest.approx(model.utility(50))
    assert np.argmax(model.utility(domain)) == 50 - 1  # index of value 50


def test_utility_is_symmetric_about_the_peak():
    """Worth depends only on distance from 50, so v and (100 - v) match... about 50."""
    for distance in range(0, 50):
        assert model.utility(50 - distance) == pytest.approx(model.utility(50 + distance))


def test_utility_decreases_monotonically_away_from_peak():
    """Moving away from 50 in either direction strictly lowers worth."""
    rising_side = model.utility(np.arange(1, 51))      # 1..50, should be increasing
    falling_side = model.utility(np.arange(50, 101))   # 50..100, should be decreasing
    assert np.all(np.diff(rising_side) > 0)
    assert np.all(np.diff(falling_side) < 0)


def test_utility_is_vectorized_and_shape_preserving():
    """A scalar in gives a scalar-worth; an array in gives the same-shaped array out."""
    assert model.utility(np.array([50, 30, 90])) == pytest.approx([100.0, 60.653066, 13.533528], abs=1e-5)
    grid = np.array([[1, 50], [90, 100]])
    assert model.utility(grid).shape == grid.shape


def test_utility_is_a_pure_gaussian_with_no_clipping():
    """Spec-silent edge: values outside {1..100} are NOT clamped - it's a bare Gaussian.
    We assert the current (documented) behavior so a future clip can't slip in unnoticed.
    """
    # v=0 is 50 below the peak, same as v=100 -> equal worths, both computed by hand.
    assert model.utility(0) == pytest.approx(model.utility(100))
    assert model.utility(0) == pytest.approx(4.393693, abs=1e-5)
    # Just outside the domain still evaluates to a positive Gaussian value (no clip to 0).
    assert model.utility(150) > 0.0
    assert model.utility(150) == pytest.approx(100.0 * math.exp(-((150 - 50) ** 2) / 800.0))
    # Extremely far out simply underflows to 0.0 via floating point (still no error/clip).
    assert model.utility(1000) == 0.0


# ==========================================================================
# sample_competitions(number_of_competitions, rng)
# ==========================================================================

def test_sample_competitions_has_expected_shape():
    """Ordinary case: (n rows, 3 territories per competition)."""
    result = model.sample_competitions(1000, np.random.default_rng(0))
    assert result.shape == (1000, model.TERRITORIES_PER_COMPETITION)


def test_sample_competitions_stays_within_the_resource_range():
    """Every drawn value lies in {1..100}; nothing spills below 1 or above 100."""
    result = model.sample_competitions(50_000, np.random.default_rng(1))
    assert result.min() >= 1
    assert result.max() <= model.MAX_RESOURCE_VALUE


def test_sample_competitions_can_reach_both_endpoints():
    """The off-by-one trap: 1 AND 100 must both be attainable.
    If the high bound were wrong (half-open misuse), 100 would never appear.
    """
    result = model.sample_competitions(200_000, np.random.default_rng(2))
    values_seen = set(np.unique(result).tolist())
    assert 1 in values_seen, "lowest value 1 should be reachable"
    assert 100 in values_seen, "highest value 100 should be reachable"
    assert 0 not in values_seen and 101 not in values_seen


def test_sample_competitions_returns_integers():
    """Resource values are discrete counts, not floats."""
    result = model.sample_competitions(10, np.random.default_rng(3))
    assert np.issubdtype(result.dtype, np.integer)


def test_sample_competitions_is_reproducible_for_a_given_seed():
    """Same seed -> identical draws; different seed -> different draws."""
    same_a = model.sample_competitions(1000, np.random.default_rng(42))
    same_b = model.sample_competitions(1000, np.random.default_rng(42))
    different = model.sample_competitions(1000, np.random.default_rng(43))
    assert np.array_equal(same_a, same_b)
    assert not np.array_equal(same_a, different)


def test_sample_competitions_marginal_is_roughly_uniform():
    """Statistical sanity: a flat draw over 1..100 has mean ~50.5 (loose tolerance)."""
    result = model.sample_competitions(500_000, np.random.default_rng(7))
    assert result.mean() == pytest.approx(50.5, abs=0.2)


def test_sample_competitions_handles_the_single_competition_case():
    """The 'one' case: n=1 is valid and shaped (1, 3), not squeezed away."""
    result = model.sample_competitions(1, np.random.default_rng(0))
    assert result.shape == (1, model.TERRITORIES_PER_COMPETITION)


@pytest.mark.parametrize("bad_count", [0, -1, -100])
def test_sample_competitions_rejects_nonpositive_counts(bad_count):
    """The zero/negative edge: must raise, not silently return an empty array."""
    with pytest.raises(ValueError):
        model.sample_competitions(bad_count, np.random.default_rng(0))


@pytest.mark.parametrize("bad_rng", [0, 12345, "seed", None, np.random.RandomState(0)])
def test_sample_competitions_rejects_non_generator_rng(bad_rng):
    """Guard against passing a bare seed / legacy RandomState instead of a Generator."""
    with pytest.raises(TypeError):
        model.sample_competitions(10, bad_rng)
