"""
Tests for sim/competition.py (Chunk 2: the game + the Truth strategy).

Same philosophy as test_model.py: every function checked on the ordinary case and
the edges, against values computed INDEPENDENTLY of the engine (hand cases, and a
sort-based oracle for Truth's picks) - not just "the code agrees with itself".

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


def fresh_rng(seed=0):
    return np.random.default_rng(seed)


# ==========================================================================
# truth_preferences(resource_values)
# ==========================================================================

def test_truth_preferences_equal_the_utilities():
    """Truth ranks territories by their worth, so its preference == utility."""
    resource_values = np.array([[20, 50, 90], [1, 100, 50]])
    assert np.allclose(competition.truth_preferences(resource_values), model.utility(resource_values))


def test_truth_prefers_nearer_the_peak_not_the_larger_value():
    """The Section-8 twist: 50 outranks 90 even though 90 is the bigger number."""
    prefs = competition.truth_preferences(np.array([[90, 50]]))
    assert prefs[0, 1] > prefs[0, 0]


# ==========================================================================
# categorize(resource_values, boundaries)   -- general in the number of boundaries
# ==========================================================================

def test_categorize_two_boundary_hand_case():
    """Boundaries [30, 70] -> 3 bands; edges are open at the bottom, closed at the top."""
    values = np.array([[10, 30, 31, 70, 90]])
    bands = competition.categorize(values, np.array([30, 70]))
    assert bands.tolist() == [[0, 0, 1, 1, 2]]


def test_categorize_is_general_in_number_of_boundaries():
    """Four boundaries -> five bands (0..4), no assumption of 3 categories."""
    values = np.array([[10, 20, 50, 81, 100]])
    bands = competition.categorize(values, np.array([20, 40, 60, 80]))
    assert bands.tolist() == [[0, 0, 2, 4, 4]]


def test_categorize_single_boundary_is_red_green():
    """One boundary -> the original 2-category red/green split (green if value > b)."""
    bands = competition.categorize(np.array([[50, 51]]), np.array([50]))
    assert bands.tolist() == [[0, 1]]


def test_categorize_is_shape_preserving():
    """Works elementwise: scalar in -> scalar out, grid in -> same-shaped grid out."""
    assert int(competition.categorize(90, np.array([30, 70]))) == 2
    grid = np.array([[10, 90], [50, 31]])
    assert competition.categorize(grid, np.array([30, 70])).shape == grid.shape


@pytest.mark.parametrize("bad_boundaries", [[70, 30], [30, 30], []])
def test_categorize_rejects_bad_boundaries(bad_boundaries):
    """Unsorted, non-strictly-increasing, or empty boundaries are invalid."""
    with pytest.raises(ValueError):
        competition.categorize(np.array([[50]]), np.array(bad_boundaries, dtype=float))


# ==========================================================================
# band_expected_utilities(boundaries)
# ==========================================================================

def test_band_expected_utilities_match_independent_means():
    """For boundaries [50], each band's worth = mean utility over its integer values,
    computed independently by slicing the value range.
    """
    low_band_worth = model.utility(np.arange(1, 51)).mean()     # values 1..50
    high_band_worth = model.utility(np.arange(51, 101)).mean()  # values 51..100
    result = competition.band_expected_utilities(np.array([50]))
    assert result == pytest.approx([low_band_worth, high_band_worth])


def test_band_expected_utilities_middle_band_is_best_for_symmetric_cuts():
    """Boundaries [30, 70]: the middle band brackets the peak, so it is the most valuable."""
    result = competition.band_expected_utilities(np.array([30, 70]))
    assert np.argmax(result) == 1


@pytest.mark.parametrize("boundaries, expected_length", [([50], 2), ([30, 70], 3), ([20, 40, 60, 80], 5)])
def test_band_expected_utilities_length_tracks_boundary_count(boundaries, expected_length):
    """Number of bands is always (number of boundaries + 1)."""
    assert competition.band_expected_utilities(np.array(boundaries)).shape == (expected_length,)


def test_band_expected_utilities_empty_band_is_negative_infinity():
    """A band containing no integer value gets -inf, so it can never be preferred."""
    result = competition.band_expected_utilities(np.array([50.4, 50.6]))  # middle band has no integer
    assert np.isneginf(result[1])


# ==========================================================================
# categorical_preferences(...) and cr3_preferences(...)
# ==========================================================================

def test_categorical_preferences_cannot_distinguish_within_a_band():
    """Three territories all in the middle band get equal scores - the coarse-vision gotcha."""
    prefs = competition.categorical_preferences(np.array([[38, 62, 50]]), np.array([30, 70]))
    assert prefs[0, 0] == prefs[0, 1] == prefs[0, 2]


def test_categorical_preferences_prefers_the_middle_band():
    """Values [10, 50, 90] fall in low/middle/high; the middle (value 50) is preferred."""
    prefs = competition.categorical_preferences(np.array([[10, 50, 90]]), np.array([30, 70]))
    assert np.argmax(prefs, axis=1)[0] == 1


def test_categorical_preferences_contrast_with_truth():
    """The teaching point: on [38, 62, 50] Truth distinguishes (prefers 50), CR3 cannot."""
    scene = np.array([[38, 62, 50]])
    truth = competition.truth_preferences(scene)
    coarse = competition.categorical_preferences(scene, np.array([30, 70]))
    assert np.argmax(truth, axis=1)[0] == 2          # Truth singles out the value-50 territory
    assert len(np.unique(coarse[0])) == 1            # CR3 sees them all as one band


def test_categorical_preferences_is_general_with_four_boundaries():
    """Generality check: with 4 boundaries the value-50 territory (middle band) still wins."""
    prefs = competition.categorical_preferences(np.array([[50, 90, 10]]), np.array([20, 40, 60, 80]))
    assert np.argmax(prefs, axis=1)[0] == 0


@pytest.mark.parametrize("wrong_count_boundaries", [[50], [20, 40, 60, 80]])
def test_cr3_requires_exactly_two_boundaries(wrong_count_boundaries):
    """CR3 is defined as a 3-category strategy; anything but 2 boundaries is rejected."""
    with pytest.raises(ValueError):
        competition.cr3_preferences(np.array([[10, 50, 90]]), np.array(wrong_count_boundaries, dtype=float))


def test_cr3_preferences_delegate_to_categorical_for_two_boundaries():
    """With 2 boundaries, cr3_preferences matches the general categorical_preferences."""
    scene = np.array([[10, 50, 90], [38, 62, 50]])
    boundaries = np.array([30, 70])
    assert np.array_equal(
        competition.cr3_preferences(scene, boundaries),
        competition.categorical_preferences(scene, boundaries),
    )


def test_cr3_plugged_into_the_engine_picks_the_middle_band():
    """Integration: CR3's preferences flow through most_preferred to a sensible choice."""
    scene = np.array([[10, 50, 90]])
    prefs = competition.cr3_preferences(scene, np.array([30, 70]))
    available = np.ones_like(scene, dtype=bool)
    assert competition.most_preferred(prefs, available, fresh_rng())[0] == 1


# ==========================================================================
# IF3: label pooling (expected_utility_per_label, labeled_preferences, if3_preferences)
# ==========================================================================

IF3_BOUNDARIES = np.array([20, 40, 60, 80])   # 5 zones with the fold [0,1,2,1,0]


def test_if3_zone_labels_fold_is_non_contiguous():
    """IF3 reuses labels 0 and 1 for two separate zones each; only label 2 is contiguous."""
    assert competition.IF3_ZONE_LABELS.tolist() == [0, 1, 2, 1, 0]
    assert len(np.unique(competition.IF3_ZONE_LABELS)) == 3   # 3 labels over 5 zones


def test_expected_utility_per_label_pools_non_adjacent_zones():
    """Each label's worth = mean utility over ALL its (possibly non-adjacent) values,
    computed independently by gathering the value ranges of the pooled zones.
    """
    tails = np.concatenate([np.arange(1, 21), np.arange(81, 101)])     # zones 0 and 4 -> label 0
    shoulders = np.concatenate([np.arange(21, 41), np.arange(61, 81)]) # zones 1 and 3 -> label 1
    peak = np.arange(41, 61)                                           # zone 2 -> label 2
    expected = [model.utility(tails).mean(), model.utility(shoulders).mean(), model.utility(peak).mean()]

    result = competition.expected_utility_per_label(IF3_BOUNDARIES, competition.IF3_ZONE_LABELS)
    assert result == pytest.approx(expected)
    assert np.argmax(result) == 2   # the peak label is the most valuable


def test_expected_utility_per_label_rejects_wrong_label_count():
    """The labeling must have exactly one entry per zone (num_boundaries + 1)."""
    with pytest.raises(ValueError):
        competition.expected_utility_per_label(IF3_BOUNDARIES, np.array([0, 1, 2]))  # 3 labels, 5 zones


def test_if3_lumps_both_extremes_together():
    """The interface trick: a very-low and a very-high territory look identical to IF3."""
    prefs = competition.if3_preferences(np.array([[10, 90, 50]]), IF3_BOUNDARIES)
    assert prefs[0, 0] == prefs[0, 1]          # 10 (tail) and 90 (tail) share label 0 -> equal
    assert np.argmax(prefs, axis=1)[0] == 2    # the near-peak territory (50) is preferred


def test_if3_ranks_peak_over_shoulder_over_tail():
    """One territory per label: preference must order peak > shoulder > tail."""
    prefs = competition.if3_preferences(np.array([[10, 30, 50]]), IF3_BOUNDARIES)[0]  # tail, shoulder, peak
    assert prefs[2] > prefs[1] > prefs[0]


@pytest.mark.parametrize("wrong_count_boundaries", [[30, 70], [10, 30, 50, 70, 90]])
def test_if3_requires_exactly_four_boundaries(wrong_count_boundaries):
    """IF3 is defined by 4 boundaries (5 zones); anything else is rejected."""
    with pytest.raises(ValueError):
        competition.if3_preferences(np.array([[10, 50, 90]]), np.array(wrong_count_boundaries, dtype=float))


def test_labeled_preferences_identity_matches_categorical():
    """Sanity of the unification: identity labeling reproduces CR3's categorical_preferences."""
    scene = np.array([[10, 50, 90], [38, 62, 50]])
    boundaries = np.array([30, 70])
    identity = np.array([0, 1, 2])
    assert np.array_equal(
        competition.labeled_preferences(scene, boundaries, identity),
        competition.categorical_preferences(scene, boundaries),
    )


# ==========================================================================
# most_preferred(preferences, available, rng)
# ==========================================================================

def test_most_preferred_picks_the_highest_when_all_available():
    """Ordinary case: the top-scoring territory is chosen."""
    prefs = np.array([[10.0, 30.0, 20.0]])
    available = np.array([[True, True, True]])
    assert competition.most_preferred(prefs, available, fresh_rng())[0] == 1


def test_most_preferred_never_picks_an_unavailable_territory():
    """Even the highest-scoring territory is skipped if it is unavailable."""
    prefs = np.array([[100.0, 1.0, 5.0]])       # index 0 is by far the best...
    available = np.array([[False, True, True]])  # ...but it is taken
    choice = competition.most_preferred(prefs, available, fresh_rng())[0]
    assert choice == 2                            # 5 > 1 among the available two


def test_most_preferred_handles_a_single_available_territory():
    """The 'one' case: if only one territory is free, it must be chosen."""
    prefs = np.array([[9.0, 9.0, 9.0]])
    available = np.array([[False, True, False]])
    assert competition.most_preferred(prefs, available, fresh_rng())[0] == 1


def test_most_preferred_breaks_ties_uniformly():
    """Tied top territories are chosen ~50/50; a lower one is never chosen."""
    competitions = 200_000
    prefs = np.tile([5.0, 5.0, 1.0], (competitions, 1))   # indices 0 and 1 tie for top
    available = np.ones((competitions, 3), dtype=bool)
    choices = competition.most_preferred(prefs, available, fresh_rng(1))
    share_index_0 = np.mean(choices == 0)
    assert np.count_nonzero(choices == 2) == 0            # the loser (score 1) never wins
    assert share_index_0 == pytest.approx(0.5, abs=0.01)  # the two ties split evenly


def test_most_preferred_rejects_mismatched_shapes():
    """Guard: preferences and availability must describe the same grid."""
    with pytest.raises(ValueError):
        competition.most_preferred(np.array([[1.0, 2.0, 3.0]]), np.array([[True, True]]), fresh_rng())


def test_most_preferred_rejects_a_competition_with_nothing_available():
    """Guard: a row with no available territory is an impossible game state."""
    with pytest.raises(ValueError):
        competition.most_preferred(np.array([[1.0, 2.0, 3.0]]), np.array([[False, False, False]]), fresh_rng())


# ==========================================================================
# play_competitions(...)
# ==========================================================================

def test_play_competitions_truth_hand_computed_cases():
    """Two fully hand-worked Truth-vs-Truth competitions."""
    resource_values = np.array([[20, 50, 90],    # utils 32.5, 100, 13.5
                                [50, 55, 45]])   # utils 100, 96.9, 88.2
    prefs = competition.truth_preferences(resource_values)
    outcome = competition.play_competitions(prefs, prefs, resource_values, fresh_rng())
    assert list(outcome.first_mover_choice) == [1, 0]              # each grabs the peak territory
    assert list(outcome.second_mover_choice) == [0, 1]            # then the next-best remaining
    assert outcome.first_mover_payoff == pytest.approx([100.0, 100.0], abs=1e-3)
    assert outcome.second_mover_payoff == pytest.approx([32.465, 96.923], abs=1e-3)


def test_play_competitions_second_mover_gets_a_different_territory():
    """The two movers never end up on the same territory."""
    resource_values = model.sample_competitions(10_000, fresh_rng(2))
    prefs = competition.truth_preferences(resource_values)
    outcome = competition.play_competitions(prefs, prefs, resource_values, fresh_rng(3))
    assert np.all(outcome.first_mover_choice != outcome.second_mover_choice)


def test_play_competitions_truth_matches_sorted_top_two():
    """Independent oracle: for Truth-vs-Truth the two payoffs must equal the two
    highest utilities in each competition, computed by sorting - across 50k rows.
    """
    resource_values = model.sample_competitions(50_000, fresh_rng(4))
    utilities_sorted = np.sort(model.utility(resource_values), axis=1)   # ascending
    highest = utilities_sorted[:, -1]
    second_highest = utilities_sorted[:, -2]

    prefs = competition.truth_preferences(resource_values)
    outcome = competition.play_competitions(prefs, prefs, resource_values, fresh_rng(5))
    assert np.allclose(outcome.first_mover_payoff, highest)
    assert np.allclose(outcome.second_mover_payoff, second_highest)


def test_play_competitions_returns_named_fields_of_correct_shape():
    """The result is a CompetitionOutcome; every field is one value per competition."""
    resource_values = model.sample_competitions(2_000, fresh_rng(6))
    prefs = competition.truth_preferences(resource_values)
    outcome = competition.play_competitions(prefs, prefs, resource_values, fresh_rng(7))
    assert isinstance(outcome, competition.CompetitionOutcome)
    for field in (outcome.first_mover_choice, outcome.second_mover_choice,
                  outcome.first_mover_payoff, outcome.second_mover_payoff):
        assert field.shape == (2_000,)


def test_play_competitions_rejects_non_2d_resource_values():
    """Guard: the engine works on a (competitions, territories) grid, not a 1-D row."""
    with pytest.raises(ValueError):
        competition.play_competitions(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), np.array([20, 50, 90]), fresh_rng()
        )


# ==========================================================================
# truth_vs_truth_expected_payoff(...)  -- the oracle check
# ==========================================================================

def test_truth_vs_truth_equals_mean_of_top_two_utilities():
    """The expected payoff must equal the mean over competitions of (best +
    second-best)/2, computed independently by sorting on the same draws.
    """
    seed = 11
    number = 300_000
    # Independent computation on the same competitions the function will draw.
    competitions = model.sample_competitions(number, fresh_rng(seed))
    utilities_sorted = np.sort(model.utility(competitions), axis=1)
    independent_expected = (utilities_sorted[:, -1] + utilities_sorted[:, -2]).mean() / 2.0

    engine_expected = competition.truth_vs_truth_expected_payoff(number, fresh_rng(seed))
    assert engine_expected == pytest.approx(independent_expected, abs=1e-9)


def test_truth_vs_truth_hits_the_oracle_value():
    """Cross-check against the PAPER: Table 3 cost-0 Truth-vs-Truth entry = 63.43.
    Confirms protocol + coin-flip order + amplitude A=100 all at once.
    """
    value = competition.truth_vs_truth_expected_payoff(2_000_000, fresh_rng(20))
    assert value == pytest.approx(63.43, abs=0.2)
