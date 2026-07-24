"""
Step 2 derivation - Fig. 14 payoffs (Route B, our own Monte-Carlo).
Model spec + parameter ledger: notes/figure14-model-spec.md

CHUNK 2: turn the static world (Chunk 1) into a game, and prove it on Truth.

Design idea that will carry us all the way to the 3-strategy matrix:
  A strategy assigns a PREFERENCE score to every territory. Choosing = take the
  most-preferred territory that is still AVAILABLE, breaking ties uniformly at random.
    - Truth's preference IS the utility (it wants the highest-worth territory).
    - CR3 / IF3 (later) will prefer territories by the decision-rank of their category.

The competition itself (Mechanism #1): the first mover claims its most-preferred
territory and collects that territory's true utility; the second mover then picks
from what remains.

Tests live in sim/tests/test_competition.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from array_types import (
    AvailabilityGrid,
    BandExpectedUtilities,
    Boundaries,
    CategoryValues,
    Payoffs,
    PreferenceGrid,
    ResourceValueGrid,
    ResourceValues,
    TerritoryChoices,
    ZoneLabels,
)
from model import MAX_RESOURCE_VALUE, sample_competitions, utility


@dataclass
class CompetitionOutcome:
    """The result of a batch of competitions - one entry per competition in each field.

    Named fields so callers read `.first_mover_payoff` instead of unpacking a tuple.
    """
    first_mover_choice: TerritoryChoices
    second_mover_choice: TerritoryChoices
    first_mover_payoff: Payoffs
    second_mover_payoff: Payoffs


def truth_preferences(resource_values: ResourceValueGrid) -> PreferenceGrid:
    """Truth's preference over territories: it wants the highest-utility one.

    Sees the exact resource values, so its preference score is simply each
    territory's worth. Same shape in, same shape out.

    Example (1 competition, 3 territories):
        resource_values = [[20, 55, 90]]
        returns           [[32.5, 96.9, 13.5]]   # most wants territory 1 (nearest peak 50)
    """
    return utility(resource_values)


def categorize(resource_values: ResourceValues, boundaries: Boundaries) -> CategoryValues:
    """Group each resource value into a contiguous band defined by sorted boundaries.

    B boundaries -> B+1 bands, indexed 0 (lowest) .. B (highest). A value v lands in
    band k where boundaries[k-1] < v <= boundaries[k] (open at the bottom of the range,
    matching the paper's "green if value > boundary"). Flexible in the number of
    boundaries - 2 for CR3, 4 for IF3, any number - so nothing is hardcoded to 3 bands.
    Shape-preserving (works on a scalar, a row, or a full grid).

    Example (boundaries [30, 70]):
        values [10, 30, 31, 70, 90]  ->  bands [0, 0, 1, 1, 2]
    """
    boundaries = np.asarray(boundaries, dtype=float)
    if boundaries.ndim != 1 or boundaries.size < 1:
        raise ValueError("boundaries must be a 1-D array with at least one cut-point")
    if not np.all(np.diff(boundaries) > 0):
        raise ValueError(f"boundaries must be sorted strictly ascending, got {boundaries}")
    return np.digitize(resource_values, boundaries, right=True)


def _identity_zone_labels(number_of_boundaries: int) -> ZoneLabels:
    """The contiguous labeling: every zone is its own label (what CR3 uses)."""
    return np.arange(number_of_boundaries + 1)


def expected_utility_per_label(boundaries: Boundaries, zone_labels: ZoneLabels) -> BandExpectedUtilities:
    """Average utility of a territory carrying each perceptual label.

    Values are uniform over 1..MAX_RESOURCE_VALUE, so this is the mean worth over all
    integer values whose zone maps to that label. A label may pool several NON-adjacent
    zones (IF3's trick); with the identity labeling each zone is its own label (CR3).
    Returns one number per label; a label no value can carry gets -inf. This ranks
    labels by desirability - the basis of the utility-tuned decision rule.

    Example (boundaries [20,40,60,80], zone_labels [0,1,2,1,0]): label 0 pools both
    tails, label 1 both shoulders, label 2 the peak -> label 2 is the most valuable.
    """
    zone_labels = np.asarray(zone_labels)
    number_of_zones = np.asarray(boundaries).size + 1
    if zone_labels.shape != (number_of_zones,):
        raise ValueError(f"zone_labels needs one entry per zone ({number_of_zones}), got {zone_labels.shape}")

    all_values = np.arange(1, MAX_RESOURCE_VALUE + 1)
    worth_of_value = utility(all_values)
    label_of_value = zone_labels[categorize(all_values, boundaries)]
    number_of_labels = int(zone_labels.max()) + 1

    worth_sum_per_label = np.bincount(label_of_value, weights=worth_of_value, minlength=number_of_labels)
    count_per_label = np.bincount(label_of_value, minlength=number_of_labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count_per_label > 0, worth_sum_per_label / count_per_label, -np.inf)


def band_expected_utilities(boundaries: Boundaries) -> BandExpectedUtilities:
    """Expected utility per band for the contiguous case (each band is its own label).

    The identity-labeling special case of `expected_utility_per_label` - what CR3 uses.
    Example (boundaries [30, 70]): the middle band brackets the peak, so it wins.
    """
    return expected_utility_per_label(boundaries, _identity_zone_labels(np.asarray(boundaries).size))


def labeled_preferences(
    resource_values: ResourceValueGrid, boundaries: Boundaries, zone_labels: ZoneLabels
) -> PreferenceGrid:
    """Preference of a coarse perceiver that sees only a LABEL per territory.

    Each territory is scored by its label's expected utility, so the perceiver prefers
    the most-desirable label and CANNOT tell apart two territories sharing a label
    (they tie; `most_preferred` breaks it at random). The zone->label map decides which
    zones share a label: identity for CR3, folded for IF3. General in both the number
    of boundaries and the labeling.

    Example (boundaries [20,40,60,80], zone_labels [0,1,2,1,0], resource_values [[10,90,50]]):
        10 and 90 both carry label 0 (the tails) -> equal scores; 50 carries label 2 (peak) -> highest.
    """
    worth_of_label = expected_utility_per_label(boundaries, zone_labels)  # validates zone_labels length
    label_of_territory = np.asarray(zone_labels)[categorize(resource_values, boundaries)]
    return worth_of_label[label_of_territory]


def categorical_preferences(resource_values: ResourceValueGrid, boundaries: Boundaries) -> PreferenceGrid:
    """Contiguous coarse perceiver (each band is its own label) - the identity-labeling
    special case of `labeled_preferences`. This is what CR3 uses.

    Example (boundaries [30, 70]):
        resource_values = [[38, 62, 50]]   # all in the middle band...
        returns           equal scores     # ...so they can't be told apart
    """
    return labeled_preferences(resource_values, boundaries, _identity_zone_labels(np.asarray(boundaries).size))


def cr3_preferences(resource_values: ResourceValueGrid, boundaries: Boundaries) -> PreferenceGrid:
    """CR3's preferences: a critical realist with 3 categories (2 boundaries).

    Sees red / yellow / green (low / middle / high band) and, under the Gaussian
    utility, prefers the middle band (nearest the peak). A thin wrapper over the
    general `categorical_preferences`, fixing CR3's identity as a 3-category strategy.
    """
    number_of_boundaries = np.asarray(boundaries).size
    if number_of_boundaries != 2:
        raise ValueError(
            f"CR3 is a 3-category critical realist: expected exactly 2 boundaries, got {number_of_boundaries}"
        )
    return categorical_preferences(resource_values, boundaries)


IF3_ZONE_LABELS = np.array([0, 1, 2, 1, 0])
"""IF3's zone->label fold over 5 zones (4 boundaries): both tails -> label 0 (red),
both shoulders -> label 1 (yellow), the peak -> label 2 (green). Red and yellow are
non-contiguous (the interface trick); green is the single contiguous peak band."""


def if3_preferences(resource_values: ResourceValueGrid, boundaries: Boundaries) -> PreferenceGrid:
    """IF3's preferences: an interface strategy with 3 labels over 4 boundaries (5 zones).

    Unlike CR3, IF3 folds the two extremes into one label and the two shoulders into
    another, keeping a label for the peak. It therefore treats a very-low and a
    very-high territory as the SAME (both bad) and reserves its top preference for the
    near-peak band. A thin wrapper over `labeled_preferences` with IF3's fixed fold.
    """
    number_of_boundaries = np.asarray(boundaries).size
    if number_of_boundaries != 4:
        raise ValueError(
            f"IF3 is a 3-label interface with 4 boundaries (5 zones), got {number_of_boundaries}"
        )
    return labeled_preferences(resource_values, boundaries, IF3_ZONE_LABELS)


def most_preferred(preferences: PreferenceGrid, available: AvailabilityGrid, rng: np.random.Generator) -> TerritoryChoices:
    """Index of the most-preferred AVAILABLE territory in each competition (row).

    `preferences` and `available` are both shape (competitions, territories).
    Unavailable territories can never be chosen. Ties among the top available
    territories are broken uniformly at random.

    Returns an integer array of shape (competitions,) - the chosen column per row.

    Example (1 competition, 3 territories):
        preferences = [[32.5, 96.9, 13.5]]   # territory 1 is most wanted...
        available   = [[True, False, True]]  # ...but it is already taken
        returns       [0]                    # so it falls back to territory 0 (32.5 > 13.5)
    """
    preferences = np.asarray(preferences, dtype=float)
    available = np.asarray(available, dtype=bool)
    if preferences.shape != available.shape:
        raise ValueError(f"preferences {preferences.shape} and available {available.shape} must match")
    if not available.any(axis=-1).all():
        raise ValueError("every competition must have at least one available territory")

    # Push unavailable territories below any real preference so they never win.
    effective_preference = np.where(available, preferences, -np.inf)
    top_preference = effective_preference.max(axis=-1, keepdims=True)
    is_top = effective_preference == top_preference

    # Random tie-break: give each territory a random key in (0, 1), zeroed unless
    # it is tied for the top. The tied territory with the largest key wins; any
    # non-top territory (key 0) can never beat a tied-top one (key > 0).
    tie_break_key = rng.random(effective_preference.shape) * is_top
    return tie_break_key.argmax(axis=-1)


def play_competitions(
    first_mover_preferences: PreferenceGrid,
    second_mover_preferences: PreferenceGrid,
    resource_values: ResourceValueGrid,
    rng: np.random.Generator,
) -> CompetitionOutcome:
    """Run many turn-taking competitions at once.

    The first mover (using `first_mover_preferences`) claims its most-preferred
    territory; the second mover (using `second_mover_preferences`) then picks from the
    remaining territories. Each collects the TRUE utility of the territory it took.

    `resource_values` is shape (competitions, territories). Returns a
    CompetitionOutcome whose fields are each shape (competitions,).

    Example (1 competition, Truth vs Truth, resource_values = [[20, 55, 90]]):
        first mover takes territory 1 (worth 96.9); second mover takes territory 0
        (worth 32.5, the best of what remains) -> payoffs 96.9 and 32.5.
    """
    resource_values = np.asarray(resource_values)
    if resource_values.ndim != 2:
        raise ValueError(f"resource_values must be 2-D (competitions, territories), got {resource_values.shape}")
    number_of_competitions = resource_values.shape[0]
    rows = np.arange(number_of_competitions)

    available = np.ones_like(resource_values, dtype=bool)
    first_mover_choice = most_preferred(first_mover_preferences, available, rng)

    available[rows, first_mover_choice] = False  # the first mover's territory is taken
    second_mover_choice = most_preferred(second_mover_preferences, available, rng)

    first_mover_payoff = utility(resource_values[rows, first_mover_choice])
    second_mover_payoff = utility(resource_values[rows, second_mover_choice])
    return CompetitionOutcome(
        first_mover_choice=first_mover_choice,
        second_mover_choice=second_mover_choice,
        first_mover_payoff=first_mover_payoff,
        second_mover_payoff=second_mover_payoff,
    )


def truth_vs_truth_expected_payoff(number_of_competitions: int, rng: np.random.Generator) -> float:
    """Expected payoff to a Truth agent competing against other Truth agents.

    Turn order is a coin flip (equal-speed rule), so a Truth agent is equally
    likely to play the first-mover or second-mover role. Its expected payoff is
    therefore the mean over BOTH roles, averaged over many competitions.

    This is our oracle check: it should land on Table 3's cost-0 Truth-vs-Truth
    entry, 63.43.
    """
    competitions = sample_competitions(number_of_competitions, rng)
    preferences = truth_preferences(competitions)
    outcome = play_competitions(preferences, preferences, competitions, rng)
    payoffs_from_both_roles = np.concatenate([outcome.first_mover_payoff, outcome.second_mover_payoff])
    return float(payoffs_from_both_roles.mean())
