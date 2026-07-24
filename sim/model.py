"""
Step 2 derivation - Fig. 14 payoffs (Route B, our own Monte-Carlo).
Model spec + parameter ledger: notes/figure14-model-spec.md

CHUNK 1: the world the agents live in - nothing here depends on strategies yet.

Two ideas:
  1. `utility(resource_value)`  - how much a resource amount is *worth* (a bell curve
                                  peaked at 50; extremes are near-worthless).
  2. `sample_competitions(...)` - roll the dice to generate many competitions, each
                                  being a set of territories with random resource values.

Resource values are drawn FLAT (every value equally likely) but their worth is PEAKED.
That tension - a near-50 territory is a lucky find - is the engine of Section 8.

Tests live in sim/tests/test_model.py.
"""
from __future__ import annotations

import numpy as np

from array_types import ResourceValueGrid, ResourceValues, UtilityValues

# --- Locked model parameters (see the parameter ledger in the spec note) ---
# A territory's resource value is an integer in {1, ..., MAX_RESOURCE_VALUE}.
MAX_RESOURCE_VALUE = 100                 # paper: resource scale 1..100
TERRITORIES_PER_COMPETITION = 3          # paper: Fig. 14 uses 3 territories

# The utility (worth) of a resource value is a Gaussian bump:
#   utility(v) = PEAK_UTILITY * exp( -(v - PEAK_RESOURCE_VALUE)^2 / (2 * UTILITY_SPREAD^2) )
PEAK_RESOURCE_VALUE = 50.0               # paper caption: Gaussian mean (where worth peaks)
UTILITY_SPREAD = 20.0                    # paper caption: Gaussian standard deviation
PEAK_UTILITY = 100.0                     # amplitude A -- WORKING ASSUMPTION (oracle to confirm);
                                         # equals the worth at the very peak (resource value == 50).


def utility(resource_value: ResourceValues) -> UtilityValues:
    """Worth of a resource value - a Gaussian peaked at PEAK_RESOURCE_VALUE.

    Deterministic in the resource value and fully vectorized: accepts a scalar or
    any array shape and returns the same shape (hence the flexible ResourceValues
    type, not a fixed grid/vector). A value of 50 is worth the most; both very low
    and very high values are worth almost nothing.

    Example:
        utility(50)            -> 100.0
        utility([20, 55, 90])  -> [32.5, 96.9, 13.5]   (90 is big but nearly worthless)
    """
    resource_value = np.asarray(resource_value, dtype=float)
    squared_distance_from_peak = (resource_value - PEAK_RESOURCE_VALUE) ** 2
    return PEAK_UTILITY * np.exp(-squared_distance_from_peak / (2.0 * UTILITY_SPREAD ** 2))


def sample_competitions(number_of_competitions: int, rng: np.random.Generator) -> ResourceValueGrid:
    """Generate random competitions, each a fresh set of territories.

    Returns an integer array of shape
        (number_of_competitions, TERRITORIES_PER_COMPETITION)
    where every entry is a resource value drawn uniformly from {1, ..., MAX_RESOURCE_VALUE},
    independently. One row = the territories on offer in one competition.

    `rng` is passed in (not created here) so that callers control the seed and
    results are reproducible.
    """
    if number_of_competitions < 1:
        raise ValueError(f"number_of_competitions must be >= 1, got {number_of_competitions}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator, e.g. np.random.default_rng(seed)")

    lowest_value, one_past_highest_value = 1, MAX_RESOURCE_VALUE + 1  # integers() is half-open on the high end
    return rng.integers(
        lowest_value,
        one_past_highest_value,
        size=(number_of_competitions, TERRITORIES_PER_COMPETITION),
    )
