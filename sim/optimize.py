"""
Step 2 derivation - Chunk 5: the boundary optimizer.
Model spec + parameter ledger: notes/figure14-model-spec.md

We find the boundary positions that maximize each strategy's SOLO foraging payoff
(the opponent-free reference we locked). Because a boundary only changes which
INTEGER resource values fall in which band when it crosses an integer, the payoff
is piecewise-constant in the boundaries - so an exact grid over integer boundaries
misses nothing. CR3 is searched unconstrained (2-D grid, small) to verify its
optimum is symmetric; IF3 is searched over the symmetric family (a full 4-D grid
would be ~96M points).

All candidates are scored on the SAME fixed batch of foraging rounds and the same
tie-break seed (common random numbers), so the comparison between candidates is
clean and the whole search is deterministic.

Tests: sim/tests/test_optimize.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from array_types import Boundaries, PreferenceGrid, ResourceValueGrid, UtilityValues
from competition import cr3_preferences, if3_preferences, most_preferred
from model import MAX_RESOURCE_VALUE, PEAK_RESOURCE_VALUE

# A strategy's preference-builder: (foraging_rounds, boundaries) -> preference per territory.
PreferenceBuilder = Callable[[ResourceValueGrid, Boundaries], PreferenceGrid]


@dataclass
class BoundarySearchResult:
    """Outcome of a boundary grid search."""
    optimal_boundaries: Boundaries          # the payoff-maximizing boundaries
    optimal_payoff: float                    # its solo foraging payoff
    candidate_boundaries: NDArray            # every candidate tried, shape (num_candidates, num_boundaries)
    candidate_payoffs: NDArray[np.floating]  # each candidate's payoff, shape (num_candidates,)


def solo_foraging_payoff(
    preferences: PreferenceGrid, round_utilities: UtilityValues, rng: np.random.Generator
) -> float:
    """Expected worth a lone forager collects.

    With no opponent, the forager simply takes its most-preferred territory (all are
    available; ties broken at random) and collects that territory's utility. Averaged
    over every round.

    `preferences` and `round_utilities` are both shape (rounds, territories);
    `round_utilities` is utility(foraging_rounds), precomputed once so the search does
    not recompute it for every candidate.
    """
    available = np.ones_like(preferences, dtype=bool)
    choice = most_preferred(preferences, available, rng)
    rows = np.arange(preferences.shape[0])
    return float(round_utilities[rows, choice].mean())


def optimize_boundaries(
    build_preferences: PreferenceBuilder,
    candidate_boundaries: NDArray,
    foraging_rounds: ResourceValueGrid,
    round_utilities: UtilityValues,
    tie_break_seed: int = 0,
) -> BoundarySearchResult:
    """Grid search: score every candidate's solo foraging payoff, keep the best.

    Each candidate is scored on the SAME `foraging_rounds` and the same tie-break seed
    (common random numbers), so payoff differences reflect the boundaries, not sampling
    noise, and the result is deterministic.
    """
    candidate_payoffs = np.empty(len(candidate_boundaries), dtype=float)
    for index, boundaries in enumerate(candidate_boundaries):
        preferences = build_preferences(foraging_rounds, boundaries)
        rng = np.random.default_rng(tie_break_seed)   # identical tie-breaks for every candidate
        candidate_payoffs[index] = solo_foraging_payoff(preferences, round_utilities, rng)

    winner = int(np.argmax(candidate_payoffs))
    return BoundarySearchResult(
        optimal_boundaries=candidate_boundaries[winner],
        optimal_payoff=float(candidate_payoffs[winner]),
        candidate_boundaries=candidate_boundaries,
        candidate_payoffs=candidate_payoffs,
    )


def cr3_candidate_boundaries() -> NDArray:
    """Every ordered pair of integer boundaries 1 <= b1 < b2 <= MAX-1 (unconstrained)."""
    highest = MAX_RESOURCE_VALUE - 1  # a boundary at MAX would leave the top band empty
    pairs = [(b1, b2) for b1 in range(1, highest + 1) for b2 in range(b1 + 1, highest + 1)]
    return np.array(pairs, dtype=float)


def if3_symmetric_candidate_boundaries() -> NDArray:
    """Integer boundaries symmetric about the peak: [peak-w2, peak-w1, peak+w1, peak+w2]
    for 1 <= w1 < w2, staying inside [1, MAX-1]. Two free half-widths (green, outer)."""
    peak = int(PEAK_RESOURCE_VALUE)
    widest = min(peak - 1, MAX_RESOURCE_VALUE - 1 - peak)  # keep all four boundaries in range
    candidates = [
        [peak - w2, peak - w1, peak + w1, peak + w2]
        for w1 in range(1, widest + 1)
        for w2 in range(w1 + 1, widest + 1)
    ]
    return np.array(candidates, dtype=float)


def optimal_cr3_boundaries(
    foraging_rounds: ResourceValueGrid, round_utilities: UtilityValues
) -> BoundarySearchResult:
    """Search CR3's 2 boundaries unconstrained (used to verify the optimum is symmetric)."""
    return optimize_boundaries(cr3_preferences, cr3_candidate_boundaries(), foraging_rounds, round_utilities)


def optimal_if3_boundaries(
    foraging_rounds: ResourceValueGrid, round_utilities: UtilityValues
) -> BoundarySearchResult:
    """Search IF3's 4 boundaries over the symmetric family (justified by CR3's symmetry)."""
    return optimize_boundaries(if3_preferences, if3_symmetric_candidate_boundaries(), foraging_rounds, round_utilities)
