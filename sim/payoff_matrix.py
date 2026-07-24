"""
Step 2 derivation - Chunk 6: the cost-0 payoff matrix.
Model spec + parameter ledger: notes/figure14-model-spec.md

Assemble the full 3x3 payoff matrix for {Truth, CR3, IF3} from our validated model -
frozen derived boundaries + coin-flip turn order - and reproduce the paper's Table 3
(Appendix B, cost = 0). Chunk 7 will subtract each strategy's perceptual cost (Eq. 20)
to get the cost-dependent matrices (Tables 4/5) that drive Fig. 14.

Tests: sim/tests/test_payoff_matrix.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from array_types import PreferenceGrid, ResourceValueGrid
from competition import cr3_preferences, if3_preferences, play_competitions, truth_preferences
from model import sample_competitions

# Canonical strategy order - matches the replicator state {Truth, CR3, IF3} in spec.txt.
STRATEGY_NAMES: tuple[str, ...] = ("Truth", "CR3", "IF3")

# Boundaries derived by sim/boundary_search.py (solo-foraging optimum), validated
# against Fig. 13 (CR3 ~ 30/70) and Table 3. Frozen so the matrix is reproducible.
CR3_BOUNDARIES = np.array([31, 71])
IF3_BOUNDARIES = np.array([22, 36, 64, 78])


@dataclass
class PayoffMatrix:
    """A payoff matrix labeled by strategy.

    `values[i, j]` = expected payoff to `strategies[i]` when paired against `strategies[j]`.
    """
    strategies: tuple[str, ...]
    values: NDArray[np.floating]

    def payoff_of(self, focal: str, opponent: str) -> float:
        """Expected payoff to `focal` when it plays `opponent` (lookup by name)."""
        return float(self.values[self.strategies.index(focal), self.strategies.index(opponent)])

    def __str__(self) -> str:
        header = "        " + "".join(f"{name:>8}" for name in self.strategies)
        rows = [
            f"  {name:<5}" + "".join(f"{value:8.2f}" for value in row)
            for name, row in zip(self.strategies, self.values)
        ]
        return "\n".join([header, *rows])


def strategy_preferences(name: str, resource_values: ResourceValueGrid) -> PreferenceGrid:
    """Preferences of the named strategy over its territories, using its frozen boundaries."""
    if name == "Truth":
        return truth_preferences(resource_values)
    if name == "CR3":
        return cr3_preferences(resource_values, CR3_BOUNDARIES)
    if name == "IF3":
        return if3_preferences(resource_values, IF3_BOUNDARIES)
    raise ValueError(f"unknown strategy {name!r}; expected one of {STRATEGY_NAMES}")


def expected_payoff(
    focal_preferences: PreferenceGrid,
    opponent_preferences: PreferenceGrid,
    resource_values: ResourceValueGrid,
    rng: np.random.Generator,
) -> float:
    """Expected payoff to the focal strategy against the opponent, over many competitions.

    Turn order is a 50/50 coin flip (Mechanism #1b, oracle-validated): the focal is
    equally likely to move first or second, so its expected payoff is the mean of the
    two roles.
    """
    focal_first = play_competitions(focal_preferences, opponent_preferences, resource_values, rng)
    focal_second = play_competitions(opponent_preferences, focal_preferences, resource_values, rng)
    return 0.5 * (float(focal_first.first_mover_payoff.mean()) + float(focal_second.second_mover_payoff.mean()))


def base_payoff_matrix(number_of_competitions: int, rng: np.random.Generator) -> PayoffMatrix:
    """The cost-0 payoff matrix for {Truth, CR3, IF3} - reproduces the paper's Table 3.

    Every cell is computed on the SAME sampled competitions (common random numbers), so
    the matrix is internally consistent and lower-variance.
    """
    competitions = sample_competitions(number_of_competitions, rng)
    preferences = {name: strategy_preferences(name, competitions) for name in STRATEGY_NAMES}

    values = np.empty((len(STRATEGY_NAMES), len(STRATEGY_NAMES)))
    for i, focal in enumerate(STRATEGY_NAMES):
        for j, opponent in enumerate(STRATEGY_NAMES):
            values[i, j] = expected_payoff(preferences[focal], preferences[opponent], competitions, rng)
    return PayoffMatrix(strategies=STRATEGY_NAMES, values=values)
