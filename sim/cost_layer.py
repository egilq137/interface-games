"""
Step 2 derivation - Chunk 7: the perceptual-cost layer (Eq. 20).
Model spec + intuition: notes/figure14-model-spec.md (Mechanism #4), notes/cost-layer-eq20.md

Charge each strategy for its perception and subtract that cost from the cost-0 matrix
(payoff_matrix.base_payoff_matrix, reproduces Table 3). This yields the cost-adjusted
matrices that drive Fig. 14 - validated against the paper's Tables 4 (1%) and 5 (10%).

Eq. 20 cost = ce*t*r*log2(q) + ck*r*q*nb, split into:
  - seeing  : classify a territory into one of q categories -> log2(q) bits (a search).
  - knowing : store the utility of each category            -> q * nb bits  (a lookup table).
The cost-per-bit ce cancels out (see truth_cost_percent below), so we carry cost in
bits-equivalent units: seeing bits + KNOWLEDGE_COST_RATIO * knowing bits.

Cost is a property of the strategy, not the opponent, so a strategy's single cost is
subtracted from its whole ROW of the matrix (net = raw payoff - cost).

Tests: sim/tests/test_cost_layer.py
"""
from __future__ import annotations

import numpy as np

from payoff_matrix import PayoffMatrix

# Eq. 20 environment constants (Fig. 14: Gaussian, 3 territories, 1 resource each).
TERRITORIES = 3               # t
RESOURCES_PER_TERRITORY = 1   # r
KNOWLEDGE_COST_RATIO = 0.1    # ck / ce - the paper sets knowing's per-bit cost to a tenth of seeing's.

# Per strategy: (number_of_categories q, number_of_utility_values whose log2 is nb).
# Truth resolves all 100 quantities; CR3/IF3 rank only their 3 categories. The two are kept
# independent because the paper's 30-resource variant has q=3 but nb=log2(100).
STRATEGY_PERCEPTION: dict[str, tuple[int, int]] = {
    "Truth": (100, 100),
    "CR3": (3, 3),
    "IF3": (3, 3),
}


def perception_cost_in_bits(number_of_categories: int, number_of_utility_values: int) -> float:
    """Eq. 20 cost with the cost-per-bit ce factored out (so ce cancels downstream).

    Bits-equivalent: seeing bits plus the knowing bits scaled by KNOWLEDGE_COST_RATIO.
    Only the RATIO between strategies matters, since truth_cost_percent fixes the scale.

    >>> round(perception_cost_in_bits(100, 100), 2)   # Truth
    86.37
    >>> round(perception_cost_in_bits(3, 3), 2)        # CR3 == IF3
    5.23
    """
    seeing = TERRITORIES * RESOURCES_PER_TERRITORY * np.log2(number_of_categories)
    knowing = KNOWLEDGE_COST_RATIO * RESOURCES_PER_TERRITORY * number_of_categories * np.log2(number_of_utility_values)
    return float(seeing + knowing)


def truth_expected_payoff(base_matrix: PayoffMatrix) -> float:
    """Truth's expected payoff averaged over its competitions with all three strategies.

    This is the reference the cost percentage is measured against (the mean of Truth's
    row in the cost-0 matrix; ~64.5 for our base matrix).
    """
    truth_row = base_matrix.values[base_matrix.strategies.index("Truth")]
    return float(truth_row.mean())


def strategy_costs_at(truth_cost_percent: float, base_matrix: PayoffMatrix) -> dict[str, float]:
    """Each strategy's cost in payoff-points at the given cost setting.

    `truth_cost_percent` sets truth's cost as a percentage of truth's expected payoff;
    every other strategy's cost scales from truth's by the ratio of perception costs
    (so CR3/IF3 always land at ~6% of truth's cost).
    """
    truth_cost = truth_cost_percent / 100.0 * truth_expected_payoff(base_matrix)
    bits = {name: perception_cost_in_bits(*STRATEGY_PERCEPTION[name]) for name in base_matrix.strategies}
    return {name: truth_cost * bits[name] / bits["Truth"] for name in base_matrix.strategies}


def cost_adjusted_matrix(base_matrix: PayoffMatrix, truth_cost_percent: float) -> PayoffMatrix:
    """The cost-0 matrix with each strategy's perceptual cost subtracted from its own row.

    Reproduces the paper's Tables 4 (truth_cost_percent=1) and 5 (=10) from Table 3.
    `truth_cost_percent=0` returns an unchanged copy.
    """
    costs = strategy_costs_at(truth_cost_percent, base_matrix)
    values = base_matrix.values.copy()
    for i, name in enumerate(base_matrix.strategies):
        values[i] -= costs[name]
    return PayoffMatrix(strategies=base_matrix.strategies, values=values)
