"""
Tests for sim/cost_layer.py (Chunk 7: the perceptual-cost layer, Eq. 20).

The headline tests reproduce the paper's Tables 4 (cost 1%) and 5 (cost 10%) by
subtracting cost from the paper's OWN Table 3 - so the oracle is exact, independent of
the small residual between our simulated base matrix and the paper's. The rest check the
pieces: the Eq. 20 bit counts, the CR3==IF3 equality, the percent->cost conversion, and
the row-subtraction properties.

Run:  python -m pytest sim/tests -q
"""
import os
import sys

import numpy as np
import pytest

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SIM_DIR)

import cost_layer  # noqa: E402
from payoff_matrix import PayoffMatrix  # noqa: E402

STRATEGIES = ("Truth", "CR3", "IF3")


def matrix_from_cells(cells: dict) -> PayoffMatrix:
    """Build a PayoffMatrix from a {(focal, opponent): payoff} dict in canonical order."""
    values = np.array([[cells[(focal, opp)] for opp in STRATEGIES] for focal in STRATEGIES])
    return PayoffMatrix(strategies=STRATEGIES, values=values)


# Paper Appendix B, Table 3 (cost = 0) - the base we subtract cost from.
PAPER_TABLE_3 = matrix_from_cells({
    ("Truth", "Truth"): 63.43, ("Truth", "CR3"): 65.72, ("Truth", "IF3"): 64.46,
    ("CR3", "Truth"): 58.11, ("CR3", "CR3"): 60.15, ("CR3", "IF3"): 59.05,
    ("IF3", "Truth"): 60.83, ("IF3", "CR3"): 63.19, ("IF3", "IF3"): 61.77,
})

# Paper Table 4 (cost = 1%). We use 59.01 for CR3-vs-IF3, not the paper's printed 58.02:
# the paper (Appendix B) states Tables 4/5 are Table 3 minus a per-strategy cost, so all of
# CR3's row must shift by the same amount. Its other two cells drop ~0.03 (60.15->60.12,
# 58.11->58.08); a 1.03 drop on this one cell is inconsistent with the paper's own method and
# row, so we take the method-consistent value 59.05 - 0.04 = 59.01. (Cause of the printed
# 58.02 is unknown - we record the discrepancy, we don't diagnose it.)
PAPER_TABLE_4 = {
    ("Truth", "Truth"): 62.78, ("Truth", "CR3"): 65.08, ("Truth", "IF3"): 63.82,
    ("CR3", "Truth"): 58.08, ("CR3", "CR3"): 60.12, ("CR3", "IF3"): 59.01,
    ("IF3", "Truth"): 60.80, ("IF3", "CR3"): 63.15, ("IF3", "IF3"): 61.73,
}

# Paper Table 5 (cost = 10%).
PAPER_TABLE_5 = {
    ("Truth", "Truth"): 56.97, ("Truth", "CR3"): 59.27, ("Truth", "IF3"): 58.01,
    ("CR3", "Truth"): 57.73, ("CR3", "CR3"): 59.76, ("CR3", "IF3"): 58.66,
    ("IF3", "Truth"): 60.44, ("IF3", "CR3"): 62.80, ("IF3", "IF3"): 61.38,
}


# ==========================================================================
# perception_cost_in_bits(number_of_categories, number_of_utility_values)
# ==========================================================================

def test_perception_cost_in_bits_matches_eq20():
    """Eq. 20 (ce factored out): Truth ~ 86.37 bits-equiv, CR3/IF3 ~ 5.23."""
    assert cost_layer.perception_cost_in_bits(100, 100) == pytest.approx(86.37, abs=0.01)
    assert cost_layer.perception_cost_in_bits(3, 3) == pytest.approx(5.23, abs=0.01)


def test_perception_cost_splits_into_seeing_and_knowing():
    """The two Eq. 20 terms add up: 3*log2(q) seeing + 0.1*q*log2(nb) knowing (Truth)."""
    seeing = 3 * np.log2(100)                      # ~19.93
    knowing = 0.1 * 100 * np.log2(100)             # ~66.44
    assert cost_layer.perception_cost_in_bits(100, 100) == pytest.approx(seeing + knowing, abs=1e-9)


def test_cr3_and_if3_have_identical_cost():
    """Same categories + utility values -> cost cannot separate CR3 from IF3."""
    cr3 = cost_layer.perception_cost_in_bits(*cost_layer.STRATEGY_PERCEPTION["CR3"])
    if3 = cost_layer.perception_cost_in_bits(*cost_layer.STRATEGY_PERCEPTION["IF3"])
    assert cr3 == if3


def test_truth_is_about_16x_cr3():
    """Truth's ~16.5x cost is the knowing term (q=100) blowing up."""
    truth = cost_layer.perception_cost_in_bits(100, 100)
    cr3 = cost_layer.perception_cost_in_bits(3, 3)
    assert truth / cr3 == pytest.approx(16.5, abs=0.1)


# ==========================================================================
# truth_expected_payoff / strategy_costs_at
# ==========================================================================

def test_truth_expected_payoff_is_the_truth_row_mean():
    """Reference payoff = mean of Truth's row in the base matrix (~64.54 for Table 3)."""
    assert cost_layer.truth_expected_payoff(PAPER_TABLE_3) == pytest.approx(64.537, abs=0.001)


def test_strategy_costs_at_one_percent():
    """At 1%, truth's cost is 1% of its reference payoff; CR3/IF3 scale down by the bit ratio."""
    costs = cost_layer.strategy_costs_at(1, PAPER_TABLE_3)
    assert costs["Truth"] == pytest.approx(0.6454, abs=0.001)
    assert costs["CR3"] == pytest.approx(0.0391, abs=0.001)
    assert costs["CR3"] == costs["IF3"]
    assert costs["CR3"] / costs["Truth"] == pytest.approx(0.0606, abs=0.0005)


# ==========================================================================
# cost_adjusted_matrix - the headline: reproduce Tables 4 and 5
# ==========================================================================

@pytest.mark.parametrize("percent, paper_table", [(1, PAPER_TABLE_4), (10, PAPER_TABLE_5)])
def test_cost_adjusted_matrix_reproduces_paper_tables(percent, paper_table):
    """Table 3 minus cost reproduces the paper's Tables 4 (1%) and 5 (10%) to the decimal.

    Tolerance is 0.02: the paper's base cells are printed to 2 decimals, so subtracting the
    (larger) 10% cost inherits up to ~0.01 of rounding slack on top of the cost estimate.
    """
    adjusted = cost_layer.cost_adjusted_matrix(PAPER_TABLE_3, percent)
    for (focal, opponent), paper_value in paper_table.items():
        assert adjusted.payoff_of(focal, opponent) == pytest.approx(paper_value, abs=0.02), (
            f"{focal} vs {opponent} at {percent}%: got {adjusted.payoff_of(focal, opponent):.3f}, "
            f"paper {paper_value}"
        )


def test_zero_percent_is_a_no_op():
    """No cost -> the matrix is unchanged (a copy, not the same object)."""
    adjusted = cost_layer.cost_adjusted_matrix(PAPER_TABLE_3, 0)
    assert np.array_equal(adjusted.values, PAPER_TABLE_3.values)
    assert adjusted.values is not PAPER_TABLE_3.values


def test_cr3_and_if3_rows_shift_by_the_same_amount():
    """Equal cost -> the CR3 and IF3 rows drop by an identical delta from the base."""
    adjusted = cost_layer.cost_adjusted_matrix(PAPER_TABLE_3, 10)
    cr3_delta = PAPER_TABLE_3.values[STRATEGIES.index("CR3")] - adjusted.values[STRATEGIES.index("CR3")]
    if3_delta = PAPER_TABLE_3.values[STRATEGIES.index("IF3")] - adjusted.values[STRATEGIES.index("IF3")]
    assert np.allclose(cr3_delta, if3_delta)


def test_cost_is_monotonically_non_increasing_in_percent():
    """Raising the cost percentage never raises any cell's payoff."""
    low = cost_layer.cost_adjusted_matrix(PAPER_TABLE_3, 2).values
    high = cost_layer.cost_adjusted_matrix(PAPER_TABLE_3, 8).values
    assert np.all(high <= low + 1e-12)


def test_at_100_percent_truth_row_sums_to_zero():
    """At 100%, truth's cost equals the mean of its row, so the adjusted row is that row's
    deviations from its own mean - which sum to zero (some cells positive, some negative).
    """
    adjusted = cost_layer.cost_adjusted_matrix(PAPER_TABLE_3, 100)
    truth_row = adjusted.values[STRATEGIES.index("Truth")]
    assert truth_row.sum() == pytest.approx(0.0, abs=1e-9)
    assert truth_row.min() < 0 < truth_row.max()   # genuinely straddles zero, not all ~0


def test_truth_row_shifts_most():
    """Truth pays the largest cost, so its row drops more than CR3's or IF3's."""
    adjusted = cost_layer.cost_adjusted_matrix(PAPER_TABLE_3, 10)
    delta = PAPER_TABLE_3.values - adjusted.values
    truth_drop = delta[STRATEGIES.index("Truth")].mean()
    cr3_drop = delta[STRATEGIES.index("CR3")].mean()
    assert truth_drop > cr3_drop
