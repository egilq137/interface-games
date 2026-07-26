"""
Generate sim/figures/bifurcation.png: the bifurcation diagram.

Sweep the perceptual cost and, at each cost, run the replicator flow to equilibrium and
plot where the population settles. Truth owns the population at low cost, its corner
collapses at the transition, and IF3 takes over; CR3 stays extinct throughout. The shaded
strip is the coexistence band from bifurcation.bifurcation_band. Reproduces the paper's
~4.27% transition (Mark, Marion & Hoffman 2010, Fig. 14).

Run:  python sim/figures/plot_bifurcation.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bifurcation import CENTER, bifurcation_band
from cost_layer import cost_adjusted_matrix
from payoff_matrix import base_payoff_matrix
from stepper import euler_step

STRATEGIES = ("Truth", "CR3", "IF3")
COLORS = {"Truth": "#1b9e77", "CR3": "#7570b3", "IF3": "#d95f02"}

TIME_STEP = 0.1
GENERATIONS = 4000          # long enough to reach equilibrium outside the slow-converging band
MAX_COST_PERCENT = 8.0
COST_STEP_PERCENT = 0.1


def equilibrium_shares(matrix) -> np.ndarray:
    """Run the flow from the centre to equilibrium and return the final shares."""
    state = np.asarray(CENTER, dtype=float)
    state = state / state.sum()
    for _ in range(GENERATIONS):
        state = euler_step(state, matrix, TIME_STEP)
    return state


def main() -> None:
    base = base_payoff_matrix(200_000, np.random.default_rng(5))
    lower, upper = bifurcation_band(base)

    costs = np.arange(0.0, MAX_COST_PERCENT + 1e-9, COST_STEP_PERCENT)
    equilibria = np.array([equilibrium_shares(cost_adjusted_matrix(base, c)) for c in costs])

    fig, ax = plt.subplots(figsize=(9, 5))
    for column, strategy in enumerate(STRATEGIES):
        ax.plot(costs, equilibria[:, column], color=COLORS[strategy], lw=2.5, label=strategy)
    ax.axvspan(lower, upper, color="0.5", alpha=0.18, label=f"coexistence band ({lower:.2f}–{upper:.2f}%)")

    ax.set_xlabel("perceptual cost  (% of truth's payoff)")
    ax.set_ylabel("equilibrium population share")
    ax.set_title("Bifurcation diagram: where the population settles vs. perceptual cost", fontweight="bold")
    ax.set_xlim(0, MAX_COST_PERCENT)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.annotate("Truth wins", (2.0, 0.9), color=COLORS["Truth"], fontweight="bold", ha="center")
    ax.annotate("IF3 wins", (6.3, 0.9), color=COLORS["IF3"], fontweight="bold", ha="center")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.5), framealpha=0.95)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bifurcation.png")
    fig.savefig(out_path, dpi=130)
    print(f"band = ({lower:.3f}%, {upper:.3f}%), midpoint {0.5 * (lower + upper):.3f}%")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
