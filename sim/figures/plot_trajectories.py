"""
Generate sim/figures/trajectories.png: replicator trajectories from an even start,
at low vs high perceptual cost, to illustrate the Euler stepper (sim/stepper.py).

Same even starting mix in both panels; only the cost parameter differs - and that alone
flips the winner (Truth at low cost, IF3 at high cost). Both show the S-curve takeover.

Run:  python sim/figures/plot_trajectories.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_layer import cost_adjusted_matrix
from payoff_matrix import base_payoff_matrix
from stepper import trajectory

STRATEGIES = ("Truth", "CR3", "IF3")
COLORS = {"Truth": "#1b9e77", "CR3": "#7570b3", "IF3": "#d95f02"}

START = np.array([1 / 3, 1 / 3, 1 / 3])
TIME_STEP = 0.1
GENERATIONS = 80
SCENARIOS = {"1% cost (low)": 1, "10% cost (high)": 10}


def main() -> None:
    base = base_payoff_matrix(200_000, np.random.default_rng(5))
    paths = {
        name: trajectory(START, cost_adjusted_matrix(base, percent), TIME_STEP, GENERATIONS)
        for name, percent in SCENARIOS.items()
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, path) in zip(axes, paths.items()):
        for column, strategy in enumerate(STRATEGIES):
            ax.plot(path[:, column], label=strategy, color=COLORS[strategy], lw=2)
        ax.set_title(name)
        ax.set_xlabel("generation")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("population share")
    axes[0].legend(loc="center right")
    fig.suptitle("Replicator trajectories from an even start (Euler stepping)", fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories.png")
    fig.savefig(out_path, dpi=130)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
