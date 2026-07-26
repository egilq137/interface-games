"""
Chunk 8b: Euler stepping - integrate the replicator velocity into a trajectory.
Model + intuition: notes/euler-stepping.md

Turn the instantaneous velocity (replicator.replicator_velocity) into an actual path
across the simplex over generations. This is what Build B animates: pick a starting mix,
watch it slide toward the winning corner along an S-curve.

One step = Euler move + two clean-ups (euler_step owns all three):
    x = x + dx * time_step     # straight-line move (velocity assumed frozen for the step)
    x = max(x, 0)              # clamp: a too-big step can push a share slightly negative
    x = x / x.sum()            # renormalize: force the three back onto the simplex (sum = 1)

Attractor detection + the ~4.27% bifurcation are the NEXT slice.

Tests: sim/tests/test_stepper.py
"""
from __future__ import annotations

import numpy as np

from array_types import PopulationShares, SimplexTrajectory
from payoff_matrix import PayoffMatrix
from replicator import replicator_velocity


def euler_step(population_shares: PopulationShares, matrix: PayoffMatrix, time_step: float) -> PopulationShares:
    """Advance the population one Euler step, returning a valid on-simplex state.

    Moves along the replicator velocity, then clamps any (overshoot) negative share to 0 and
    renormalizes so the shares still sum to 1. A negative share pre-clamp means `time_step` was
    too large for that region (the true dynamics never leave the simplex).
    """
    shares = np.asarray(population_shares, dtype=float)
    stepped = shares + replicator_velocity(shares, matrix) * time_step
    stepped = np.maximum(stepped, 0.0)
    return stepped / stepped.sum()


def trajectory(
    initial_shares: PopulationShares,
    matrix: PayoffMatrix,
    time_step: float,
    generations: int,
) -> SimplexTrajectory:
    """The population's path from `initial_shares` over `generations` Euler steps.

    Returns shape (generations + 1, strategies): row 0 is the (normalized) start, and each later
    row is one `euler_step` on from the previous. A fixed step count - stopping early at
    convergence belongs with the attractor slice.
    """
    start = np.asarray(initial_shares, dtype=float)
    states = np.empty((generations + 1, start.size))
    states[0] = start / start.sum()
    for generation in range(1, generations + 1):
        states[generation] = euler_step(states[generation - 1], matrix, time_step)
    return states
