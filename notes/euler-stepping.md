# Euler stepping — turning velocity into a trajectory

Learning notes for the Euler-integration sub-chunk (Build B's animated path).
Plain-text math. Builds on [[replicator-equation]] (the velocity dx) and [[replicator-dynamics]].
This slice = integrate dx over time to trace a population's path across the simplex.

## The idea

Velocity `dx` is just an arrow: direction + speed at the current point. To get a *journey*,
take small steps. Euler's method = assume the velocity is frozen for a tiny time slice `dt`,
move in a straight line, then recompute the arrow and repeat.

```
x_new = x_old + dx * dt          # Euler step: one straight-line move
```

Chaining these points IS the trajectory from a given starting mix. This is what Build B animates.

## One full step = update + two clean-ups (in order)

```
x = x + dx * dt        # 1. Euler move
x = max(x, 0)          # 2. clamp: no negative shares (see below)
x = x / x.sum()        # 3. renormalize: force the three back to summing to 1
```

- **Why renormalize?** In exact math `dx` sums to 0, so `x` still sums to 1 - but floating-point
  rounding leaves it at `1 +/- a hair`. Dividing by the sum cleans that drift.
- **Why clamp?** See the overshoot below - a too-big step can push a share slightly negative,
  which is physically impossible. Clamp to 0, then renormalize soaks up the rest.

## The negative-share overshoot (the Euler artifact)

In the TRUE dynamics a share can never go negative: `dx_i` has an `x_i` factor, so as a share
shrinks toward 0 its velocity shrinks too - it decelerates and only *approaches* 0, never crosses.
A negative value after a step is therefore purely an Euler artifact: the straight-line step
overshot the curve. When you see `x_i < 0`, it means **`dt` was too large** for that region.

## dt = the accuracy vs cost dial

- **Smaller dt** -> steps hug the true curve, no overshoot, but MORE steps to compute (slower).
- **Larger dt** -> fewer steps (faster), but jerky and risks overshoot.
- Universal tradeoff of any explicit integrator. Practical rule for Build B animation:
  **pick the largest dt that still looks smooth and never overshoots** - smoothness/stability is
  a ceiling on dt; compute pushes you up toward it. (Clamp+renormalize is the cheap safety net
  for the rare blip.)

## What the path looks like: the S-curve

A winning strategy's takeover is **slow -> fast -> slow** (logistic / sigmoid). Both brakes are
visible right in `dx_i = x_i * (f_i - fbar)`:
- **rare** (`x_i` tiny) -> the `x_i` factor stalls it;
- **dominant** (`fbar -> f_i`) -> the `(f_i - fbar)` factor stalls it;
- **~half** -> both terms healthy -> fastest motion across the simplex.

## Scope note
This slice = the stepper (one step + trajectory). Attractor detection and the ~4.27% bifurcation
come next; the interactive Build A/B come after that.
