# Replicator equation — how we compute the change per generation

Learning notes for the replicator sub-chunk (the pure velocity, Eq. 21). Plain-text math.
Companion to [[replicator-dynamics]] (geometry/intuition) and [[cost-layer-eq20]] (the matrix's cost).
This note is the *recipe we code*: turn a payoff matrix + population mix into a rate of change.

## The setup

- Population = shares `x = (x_Truth, x_CR3, x_IF3)`, all >= 0, summing to 1. One point in the simplex.
- `P` = a payoff matrix (canonical order Truth, CR3, IF3); `P[i][j]` = payoff to strategy i vs j.
  Each cell is ALREADY a Monte-Carlo average over all territory draws, so the dynamics never
  touches territories again.

## Three quantities, two dot-products

1. **fitness** of strategy i = its expected payoff against the current population.
   You do NOT re-simulate: the matrix already stores each matchup's average, so just weight
   strategy i's row by how likely each opponent is (= the shares):
   ```
   f_i = sum_j P[i][j] * x_j          # row i . x   -> f = P @ x
   ```
   This is *frequency-dependent*: who you meet depends on the mix.

2. **mean fitness** = the population average, weighting each strategy's fitness by its share:
   ```
   fbar = sum_i x_i * f_i             # x . f
   ```
   An extinct strategy has share 0, so it drops out automatically — no need to special-case
   "only active strategies."

3. **velocity** (the change) = share times how far you beat the average:
   ```
   dx_i = x_i * (f_i - fbar)          # elementwise
   ```

So the whole recipe is: `f = P @ x`,  `fbar = x . f`,  `dx = x * (f - fbar)`.

## Worked example (from the session)

x = (0.5, 0.3, 0.2). Truth row [63.4, 65.7, 64.5] -> f_Truth = 0.5*63.4+0.3*65.7+0.2*64.5 = 64.31.
With f_CR3=59.4, f_IF3=61.9: fbar = 0.5*64.31+0.3*59.4+0.2*61.9 = 62.36.
dx_Truth = 0.5*(64.31-62.36) = +0.975 -> Truth is above average, its share climbs.

## Two properties (these become tests)

- **Conservation: sum of dx = 0.** Winners' gains exactly cancel losers' losses (the `-fbar` term
  is what makes this hold). Since "in the triangle" IS the equation `sum of shares = 1`, and the sum
  changes by `sum of dx`, a total of 0 means the sum stays 1 forever -> the point never leaves the
  triangle. (Bucket image: 1 liter split across 3 buckets; if every drop leaving one enters another,
  the total is always 1 liter. Conservation of the sum = confinement to the constraint surface.)
- **Extinction is sticky: x_i = 0 => dx_i = 0.** The `x_i` out front means a strategy at 0% can never
  spontaneously appear. Together with conservation this traps the motion inside the triangle (sum=1
  holds the plane, x_i>=0 holds the edges).
- Corollary: every pure corner is a fixed point (all dx = 0).

## Where costs enter

The replicator equation has **no cost term**. Cost enters *only* through the matrix: feed it a
`cost_adjusted_matrix` (Eq. 20 already subtracted). Causal chain when the cost dial goes up:

```
cost up  ->  P entries drop (Truth's most)  ->  fitnesses f_i shift
         ->  the (f_i - fbar) gaps flip sign/size  ->  the velocity arrow re-points & rescales
```

Cost touches only the first link; everything after is the plain replicator equation. This is why the
cost layer and the dynamics are cleanly separate pieces of code.

## Scope note
This sub-chunk = the instantaneous velocity only. Turning `dx` into a trajectory over generations
(Euler step `x += dx*dt` + renormalize, for Build B) is the NEXT slice. Bifurcation + interactive
builds are later still.
