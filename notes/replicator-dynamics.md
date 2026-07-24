# Replicator Dynamics (engine behind Fig. 14, Mark, Marion & Hoffman 2010)

## The one equation
$$\dot x_i = x_i\,(f_i - \bar f)$$
- $x_i$ = share (fraction) of strategy $i$ in the population.
- $f_i$ = payoff/fitness of strategy $i$.
- $\bar f = \sum_j x_j f_j$ = population-average payoff.
- **Plain words:** a strategy's share grows in proportion to how far its payoff *beats the average*.

## Why "beats the AVERAGE", not just $f_i$
- Subtracting $\bar f$ enforces **conservation**: winners' gains exactly cancel losers' losses, so shares always re-sum to 1.
- Raw $f_i$ would inflate the total past 100%.
- A strategy earning exactly the average → share unchanged.

## The $x_i$ out front = extinction is sticky
- A strategy at 0% share stays at 0% (0 × anything = 0).
- **Gotcha for the sim:** replicator dynamics *cannot resurrect or invent* a strategy that starts at zero share. It only reshuffles what's already present. Seed all strategies at >0 if you want them able to win.

## Dynamics shape → arrow thickness
- Takeover is **logistic / S-curve**: slow (small share) → fast (mid) → slow (lead over average vanishes).
- **Arrow thickness/length ∝ $|f_i - \bar f|$** (speed of flow).
  - Near a corner: average is dragged toward the dominant strategy → gaps shrink → **thin** arrows (corners are attractors/repellers, flow decelerates).
  - Interior (big payoff gaps): **thick** arrows.

## The state space = simplex (triangle)
- 3 strategies, shares sum to 1 → **2 degrees of freedom** → 2D triangle.
- **Corner** = pure population (100% one strategy: truth / CR3 / IF3).
- **Edge** = only two strategies present (third extinct).
- **Interior** = all three coexist.
- A population state = one point; replicator dynamics makes it drift → that drift field = Fig. 14's arrows.

## Cost = the dial
- Perceptual precision costs bits; cost is subtracted from payoff. Truth = most expensive (resolves everything), IF3 = cheap (lumps states).
- Turning cost reshuffles the $f_i$ → reshuffles $(f_i - \bar f)$ → re-aims every arrow.
- **Low cost:** truth has top payoff → arrows converge on the **truth** corner (attractor).
- **High cost:** IF3 wins → arrows converge on the **IF3** corner.

## Bifurcation = "realist perception goes extinct"
- There's a **critical cost** where the flow reorganizes: truth corner flips from attractor → repeller, IF3 corner becomes the attractor.
- Not gradual mush — a tipping point. This is the literal moment the paper's headline claim becomes a fact about arrow direction.

See also: [[mark-2010-interface-games]], [[hoffman-2013-interface-perception]]
