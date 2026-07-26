# Finding the bifurcation cost (~4.27%)

Learning notes for the bifurcation sub-chunk. Builds on [[euler-stepping]] (trajectories),
[[replicator-equation]] (velocity), [[cost-layer-eq20]] (the cost parameter).
Goal: pin the cost where the replicator flow's winner flips from Truth to IF3.

## The method (three layers)

1. **Winner check at a fixed cost** — run a trajectory from a central start (e.g. 1/3,1/3,1/3)
   for enough generations, then read the corner it converged to = the share sitting at ~1.
   Three verdicts, not two (see coexistence below): Truth, IF3, or coexistence.
2. **Coarse sweep** — evaluate the winner across a grid of costs (e.g. 0%..10%). This does two
   jobs: it CONFIRMS the winner flips exactly once (monotone), and it BRACKETS the flip to an
   interval. Without it, we'd be assuming monotonicity blindly.
3. **Binary search inside the bracket** — halve the interval repeatedly (winner below vs above the
   midpoint) to pin the threshold fast. Valid ONLY because the sweep confirmed a single flip.

## Why the sweep is not optional

Binary search finds a boundary only if the yes/no answer is **monotone** (one flip). Nothing in the
replicator equation guarantees that a priori - it's an assumption we must VERIFY. We *expect* it here
because as cost rises, truth's cost climbs ~16x faster than IF3's, so truth only ever loses ground to
IF3 - a one-way tilt. The coarse sweep turns "expect" into "checked."

## It's a coexistence BAND, not a knife-edge

A single winner is not guaranteed: replicator dynamics can settle to a mix - an interior equilibrium
(all coexist) or an edge one (two survive). This game has exactly that near the transition:
- **below ~4.25%** -> Truth wins outright;
- **~4.25-4.29%** -> Truth and IF3 **coexist** (neither reaches its corner) - the thin band;
- **above ~4.29%** -> IF3 wins outright.
So the transition has two edges (a band), which is why the winner check needs a third "coexistence"
verdict. (Separately, CR3 goes extinct whenever IF3 is present, at any cost - that part is clean.)

Practical: declare a winner only if some share crosses a threshold (e.g. > 0.99) after enough
generations; otherwise it's coexistence. Near the band, convergence slows, so allow enough generations.

## Scope note
This slice = detect the winner + find the band edges. The interactive Build A/B come after.
