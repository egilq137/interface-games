# interface-games

Reproducing **Figure 14** of Mark, Marion & Hoffman (2010), *Natural selection and
veridical perceptions* (J. Theor. Biol. 266, 504–515): the replicator-dynamics flow
over a triangle (simplex) of three perceptual strategies — **truth**, **CR3**, and
**IF3** — and how a perceptual-**cost** dial moves the population from "truth wins" to
"interface (IF3) wins", the bifurcation where realist perception goes extinct.

> **Status: work in progress.** We are currently *deriving the payoffs* (see below).
> The interactive simplex visualizations described in `spec.txt` are not built yet.

## Approach

The payoffs that drive Figure 14 are **not** given as closed-form formulas in the paper
for the three-strategy game — the paper obtains them by Monte-Carlo simulation and
publishes the resulting payoff matrices (Appendix B, Table 3). We reproduce that
derivation ourselves ("Route B"): rebuild the model — Gaussian utility, three
territories, the turn-taking foraging game, and each strategy's perception — and
compute the 3×3 payoff matrix by simulation, **validated against the paper's Table 3**
as an oracle.

Every model parameter is treated as an explicit, named assumption with a validation
status; the full walkthrough and parameter ledger live in
[`notes/figure14-model-spec.md`](notes/figure14-model-spec.md).

## Where we are (built in small, test-first chunks)

- **The world** — Gaussian utility + environment sampler (`sim/model.py`).
- **The game engine + Truth** — turn-taking competition, oracle-validated
  (Truth-vs-Truth simulates to 63.45 vs. the paper's 63.43).
- **CR3** — a critical realist: 2 boundaries → 3 contiguous categories.
- **IF3** — an interface strategy: 4 boundaries → 3 non-contiguous labels
  (both extremes folded together; the peak kept separate).

- **Boundary optimizer + full payoff matrix** — derived boundaries reproduce the paper's
  cost-0 Table 3 (`sim/boundary_search.py`, `sim/payoff_matrix.py`).
- **Perceptual-cost layer** — Eq. 20 subtracted per strategy; reproduces the paper's
  cost-1% and cost-10% matrices, Tables 4 & 5 (`sim/cost_layer.py`).
- **Replicator equation** — the change-per-generation velocity `dx_i = x_i(f_i − f̄)`
  over the simplex, conserved and extinction-sticky (`sim/replicator.py`).
- **Euler stepping** — integrate the velocity into a trajectory across the simplex
  (S-curve takeover, clamp + renormalize each step), for animated paths (`sim/stepper.py`).
- **Bifurcation search** — detect which corner the flow settles on (`attractor`) and find the
  cost band where Truth's win flips to IF3's; reproduces the paper's ~4.27% (`sim/bifurcation.py`).
- **Interactive Figure 14** — a vanilla-JS/Canvas app (`web/index.html`) combining both builds
  from `spec.txt` on one simplex: the **flow field** of arrows (Build A) with a ring on the
  attractor, and a seed-and-**play trajectory** (Build B) riding the flow. A cost slider (0–100%)
  and presets (1% truth / 4.36% coexistence / 10% IF3) morph the field live; the arrows toggle
  off for a clean trajectory, with an optional shimmer that animates the flow direction. Math
  core (`web/model.js`, `web/geometry.js`) ports the validated
  `sim/` and is Node-tested; the cost flip (Truth → IF3) and the coexistence band are both live.

The whole paper is now reproduced end to end — from the Monte-Carlo payoff derivation to an
interactive Figure 14.

## Project layout

- `sim/` — the derivation code.
  - `model.py` — the world: Gaussian utility and the competition sampler.
  - `competition.py` — the turn-taking game plus the perceptual strategies.
  - `array_types.py` — semantic array type aliases (shapes documented on hover).
  - `tests/` — the pytest suite (131 tests, every function checked independently).
- `web/` — the interactive builds (vanilla JS + Canvas). `model.js`/`geometry.js` mirror the
  validated `sim/` math core, with Node tests (`cd web && node --test`); `index.html` is the
  Figure-14 app (flow field + trajectory); `geometry-check.html` is a visual geometry sanity page.
  Serve locally to open (`python -m http.server` from `web/`, then browse to the root), since
  ES-module imports need `http://`, not `file://`.
- `notes/` — model-spec walkthrough and reading notes on the source papers.
- `spec.txt` — spec for the eventual interactive Figure-14 visualizations.
- `papers/` — source PDFs (local only; git-ignored, as they are copyrighted).

## Running the tests

```
pip install numpy pytest      # matplotlib is optional, only for the utility-curve figure
python -m pytest sim/tests -q
```

## Reference

Mark, J. T., Marion, B. B., & Hoffman, D. D. (2010). Natural selection and veridical
perceptions. *Journal of Theoretical Biology*, 266(4), 504–515.
