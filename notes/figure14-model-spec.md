# Fig. 14 Model Spec — deriving the payoffs (Route B: our own Monte-Carlo)

*Living document. We are reconstructing the game mechanics of Mark, Marion & Hoffman (2010),
Sec. 8.2 / Fig. 14, so we can compute the 3×3 payoff matrix ourselves and validate it against
the paper's published Table 3.*

Source paper: `papers/mark_hoffman_2010.pdf`.
See also: [[mark-2010-interface-games]], [[replicator-dynamics]].

> **Method note (our philosophy):** every constant is a *named* parameter with a stated source
> and a validation status. Working assumptions are marked ⚙️ and must be confirmed by the oracle
> (the paper's Table 3). Nothing is a black box.

---

## Status

**Step 1 — reconstruct the model mechanics. ✅ CLOSED.** All mechanisms locked: competition
protocol (#1), turn order (#1b), utility (#2), perceptual maps + decision rules (#3). Cost (#4)
deferred to Step 6 with a known answer. Boundary *values* are outputs of Step 2.

**Step 2 — derive CR3/IF3 boundaries. ← CURRENT.** Built in chunks (Gaussian μ=50 σ=20, 3 terr, 1 res):
- Chunk 1 ✅ world: Gaussian utility + environment sampler (`sim/model.py`).
- Chunk 2 ✅ game engine + Truth: turn-taking competition, oracle-validated (Truth-vs-Truth = 63.45 vs 63.43).
- Chunk 3 ✅ CR3: general `categorize` (B boundaries → B+1 bands) + preference-by-band-expected-utility.
- Chunk 4 ← IF3: same machinery + a **zone→label map** for non-contiguous labels (tails/shoulders/peak).
  CR3 = identity map; IF3 = folded map [0,1,2,1,0]. (Reordered before the optimizer so it covers both.)
- Chunk 5 ✅ boundary optimizer (solo-foraging grid search): CR3 → [31,71] (Fig. 13 anchor confirmed),
  IF3 → [22,36,64,78]. With derived boundaries + coin-flip order, the full cost-0 3×3 matrix reproduces
  **Table 3 to max |diff| 0.22, mean 0.06** — the whole Route-B reconstruction is validated end to end.
- Chunk 6 ✅ formalized the cost-0 matrix in `sim/payoff_matrix.py` (`base_payoff_matrix`, frozen
  boundaries + coin-flip order, `PayoffMatrix` dataclass). Test reproduces Table 3 to ±0.5.
- Chunk 7 ✅ cost layer (Eq. 20) in `sim/cost_layer.py`: `cost_adjusted_matrix` reproduces the paper's
  Tables 4 (1%) and 5 (10%) from Table 3 (±0.02). Bifurcation ~4.27% moved to Chunk 8 (needs replicator
  dynamics). Goal + intuition: Mechanism #4 below, [[cost-layer-eq20]].

Open knobs to settle at the top of Step 2:
- **Reference for optimization** ("optimal against whom"): solo foraging / specific opponent / mixed field.
- **IF3 structure:** adopt symmetric red=tails, yellow=shoulders, green=peak (3 labels, 5 zones)?
  Enforce symmetry in the search or let all 4 float?
- **Decision order:** derive by expected-utility-per-band (fixed) vs. jointly optimize?
- **Tooling:** derivation is offline (one-time) → Python+numpy prototype, OR JS to share with the app.
  (The live app only needs the final matrix, not the Monte-Carlo — so language is free here.)

Remaining plan: Step 3 lock oracle + tolerance → Step 4 full Monte-Carlo matrix → Step 5 validate
vs Table 3 → Step 6 add cost layer (Eq. 20), check Tables 4/5 + bifurcation (~4.27%).

---

## Mechanism #1 — the pairwise competition protocol  ✅ locked

A single competition (one entry sampled for the payoff matrix) works like this:

1. Draw **3 territories**; each holds one resource quantity $v \sim \text{Uniform}\{1,\dots,100\}$,
   independent, drawn fresh each competition.
2. Two agents are paired. **One chooses first** (see Mechanism #1b): it perceives all 3 territories
   through its perceptual map, picks the best available *by its own decision rule*, and **claims it**.
3. That agent's payoff = the **true utility** of the territory it claimed.
4. The second agent picks between the **2 remaining** territories, and gets that territory's true utility.
5. **Cost** (perceptual bits, Eq. 20) is subtracted from each agent's utility. *(Added in Step 6.)*

The payoff-matrix entry $P_{ij}$ = expected fitness of strategy $i$ paired against strategy $j$,
averaged over random environments **and** over turn order.

*(Paper: Sec. 3–4, lines ~129–173.)*

---

## Mechanism #1b — turn order  ✅ locked: **coin-flip every pairing** (oracle-validated)

The paper states two concrete cases:
- **2-strategy simple-vs-truth game:** the faster (cheaper) perceiver chooses first, deterministically
  ("seeing more data takes more time"; simple precedes truth).
- **Equal speed → coin flip**; expected payoff = average of choosing-first and choosing-second (Eq. 11).

For the **3-strategy game (Fig. 14)** the paper only says "priority of choice is settled
**probabilistically**" — never a deterministic rule.

**We first assumed *faster-first* — the oracle falsified it.** A preliminary Table-3 recreation
(eyeballed boundaries, cost 0) showed faster-first makes **Truth earn the *least*** (it is slowest,
so always moves second), whereas the paper has Truth earn the *most* — max |diff| ≈ 14.7. Switching
to a **plain 50/50 coin flip for every pairing** dropped max |diff| to ≈ 2.8 and mean |diff| to ≈ 1.0,
matching the whole Truth row/column to < 0.8. So "probabilistic priority" = random order.

**Locked rule:** *every pairing's turn order is a 50/50 coin flip* → expected payoff to strategy i vs j
= ½·(i-moves-first payoff + i-moves-second payoff), averaged over competitions.

*(Paper: lines ~166–169, 242, 254–256. Validation: preliminary Table-3 recreation, this repo.)*

---

## Mechanism #2 — utility function  ✅ locked (amplitude ⚙️ working assumption)

Fig. 14 replaces "more is better" with a **truncated Gaussian** bump: mid-range quantity is best,
both extremes bad. The best territory is the one nearest $v=50$, **not** the one with the most resource.

$$U(v) = A \cdot \exp\!\left(-\frac{(v-\mu)^2}{2\sigma^2}\right), \qquad v \in \{1,\dots,100\}$$

- $\mu = 50$ — peak location *(paper, Fig. 14 caption)*
- $\sigma = 20$ — spread *(paper, Fig. 14 caption)*
- $A = 100$ — amplitude ⚙️ **working assumption** (paper never states it). Rationale: matches the
  resource scale (peak utility = 100), and a magnitude check lands the whole matrix in the observed
  56–66 band (single-territory mean utility $\approx A\cdot 0.50 \approx 50$; best-of-3 → low-60s,
  vs Table 3 Truth-vs-Truth = 63.43). $A$ is a pure multiplier on every entry, so the oracle can
  fix it exactly from a single Table 3 value.
- "Truncated" = only evaluated on $v\in\{1,\dots,100\}$ (a Gaussian is natively defined on
  $(-\infty,\infty)$; we chop it to the resource range). Tails ≈ 5% of peak. Deterministic in $v$.

> **Keep two things separate:** quantities $v$ are **sampled Uniform{1..100}** (flat — every
> quantity equally likely), but the **worth** of a quantity is the **Gaussian bump** (peaked at 50).
> So a territory is equally likely to hold any quantity, yet quantities near 50 are worth the most.
> That tension is the entire engine of Section 8.

Payoff collected = $U(v_{\text{chosen}}) - \text{cost}$.

*(Paper: Sec. 8.1, lines ~636–642; caption line ~716.)*

---

## Mechanism #3 — perceptual maps + decision rules  ✅ locked (boundary *values* to be optimized)

A **boundary** is a threshold on the resource quantity that turns a number into a category. It is
the only thing that defines a coarse perceiver. Truth has none; CR3 has 2; IF3 has 4.

**How a boundary enters the model (worked example).** Scene: three territories, quantities
$v=[20,55,90]$. Gaussian worth (A=100): $U(20)=32.5$, $U(55)=96.9$, $U(90)=13.5$ — so the
*best* territory is #2 ($v=55$, nearest the peak 50), even though #3 has the most resource.

- **Truth** (no boundaries): reads exact $[20,55,90]$, picks max utility → #2 → collects **96.9**.
- **CR3** (2 boundaries, e.g. at 35 & 65): sees only the *band*: $20\to$red, $55\to$yellow, $90\to$green,
  i.e. `[red, yellow, green]` — it has no idea 55 beats 90. Decision order (utility-tuned) = **yellow > green > red**,
  so it picks the yellow territory → #2 → **96.9**. Matches Truth *here*.
- **Why position matters:** scene $v=[50,40,95]$ with boundaries (35,65): both 50 and 40 fall in
  yellow → CR3 can't tell them apart → picks randomly → expected $(100+88.2)/2=94.1$, losing ~6 vs
  Truth's 100. Tighten to (45,55): now $40\to$red, so yellow=50 is unique → CR3 gets 100. Same
  organism, different cuts, different payoff. **That is the knob the optimizer turns.**

The three strategies:
- **Truth** — sees exact $v$; picks genuinely best-utility available territory. No boundaries.
- **CR3** — 2 boundaries → **3 contiguous bands** (a homomorphism: order preserved). Decision order
  tuned to utility (prefer the middle/peak band). Contiguity is forced: 2 cuts can't give low & high the same label.
- **IF3** — 4 boundaries → 5 zones but only 3 labels, with **≥1 label reused non-contiguously**
  (e.g. low tail *and* high tail both "red" — both are bad), reserving a band for the peak. This
  breaks the homomorphism and lets IF3 track utility directly. CR3 *cannot* do this.

### How the boundaries are obtained  ✅ decided: **we optimize them ourselves**

The paper never prints CR3/IF3 boundary values — it obtains them by **maximizing expected payoff**:
analytically only in the trivial 2-category linear case ($b^*=m/\sqrt3\approx57.7$ by setting the
derivative to zero); by **Monte-Carlo search over boundary positions** everywhere else, including
the Gaussian strategies of Fig. 13. So for Route B we **replicate that search**: an inner optimizer
that sweeps boundary positions and keeps the payoff-maximizing set. (Figs. 9 & 10 display *positions*
— the search's argmax; Fig. 11 is a *different* study, payoff vs. **number** of evenly-spaced bands.)

**Context that applies to us — Fig. 13's world, NOT Fig. 10's:**

| | Fig. 10 (IGNORE) | Fig. 13 / Fig. 14 (OURS) |
|---|---|---|
| utility | linear ("more=better") | **Gaussian μ=50, σ=20** |
| resources/territory | 2 | **1** |
| territories | 3 | **3** |
| decision order | = perceptual order (prefer highest) | **utility-tuned (prefer peak band)** |
| CR3 boundaries shown | ~40s / mid-70s (high) | **~early-30s / early-70s (bracket the peak)** |

⚙️ **Free anchor:** our optimizer, run in the Gaussian/3-terr/1-res context, should reproduce CR3's
Fig. 13 cuts (~30/70 ≈ peak ± 1σ). Cross-check the optimizer against Fig. 13 *before* trusting the matrix.

**Still open (small):** "optimal against whom?" — the paper defined simple's optimum as payoff
*against truth*. For the 3-strategy case we must pick a reference (solo foraging / specific opponent /
mixed field) — deferred to Step 2, treated as a falsifiable knob.

## Mechanism #4 — cost layer (Eq. 20)  ✅ CHUNK 7 (done)

Full walkthrough + intuition: [[cost-layer-eq20]]. Cost subtracts `net = raw − cost` per strategy,
where each strategy's cost is a single number applied to its whole row of the matrix.

**Eq. 20** (cost = `ce·t·r·log2(q) + ck·r·q·nb`, with `t=3`, `r=1`, `ck = ce/10`) splits into
*seeing* (classify → `log2(q)` bits) and *knowing* (store each category's utility → linear in `q`).
The `ce` unit cancels — only the ratio of strategies' costs matters. Cost is verified to reproduce
Tables 4/5 exactly (per-row constant subtraction; truth ≈ 86.37 bits-equiv, CR3 = IF3 ≈ 5.23, ratio 0.0606).
The `truth ≈ 16.5×` figure is `86.37 / 5.23` — the knowing term (q=100) is what makes truth expensive.

**The dial is a parameter, `truth_cost_percent`:** cost expressed as a percentage of truth's expected
payoff (mean of the base matrix's Truth row ≈ 64.5). Fixes truth's cost directly; CR3/IF3 scale from it
by 0.0606. The paper's Tables 4/5 are the 1% and 10% settings.

### Chunk 7 goal — implement the cost layer, validate vs Tables 4/5

Scope: cost layer only. Replicator dynamics, attractor detection, and the ~4.27% bifurcation are
**Chunk 8** (they need the shared math core, not yet built).

New files: `sim/cost_layer.py`, `sim/tests/test_cost_layer.py`.

Module constants: `TERRITORIES = 3`, `RESOURCES_PER_TERRITORY = 1`, `KNOWLEDGE_COST_RATIO = 0.1`
(= `ck/ce`), and the per-strategy `(number_of_categories, number_of_utility_values)`:
Truth `(100, 100)`, CR3 `(3, 3)`, IF3 `(3, 3)`. Categories `q` (for *seeing*) and utility resolution
(for *knowing*) are kept independent — the paper's 30-resource variant has `q=3` but `nb=log2(100)`.

Functions (each an independent unit-test target):

| function | signature | returns |
|----------|-----------|---------|
| `perception_cost_in_bits` | `(number_of_categories, number_of_utility_values) -> float` | `TERRITORIES*log2(q) + KNOWLEDGE_COST_RATIO*q*log2(utility_values)`; bits-equivalent (knowing term scaled). Truth→86.37, CR3/IF3→5.23. |
| `truth_expected_payoff` | `(base_matrix) -> float` | mean of the matrix's Truth row (our base ≈ 64.5) |
| `strategy_costs_at` | `(truth_cost_percent, base_matrix) -> dict[str, float]` | per-strategy cost in payoff-points; CR3/IF3 scaled from truth by the cost-in-bits ratio |
| `cost_adjusted_matrix` | `(base_matrix, truth_cost_percent) -> PayoffMatrix` | new matrix, `C_i` subtracted from row `i` |

Validation (oracle):
1. `perception_cost_in_bits`: Truth ≈ 86.37, CR3 = IF3 ≈ 5.23; ratio ≈ 0.0606.
2. `cost_adjusted_matrix(paper_table3, 1)` → **Table 4** (±0.02). One cell we take as `59.01` not the
   paper's printed `58.02`: cost is one number per strategy (Appendix B), so all of CR3's row must shift
   equally — its other cells drop ~0.03, so `59.05 − 0.04 = 59.01`. The printed `58.02` is inconsistent
   with the paper's own row/method; cause unknown, recorded not diagnosed.
3. `cost_adjusted_matrix(paper_table3, 10)` → **Table 5** (±0.02).
4. Properties: `truth_cost_percent=0` is a no-op; CR3 and IF3 rows shift by the *same* amount;
   every cell is monotonically non-increasing in `truth_cost_percent`.

Tables 4/5 are checked against the **paper's** Table 3 (their published cells) minus cost, so the oracle
stays exact despite our simulated base matrix differing slightly (their 63.43 vs our 63.45).

---

## Parameter ledger

| symbol | meaning | value | source | status |
|--------|---------|-------|--------|--------|
| $t$ | territories per competition | 3 | paper | ✅ |
| $r$ | resources per territory | 1 | paper (Fig. 14) | ✅ |
| $m$ | max resource quantity | 100 | paper | ✅ |
| — | quantity distribution | Uniform{1..100}, iid | paper | ✅ |
| turn order | who picks first | coin-flip 50/50, every pairing | oracle-validated vs Table 3 | ✅ (faster-first falsified) |
| $\mu$ | utility mean | 50 | paper caption | ✅ |
| $\sigma$ | utility std dev | 20 | paper caption | ✅ |
| $A$ | utility amplitude | 100 | ⚙️ working assumption | ✅ confirmed (Table 3 reproduced) |
| decision order | which band an agent prefers | utility-tuned (prefer peak band) | paper (Sec. 8, Fig. 13) | ✅ |
| CR3 boundaries | 2 cut-points | [31, 71] (≈ Fig. 13 [30,70]) | solo-foraging grid search | ✅ derived; anchor confirmed |
| IF3 boundaries | 4 cut-points | [22, 36, 64, 78] (symmetric fold) | solo-foraging grid search | ✅ derived |
| optimize against | reference for boundary search | solo foraging | our choice | ✅ (reproduces Table 3) |
| cost layer | Eq. 20 | see Step 6 | paper, reverse-checked | ⏳ deferred |

**Oracle (validation target) — Table 3, cost = 0** (rows = focal, cols = opponent):

| focal \ opp | CR3 | IF3 | Truth |
|-------------|-----|-----|-------|
| CR3   | 60.15 | 59.05 | 58.11 |
| IF3   | 63.19 | 61.77 | 60.83 |
| Truth | 65.72 | 64.46 | 63.43 |
