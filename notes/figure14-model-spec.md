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
- Chunk 5: **general boundary optimizer** (solo foraging) — search boundary positions maximizing each
  strategy's expected payoff; anchor-check CR3 ≈ 30/70 and IF3's shape against Fig. 13. Freeze into ledger.
- Chunk 6: assemble full 3×3 matrix (with faster-first turn order) → validate vs Table 3.
- Chunk 7: add cost layer (Eq. 20) → check Tables 4/5 + bifurcation (~4.27%).

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

## Mechanism #1b — turn order  ✅ locked (⚙️ falsifiable against oracle)

The paper states only two concrete cases:
- **Unequal perceptual speed → the faster (cheaper) perceiver chooses first, deterministically**
  ("seeing more data takes more time"; simple always precedes truth).
- **Equal speed → coin flip**; expected payoff is the average of choosing-first and choosing-second
  (simple vs simple, Eq. 11).

The general "priority settled probabilistically" is never expanded into a formula. Speed = fewer
perceptual categories. For our three strategies: **Truth** is slowest; **CR3** and **IF3** tie (both 3 categories).

**Working rule:** *faster-first, tie → coin-flip.*

| pairing        | who chooses first |
|----------------|-------------------|
| Truth vs CR3   | CR3 (faster)      |
| Truth vs IF3   | IF3 (faster)      |
| CR3 vs IF3     | coin flip (tie)   |
| any self-pair  | coin flip         |

⚙️ First suspect if Step 5 misses Table 3; alternatives to test then = always-coin-flip, or cost-weighted priority.

*(Paper: lines ~166–169, 242, 254–256.)*

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

## Mechanism #4 — cost layer (Eq. 20)  ⏳ deferred to Step 6

Already reverse-engineered from Tables 3–5 (cost subtracted per focal strategy: truth ≈ 16.5× CR3/IF3).
Bolt on after the cost-0 matrix validates against Table 3.

---

## Parameter ledger

| symbol | meaning | value | source | status |
|--------|---------|-------|--------|--------|
| $t$ | territories per competition | 3 | paper | ✅ |
| $r$ | resources per territory | 1 | paper (Fig. 14) | ✅ |
| $m$ | max resource quantity | 100 | paper | ✅ |
| — | quantity distribution | Uniform{1..100}, iid | paper | ✅ |
| turn order | who picks first | faster-first, tie→coin-flip | paper (partial) + ⚙️ | 🔶 falsifiable |
| $\mu$ | utility mean | 50 | paper caption | ✅ |
| $\sigma$ | utility std dev | 20 | paper caption | ✅ |
| $A$ | utility amplitude | 100 | ⚙️ working assumption | 🔶 oracle to confirm |
| decision order | which band an agent prefers | utility-tuned (prefer peak band) | paper (Sec. 8, Fig. 13) | ✅ |
| CR3 boundaries | 2 cut-points | optimize ourselves (~30/70 expected) | our search; anchored to Fig. 13 | 🔶 to compute |
| IF3 boundaries | 4 cut-points | optimize ourselves | our search; anchored to Fig. 13 | 🔶 to compute |
| optimize against | reference opponent for boundary search | TBD (solo / opponent / mixed) | ⚙️ | ⬜ open (Step 2) |
| cost layer | Eq. 20 | see Step 6 | paper, reverse-checked | ⏳ deferred |

**Oracle (validation target) — Table 3, cost = 0** (rows = focal, cols = opponent):

| focal \ opp | CR3 | IF3 | Truth |
|-------------|-----|-----|-------|
| CR3   | 60.15 | 59.05 | 58.11 |
| IF3   | 63.19 | 61.77 | 60.83 |
| Truth | 65.72 | 64.46 | 63.43 |
