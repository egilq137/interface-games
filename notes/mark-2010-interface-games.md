# Natural Selection and Veridical Perceptions — Mark, Marion & Hoffman (2010)

*Journal of Theoretical Biology 266, 504–515. Paper: `papers/mark_hoffman_2010.pdf`*

The **formal, closed-form proof** behind the 2013 keynote's claim. Where 2013 *asserted*
interface strategies win, this paper *derives* it with probability theory, payoff matrices,
and evolutionary game theory (replicator dynamics) — no genetic algorithm needed.

> **What this paper adds over the 2013 keynote:** a rigorous mathematical formalism and
> closed-form demonstration that truth goes extinct, rather than just a GA simulation result.

---

## The model (same skeleton as 2013, made precise)
- **World:** a set of **territories**, each holding a resource (e.g. food) with quantity in
  V = {1,…,m}, m = 100. World space W = V₁ × V₂ × V₃ (for 3 territories).
- **A perceptual strategy** is still a map **g: W → X**. Strategy classes (nested subsets):
  **naive realist ⊂ critical realist ⊂ interface** (Fig. 1).
- **Two headline strategies:**
  - **truth** (naive realist): sees the *exact* quantity, always picks the best available territory.
  - **simple** (critical realist): sees only **red/green** via one threshold **β** — green if
    food > β, else red. Picks a green territory if any, else random.

### Key mapping to the genetic-algorithm papers
| Robby / water-Robby (GA) | This paper (closed-form) |
|--------------------------|--------------------------|
| Cost per **direction** Robby can see | **Cost per bit** of information, `cₑ` |
| Evolved **red/green water-boundary** | **simple's β** (the evolvable cut-point) |
| red/green/... percepts | perceptual categories (labels) |
| Non-monotonic water payoff (0,1,3,6,9,10,9,6,3,1,0) | **Gaussian (bell-curve) utility** |
| 0,1,9,10 → same color (extremes merged) | **IF3**: far-low & far-high → same label |
| GA needed (search space 7²⁴³, intractable) | Closed-form/replicator (small space, solvable) |

**What evolves — the papers invert each other:**
- Mitchell's Robby: perception **fixed & truthful**, only the **243-entry decision table** evolves.
- This paper's simple game: decision rule **fixed**, only **perception (β)** evolves → genome = 1 number.
- water-Robby & this paper's later sections (nCat, IF3): **both P and D coevolve.**
- With a 1-parameter genome (β), no GA is needed — just take the derivative. Optimal β = **m/√3 ≈ 57.7**.

---

## Cost = information, measured in bits
- **Cost = cₑ × (number of bits used).** A "bit" = one yes/no distinction needed to tell your
  states apart. To identify 1-of-N categories you need **log₂(N)** bits (this counts *labels to
  distinguish*, NOT digits to write the numeral — 4 labels = log₂4 = 2 bits).
- **simple:** 2 colors = **1 bit/territory** → 3 bits total (3 territories).
- **truth:** 1-of-100 = **log₂(100) ≈ 6.6 bits/territory** → ~20 bits total (~7× simple's cost).
- Net fitness = **expected payoff − cost**. Truth's *raw* payoff is high (usually finds the best
  territory) but its cost eats the advantage.

---

## Who wins — the payoff matrix
|         | vs Simple | vs Truth |
|---------|-----------|----------|
| **Simple** | a | b |
| **Truth**  | c | d |

- **Simple wins** if a > c AND b > d. **Truth wins** if a < c AND b < d. Else bistable/coexist.
- **Win condition in plain form (learner's inequality):** truth wins iff
  **(benefit_truth − benefit_simple) > (cost_truth − cost_simple)** — extra accuracy must
  outweigh extra cost.

### Result — the cₑ dial (Fig. 3)
- x-axis = **β** (simple's threshold, its whole genome); y-axis = **cₑ** (cost per bit).
- **cₑ small → truth wins** (accuracy edge not yet offset). **cₑ modestly larger → simple wins
  across almost the whole β range.** Truth survives only in a thin cheap-perception sliver.
- Same shape as water-Robby's 2nd experiment: charge more per direction → Robby evolves to see less.

---

## Environmental complexity does NOT rescue truth (Sec. 6–7)
- **More territories:** helps truth only up to ~**8**; beyond that truth's cost (scales with every
  territory inspected) outgrows the benefit → simple pulls ahead again.
- **More resources/territory:** brief benefit, then decline — territories become **homogeneous**
  (harder to find a standout) while truth's cost climbs linearly.
- **Correlated resources:** helps **simple** — it gets "free" info about water by only looking at food.
- **Takeaway:** complexity raises truth's *bill* faster than its *payout*; simple's bill stays flat
  (1 bit/territory). So richer real environments make the interface case **stronger**, not weaker.

---

## Non-monotonic (Gaussian) utility → interface becomes necessary (Sec. 8)
- Replace linear "more = better" with a **bell-curve utility**: mid-range quantity is best, both
  extremes bad (formal version of water-Robby's sweet-spot).
- **Contiguous** category = one unbroken chunk of the number line; **non-contiguous** = two+
  separated chunks sharing a label (e.g. red = {0–20} ∪ {80–100}).
- **1 boundary → always 2 contiguous zones** — can never make a gap; can only crudely place its
  cut just below the peak. **Non-contiguity requires MORE boundaries than labels.**
  - **CR3** (critical realist): 2 boundaries, 3 labels → each label one contiguous zone → still a
    **homomorphism** (order preserved), just coarse.
  - **IF3** (interface): **4 boundaries, 3 labels** → 5 zones, ≥1 label reused for two separate
    zones → **non-contiguous → breaks the homomorphism** → free to track *utility* directly.
    This is exactly water-Robby merging 0,1,9,10 into one color.

---

## Three-strategy proof — replicator dynamics (Sec. 8.2)
- With 3 strategies (truth, CR3, IF3) a single inequality won't do; use the **replicator equation**:
  each strategy's share grows ∝ (its fitness − population-average fitness). *Same principle as the
  GA's "fitter parents reproduce more," written as a continuous equation.*
- Visualized as flows on a **triangle (simplex)** — corners = pure populations, arrows = how the mix
  evolves (Fig. 14).
- **Results (Gaussian utility, 3 territories, 1 resource):**
  - Low cost → flows to **truth**. Cost > ~4.25% of truth's payout → flows to **IF3**.
  - **CR3 goes extinct whenever IF3 is present, at any cost** — no safe middle ground for partial truth.
  - ⚠️ **Nuance (self-corrected):** in a *truth-vs-CR3 world with no IF3*, **truth dominates CR3 for
    cost < 9.9%**, they coexist 9.3–9.9%, CR3 wins only above 9.3%. So "partial truth beats whole
    truth" is **cost-dependent**, not automatic — it's the *interface* that reliably kills both.
  - Robustness: adding territories (30) or resources (30/territory) → IF3 still drives both extinct
    at costs as low as 0.5–2.8% of truth's payout.

---

## Honest limits the authors flag (Sec. 9)
Selection *can* (not *always* does) drive truth extinct. Their sims assume: static environment,
infinite well-mixed populations, deterministic perception maps, precise category boundaries,
fixed lifetime strategy (no learning), classical (non-quantum) computation. Real biology relaxes
all of these — future work needed, especially **critical-realist vs interface** competitions.

**Bottom line:** Fitness and truth are logically distinct. Perception is shaped by natural selection
to track **utility, not reality** — and once a genuine interface strategy exists, neither whole
truth (naive realism) nor partial truth (critical realism) reliably survives.
