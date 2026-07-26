# Cost layer (Eq. 20) — how perception is priced

Learning notes for Chunk 7. Plain-text math (no LaTeX). Cost is the dial that turns
the cost-0 payoff matrix (Table 3) into Figure 14's "truth wins → IF3 wins" story.

## 1. The unit: bits

- A **bit** = one yes/no distinction. Telling apart N possibilities costs `log2(N)` bits.
- Why `log2` and not N: each yes/no question **halves** what's left (N → N/2 → N/4 → 1).
  It's literally **binary search**. `log2(N)` = how many times you halve N to reach 1.
- General rule for future modeling: when a cost grows by *repeated division into b parts*,
  reach for `log_b`. Base = how many answers each question has (yes/no = base 2).
- Careful: `log` is "of what's left," not "of the original." Each cut discards a fraction
  of the *remaining* pool, not the starting pool.

## 2. Every strategy pays for TWO things

```
cost = ce*t*r*log2(q)   +   ck*r*q*nb
       \___ seeing ___/      \__ knowing __/
```

- **seeing** = classify a territory into one of `q` categories. A *search* → scales with `log2(q)`.
- **knowing** = store the utility of *each* category. A *lookup table* → scales **linearly with q**.
- Symbols (Fig. 14 values): `t`=territories=3, `r`=resources/terr=1, `ce`=cost/bit (the knob),
  `ck`=ce/10 (knowing is cheaper per bit), `q`=# categories, `nb`=bits to store one utility number.
- **Key asymmetry:** `seeing` carries a `t`, `knowing` does NOT — you learn the utility table
  once and reuse it across all territories. So more territories tilt the balance toward seeing,
  and make truth relatively more expensive (the paper exploits this: 30 territories → IF3 wins sooner).

## 3. Plugging in the numbers (cost in units of ce)

| strategy | q | nb = log2(utility levels) | seeing = 3*log2(q) | knowing = 0.1*q*nb | TOTAL |
|----------|---|---------------------------|--------------------|--------------------|-------|
| Truth    | 100 | log2(100) ≈ 6.6         | 19.9               | 66.4               | ≈ 86 * ce |
| CR3      | 3   | log2(3) ≈ 1.6           | 4.8                | 0.48               | ≈ 5.3 * ce |
| IF3      | 3   | log2(3) ≈ 1.6           | 4.8                | 0.48               | ≈ 5.3 * ce |

- Truth costs **~16× more** than CR3/IF3 (86 / 5.3).
- **Why knowing dominates truth:** it's not the rate (knowing is 10× *cheaper* per bit) — it's the
  bit COUNT. Truth has ~664 knowing-bits vs ~20 seeing-bits (33×). 33× bits at 1/10 rate → ~3.3× cost.

## 4. THE punchline for this project

- **CR3 and IF3 get the identical cost** (same q=3, same nb). So cost can NEVER separate them.
- Therefore any CR3-vs-IF3 difference lives entirely in the **cost-0 raw payoffs** (Table 3).
- IF3 beats CR3 not by seeing *more* (same 3 categories) but by **placing** those categories
  smarter: it drops the fixed order (non-homomorphic) so it can bracket the utility peak.
  CR3 [31,71] = 3 contiguous bands; IF3 [22,36,64,78] = folds both extremes, isolates the peak.

## 5. Applying cost = net fitness per strategy

- Cost is a property of the **strategy** (its own perception), not of the opponent.
- So subtract each strategy's one cost number from that strategy's **entire row** of the matrix:
  `net[i][j] = raw[i][j] - C_i`. Net fitness = raw payoff − cost. That's the whole cost layer.

## 6. The dial: % of truth's payoff, not ce

- `ce` alone is meaningless ("is ce=0.03 big?"). So the paper reports the dial as
  **truth's cost as a percentage of truth's raw payoff**.
- Truth's raw payoff = average of truth's row at cost 0 = mean(65.72, 64.46, 63.43) ≈ **64.5**.
  (Reference is truth because the paper's whole question is "does perceiving reality win?"
  Truth's payoff ≈ the ceiling of raw benefit, so % = "how much of best-case benefit cost eats.")
- Convert: `X%` → `C_truth = X/100 * 64.5`. E.g. 10% → 6.45 points; 4.27% → ~2.75 points.
  Back out the knob if needed: `ce = C_truth / 86`. CR3/IF3 cost = (5.3/86) * C_truth ≈ 6% of it.
- **Bifurcation:** IF3 dominates for cost > ~4.25%, truth for < ~4.29%, coexist between (~4.27%).
  CR3 goes extinct whenever IF3 is present, at any cost.

## Chunk 7 to-do (implied by these notes)
- Function: strategy cost in ce-units via Eq. 20 (truth ≈ 86, CR3 ≈ IF3 ≈ 5.3).
- Dial in %: given X%, compute C_truth from truth's mean raw payoff, derive ce, get all three costs.
- Apply: subtract C_i from row i of the base matrix → cost-adjusted matrix.
- Validate: reproduce Tables 4/5 and the ~4.27% bifurcation.
