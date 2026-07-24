# Does Evolution Favor True Perceptions? — Hoffman, Singh & Mark (2013)

*Proc. SPIE 8651, Human Vision and Electronic Imaging XVIII. Paper: `papers/hoffman_singh_2013.pdf`*

Study notes — the model, the methodology, and the conclusion.

---

## The big question
Does natural selection favor **veridical** (truthful) perception? Most vision scientists
assume yes (the **True-Vision Argument**: truer perceptions are fitter, so evolution shapes
vision toward truth). This paper tests that assumption formally — and finds it fails.

**Core distinction: fitness ≠ truth.**
- **Truth** = how accurately perception mirrors the world (P is an isomorphism/homomorphism).
- **Fitness** = the reproductive payoff of the *behavior* a perception guides.
- A trait is selected for *fitness*, not accuracy. They coincide only by accident.
- Analogy: the peacock's tail is costly for survival yet selected — proof that
  "selected" ≠ "accurate" ≠ "optimal-for-survival."

---

## The model: perceptual strategies as mappings

- **W** = set of objective world-states. **X** = set of possible perceptions.
- **Key move:** do NOT assume W = X. A **perceptual strategy** is a mapping **P: W → X**.
- Perception typically *compresses* a rich W into a small X (many-to-one) → information is
  lost; you can't recover the world-state from the perception.

### The realist → interface ladder (weakening the truthfulness requirement)
| Rung | Strategy | Requirement | Meaning |
|------|----------|-------------|---------|
| 1 | **Naïve Realist** | W = X, P isomorphism | Perception *is* reality, exactly. |
| 2 | **Strong Critical Realist** | X ⊂ W, isomorphism onto X | Perceive a faithful *slice* of reality itself. |
| 3 | **Weak Critical Realist** | P is a **homomorphism** | Different medium, but **preserves structure/order** (e.g. temp→color, hotter always redder). *Most vision scientists live here.* |
| 4 | **Interface Strategy** | just a well-defined mapping (Markov kernel) | Preserves **no** structure of W. Like a **desktop icon**: linked to the file but shares none of its real properties. |

- **Homomorphism (rung 3)** preserves *ordering*: hotter → redder, no exceptions.
- It **breaks (→ rung 4)** when the mapping is **non-monotonic**: e.g. 20°C→red AND 80°C→red,
  but 50°C→green. "Red" then tells you nothing about the true magnitude.

### Interface Theory of Perception
Evolution builds perception like a **desktop interface**: cheap, species-specific symbols
that *hide* reality (voltages, circuits) and guide useful action. Perception is a
**species-specific desktop, not a window onto reality**. Not true — but useful.

### The PDA loop (formal scaffold)
Organism = a cycle of three mappings:
- **P** (Perception): W → X
- **D** (Decision): X → G   *(G = set of possible actions)*
- **A** (Action): G → W   *(acting changes the world → loop closes back to W)*

All three are **Markov kernels** — generalizations of functions where each input maps to a
*probability distribution* over outputs (captures noisy perception). "Markovian" = memoryless
(next state depends only on current state); in info-theory terms, a **channel**.
Kernels **compose like matrices** (PD, AP, DAP…), so the whole W→X→G→W loop is one object
that evolution tunes **together**. A deterministic organism (like Robby) is the
**dispersion-free** special case (all probability on one outcome).

---

## The methodology: genetic algorithms (evolving Robby)

A **genetic algorithm (GA)** = simulated evolution: random population → score by fitness →
fittest breed (crossover + mutation) → repeat for many generations → good strategies emerge
without being designed.

### Mitchell's original "Robby the robot"
- **W:** 10×10 grid, walls around edge, soda cans randomly placed.
- **X:** sees 5 squares (self + N/S/E/W), each ∈ {empty, can, wall} → 3⁵ = **243** perceptual states.
- **G:** 7 actions (move N/S/E/W, move random, stay, pick-up-can).
- **Genome:** a lookup table (243 perceptions → 1 action each) → **7²⁴³** possible strategies
  (astronomically large — impossible to hand-design; hence use evolution as the search).
- **Loop:** 200 random Robbies → score by avg foraging points → breed (crossover + mutation)
  → 1000 generations. Early gens forage terribly; final gens are excellent.
- **Limitation:** perception is *fixed and truthful* — tests evolving behavior, not perception.

### Hoffman et al.'s modification — evolving perception itself
- Replace cans with **water**, quantity **0–10**.
- **Non-monotonic fitness:** points = (0,1,3,6,9,**10**,9,6,3,1,0). 5 units = ideal;
  0 and 10 = death. (Too little → thirst; too much → drowning.)
- Robby does **not** see quantity — he sees only **two colors (red/green)**, and *which
  quantities look red vs. green is encoded in his genes and left to evolve.*

**Result (after ~500 generations):**
- Quantities **0, 1, 9, 10 → one color; middle → the other.**
- Color encodes **fitness ("good/bad to drink"), NOT magnitude.** Opposite extremes of
  reality (empty vs. drowning) collapse to the **same percept** — same payoff → same perception.
- This is a **strict interface strategy** — perception tuned to fitness, not truth.
- *Which* color meant "good" was **genetic drift** — arbitrary (≈half the runs each way).
- **Second experiment:** charge an energy cost per viewing-direction → Robby **evolves to
  see less** as seeing gets more expensive. Truth economized away.

---

## The conclusion & when realism survives

- In their simulations, **truthful (realist) perception goes extinct**, out-competed by cheap
  fitness-tuned interface strategies.
- **When does the realist survive?** Only when **fitness is monotonic** in the world-quantity
  (more of the real thing always ≥ as good). Then tracking truth = tracking fitness, so
  truth costs nothing extra.
- **Why that's rare:** almost every real resource has a *sweet-spot* (salt, water, oxygen,
  heat, sunlight, food) — deadly at both extremes → **non-monotonic**. So the realist's
  safe haven almost never exists in real environments.

### "Isn't this just a rigged toy world?" — the two-part reply
1. The feature that makes false perception win is **non-monotonic fitness**.
2. That feature is **not an artifact** — it reflects real environments (resource sweet-spots).

**Real-world evidence (perception firing on cheap fitness-triggers, not truth):**
- **Jewel beetle** (*Julodimorpha bakewelli*): males perceive "bumpy, glossy, brown" as *mate*;
  beer bottles are a **supernormal stimulus**, so males mate with bottles instead of females.
- **Brood parasitism** (cowbird): host feeds an oversized foreign chick preferentially — a
  supernormal "feed-me" trigger exploited across 200+ host species.
- **Frog:** "does not detect flies — it detects small, moving black spots of about the right size."

**Bottom line:** *Perception is tuned to fitness, not truth.* An organism can act adaptively in
a world it does not perceive veridically — exactly as a computer user works effectively while
ignorant of the machine's true nature.
