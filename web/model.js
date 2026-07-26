// The Figure-14 math core, ported from the validated Python (sim/). Pure functions, no DOM.
//
// The app does NOT run the Monte-Carlo derivation; it carries the finished cost-0 payoff matrix
// (the paper's Table 3) as a constant and applies the cost layer + replicator dynamics live.
// Everything is in canonical strategy order [Truth, CR3, IF3]; a matrix M has M[i][j] = payoff
// to strategy i when it plays strategy j; shares are [truth, cr3, if3] summing to 1.

export const STRATEGIES = ["Truth", "CR3", "IF3"];

// Paper Table 3 (Appendix B, cost 0) - our simulation reproduces this; see sim/payoff_matrix.py.
export const BASE_MATRIX = [
  [63.43, 65.72, 64.46], // Truth vs [Truth, CR3, IF3]
  [58.11, 60.15, 59.05], // CR3
  [60.83, 63.19, 61.77], // IF3
];

// --- cost layer (Eq. 20), mirrors sim/cost_layer.py ---

const TERRITORIES = 3;
const KNOWLEDGE_COST_RATIO = 0.1; // ck / ce
// (categories q, utility values whose log2 is nb) per strategy: Truth resolves 100, CR3/IF3 rank 3.
const PERCEPTION = [[100, 100], [3, 3], [3, 3]];

/** Eq. 20 cost with the per-bit cost factored out (bits-equivalent). Truth ~= 86.37, CR3/IF3 ~= 5.23. */
export function perceptionCostInBits(categories, utilityValues) {
  const seeing = TERRITORIES * Math.log2(categories);
  const knowing = KNOWLEDGE_COST_RATIO * categories * Math.log2(utilityValues);
  return seeing + knowing;
}

/** Truth's expected payoff averaged over its three matchups (mean of Truth's row) - the cost reference. */
export function truthExpectedPayoff(matrix) {
  const [a, b, c] = matrix[0];
  return (a + b + c) / 3;
}

/** Each strategy's cost in payoff-points at the given percent-of-truth's-payoff setting. */
export function strategyCostsAt(truthCostPercent, matrix) {
  const truthCost = (truthCostPercent / 100) * truthExpectedPayoff(matrix);
  const bits = PERCEPTION.map(([q, u]) => perceptionCostInBits(q, u));
  return bits.map((b) => (truthCost * b) / bits[0]);
}

/** The cost-0 matrix with each strategy's cost subtracted from its own row (reproduces Tables 4/5). */
export function costAdjustedMatrix(matrix, truthCostPercent) {
  const costs = strategyCostsAt(truthCostPercent, matrix);
  return matrix.map((row, i) => row.map((value) => value - costs[i]));
}

// --- replicator dynamics, mirrors sim/replicator.py and sim/stepper.py ---

/** Frequency-dependent fitness of each strategy: f = M @ shares (row . shares). */
export function strategyFitness(shares, matrix) {
  return matrix.map((row) => row[0] * shares[0] + row[1] * shares[1] + row[2] * shares[2]);
}

/** Population-average fitness: shares . fitness. */
export function meanFitness(shares, matrix) {
  const f = strategyFitness(shares, matrix);
  return shares[0] * f[0] + shares[1] * f[1] + shares[2] * f[2];
}

/** Replicator velocity dx_i = x_i (f_i - fbar); components sum to 0. */
export function replicatorVelocity(shares, matrix) {
  const f = strategyFitness(shares, matrix);
  const fbar = shares[0] * f[0] + shares[1] * f[1] + shares[2] * f[2];
  return shares.map((x, i) => x * (f[i] - fbar));
}

/** One Euler step, returning a valid on-simplex state: move, clamp negatives, renormalize. */
export function eulerStep(shares, matrix, timeStep) {
  const velocity = replicatorVelocity(shares, matrix);
  const moved = shares.map((x, i) => Math.max(x + velocity[i] * timeStep, 0));
  const total = moved[0] + moved[1] + moved[2];
  return moved.map((x) => x / total);
}
