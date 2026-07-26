// Tests for web/model.js - checks the JS math core matches the validated Python (sim/).
// Run:  cd web && node --test
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  BASE_MATRIX,
  perceptionCostInBits,
  costAdjustedMatrix,
  replicatorVelocity,
  eulerStep,
} from "./model.js";

function close(actual, expected, epsilon, message) {
  assert.ok(Math.abs(actual - expected) < epsilon, `${message}: ${actual} vs ${expected}`);
}

// ---- cost layer ----

test("perceptionCostInBits matches Eq. 20 (Truth ~86.37, CR3/IF3 ~5.23)", () => {
  close(perceptionCostInBits(100, 100), 86.37, 0.01, "Truth");
  close(perceptionCostInBits(3, 3), 5.23, 0.01, "CR3/IF3");
});

test("costAdjustedMatrix reproduces the paper's Table 5 (10%) from Table 3", () => {
  const table5 = costAdjustedMatrix(BASE_MATRIX, 10);
  // Truth row, paper Table 5: 56.97, 59.27, 58.01
  close(table5[0][0], 56.97, 0.02, "Truth vs Truth");
  close(table5[0][1], 59.27, 0.02, "Truth vs CR3");
  close(table5[0][2], 58.01, 0.02, "Truth vs IF3");
  // CR3 row: 57.73, 59.76, 58.66
  close(table5[1][0], 57.73, 0.02, "CR3 vs Truth");
  close(table5[1][2], 58.66, 0.02, "CR3 vs IF3");
});

test("zero cost leaves the matrix unchanged", () => {
  const same = costAdjustedMatrix(BASE_MATRIX, 0);
  same.forEach((row, i) => row.forEach((v, j) => close(v, BASE_MATRIX[i][j], 1e-9, `cell ${i},${j}`)));
});

// ---- replicator dynamics ----

test("replicator velocity components sum to zero (conservation)", () => {
  for (const shares of [[0.5, 0.3, 0.2], [0.2, 0.2, 0.6], [0.8, 0.1, 0.1]]) {
    const v = replicatorVelocity(shares, BASE_MATRIX);
    close(v[0] + v[1] + v[2], 0, 1e-12, `sum for ${shares}`);
  }
});

test("an extinct strategy has zero velocity (sticky extinction)", () => {
  const v = replicatorVelocity([0.6, 0.4, 0.0], BASE_MATRIX);
  close(v[2], 0, 1e-12, "IF3 velocity");
});

test("eulerStep returns a valid simplex point (non-negative, sums to 1)", () => {
  const next = eulerStep([0.5, 0.3, 0.2], BASE_MATRIX, 0.5);
  assert.ok(next.every((x) => x >= 0), "non-negative");
  close(next[0] + next[1] + next[2], 1, 1e-12, "sum");
});

// ---- end-to-end: the cost flip ----

function settle(shares, truthCostPercent, steps = 3000) {
  const matrix = costAdjustedMatrix(BASE_MATRIX, truthCostPercent);
  let state = shares.slice();
  for (let s = 0; s < steps; s++) state = eulerStep(state, matrix, 0.1);
  return state;
}

test("low cost flows to Truth, high cost flows to IF3", () => {
  const low = settle([1 / 3, 1 / 3, 1 / 3], 1);
  const high = settle([1 / 3, 1 / 3, 1 / 3], 10);
  assert.ok(low[0] > 0.99, `Truth wins at 1% (got ${low})`);
  assert.ok(high[2] > 0.99, `IF3 wins at 10% (got ${high})`);
});
