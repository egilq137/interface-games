// Tests for web/geometry.js. Run:  cd web && node --test
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  barycentricToPixel,
  pixelToBarycentric,
  isInsideTriangle,
  clampToSimplex,
} from "./geometry.js";

// Canonical triangle: Truth top, CR3 bottom-left, IF3 bottom-right.
const TRIANGLE = [
  [200, 20],   // Truth
  [20, 380],   // CR3
  [380, 380],  // IF3
];
const CENTROID = [(200 + 20 + 380) / 3, (20 + 380 + 380) / 3];

function assertClose(actual, expected, message, epsilon = 1e-9) {
  actual.forEach((value, k) =>
    assert.ok(Math.abs(value - expected[k]) < epsilon, `${message}: [${actual}] vs [${expected}]`),
  );
}

// ---- barycentricToPixel ----

test("each pure corner maps to its own pixel", () => {
  assertClose(barycentricToPixel([1, 0, 0], TRIANGLE), TRIANGLE[0], "Truth corner");
  assertClose(barycentricToPixel([0, 1, 0], TRIANGLE), TRIANGLE[1], "CR3 corner");
  assertClose(barycentricToPixel([0, 0, 1], TRIANGLE), TRIANGLE[2], "IF3 corner");
});

test("the even mix maps to the centroid", () => {
  assertClose(barycentricToPixel([1 / 3, 1 / 3, 1 / 3], TRIANGLE), CENTROID, "centroid");
});

test("an edge midpoint sits halfway between two corners", () => {
  // 50/50 Truth-IF3, no CR3 -> midpoint of the Truth-IF3 edge.
  const expected = [(200 + 380) / 2, (20 + 380) / 2];
  assertClose(barycentricToPixel([0.5, 0, 0.5], TRIANGLE), expected, "Truth-IF3 midpoint");
});

// ---- pixelToBarycentric (inverse) ----

test("pixelToBarycentric inverts each corner to its share vector", () => {
  assertClose(pixelToBarycentric(TRIANGLE[0], TRIANGLE), [1, 0, 0], "Truth");
  assertClose(pixelToBarycentric(TRIANGLE[1], TRIANGLE), [0, 1, 0], "CR3");
  assertClose(pixelToBarycentric(TRIANGLE[2], TRIANGLE), [0, 0, 1], "IF3");
});

test("recovered shares always sum to 1", () => {
  for (const point of [[123, 210], [300, 100], [50, 370], [999, -40]]) {
    const shares = pixelToBarycentric(point, TRIANGLE);
    assert.ok(Math.abs(shares[0] + shares[1] + shares[2] - 1) < 1e-9, `sum for ${point}`);
  }
});

test("round trip shares -> pixel -> shares is the identity", () => {
  for (const shares of [[0.5, 0.3, 0.2], [0.1, 0.1, 0.8], [0.25, 0.7, 0.05]]) {
    const recovered = pixelToBarycentric(barycentricToPixel(shares, TRIANGLE), TRIANGLE);
    assertClose(recovered, shares, "round trip", 1e-9);
  }
});

// ---- isInsideTriangle ----

test("centroid and corners count as inside; a far point is outside", () => {
  assert.equal(isInsideTriangle(CENTROID, TRIANGLE), true);
  assert.equal(isInsideTriangle(TRIANGLE[0], TRIANGLE), true);   // corner (boundary)
  assert.equal(isInsideTriangle([0, 0], TRIANGLE), false);        // above the top edge
  assert.equal(isInsideTriangle([200, 500], TRIANGLE), false);    // below the base
});

// ---- clampToSimplex ----

test("clampToSimplex leaves a valid mix unchanged", () => {
  assertClose(clampToSimplex([0.5, 0.3, 0.2]), [0.5, 0.3, 0.2], "already valid");
});

test("clampToSimplex snaps a negative share to a valid population", () => {
  const snapped = clampToSimplex([1.2, -0.3, 0.1]);   // outside-the-triangle click
  assert.ok(snapped.every((s) => s >= 0), "no negatives");
  assert.ok(Math.abs(snapped[0] + snapped[1] + snapped[2] - 1) < 1e-9, "sums to 1");
  assert.equal(snapped[1], 0, "the negative share becomes 0");
});
