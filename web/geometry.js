// Simplex geometry for the interactive Figure-14 builds.
//
// The population lives on a 2-simplex (a triangle): three shares (Truth, CR3, IF3) that are
// each >= 0 and sum to 1. This module converts between those shares (barycentric coordinates)
// and pixel positions on the canvas, both directions:
//   - barycentricToPixel: shares -> (x, y)   [for drawing a population as a point/arrow]
//   - pixelToBarycentric: (x, y) -> shares    [for reading a mouse click as a population]
//
// A `triangle` is the three corner pixels in canonical strategy order [Truth, CR3, IF3],
// e.g. [[200, 20], [20, 380], [380, 380]] (Truth top, CR3 bottom-left, IF3 bottom-right).
// A `point` is [x, y]; `shares` is [truth, cr3, if3].

/**
 * Pixel position of a population mix: the corners weighted by their shares.
 * @param {number[]} shares - [truth, cr3, if3] (need not be normalized; weights are used as given)
 * @param {number[][]} triangle - corner pixels [Truth, CR3, IF3]
 * @returns {number[]} [x, y]
 */
export function barycentricToPixel(shares, triangle) {
  const [t, c, i] = shares;
  const [[tx, ty], [cx, cy], [ix, iy]] = triangle;
  return [t * tx + c * cx + i * ix, t * ty + c * cy + i * iy];
}

/**
 * Shares corresponding to a pixel position (the inverse of barycentricToPixel).
 * Returns [truth, cr3, if3] summing to 1. Values may be negative if the point is outside
 * the triangle - use isInsideTriangle to test, or clampToSimplex to snap onto it.
 * @param {number[]} point - [x, y]
 * @param {number[][]} triangle - corner pixels [Truth, CR3, IF3]
 * @returns {number[]} [truth, cr3, if3]
 */
export function pixelToBarycentric(point, triangle) {
  const [px, py] = point;
  const [[tx, ty], [cx, cy], [ix, iy]] = triangle;
  // Solve the 2x2 system for the CR3 and IF3 weights relative to the Truth corner; Truth = rest.
  const v0x = cx - tx, v0y = cy - ty;   // Truth -> CR3
  const v1x = ix - tx, v1y = iy - ty;   // Truth -> IF3
  const v2x = px - tx, v2y = py - ty;   // Truth -> point
  const denominator = v0x * v1y - v1x * v0y;
  const cr3 = (v2x * v1y - v1x * v2y) / denominator;
  const if3 = (v0x * v2y - v2x * v0y) / denominator;
  const truth = 1 - cr3 - if3;
  return [truth, cr3, if3];
}

/**
 * Is a pixel inside (or on the edge of) the triangle? True iff every share is >= -epsilon.
 * @param {number[]} point - [x, y]
 * @param {number[][]} triangle - corner pixels [Truth, CR3, IF3]
 * @param {number} [epsilon] - tolerance so exact edge/corner points count as inside
 * @returns {boolean}
 */
export function isInsideTriangle(point, triangle, epsilon = 1e-9) {
  return pixelToBarycentric(point, triangle).every((share) => share >= -epsilon);
}

/**
 * Snap any share vector onto the simplex: clamp negatives to 0, then renormalize to sum 1.
 * Maps an outside-the-triangle click to a valid population (same rule euler_step uses).
 * @param {number[]} shares - [truth, cr3, if3]
 * @returns {number[]} valid shares, each >= 0, summing to 1
 */
export function clampToSimplex(shares) {
  const clamped = shares.map((share) => Math.max(share, 0));
  const total = clamped[0] + clamped[1] + clamped[2];
  return clamped.map((share) => share / total);
}
