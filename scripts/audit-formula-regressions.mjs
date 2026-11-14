// Deterministic arithmetic smoke checks for audited formulas in Chapters 18/21.
// This is a narrow, dependency-free transcription of the reviewed Go operations,
// not execution of the upstream simulation or reproduction of a paper experiment.
// Upstream retrieve.go blob: e228294e9b3c29911993eb0d63ea7fae5c9b45a2
import assert from "node:assert/strict";

function normalizeMinMax(values) {
  const nums = values.filter(Number.isFinite);
  if (nums.length !== values.length || nums.length === 0) {
    throw new Error("Test fixtures must contain at least one finite score");
  }
  const min = Math.min(...nums);
  const max = Math.max(...nums);
  // The upstream normalizeMap(..., 0, 1) uses 0.5 for constant inputs.
  return values.map((value) => max === min ? 0.5 : (value - min) / (max - min));
}

function valenceTerm(signedValences, negativityBias = 1.5) {
  const normalized = normalizeMinMax(signedValences.map(Math.abs));
  // The upstream condition checks the ORIGINAL signed memory valence.
  return normalized.map((value, i) =>
    signedValences[i] < 0 ? value * negativityBias : value,
  );
}

function retrievalScore({ recency, relevance, importance, valence, valenceWeight }) {
  // Here the three baseline component weights are 1; valenceWeight is the
  // configurable switch in scratch.json, distinct from negativityBias.
  return recency + relevance + importance + valence * valenceWeight;
}

function patchingRecovery({ clean, corrupt, patched, epsilon = 1e-8 }) {
  // An explicitly editorial guard for Equation (21.7), NOT in the source.
  const denominator = clean - corrupt;
  if (Math.abs(denominator) < epsilon) return null;
  return (patched - corrupt) / denominator;
}

const close = (actual, expected, message) =>
  assert.ok(Math.abs(actual - expected) < 1e-12, message + `: ${actual} != ${expected}`);

let checks = 0;

const v = valenceTerm([0, -2, 4, -8]);
for (const [index, expected] of [0, 0.375, 0.5, 1.5].entries()) {
  close(v[index], expected, "absolute-normalize-then-bias " + index);
}
checks++;

// The post-normalization negative score may exceed 1.0.
assert.ok(v[3] > 1);
checks++;

// Equal absolute valences take the special 0.5 normalization branch.
const equal = valenceTerm([-4, 4]);
close(equal[0], 0.75, "constant negative");
close(equal[1], 0.5, "constant positive");
checks++;

// With the initial config's valence_w=0, a 1.5× bias has no total effect.
const original = { recency: 0.2, relevance: 0.3, importance: 0.4 };
close(retrievalScore({ ...original, valence: equal[0], valenceWeight: 0 }),
      retrievalScore({ ...original, valence: equal[1], valenceWeight: 0 }), "disabled valence");
checks++;

// The negative weight must affect total score once valence_w=1 is enabled.
assert.ok(retrievalScore({ ...original, valence: equal[0], valenceWeight: 1 }) >
          retrievalScore({ ...original, valence: equal[1], valenceWeight: 1 }));
checks++;

// Chapter 21's denominator can be zero / nearly zero: a ratio alone is unsafe.
close(patchingRecovery({ clean: 0.9, corrupt: 0.1, patched: 0.5 }), 0.5, "normalized recovery");
assert.equal(patchingRecovery({ clean: 0.5, corrupt: 0.5, patched: 0.6 }), null);
assert.equal(patchingRecovery({ clean: 0.5000000001, corrupt: 0.5, patched: 0.6 }), null);
checks++;

console.log(`PASS ${checks} arithmetic smoke checks (Ch18 retrieval, Ch21 recovery)`);
console.log("Note: these DO NOT run original Go/Python code, LLMs, models, or simulations.");
