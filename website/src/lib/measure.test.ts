import { test } from "node:test";
import assert from "node:assert/strict";
import { formatMeasure, formatDifference } from "./measure.ts";

test("a rate is a percentage and its difference is in percentage points", () => {
  assert.equal(formatMeasure("v.allele_in_call_set", 0.9734), "97.3%");
  assert.equal(formatDifference("v.allele_in_call_set", 0.0123, [0.01, 0.0146]), "+1.2 pp [1.0, 1.5]");
});

test("an error measure is a mean in its own unit, not a percentage", () => {
  assert.equal(formatMeasure("junction.start_abs_err", 0.4321), "0.43");
  assert.equal(formatDifference("junction.start_abs_err", -0.25, [-0.31, -0.19]), "−0.25 [−0.31, −0.19]");
});
