import { test } from "node:test";
import assert from "node:assert/strict";
import { inlineParts } from "./inline.ts";

test("backtick spans of help and docstring text become code, the rest stays text", () => {
  assert.deepEqual(inlineParts("the GenotypeSet `alignair genotype` writes"), [
    { code: false, text: "the GenotypeSet " }, { code: true, text: "alignair genotype" }, { code: false, text: " writes" }]);
  assert.deepEqual(inlineParts("truncating via ``zip`` and losing reads"), [
    { code: false, text: "truncating via " }, { code: true, text: "zip" }, { code: false, text: " and losing reads" }]);
});

test("emphasis asterisks are dropped, allele names keep theirs", () => {
  assert.deepEqual(inlineParts("checked *before* the run, as IGHV1-2*02"), [
    { code: false, text: "checked before the run, as IGHV1-2*02" }]);
});
