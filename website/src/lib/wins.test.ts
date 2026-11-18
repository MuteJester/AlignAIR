import { test } from "node:test";
import assert from "node:assert/strict";
import { winsByFamily } from "./wins.ts";

const row = (measure: string, igblast: number, alignair: number, verdict: string) =>
  ({ measure, igblast, alignair, delta: alignair - igblast, ci95: [0, 0] as [number, number], verdict });
const loci = {
  IGH: { reference: "HUMAN_IGH_OGRDB", suites: [
    { name: "S1", title: "Core repertoire", cases: 2000, rows: [row("j.top1_in_set", 0.982, 0.994, "b_better"),
      row("v.top1_in_set", 0.99, 0.99, "tie_or_inconclusive"), row("productive.correct", 0.99, 0.98, "a_better")] },
    { name: "S5", title: "V indels", cases: 2100, rows: [row("j.top1_in_set", 0.97, 0.97, "tie_or_inconclusive")] }] },
  TRB: { reference: "HUMAN_TCRB_IMGT", suites: [
    { name: "S1", title: "Core repertoire", cases: 2000, rows: [row("j.top1_in_set", 0.95, 0.99, "b_better")] }] },
};

test("each family counts AlignAIR's wins out of every comparison of its measures, with the loci that won", () => {
  const j = winsByFamily(loci).find((family) => family.label === "J allele");
  assert.ok(j);
  assert.equal(j.wins, 2);
  assert.equal(j.comparisons, 3);
  assert.deepEqual(j.loci, ["IGH", "TRB"]);
  assert.equal(j.example.locus, "TRB");                     // the largest gain shown as the example
  assert.equal(j.example.suite, "Core repertoire");
});

test("a family without a win is left out, and productivity is never listed as a win", () => {
  const labels = winsByFamily(loci).map((family) => family.label);
  assert.ok(!labels.includes("V allele"));
  assert.ok(!labels.some((label) => /productiv/i.test(label)));
});
