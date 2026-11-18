import { test } from "node:test";
import assert from "node:assert/strict";
import { rank, type SearchEntry } from "./search.ts";

const index: SearchEntry[] = [
  { kind: "page", title: "Align with a donor genotype", text: "Use on your data", href: "#/docs/donor-genotype" },
  { kind: "option", title: "--genotype-out", text: "alignair predict: where --infer-genotype writes", href: "#/docs/cli?h=predict-genotype-out" },
  { kind: "option", title: "--genotype", text: "alignair predict: a donor's genotype", href: "#/docs/cli?h=predict-genotype" },
  { kind: "column", title: "v_call", text: "The V allele call", href: "#/docs/column-dictionary?h=v_call" },
  { kind: "column", title: "v_call_set", text: "The candidate set", href: "#/docs/column-dictionary?h=v_call_set" },
  { kind: "model", title: "alignair4-trb-human", text: "TRB · HUMAN_TCRB_IMGT", href: "#/docs/model-cards?h=alignair4-trb-human" },
  { kind: "error", title: "ALIGNAIR_MODEL_NOT_FOUND", text: "No such model", href: "#/docs/error-codes?h=alignair_model_not_found" },
];

test("a flag with its dashes finds that option first", () => {
  assert.equal(rank(index, "--genotype")[0].title, "--genotype");
});

test("a column, a model id and an error code are found in any case", () => {
  assert.equal(rank(index, "V_CALL")[0].title, "v_call");
  assert.equal(rank(index, "alignair4-trb")[0].title, "alignair4-trb-human");
  assert.equal(rank(index, "alignair_model_not_found")[0].title, "ALIGNAIR_MODEL_NOT_FOUND");
});

test("a word finds pages by their title and text", () => {
  assert.ok(rank(index, "donor").some((entry) => entry.kind === "page"));
});

test("a query that matches nothing, or nothing at all, returns no results", () => {
  assert.deepEqual(rank(index, "zebrafish"), []);
  assert.deepEqual(rank(index, "   "), []);
});

test("a locus code finds its model before the pages whose title merely starts with it", () => {
  const withTrain: SearchEntry[] = [
    { kind: "page", title: "Train a model for your germline set", text: "Train or fine-tune", href: "#/docs/train" },
    { kind: "model", title: "alignair4-tra-human", text: "TRA · HUMAN_TCRA_IMGT", href: "#/docs/model-cards?h=alignair4-tra-human" },
  ];
  assert.equal(rank(withTrain, "TRA")[0].title, "alignair4-tra-human");
});

test("a phrase only in a section's prose finds that section, after every title match", () => {
  const withBody: SearchEntry[] = [
    ...index,
    { kind: "heading", title: "A 10x example", text: "Join sample and cell metadata", href: "#/docs/metadata-join?h=10x",
      body: "Cell Ranger's contig file mixes the chains of a cell" },
  ];
  assert.equal(rank(withBody, "cell ranger")[0].href, "#/docs/metadata-join?h=10x");
  assert.equal(rank(withBody, "genotype").at(-1)?.kind !== "heading", true);
});
