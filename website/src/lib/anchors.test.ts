import { test } from "node:test";
import assert from "node:assert/strict";
import { headingHref, headingPath, parseHeading } from "./anchors.ts";

test("a heading link keeps the hash route and names the heading", () => {
  assert.equal(headingHref("choose-a-model", "naming"), "#/docs/choose-a-model?h=naming");
  assert.equal(headingPath("choose-a-model", "naming"), "/docs/choose-a-model?h=naming");
});

test("the heading is read back from the route's query", () => {
  assert.equal(parseHeading("?h=naming"), "naming");
  assert.equal(parseHeading("?x=1&h=v_call"), "v_call");
  assert.equal(parseHeading(""), null);
});
