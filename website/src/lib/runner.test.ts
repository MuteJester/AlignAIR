import { test } from "node:test";
import assert from "node:assert/strict";

test("the site test runner runs TypeScript", () => {
  const n: number = 1;
  assert.equal(n + 1, 2);
});
