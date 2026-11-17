import { test } from "node:test";
import assert from "node:assert/strict";
import { readTab, writeTab, TAB_KEY, commandText } from "./tabs.ts";

function storage(initial: Record<string, string> = {}) {
  const data = { ...initial };
  return { getItem: (k: string) => (k in data ? data[k] : null), setItem: (k: string, v: string) => { data[k] = v; }, data };
}

test("the chosen tab is remembered across pages", () => {
  const store = storage();
  writeTab(store, "python");
  assert.equal(store.data[TAB_KEY], "python");
  assert.equal(readTab(store), "python");
  assert.equal(TAB_KEY, "alignair.codeTab");
});

test("no storage, or a value it does not know, falls back to the CLI tab", () => {
  assert.equal(readTab(null), "cli");
  assert.equal(readTab(storage({ [TAB_KEY]: "rust" })), "cli");
  const broken = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); } };
  assert.equal(readTab(broken), "cli");
  writeTab(broken, "python");
});

test("copying a transcript copies the command only: no prompt, no output, continuations kept", () => {
  assert.equal(commandText("$ alignair a \\\n  --b x\naligned 2 reads"), "alignair a \\\n  --b x");
  assert.equal(commandText("$ pip install alignair\n$ alignair doctor\nPython 3.12"), "pip install alignair\nalignair doctor");
});

test("code without prompts is copied as it is", () => {
  const python = "from alignair import Aligner\naligner = Aligner.from_pretrained(\"x\")";
  assert.equal(commandText(python), python);
});
