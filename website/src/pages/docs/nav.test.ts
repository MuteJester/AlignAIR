import { test } from "node:test";
import assert from "node:assert/strict";
import { USER_SECTIONS, DEVELOPER_SECTIONS, audienceOf, sectionsFor, adjacent } from "./nav.ts";

const pages = [
  { slug: "architecture", title: "Architecture", section: "Developers" },
  { slug: "cli", title: "CLI options", section: "Reference" },
  { slug: "install", title: "Install", section: "Get started" },
  { slug: "first-alignment", title: "Your first alignment", section: "Get started" },
  { slug: "cite", title: "Cite", section: "Project" },
  { slug: "testing", title: "Testing", section: "Developers" },
];

test("the user sections come in the reader's order", () => {
  assert.deepEqual(USER_SECTIONS, ["Get started", "Coming from…", "Use on your data", "Understand the results",
    "Reference", "Train or fine-tune", "Project"]);
  assert.deepEqual(sectionsFor(pages, "user").map((s) => s.title), ["Get started", "Reference", "Project"]);
  assert.deepEqual(sectionsFor(pages, "developer").map((s) => s.title), DEVELOPER_SECTIONS);
});

test("every page belongs to its section's audience", () => {
  assert.equal(audienceOf("Developers"), "developer");
  assert.equal(audienceOf("Get started"), "user");
  assert.ok(sectionsFor(pages, "user").every((s) => s.pages.every((p) => audienceOf(p.section) === "user")));
});

test("Next never walks from the user pages into the developer pages, nor back", () => {
  assert.equal(adjacent(pages, "install").next?.slug, "first-alignment");
  assert.equal(adjacent(pages, "first-alignment").next?.slug, "cli");
  assert.equal(adjacent(pages, "cite").next, undefined);
  assert.equal(adjacent(pages, "architecture").prev, undefined);
  assert.equal(adjacent(pages, "architecture").next?.slug, "testing");
});

test("a section the order does not know still shows, after the known ones", () => {
  const extra = [...pages, { slug: "old", title: "Old page", section: "Using AlignAIR" }];
  assert.deepEqual(sectionsFor(extra, "user").map((s) => s.title), ["Get started", "Reference", "Project", "Using AlignAIR"]);
});
