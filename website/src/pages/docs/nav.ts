/**
 * The docs' navigation: which sections a reader sees, in what order, and where Previous/Next lead. Pure (no React),
 * so `npm test` checks it. User pages and developer pages are two separate journeys: the sidebar shows the current
 * one, and Previous/Next never cross from one to the other.
 */
export const USER_SECTIONS = ["Get started", "Coming from…", "Use on your data", "Understand the results",
  "Reference", "Train or fine-tune", "Project"];
export const DEVELOPER_SECTIONS = ["Developers"];

export type Audience = "user" | "developer";

export interface NavPage {
  slug: string;
  title: string;
  section: string;
}

export function audienceOf(section: string): Audience {
  return DEVELOPER_SECTIONS.includes(section) ? "developer" : "user";
}

/** The audience's sections in reading order (a section the order does not name comes after, as first seen). */
export function sectionsFor<T extends NavPage>(pages: T[], audience: Audience): { title: string; pages: T[] }[] {
  const order = audience === "user" ? USER_SECTIONS : DEVELOPER_SECTIONS;
  const titles: string[] = [];
  for (const page of pages) {
    if (audienceOf(page.section) === audience && !titles.includes(page.section)) titles.push(page.section);
  }
  const rank = (title: string) => (order.includes(title) ? order.indexOf(title) : order.length + titles.indexOf(title));
  return titles
    .sort((a, b) => rank(a) - rank(b))
    .map((title) => ({ title, pages: pages.filter((page) => page.section === title) }));
}

/** The pages before and after `slug` in its audience's reading order. */
export function adjacent<T extends NavPage>(pages: T[], slug: string): { prev?: T; next?: T } {
  const current = pages.find((page) => page.slug === slug);
  if (!current) return {};
  const ordered = sectionsFor(pages, audienceOf(current.section)).flatMap((section) => section.pages);
  const i = ordered.findIndex((page) => page.slug === slug);
  return { prev: i > 0 ? ordered[i - 1] : undefined, next: i + 1 < ordered.length ? ordered[i + 1] : undefined };
}

/**
 * Docs slugs from before the restructure, and the page that replaced each, so old links and bookmarks still arrive
 * somewhere useful. A slug that is live again must leave this table (test_links checks both sides).
 */
export const MOVED: Record<string, string> = {
  "getting-started": "first-alignment", "worked-example": "first-alignment", "migrating-from-igblast": "from-igblast",
  models: "choose-a-model", integrations: "from-immcantation", "airr-fields": "column-dictionary",
  errors: "error-codes", "model-contract": "trust-the-calls", "known-failure-modes": "trust-the-calls",
  troubleshooting: "fix-common-problems", training: "train", "training-workflow": "train", benchmarks: "comparison",
  performance: "run-at-scale", "citation-support": "cite", design: "architecture", "model-composition": "architecture",
  "model-setup": "architecture", "curriculum-sources": "architecture", "training-tasks": "architecture",
  "training-batches": "architecture", "training-engine": "architecture", "model-inference": "architecture",
};

// The retired Learn tracks, each to the page that now teaches it.
const LEARN_TRACKS: Record<string, string> = {
  foundations: "concepts", predicting: "first-alignment", integrating: "from-immcantation", training: "train",
  benchmarking: "comparison",
};

/** Where an address from before the restructure now leads (`/docs/<slug>` or `/docs`), or undefined if it is live. */
export function movedTo(path: string): string | undefined {
  const [, area, first] = path.split("/");
  if (area === "learn") return first && first in LEARN_TRACKS ? `/docs/${LEARN_TRACKS[first]}` : "/docs";
  if (area === "docs" && first && first in MOVED) return `/docs/${MOVED[first]}`;
  return undefined;
}
