import type { DocPage } from "./doc-kit";
import { getStartedPages, conceptsPages } from "./pages-getstarted";
import { migratePages } from "./pages-migrate";
import { tutorialPages } from "./pages-tutorial";
import { guidesPages, trainingPages } from "./pages-guides";
import { guides2Pages } from "./pages-guides2";
import { airrFieldsPages } from "./pages-airrfields";
import { referencePages } from "./pages-reference";
import { evalPages } from "./pages-eval";
import { maintainerPages } from "./pages-maintainer";
import { communityPages } from "./pages-community";

/**
 * All reference pages in section-grouped reading order, so the sidebar order and the
 * Previous/Next flow agree. The primary journey lives in "Get started": install and predict
 * (getting-started) -> migrate an existing pipeline -> the concepts behind the output. "Using
 * AlignAIR" is everyday usage; Training and Evaluation are deliberately separate activities.
 */
export const DOCS: DocPage[] = [
  ...getStartedPages, // getting-started
  ...migratePages, // migrating-from-igblast
  ...tutorialPages, // worked-example
  ...conceptsPages, // concepts
  ...guidesPages, // models  [Using AlignAIR]
  ...guides2Pages, // cli, python-api, integrations  [Using AlignAIR]
  ...airrFieldsPages, // airr-fields  [Reference]
  ...referencePages, // model-contract, known-failure-modes, troubleshooting
  ...trainingPages, // training  [Training]
  ...evalPages, // benchmarks, performance  [Evaluation]; design  [Design]
  ...maintainerPages, // publishing  [Maintainer]
  ...communityPages, // citation-support  [Community]
];

const SECTION_ORDER = ["Get started", "Using AlignAIR", "Reference", "Training", "Evaluation", "Design", "Maintainer", "Community"];

export interface DocSection {
  title: string;
  pages: DocPage[];
}

export const DOC_SECTIONS: DocSection[] = SECTION_ORDER.map((title) => ({
  title,
  pages: DOCS.filter((p) => p.section === title),
})).filter((s) => s.pages.length > 0);

export function getDoc(slug: string | undefined): DocPage | undefined {
  return DOCS.find((p) => p.slug === slug);
}

export function firstDocSlug(): string {
  return DOCS[0].slug;
}

export function adjacentDocs(slug: string): { prev?: DocPage; next?: DocPage } {
  const i = DOCS.findIndex((p) => p.slug === slug);
  return {
    prev: i > 0 ? DOCS[i - 1] : undefined,
    next: i >= 0 && i + 1 < DOCS.length ? DOCS[i + 1] : undefined,
  };
}
