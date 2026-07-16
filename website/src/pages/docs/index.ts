import type { DocPage } from "./doc-kit";
import { getStartedPages } from "./pages-getstarted";
import { guidesPages } from "./pages-guides";
import { guides2Pages } from "./pages-guides2";
import { airrFieldsPages } from "./pages-airrfields";
import { referencePages } from "./pages-reference";
import { evalPages } from "./pages-eval";
import { maintainerPages } from "./pages-maintainer";

/** All reference pages, in reading order. */
export const DOCS: DocPage[] = [
  ...getStartedPages,
  ...guidesPages,
  ...guides2Pages,
  ...airrFieldsPages,
  ...referencePages,
  ...evalPages,
  ...maintainerPages,
];

const SECTION_ORDER = ["Get started", "Guides", "Reference", "Evaluation", "Design", "Maintainer"];

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
