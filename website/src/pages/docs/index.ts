import type { DocPage } from "./doc-kit";
import { adjacent, sectionsFor } from "./nav";
import { startPages } from "./pages-start";
import { comingFromPages } from "./pages-coming-from";
import { usePages } from "./pages-use";
import { resultsPages } from "./pages-results";
import { referencePages } from "./pages-reference";
import { trainPages } from "./pages-train";
import { projectPages } from "./pages-project";
import { developerPages } from "./pages-developers";

/**
 * Every docs page. The sidebar and Previous/Next order them by section (nav.ts): the user's journey (Get started,
 * Coming from…, Use on your data, Understand the results, Reference, Train or fine-tune, Project), then, apart, the
 * developers'.
 */
export const DOCS: DocPage[] = [
  ...startPages, // install, first-alignment, concepts, how-alignair-works  [Get started]
  ...comingFromPages, // from-igblast, from-mixcr, from-immcantation  [Coming from…]
  ...usePages, // choose-a-model … fix-common-problems  [Use on your data]
  ...resultsPages, // output-columns, trust-the-calls, comparison  [Understand the results]
  ...referencePages, // cli, python-api, column-dictionary, model-cards, error-codes, cache-and-environment  [Reference]
  ...trainPages, // do-i-need-to-train, train, fine-tune, evaluate, share  [Train or fine-tune]
  ...projectPages, // cite, support, changelog  [Project]
  ...developerPages, // architecture, testing, publishing, contributing  [Developers]
];

export { audienceOf, sectionsFor } from "./nav";

export function getDoc(slug: string | undefined): DocPage | undefined {
  return DOCS.find((p) => p.slug === slug);
}

export function firstDocSlug(): string {
  return sectionsFor(DOCS, "user")[0].pages[0].slug;
}

export function adjacentDocs(slug: string): { prev?: DocPage; next?: DocPage } {
  return adjacent(DOCS, slug);
}
