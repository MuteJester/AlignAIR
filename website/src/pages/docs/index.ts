import type { DocPage } from "./doc-kit";
import { adjacent, sectionsFor } from "./nav";
import { getStartedPages, conceptsPages } from "./pages-getstarted";
import { migratePages } from "./pages-migrate";
import { tutorialPages } from "./pages-tutorial";
import { guidesPages, trainingPages } from "./pages-guides";
import { guides2Pages } from "./pages-guides2";
import { airrFieldsPages } from "./pages-airrfields";
import { referencePages } from "./pages-reference";
import { errorPages } from "./pages-errors";
import { evalPages } from "./pages-eval";
import { maintainerPages } from "./pages-maintainer";
import { testingPages } from "./pages-testing";
import { setupPages } from "./pages-setup";
import { compositionPages } from "./pages-composition";
import { streamPages } from "./pages-streams";
import { taskPages } from "./pages-tasks";
import { batchingPages } from "./pages-batching";
import { enginePages } from "./pages-engine";
import { trainingWorkflowPages } from "./pages-training-workflow";
import { inferencePages } from "./pages-inference";
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
  ...errorPages, // errors  [Reference]
  ...referencePages, // model-contract, known-failure-modes, troubleshooting
  ...trainingPages, // training  [Training]
  ...trainingWorkflowPages, // training-workflow  [Training]
  ...evalPages, // benchmarks, performance  [Evaluation]; design  [Design]
  ...compositionPages, // model-composition  [Design]
  ...setupPages, // model-setup  [Design]
  ...streamPages, // curriculum-sources  [Design]
  ...taskPages, // training-tasks  [Design]
  ...batchingPages, // training-batches  [Design]
  ...enginePages, // training-engine  [Design]
  ...inferencePages, // model-inference  [Design]
  ...maintainerPages, // publishing  [Maintainer]
  ...testingPages, // testing  [Maintainer]
  ...communityPages, // citation-support  [Community]
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
