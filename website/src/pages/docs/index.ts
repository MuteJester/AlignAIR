import type { DocPage } from "./doc-kit";
import { adjacent, sectionsFor } from "./nav";
import { startPages } from "./pages-start";
import { comingFromPages } from "./pages-coming-from";
import { usePages } from "./pages-use";
import { trainingPages } from "./pages-guides";
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
 * Every docs page. The sidebar and Previous/Next order them by section (nav.ts): the user's journey (Get started,
 * Coming from…, Use on your data, Understand the results, Reference, Train or fine-tune, Project), then, apart, the
 * developers'.
 */
export const DOCS: DocPage[] = [
  ...startPages, // install, first-alignment, concepts, how-alignair-works  [Get started]
  ...comingFromPages, // from-igblast, from-mixcr, from-immcantation  [Coming from…]
  ...usePages, // choose-a-model … fix-common-problems  [Use on your data]
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
