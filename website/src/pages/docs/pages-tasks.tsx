import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const trainingTasks: DocPage = {
  slug: "training-tasks",
  title: "Training task API",
  section: "Design",
  lead: "Use one loss, metric, and validation contract for nucleotide and protein-class alignment models.",
  body: () => (
    <>
      <p>
        <code>alignair.tasks.build_tasks(config)</code> selects alignment objectives from the model’s
        classification semantics and enabled heads. It does not select an objective by alphabet or prepare
        training examples. The <DocLink to="training-batches">batch adapters</DocLink> prepare examples
        with the reference, labels, and coordinates required by those objectives.
        Model construction is described in the <DocLink to="model-composition">composition API</DocLink>.
      </p>
      <H2 id="step">Configure a training step</H2>
      <CodeBlock lang="python" code={`from itertools import chain
import torch
from alignair.tasks import build_tasks
from alignair.train import TrainingEngine

tasks = build_tasks(model.cfg)
weights = tasks.make_weights().to(device)
optimizer = torch.optim.AdamW(
    chain(model.parameters(), weights.parameters()), lr=3e-4,
)

engine = TrainingEngine(
    model, optimizer, tasks=tasks, weights=weights, grad_clip=1.0,
)
# inputs and targets come from the matching data adapter.
total, parts, gradient_norm = engine.step((inputs, targets), step=1)`} />
      <p>
        Create task weights once and include them in the optimizer. Only enabled weighted tasks receive
        uncertainty parameters. A standard protein-class model has no uncertainty weights for its gene
        objective; enabling auxiliary or edit-state heads adds the corresponding weights. Changing the task
        set changes optimizer state requirements, so use a compatible checkpoint or start fresh training.
        Older checkpoints can also contain unused uncertainty weights that the shared task contract no
        longer allocates; their training state needs migration or fresh training. Inference weights are unaffected.
      </p>
      <H2 id="objectives">Objective and target contracts</H2>
      <DocTable head={["Objective", "Targets and reduction"]} rows={[
        ["Multilabel hierarchical", <>Uses <code>{"{gene}_allele"}</code> multi-hot targets, Gaussian boundary
          objectives, learned uncertainty weights, span regularizers, and the existing short-D penalty.</>],
        ["Categorical", <>Uses integer <code>{"{gene}_class"}</code> labels or binary
          <code>{" {gene}_compatible"}</code> sets. A compatible-set loss is the negative log probability
          assigned to that set. Boundary terms are averaged, class terms are averaged, and the two means are added.</>],
        ["Auxiliary heads", "Mutation and indel regression, productivity classification, and optional locus classification use the same implementations for either score family."],
        ["Edit states", <>Uses per-position labels with <code>-100</code> for ignored positions. A batch
          with no state labels contributes zero state loss and zero state-weight gradient.</>],
      ]} />
      <p>
        <code>{"{gene}_supervise"}</code> and <code>{"{gene}_valid"}</code> are binary row masks; when both
        exist, their intersection selects supervised rows. Categorical label <code>-1</code> excludes class
        supervision while allowing known boundary targets. The binary row mask <code>{"{gene}_class_supervise"}</code>
        also excludes classification independently of boundaries, including when compatible sets are supplied.
        An empty compatible set is permitted on an
        excluded row and rejected on a supervised row. Excluded categorical rows are removed before
        cross-entropy and logarithms, keeping empty sets from producing non-finite losses.
      </p>
      <p>
        Enabled auxiliary regression and productivity heads require their target fields. Orientation,
        locus, and state objectives run when their labels are supplied. Missing required fields and invalid
        masks raise <code>TrainingDataError</code> with field context. Missing uncertainty weights raise
        <code>ConfigurationError</code>.
      </p>
      <H2 id="metrics">Metrics and validation</H2>
      <CodeBlock lang="python" code={`from alignair.tasks import MetricAccumulator

report = tasks.metric_report(outputs, targets)
batch_metrics = report.means()
support = report.counts

accumulator = MetricAccumulator()
for outputs, targets in evaluated_batches:
    accumulator.update(tasks.metric_report(outputs, targets))
metrics = accumulator.means()
selection_score = tasks.selection_score(metrics)`} />
      <p>
        A report carries metric totals and their supervised counts. Accumulation weights by those counts:
        one correct call out of two in one batch and one out of one in another gives <code>2/3</code>,
        rather than the mean of the two batch accuracies. Gene metrics count supervised rows; edit-state
        metrics count labeled positions. Genes with no supervision are omitted. A state recall with no
        examples has zero support and a NaN value, preserving the distinction from measured zero recall.
      </p>
      <p>
        Multilabel metrics retain <code>{"{gene}_allele_top1"}</code> and <code>{"{gene}_seg_mae"}</code>.
        Categorical metrics retain <code>{"{gene}_class_acc"}</code> and separate boundary MAEs, respecting
        compatible sets and validity masks. <code>tasks.metrics(outputs, targets)</code> returns just the
        means. The default selection score averages available per-gene classification accuracies; a
        missing or non-finite score is not converted to zero.
      </p>
      <H2 id="extension">Extend task behavior</H2>
      <p>
        Trainers consume the <code>TrainingTasks</code> protocol: <code>loss</code>, <code>make_weights</code>,{" "}
        <code>metrics</code>, <code>metric_report</code>, and <code>selection_score</code>. Supply a task object
        through the explicit <code>tasks=</code> argument to customize behavior. The loss returns a scalar
        tensor and a dictionary of named tensor components. Metric reports supply totals and counts for
        correct aggregation. Keep a custom task implementation with the experiment source when resuming;
        Python task callbacks are not serialized into model files.
      </p>
      <p>
        The <DocLink to="training-engine">shared training engine</DocLink> works with either target contract. Existing gym and research runners
        retain their data preparation and study schedules. The historical <code>core.losses</code>
        facade has been removed. Import objective functions and <code>make_loss_weights</code> from
        <code> alignair.tasks.losses</code>, and Gaussian targets from <code>alignair.tasks.functional</code>.
      </p>
      <H2 id="migration">Migrate former protein loss imports</H2>
      <p>
        The <code>aa.loss</code> facade has been removed. Training code should use
        <code>build_tasks(config).loss(...)</code> to select the configured gene set and validate its
        targets. Code calling the lower-level functions directly can use these replacements.
      </p>
      <DocTable head={["Former import", "Replacement"]} rows={[
        ["alignair.aa.loss.aa_loss", "alignair.tasks.losses.categorical_loss"],
        ["alignair.aa.loss.masked_soft_gaussian", "alignair.tasks.losses.masked_soft_gaussian"],
      ]} />
      <p>
        <code>categorical_loss</code> retains its <code>(outputs, targets, genes=("v", "j"))</code>
        signature and scalar-total/component-dictionary return contract. Select D explicitly in
        that function&apos;s gene tuple when needed; the task interface derives it from configuration.
      </p>
    </>
  ),
};

export const taskPages = [trainingTasks];
