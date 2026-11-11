import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const trainingTasks: DocPage = {
  slug: "training-tasks",
  title: "Training task API",
  section: "Design",
  lead: "The chip's training objective, its metrics with support counts, and the task contract a trainer consumes.",
  body: () => (
    <>
      <p>
        <code>alignair.tasks.build_tasks(config)</code> returns the configured architecture&apos;s own objective, the
        <code> objective</code> factory of its <DocLink to="model-composition">registered components</DocLink>. For
        <code> chip-v1</code> this is <code>alignair.network.objective.ChipObjective</code>, which supervises the whole
        gene set; passing <code>genes=</code> raises <code>ConfigurationError</code> (&ldquo;an architecture&apos;s own
        objective supervises its whole gene set&rdquo;). The <DocLink to="training-batches">batch adapter</DocLink>
        prepares the targets the objective reads.
      </p>
      <H2 id="step">Configure a training step</H2>
      <CodeBlock lang="python" code={`import torch
from alignair.tasks import build_tasks
from alignair.train import TrainingEngine

tasks = build_tasks(model.cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
engine = TrainingEngine(model, optimizer, tasks=tasks, grad_clip=1.0)

# inputs and targets come from NucleotideBatchAdapter.
total, parts, gradient_norm = engine.step((inputs, targets), step=1)`} />
      <p>
        The objective has no learned parameters: each loss term is divided by its chance level, so every term starts
        near 1, and the terms are summed with fixed weights (<code>LOSS_WEIGHTS</code>, all 1). The optimizer
        therefore holds the model&apos;s parameters only.
      </p>
      <H2 id="objectives">Loss terms</H2>
      <DocTable head={["Term (parts key)", "Targets", "Loss and normalisation"]} rows={[
        [<code>orientation</code>, <code>orientation</code>, "Cross entropy over the 4 orientations, / log 4."],
        [<code>parse</code>, <><code>region</code>, <code>region_allowed</code></>, "The parser's negative log probability of the region paths the read cannot tell apart from the true one, / ((the true path's region changes + 1) x log(read length))."],
        [<><code>v_anchor</code>, <code>j_anchor</code></>, <><code>v_anchor</code>, <code>j_anchor</code> and their <code>_supervise</code> masks</>, "Cross entropy over the read's positions and the \"not in read\" slot, / log(window + 1)."],
        [<code>shm</code>, <><code>state_labels</code>, <code>region</code></>, "Unweighted cross entropy over germline / substitution / insertion on V, D and J bases (deletion-tagged bases left out), / log 3."],
        [<code>coord</code>, <code>{"coord_{gene}"}</code>, "Cross entropy over each gene's frame columns on the bases its alignment maps, / log(width), averaged over the genes."],
        [<code>{"scan_{gene}"}</code>, <code>{"{gene}_allele"}</code>, "Minus the log posterior mass on the true allele set over the bank rows the batch allows, / log(alleles allowed)."],
        [<code>d_presence</code>, <><code>d_presence</code>, <code>d_presence_supervise</code></>, "Cross entropy over absent / too short / callable, / log 3."],
      ]} />
      <p>
        <code>{"{gene}_supervise"}</code> is a binary row mask (all rows when absent); any other value raises
        <code> TrainingDataError</code> (&ldquo;v_supervise must be a binary row mask of shape (B,)&rdquo;). A row
        contributes a scan term only when it is supervised, its bank allows at least 2 alleles and a true allele is
        among them. Anchor and D presence terms skip rows whose <code>_supervise</code> mask is false, such as reads
        whose alignment cannot be mapped. A term with no supervised rows contributes zero. The terms present depend on
        the configuration: no anchor terms without <code>anchor_heads</code>, no D terms without a D.
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
        A <code>MetricReport</code> carries metric totals and their supervised counts, and accumulation weights by
        those counts: one correct call out of two in one batch and one out of one in another gives <code>2/3</code>,
        not the mean of the two batch accuracies. Genes with no supervised rows are omitted; a metric with zero support
        is NaN. <code>tasks.metrics(outputs, targets)</code> returns just the means.
      </p>
      <DocTable head={["Metric", "Meaning"]} rows={[
        [<code>{"{gene}_allele_top1"}</code>, <>The top <code>{"{gene}_allele"}</code> score is in the read&apos;s true allele set.</>],
        [<code>{"{gene}_seg_mae"}</code>, "Mean absolute error of the decoded start and end, in positions."],
        [<code>{"parse_{gene}_{start,end}_exact"}</code>, "The parser's decoded boundary equals the true one."],
        [<code>{"scan_{gene}_top1"}</code>, <>The argmax of <code>{"{gene}_log_posterior"}</code> is in the true allele set.</>],
        [<code>{"bank_{gene}_size"}</code>, "Mean number of alleles the batch's bank allows."],
        [<code>{"coord_{gene}_top1"}</code>, "The reader's top frame column is the true one, over mapped bases."],
        [<><code>anchor_v_exact</code>, <code>anchor_j_exact</code></>, "The decoded anchor position (or \"not in read\") is the true one."],
        [<code>shm_brier</code>, "Brier score of the per-base edit-state probabilities."],
        [<code>d_presence_acc</code>, "Accuracy of the 3-state D presence call."],
        [<code>orientation_acc</code>, "Accuracy of the orientation head."],
        [<><code>mutation_mae</code>, <code>indel_mae</code></>, "Absolute error of the mutation-rate and indel-count estimates."],
      ]} />
      <p>
        <code>selection_score</code> is the mean of the available <code>{"scan_{gene}_top1"}</code> values; it is
        NaN when none is present or any is non-finite, never converted to zero. The trainer keeps the checkpoint with
        the highest score as its best model.
      </p>
      <H2 id="extension">Extend task behavior</H2>
      <p>
        Trainers consume the <code>TrainingTasks</code> protocol: <code>loss(outputs, targets)</code> returning a scalar
        tensor and a dictionary of named tensor parts, <code>metrics</code>, <code>metric_report</code> and
        <code> selection_score</code>. Supply a task object through the <code>tasks=</code> argument of the
        <DocLink to="training-engine">training engine</DocLink>, or make it a new architecture&apos;s
        <code> objective</code> factory. Task code is not serialized into model files, so keep a custom implementation
        with the experiment source when resuming.
      </p>
    </>
  ),
};

export const taskPages = [trainingTasks];
