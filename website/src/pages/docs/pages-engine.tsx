import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const trainingEngine: DocPage = {
  slug: "training-engine",
  title: "Training engine API",
  section: "Design",
  lead: "Assemble training state, iterate batches, run optimizer and validation steps, and write checkpoints.",
  body: () => (
    <>
      <p>
        For the public build/train/checkpoint/reload entry point, see the
        <DocLink to="training-workflow"> training workflow API</DocLink>.
        <code> alignair.train.TrainingEngine</code> consumes the shared{" "}
        <DocLink to="training-batches">batch contract</DocLink> and{" "}
        <DocLink to="training-tasks">task interface</DocLink>. A runner selects the record stream, schedules and
        checkpoint timing; the engine owns each optimization step and support-weighted validation.
      </p>
      <H2 id="step">Construct and run an engine</H2>
      <CodeBlock lang="python" code={`from alignair.train import build_training_engine

# model can come from alignair.core.prepare_model(...).build_model().
engine = build_training_engine(model, device=device, lr=3e-4, grad_clip=1.0)

for step, batch in enumerate(training_batches, start=1):
    result = engine.step(batch.to(device), step=step)
    print(result.total, result.parts, result.gradient_norm)`} />
      <p>
        <code>build_training_engine(model, *, tasks=None, device=None, lr=3e-4, grad_clip=None,
        optimizer_factory=torch.optim.AdamW, scheduler_factory=None)</code> returns a <code>TrainingEngine</code>
        holding the model, the tasks, a fresh optimizer and an optional scheduler. Default tasks come from
        <code> build_tasks(model.cfg)</code>; a model without an <code>AlignAIRConfig</code> needs explicit
        <code> tasks=</code>.
      </p>
      <H2 id="initialization">Placement and parameter ownership</H2>
      <DocTable head={["Argument", "Contract"]} rows={[
        ["device", "A Torch device, or None to keep the model's current single device (CPU for a tensor-free model). A model with tensors on several devices needs an explicit device."],
        ["tasks", "An optional TrainingTasks implementation; a supplied object is used as is. It must provide loss."],
        ["lr / grad_clip", "A positive finite learning rate; clipping is a positive finite value or None. Invalid requests fail before placement."],
      ]} />
      <p>
        The factory moves the model in place before creating the optimizer and passes it the model&apos;s parameters,
        each once, however many modules register it. Default groups keep frozen parameters. If placement splits shared
        parameter aliases, construction fails before optimizer creation; place and tie those modules on the target
        device first. The factory creates fresh optimizer state and does not load checkpoints.
      </p>
      <H2 id="factories">Select optimizers and schedulers</H2>
      <CodeBlock lang="python" code={`import torch
from alignair.train import build_training_engine

def optimizer_factory(parameters, *, lr):
    return torch.optim.SGD(parameters, lr=lr, momentum=0.9)

engine = build_training_engine(
    model, device=device, lr=0.01,
    optimizer_factory=optimizer_factory,
    scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1000, gamma=0.5,
    ),
)`} />
      <p>
        An <code>OptimizerFactory</code> receives a parameter tuple and a keyword learning rate. It can define
        parameter groups but must include every trainable model parameter exactly once and no foreign parameter; it
        may omit frozen ones. Otherwise <code>ConfigurationError</code> (&ldquo;optimizer must own every trainable
        model parameter exactly once without foreign parameters&rdquo;). A scheduler factory receives that optimizer
        and must return a step scheduler bound to it; <code>ReduceLROnPlateau</code> is rejected, so manage
        metric-driven schedules in the runner. The engine calls the scheduler&apos;s <code>step()</code> after each
        successful update. Construct <code>TrainingEngine(model, optimizer, tasks=..., grad_clip=...,
        scheduler=...)</code> directly to use an existing optimizer.
      </p>
      <p>
        A step accepts a <code>TrainingBatch</code> or an <code>(inputs, targets)</code> pair. It enters training mode,
        clears gradients, computes the task loss, checks it, and backpropagates. A non-finite loss raises
        <code> NonFiniteLossError</code> with the per-term values before backpropagation; a non-finite gradient norm
        raises it too, naming the affected parameters, after clearing the rejected gradients and before the optimizer
        and scheduler advance. A loss with no gradient graph, or with no gradient for any optimized parameter, raises
        <code> TrainingError</code>.
      </p>
      <p>
        <code>StepResult(total, parts, gradient_norm)</code> holds detached Python values and supports tuple
        unpacking. <code>gradient_norm</code> covers every optimized parameter and is measured before clipping; a
        <code> grad_clip</code> clips the same parameter set.
      </p>
      <H2 id="iterations">Run an explicit step window</H2>
      <CodeBlock lang="python" code={`from alignair.data import NucleotideBatchAdapter
from alignair.train import training_iterations

adapter = NucleotideBatchAdapter(reference, model.cfg, device=device)

completed_step = start_step
for iteration in training_iterations(
    engine, records, adapter,
    start_step=start_step, stop_step=10_000, batch_size=32,
):
    completed_step = iteration.step
    print(completed_step, iteration.result.total)
    # Evaluate, checkpoint, inspect iteration.records, or break before the next update.

# A final checkpoint should record completed_step, including when records end early.`} />
      <p>
        <code>training_iterations</code> takes an engine, an iterable of records and a
        <DocLink to="training-batches"> batch adapter</DocLink>, and yields
        <code> TrainingIteration(step, result, records)</code> after each successful engine step; <code>records</code>
        is the tuple of that update&apos;s source records. The iterator does not keep the adapted tensor batch while
        runner code runs. The <DocLink to="curriculum-sources">curriculum source API</DocLink> builds the record stream.
      </p>
      <DocTable head={["Argument", "Contract"]} rows={[
        ["start_step", "Last completed optimizer step; defaults to zero."],
        ["stop_step", "Absolute final step, inclusive. A resumed window 50 to 55 performs five updates."],
        ["batch_size", "Positive integer. Only complete batches are adapted and trained."],
        ["records", "Lazy iterable of source records. The adapter determines their input and target semantics."],
      ]} />
      <p>
        Count and adapter validation runs when the function is called; source iteration begins on the first request
        for an update, so an empty window never starts the source. Each request reads one batch, with no prefetch
        beyond the requested steps. If the source ends, its incomplete final batch is consumed and discarded. AlignAIR
        errors created while reading, adapting or training carry the attempted step; that scope ends before control
        returns to the runner. The function does not save checkpoints or restore source positions or RNG state.
      </p>
      <H2 id="validation">Validate without changing training modes</H2>
      <CodeBlock lang="python" code={`report = engine.validate(validation_batches)
print(report.metrics, report.score)

# Wrap a custom evaluator with the same mode isolation.
from alignair.train import evaluation_mode

with evaluation_mode(model):
    study_result = evaluate_study(model)`} />
      <p>
        Validation runs without gradients, accumulates the task&apos;s metric totals and supervised counts, and returns
        <code> ValidationResult(metrics, score)</code>, where <code>score</code> is the task&apos;s
        <code> selection_score</code>. <code>evaluation_mode</code> restores every module&apos;s previous
        training/evaluation flag on success or failure, including mixed modes.
      </p>
      <H2 id="checkpoints">Checkpoints</H2>
      <DocTable head={["API in alignair.train.checkpoints", "Behavior"]} rows={[
        ["save_checkpoint(path, cfg, model, step, opt=None, *, dataconfigs, train_args, label_schema, scheduler)", <>Writes a
          self-contained <code>.alignair</code> training checkpoint: configuration, weights, the reference (FASTA and
          pickle-free JSON), the label schema, and the trusted pickle sections <code>dataconfig/i</code> and
          <code> train_state</code> (optimizer, scheduler, process RNG state, step and <code>train_args</code>).</>],
        ["load_training_checkpoint(path, *, device)", <>Decodes a checkpoint into a <code>ResumeState</code> without
          allocating a model and without applying its RNG state. An AlignAIR 3 <code>.pt</code> is refused with the
          version that reads it.</>],
        ["save_rotating(base_path, cfg, model, step, opt, keep=3, ...)", <>Writes <code>{"<stem>.step<N>.alignair"}</code>,
          refreshes the base-path copy and keeps the newest <code>keep</code> numbered files. Retention must be a
          positive integer.</>],
        ["latest_checkpoint(base_path)", "The numbered checkpoint with the highest step beside base_path, else base_path if it exists, else None."],
      ]} />
      <CodeBlock lang="python" code={`from alignair.train.checkpoints import save_checkpoint

save_checkpoint(
    "run/model.alignair", model.cfg, model, step, engine.optimizer,
    dataconfigs=["HUMAN_IGH_OGRDB"],
    train_args={"lr": 3e-4, "batch_size": 32, "seed": 0},
    label_schema=setup.label_schema(), scheduler=engine.scheduler,
)`} />
      <p>
        Writers serialize to a temporary sibling file and replace the destination only after the write completes. If
        the base-path refresh fails, the previous base stays intact and the numbered checkpoint is available to
        <code> latest_checkpoint</code>. Retention runs after publication and logs a warning when an older file cannot
        be removed. A training checkpoint holds pickle sections, so open only checkpoints you trust; to distribute a
        model, export a pickle-free bundle (see the <DocLink to="training-workflow">training workflow</DocLink>).
      </p>
      <H2 id="resume">RNG state and continuation limits</H2>
      <p>
        <code>train.random_state.capture_rng_state()</code> captures Python, NumPy&apos;s process generator, Torch CPU,
        and available CUDA generator states; <code>restore_rng_state(state)</code> restores the supplied fields. Malformed
        state raises <code>TrainingError</code> and rolls back the partial restoration. CUDA state is skipped on a CPU
        host; missing state in a file stays missing.
      </p>
      <p>
        Restoring these generators and the optimizer moments does not restore a simulator stream. A resumed run builds
        new record streams for its start step, which draw seed blocks of their own: continuation is statistical, not an
        exact replay of the next uninterrupted batch.
      </p>
      <H2 id="ownership">Implementation ownership</H2>
      <p>
        The trainer (<code>train.trainer.train</code>) assembles its engine with <code>build_training_engine</code>,
        steps it through <code>training_iterations</code>, validates with the engine and writes checkpoints with
        <code> save_checkpoint</code> and <code>save_rotating</code>. Its batches come from the adapters of
        <code> prepare_training_data</code>, built in the training process or, with <code>data_workers</code> above 0,
        in worker processes that produce the same batches.
      </p>
    </>
  ),
};

export const enginePages = [trainingEngine];
