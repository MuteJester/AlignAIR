import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const trainingEngine: DocPage = {
  slug: "training-engine",
  title: "Training engine API",
  section: "Design",
  lead: "Assemble training state, iterate batches, and run optimizer and validation steps through shared nucleotide and protein APIs.",
  body: () => (
    <>
      <p>
        For the public build/train/checkpoint/reload entry point, see the
        <DocLink to="training-workflow"> training workflow API</DocLink>.
        <code>alignair.train.TrainingEngine</code> consumes the shared{" "}
        <DocLink to="training-batches">batch contract</DocLink> and{" "}
        <DocLink to="training-tasks">task interface</DocLink>. A runner selects data, augmentation,
        experiment schedules, and checkpoint timing. The engine owns each optimization step and
        support-weighted metric validation.
      </p>
      <H2 id="step">Construct and run an engine</H2>
      <CodeBlock lang="python" code={`from alignair.train import build_training_engine

# model can come from alignair.core.prepare_model(...).build_model().
engine = build_training_engine(model, device=device, lr=3e-4, grad_clip=1.0)

for step, batch in enumerate(training_batches, start=1):
    result = engine.step(batch.to(device), step=step)
    print(result.total, result.parts, result.gradient_norm)`} />
      <p>
        <code>build_training_engine</code> returns a <code>TrainingEngine</code> containing the model,
        task implementation, task weights, optimizer, and optional scheduler. It selects default tasks
        from <code>model.cfg</code>, calls <code>tasks.make_weights()</code> once when weights are omitted,
        and creates AdamW with the requested learning rate and its established library defaults.
        Models without an <code>AlignAIRConfig</code> need explicit <code>tasks=</code>.
      </p>
      <H2 id="initialization">Placement and parameter ownership</H2>
      <DocTable head={["Argument", "Contract"]} rows={[
        ["device", "Explicit Torch device, or None to retain the model’s current single device. Tensor-free models default to CPU. Mixed-device models require an explicit target or manual assembly."],
        ["tasks", "An optional TrainingTasks implementation. A supplied task object is retained."],
        ["weights", "An optional ModuleDict or mapping from names to modules. The supplied container and modules are retained; an empty mapping remains empty."],
        ["lr / grad_clip", "Positive finite learning rate; clipping is a positive finite value or None. Invalid requests fail before placement."],
      ]} />
      <p>
        The factory moves model and task modules in place before creating the optimizer. Parameter
        order is model first, followed by task weights, with shared identities included once. Default
        groups retain frozen parameters, preserving the existing checkpoint order. Parameters registered
        directly on a task <code>ModuleDict</code> are included as well as child-module parameters.
        If placement splits shared parameter aliases, construction fails before optimizer creation;
        place and tie those modules on the target device before assembly.
      </p>
      <p>
        Assembly preserves training/evaluation flags; each engine step enters training mode. Reusing
        supplied weights avoids reinitializing learned values. The factory creates fresh optimizer state
        and does not load checkpoints or alter a runner’s resume learning-rate policy. Build the engine
        once per run. The existing nucleotide trainer now creates task weights when its optional
        <code> logvars</code> argument is omitted; supplied weights retain their existing behavior.
      </p>
      <H2 id="factories">Select optimizers and update schedulers</H2>
      <CodeBlock lang="python" code={`import torch
from alignair.train import build_training_engine

def optimizer_factory(parameters, *, lr):
    return torch.optim.SGD(parameters, lr=lr, momentum=0.9)

engine = build_training_engine(
    model, tasks=custom_tasks, device=device, lr=0.01,
    optimizer_factory=optimizer_factory,
    scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1000, gamma=0.5,
    ),
)`} />
      <p>
        An <code>OptimizerFactory</code> receives a parameter tuple and keyword learning rate. It can
        define parameter groups but must include every trainable model/task parameter exactly once,
        with no foreign parameters. It may omit frozen parameters. Extra learned values belong in a
        registered model or task module. The returned object must be a Torch optimizer.
      </p>
      <p>
        A scheduler factory receives that optimizer and must return a scheduler bound to it. The engine
        calls its argument-free <code>step()</code> after successful updates. Metric-driven
        <code> ReduceLROnPlateau</code> is rejected here; manage it in the runner with validation results.
        Epoch-based policies likewise belong to the runner. When using an existing optimizer or a more
        specialized placement policy, construct <code>TrainingEngine</code> directly.
      </p>
      <p>
        A step accepts a <code>TrainingBatch</code> or an <code>(inputs, targets)</code> pair.
        It enters training mode, clears gradients, computes the task loss, checks its value, and
        backpropagates. The gradient norm covers all parameters in the optimizer, including learned
        task weights. Non-finite loss or norm raises <code>NonFiniteLossError</code> before the optimizer
        and scheduler advance. A loss disconnected from every optimized parameter raises
        <code> TrainingError</code>.
        Gradient diagnostics name affected model/task parameters and distinguish invalid entries from
        overflow while combining otherwise finite gradients.
      </p>
      <p>
        <code>StepResult</code> contains detached Python values and supports tuple unpacking.
        <code>gradient_norm</code> reports the norm before clipping. A finite positive
        <code> grad_clip</code> clips the combined optimized parameter set; <code>None</code> measures
        without clipping. This differs from the historical NT step, which clipped only model parameters.
        Runs using learned task weights and clipping can therefore follow a different training trajectory.
      </p>
      <p>
        After the optimizer step, the engine calls the model’s optional <code>clamp_params()</code>
        hook and each task-weight module’s optional <code>apply_constraints()</code> hook. An optional
        <code> scheduler=</code> advances afterward and must belong to the same optimizer. Epoch-based
        schedulers or schedulers driven by validation metrics should be managed by the runner instead.
        Every trainable task-weight parameter must be included in the optimizer; omission is a
        <code> ConfigurationError</code>.
      </p>
      <H2 id="iterations">Run an explicit step window</H2>
      <CodeBlock lang="python" code={`from alignair.data import NucleotideBatchAdapter, CategoricalBatchAdapter
from alignair.train import training_iterations

# Choose the adapter that matches your prepared source and model configuration.
adapter = NucleotideBatchAdapter(reference, model.cfg, device=device)
# For protein-class samples, use CategoricalBatchAdapter.from_config(model.cfg, device=device).

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
        <code>training_iterations</code> accepts an engine, an iterable of records or prepared samples,
        and a <DocLink to="training-batches">batch adapter</DocLink>. It yields
        <code> TrainingIteration(step, result, records)</code> after each successful engine step.
        The result is the shared <code>StepResult</code>; records is a tuple containing references to
        that update’s source rows, available for diagnostics such as per-gene class support.
        The iterator does not retain the adapted tensor batch while runner callbacks execute.
        The <DocLink to="curriculum-sources">curriculum source API</DocLink> constructs simulator
        mixtures upstream of this iterator.
      </p>
      <DocTable head={["Argument", "Contract"]} rows={[
        ["start_step", "Last completed optimizer step; defaults to zero."],
        ["stop_step", "Absolute final step, inclusive. A resumed window 50 to 55 performs five updates."],
        ["batch_size", "Positive integer. Only complete batches are adapted and trained."],
        ["records", "Lazy iterable of source records or prepared samples. The adapter determines their label and input semantics."],
      ]} />
      <p>
        Count and adapter validation runs when the function is called. Source iteration begins on the
        first request for an update. Equal start and stop steps form an empty window and do not
        initialize the source. Each subsequent request reads one batch; there is no prefetch beyond
        the requested step budget. If the source ends, its incomplete final batch is consumed and
        discarded. Closing the iterator does not explicitly close the caller’s source.
      </p>
      <p>
        AlignAIR errors created while reading, adapting, or training carry the attempted step in scoped
        diagnostics. Engine errors retain their more specific stage. Failed iterations yield no
        completion record. The iteration scope ends before control returns to the runner, so its
        validation and checkpoint callbacks do not inherit a stale training context.
      </p>
      <p>
        The runner owns history, logging, validation timing, source rejection budgets, checkpoint
        timing, and stopping decisions. An iteration limit bounds optimizer updates; it cannot bound
        raw simulator attempts hidden inside a filtering source. The function does not restore source
        positions or RNG state. When preserving a runner’s additional-step API, pass
        <code> stop_step=start_step + steps</code>; the nucleotide trainer retains its absolute-target
        API, while the crop runner retains its additional-step API.
      </p>
      <H2 id="validation">Validate without changing training modes</H2>
      <CodeBlock lang="python" code={`report = engine.validate(validation_batches)
print(report.metrics, report.score)

# Wrap a study-specific evaluator or callback with the same mode isolation.
from alignair.train import evaluation_mode

with evaluation_mode(model):
    study_result = evaluate_study(model)`} />
      <p>
        Validation runs without gradients and aggregates task metric totals and supervised counts.
        It returns <code>ValidationResult(metrics, score)</code>, where the task implementation chooses
        the selection score. Each module’s previous training/evaluation flag is restored on success or
        failure, including mixed modes such as a frozen evaluation-only child inside a training model.
        The same context surrounds Stage-1 validation callbacks and the AA frozen/allele evaluators.
      </p>
      <H2 id="capacity">Diagnose one-batch learning capacity</H2>
      <CodeBlock lang="python" code={`from alignair.train import overfit_one_batch

# Prepare inputs and targets on the model's device through a batch adapter.
report = overfit_one_batch(model, inputs, targets, steps=300, lr=1e-3)
print(report["initial_loss"], report["final_loss"])
print(report["history"][-1])`} />
      <p>
        <code>train.diagnostics.overfit_one_batch</code> repeatedly updates the supplied model on one
        fixed batch. Default tasks follow its configuration, including every configured gene and
        auxiliary task. This supports nucleotide allele and protein categorical models through the
        same interface. Pass explicit <code>tasks=</code> to select another loss and metrics, or use
        <code> genes=</code> to select a supported subset of the built-in tasks. Those options are
        mutually exclusive. Hierarchical allele tasks require their complete canonical gene set.
      </p>
      <DocTable head={["Result", "Meaning"]} rows={[
        ["initial_loss", "The first training loss, before any optimizer update."],
        ["history", "One detached total/component-loss dictionary per successful update, measured before that update."],
        ["final_loss", "Evaluation-mode loss after the final update, using the actual learned task weights."],
        ["Other keys", "Final metrics provided by tasks.metrics; names follow the configured task semantics."],
      ]} />
      <p>
        The helper creates fresh AdamW state through <code>build_training_engine</code>. Supplied
        <code> weights=</code> are retained and optimized; omitted weights are created once. Optional
        <code> grad_clip</code> uses the engine’s combined gradient policy. Positive integer steps
        and custom metric capability are validated before training. Training failures retain their
        step diagnostics; a non-finite final total or component loss raises
        <code> NonFiniteLossError</code> with evaluation context.
      </p>
      <p>
        This operation changes model and task-weight parameters in place and leaves the model in
        evaluation mode on success. Inputs and targets remain on their supplied device. Seed the
        experiment and prepare a representative batch before running it; the helper does not fetch
        records, check diversity, or assert a universal passing threshold. Memorization on this batch
        measures learning capacity. Use a separate held-out cohort to assess generalization.
        The <DocLink to="sample-preparation">sample preparation API</DocLink> provides class-diversity
        policies and explicit completed-cohort validation. Historical <code>aa.train</code> imports
        delegate here; the shared default includes all configured genes rather than assuming V/J only.
      </p>
      <H2 id="checkpoints">Choose a checkpoint contract</H2>
      <DocTable head={["API in alignair.train.checkpoints", "Stored state and purpose"]} rows={[
        ["save_checkpoint / load_training_checkpoint", <>Self-contained <code>.alignair</code> artifacts
          with model configuration, reference, weights, task weights, and process RNG state.
          Supplying an optimizer stores its continuation state. The reader also supports explicitly
          trusted historical NT checkpoints.</>],
        ["save_research_checkpoint / load_research_checkpoint", <>Trusted Python <code>.pt</code> research
          checkpoints containing model/configuration, optimizer, scheduler, process RNG, learning rate,
          history, and provenance. Existing research field names remain supported; new files also include
          Python’s RNG state. Task-weight module state is not stored by this research helper and
          must be persisted/restored separately when the experiment needs it.</>],
        ["save_snapshot", <>Model weights, plain configuration, and completed step only. Used for a warm
          start or research weight export; optimizer and RNG continuation state are absent.</>],
        ["save_rotating / latest_checkpoint", "Publish numbered checkpoints, retain the requested number of recent states, and resolve the newest completed step. Retention must be a positive integer."],
      ]} />
      <CodeBlock lang="python" code={`from alignair.train.checkpoints import save_checkpoint, save_snapshot

save_checkpoint(
    "run/model.alignair", model.cfg, model, engine.weights, step, engine.optimizer,
    dataconfigs=dataconfigs,
    train_args={"lr": 3e-4, "batch_size": 32, "seed": 0},
)
save_snapshot("run/warm-start.pt", model, model.cfg, step)`} />
      <p>
        Checkpoint writers serialize to a temporary sibling file and replace the destination only after
        the write completes. Rotation refreshes the base-path copy the same way. If that refresh fails,
        the previous base remains intact and the completed numbered checkpoint is available to
        <code> latest_checkpoint</code>. These writers protect completed files from interrupted writes;
        retention runs after publication and logs a warning if an older file cannot be removed.
      </p>
      <p>
        Research checkpoints use Python pickle and are for trusted local inputs. The NT checkpoint
        reader requires <code>trust_pickle=True</code> for historical raw <code>.pt</code> files;
        the runner exposes this as <code>resume_trust_pickle</code>. Public inference artifacts keep
        their existing model-file loading and export contracts.
      </p>
      <H2 id="resume">RNG state and continuation limits</H2>
      <p>
        <code>train.random_state.capture_rng_state()</code> captures Python, NumPy’s process generator,
        Torch CPU, and available CUDA generator states. <code>restore_rng_state(state)</code> restores
        supplied fields. Malformed state raises <code>TrainingError</code> and rolls back partial
        restoration rather than continuing with a partly restored process. CUDA state is skipped on a
        CPU host. Missing state in an older file remains missing; it is not reconstructed.
      </p>
      <p>
        Restoring these generators and optimizer moments does not restore a simulator iterator or a
        separately constructed NumPy <code>Generator</code>. Current gym and Stage-1 runners re-seed
        their source streams at resume; Stage-1 also re-seeds its augmentation generator. They provide
        statistical continuation, while exact uninterrupted replay would additionally require restoring
        those source positions and local generator states.
      </p>
      <H2 id="migration">Implementation ownership</H2>
      <p>
        NT, AA allele, categorical, partial-label, and capacity-proof runners share this engine and
        its training-state factory. Frozen-cohort replicate training uses the same factory.
        Historical <code>train.trainer.train_step</code>, <code>validate</code>, checkpoint helpers,
        and <code>aa.resume</code> imports delegate to shared owners. Duplicate optimizer loops,
        snapshot writers, and process-RNG implementations have been removed from those runners.
        Nucleotide training, protein allele training, categorical training, and partial-coverage
        training also share <code>train.loop.training_iterations</code>; their repeated batch-fetch
        and step-window implementations have been removed. Study-specific sampling, stopping rules,
        and provenance remain with the study code.
      </p>
    </>
  ),
};

export const enginePages = [trainingEngine];
