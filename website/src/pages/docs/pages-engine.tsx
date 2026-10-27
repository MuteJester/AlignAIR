import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const trainingEngine: DocPage = {
  slug: "training-engine",
  title: "Training engine API",
  section: "Design",
  lead: "Use shared batch iteration, optimizer steps, numerical checks, and validation for nucleotide and protein alignment tasks.",
  body: () => (
    <>
      <p>
        <code>alignair.train.TrainingEngine</code> consumes the shared{" "}
        <DocLink to="training-batches">batch contract</DocLink> and{" "}
        <DocLink to="training-tasks">task interface</DocLink>. A runner selects data, augmentation,
        experiment schedules, and checkpoint timing. The engine owns each optimization step and
        support-weighted metric validation.
      </p>
      <H2 id="step">Construct and run an engine</H2>
      <CodeBlock lang="python" code={`from itertools import chain
import torch
from alignair.tasks import build_tasks
from alignair.train import TrainingEngine

model.to(device)
tasks = build_tasks(model.cfg)
weights = tasks.make_weights().to(device)
optimizer = torch.optim.AdamW(
    chain(model.parameters(), weights.parameters()), lr=3e-4,
)
engine = TrainingEngine(
    model, optimizer, tasks=tasks, weights=weights, grad_clip=1.0,
)

for step, batch in enumerate(training_batches, start=1):
    result = engine.step(batch.to(device), step=step)
    print(result.total, result.parts, result.gradient_norm)`} />
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
      <H2 id="checkpoints">Choose a checkpoint contract</H2>
      <DocTable head={["API in alignair.train.checkpoints", "Stored state and purpose"]} rows={[
        ["save_checkpoint / load_training_checkpoint", <>Self-contained <code>.alignair</code> artifacts
          with model configuration, reference, weights, task weights, and process RNG state.
          Supplying an optimizer stores its continuation state. The reader also supports explicitly
          trusted historical NT checkpoints.</>],
        ["save_research_checkpoint / load_research_checkpoint", <>Trusted Python <code>.pt</code> research
          checkpoints containing model/configuration, optimizer, scheduler, process RNG, learning rate,
          history, and provenance. Existing research field names remain supported; new files also include
          Python’s RNG state.</>],
        ["save_snapshot", <>Model weights, plain configuration, and completed step only. Used for a warm
          start or research weight export; optimizer and RNG continuation state are absent.</>],
        ["save_rotating / latest_checkpoint", "Publish numbered checkpoints, retain the requested number of recent states, and resolve the newest completed step. Retention must be a positive integer."],
      ]} />
      <CodeBlock lang="python" code={`from alignair.train.checkpoints import save_checkpoint, save_snapshot

save_checkpoint(
    "run/model.alignair", model.cfg, model, weights, step, optimizer,
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
        NT, AA allele, categorical, partial-label, and capacity-proof runners share this engine.
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
