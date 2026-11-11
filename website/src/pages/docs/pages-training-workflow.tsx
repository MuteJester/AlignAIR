import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const trainingWorkflow: DocPage = {
  slug: "training-workflow",
  title: "Training workflow API",
  section: "Training",
  lead: "Build, train, validate, checkpoint, resume and reload a model through one public Python workflow.",
  body: () => (
    <>
      <CodeBlock lang="python" code={`from alignair import TrainingConfig, run_training

request = TrainingConfig.from_genairr(
    "HUMAN_IGH_OGRDB", preset="quick",
    steps=10, batch_size=16, val_every=5, seed=7, device="cpu",
)
run = run_training(request, output_dir="runs/igh")
aligner = run.best_aligner(device="cpu")
raw = aligner.predict_raw(["CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCC"])
print(raw["v_allele"].shape)                  # (1, number of V alleles)
print(aligner.label_schema["genes"]["v"][0])  # the allele of column 0`} />
      <p>
        <code>TrainingConfig</code>, <code>TrainingRun</code> and <code>run_training</code> are owned by
        <code> alignair.train</code> and exported from the package root. A run resolves the dataconfigs to an ordered
        reference, prepares and builds the model, streams curriculum records, trains, validates and writes
        checkpoints. The short run above exercises the workflow; it does not make a usable model. With
        <code> data_workers</code> above 0, worker processes are spawned: call <code>run_training</code> under an
        <code> if __name__ == &quot;__main__&quot;:</code> guard.
      </p>
      <H2 id="spaces">Input and label spaces</H2>
      <p>
        AlignAIR trains on nucleotide reads with allele labels: <code>input_space=&quot;nucleotide&quot;</code> and
        <code> label_space=&quot;alleles&quot;</code> are the only values, and any other raises
        <code> ConfigurationError</code> (&ldquo;AlignAIR trains on nucleotide reads with allele labels&rdquo;). The
        input window defaults to 576 positions. For a custom model, use <DocLink to="model-setup">model setup</DocLink>
        with the <DocLink to="training-engine">training engine</DocLink>.
      </p>
      <H2 id="request">Configure a run</H2>
      <DocTable head={["Request", "Default", "Contract"]} rows={[
        ["dataconfigs / from_genairr", "required", "One or more built-in GenAIRR names or DataConfig objects; the ordered reference derives the allele and locus counts."],
        ["max_seq_length / has_d", "None / None", "Optional window and D-head selection; None uses 576 and the reference's D."],
        ["model_options", "{}", "Model configuration fields such as architecture, embed_dim or trunk_channels. Copied on construction; reference counts and input/label choices cannot be overridden here."],
        ["steps / batch_size / lr / grad_clip", "100,000 / 64 / 3e-4 / None", "Update budget, batch size, learning rate and gradient clipping. Invalid values raise ConfigurationError (or its subclass TrainingConfigError, from the preflight) before the model is allocated."],
        ["val_every", "0", "0 disables validation. A positive interval validates on 4 fixed held-out batches drawn from validation seed blocks and keeps the best checkpoint."],
        ["seed / init_seed", "0 / None", "A uint32 seed for the process RNGs, the record streams and, when init_seed is None, the initial weights. An init_seed draws the initial weights from its own seed and leaves the batches unchanged."],
        ["device", "\"auto\"", "CUDA, then MPS, then CPU."],
        ["short_boost", "1", "Repeats of the three amplicon profiles in the curriculum mix."],
        ["flanks", "FlankPolicy()", "Library flanks around the curriculum's reads; None trains on the bare simulated reads."],
        ["novel_alleles", "None", "A NovelAlleles draws part of the reads from synthetic novel V alleles; the model and saved reference stay the dataconfigs'."],
        ["data_workers", "0", "Generate batches in this many worker processes (at most one per curriculum stream: 28 for the default mix). The batches are those of data_workers=0."],
        ["resume_path", "None", "Continue a checkpoint: weights, optimizer, scheduler and process RNG state; the configuration and label schema must agree."],
        ["init_from", "None", "A weights-only warm start from another .alignair model, with fresh optimizer and scheduler. Mutually exclusive with resume_path."],
        ["schedule", "None", "None keeps lr constant. CosineSchedule(total_steps, min_lr=1e-5) steps after each successful update."],
        ["study", "None", "A TrainingStudy with absolute checkpoint steps and an optional evaluation callback."],
      ]} />
      <p>
        <code>preset=</code> selects step, batch and validation defaults; explicit fields override it. Use
        <code> request.with_(...)</code> to construct a changed request. Model options select
        <DocLink to="model-composition"> registered components</DocLink>; a custom architecture must be registered
        before its model is built or reloaded.
      </p>
      <DocTable head={["Preset", "steps", "batch_size", "val_every"]} rows={[
        [<code>quick</code>, "300", "16", "100"],
        [<code>desktop</code>, "50,000", "64", "2,000"],
        [<code>full</code>, "300,000", "128", "5,000"],
      ]} />
      <p>
        A warm start reads the source&apos;s weights from its pickle-free sections without constructing its model and
        copies every tensor whose name and shape match; the rest keep their initialization. The chip&apos;s weights do
        not depend on its reference, so a source of the same widths transfers whole even when its reference differs.
        A source that is not an <code>.alignair</code> file is refused, and an AlignAIR 3 <code>.pt</code> is named as
        such.
      </p>
      <H2 id="study">Retain and evaluate study checkpoints</H2>
      <CodeBlock lang="python" code={`from alignair.train import CosineSchedule, TrainingStudy

studied = request.with_(
    steps=20, grad_clip=1.0,
    schedule=CosineSchedule(total_steps=40, min_lr=1e-5),
    study=TrainingStudy(checkpoint_dir="runs/study/selected", checkpoint_steps=(5, 10, 20)),
)
study_run = run_training(studied, output_dir="runs/study")
print(study_run.checkpoints)   # actual step -> retained .alignair checkpoint`} />
      <p>
        A <code>TrainingStudy</code> saves the requested absolute steps as <code>checkpoint-&lt;step&gt;.alignair</code>
        in <code>checkpoint_dir</code>, separately from the rotating checkpoints. <code>TrainingRun.checkpoints</code>
        lists only files written during that invocation. Each records its actual step, label identity, seed, policies
        and, for a continued run, the source checkpoint&apos;s SHA-256, alongside optimizer, scheduler and process RNG
        state.
      </p>
      <p>
        For external evaluation, supply <code>validation_every</code> and an <code>evaluate(model, step)</code> callback
        returning a metric mapping. It runs in evaluation mode, with the previous mode restored even on failure, and
        <code> TrainingRun.validations</code> keeps the mappings. An optional <code>plateau_window</code> prints a notice
        when the mapping&apos;s <code>objective</code> has not improved over that many evaluations; it does not stop
        training or select a model. Set the cosine horizon to cover every scheduler update, including later
        continuation.
      </p>
      <H2 id="observe">Observe successful updates</H2>
      <CodeBlock lang="python" code={`def observe(iteration):
    print(iteration.step, iteration.result.total, iteration.result.parts)

run = run_training(request, output_dir="runs/observed", on_iteration=observe)`} />
      <p>
        The callback receives a <code>TrainingIteration</code> after each successful optimizer update, before logging,
        validation and checkpoint work, with the detached loss and gradient results and the source records. No history
        is accumulated unless the callback collects it; callback failures propagate. With <code>data_workers</code> above
        0 on an accelerator, batches are assembled ahead from the CPU random state, so a callback that draws from the
        CPU random generators raises <code>TrainingError</code>; use <code>data_workers=0</code> for such a callback.
      </p>
      <H2 id="reload">Checkpoint, resume, and reload</H2>
      <DocTable head={["File in the output directory", "Written"]} rows={[
        [<code>model.step&lt;N&gt;.alignair</code>, "Every 2,000 steps and at the last completed step; the newest 3 are kept."],
        [<code>model.alignair</code>, "A copy of the newest numbered checkpoint: TrainingRun.model_path."],
        [<code>model.best.alignair</code>, "When validation ran: the checkpoint with the highest selection score (the mean scan_{gene}_top1). TrainingRun.best_model_path, else None."],
      ]} />
      <CodeBlock lang="python" code={`continued = run_training(
    request.with_(steps=20, resume_path=run.model_path),
    output_dir="runs/continued",
)
aligner = continued.best_aligner(device="cpu")   # the best, else the last, checkpoint`} />
      <p>
        <code>TrainingRun.steps</code> records the actual completed step. Checkpoints hold the configuration, weights,
        the reference and the label schema, plus trusted pickle sections (<code>dataconfig/0</code> and
        <code> train_state</code>) for resume. <code>resume_path</code> resolves to the newest numbered checkpoint
        beside it, else to the file itself. Resume checks the configuration, the exact label schema and the schedule before loading anything,
        restores weights, optimizer moments (the learning rate is reset to the requested one unless a schedule is
        set), scheduler phase and process RNG state, then trains to the absolute <code>steps</code>; a target not
        beyond the checkpoint step raises <code>ValueError</code>. Record streams restart from seed blocks of their own,
        so continuation does not replay the next batch of an uninterrupted run. To change the model or loss settings
        while keeping weights, use <code>init_from</code>.
      </p>
      <p>
        The saved <code>label_schema</code> records the input space, label space and each gene&apos;s ordered allele
        names. Loading validates it against the embedded reference; resume compares it exactly, so heads of the same
        size cannot be continued with different alleles.
      </p>
      <H2 id="rng-state">Capture and restore process RNG state</H2>
      <CodeBlock lang="python" code={`from alignair.train.random_state import capture_rng_state, restore_rng_state

state = capture_rng_state(include_cuda=False)
# Restore the saved generators when continuing CPU training.
restore_rng_state(state, include_cuda=False, require_process_state=True)`} />
      <p>
        The state dictionary holds the Python, NumPy and PyTorch CPU generators, and by default every CUDA generator
        when CUDA is available. Set <code>include_cuda=False</code> on both calls to exclude CUDA. Restore accepts
        sparse dictionaries and <code>None</code> by default; <code>require_process_state=True</code> requires non-null
        Python, NumPy and PyTorch CPU states before anything is applied. Invalid state raises
        <code> TrainingError</code> and rolls the process generators back to their values before the call. Local
        <code> random.Random</code> and NumPy <code>Generator</code> instances have their own state.
      </p>
      <H2 id="export">Export a distributable model</H2>
      <CodeBlock lang="python" code={`import GenAIRR as ga
from alignair.train.build import export_bundle

model_path = export_bundle(
    run.best_model_path or run.model_path, [ga.HUMAN_IGH_OGRDB], "runs/igh/bundle",
)`} />
      <p>
        <code>run_training</code> writes training checkpoints only. <code>alignair.train.build.export_bundle</code>
        writes a pickle-free bundle from a checkpoint: <code>model.alignair</code> with no pickle sections,
        <code> model_card.md</code>, <code>reference_manifest.json</code> and, unless <code>validate=False</code>,
        <code> validation_report.json</code> (the task metrics on a fixed held-out stream). The training provenance is
        read from the checkpoint. It refuses an existing bundle directory unless <code>overwrite=True</code>.
        <code> alignair train</code> calls it on the best (or last) checkpoint, into <code>&lt;out&gt;/bundle/</code>.
      </p>
      <p>
        <code>Aligner.save_pretrained</code> and <code>push_to_hub</code> refuse a file with pickle sections, so publish
        the exported <code>model.alignair</code>. <code>alignair convert</code> converts lab chip model files only
        (<code>lab-chip-v1</code>, <code>lab-chipw64-v1</code>), not a training checkpoint. The
        <DocLink to="publishing"> publishing runbook</DocLink> continues from the bundle.
      </p>
      <H2 id="predictions">Predict from a trained model</H2>
      <p>
        <code>Aligner.predict</code> returns AIRR records; <code>Aligner.predict_raw</code> returns the model&apos;s raw
        output arrays (allele scores relative to each read&apos;s best allele, model-frame positions) from the same
        forward pass, without AIRR assembly. <code>label_schema</code> names each allele column. Both are described in
        the <DocLink to="python-api">Python API</DocLink>.
      </p>
    </>
  ),
};

export const trainingWorkflowPages = [trainingWorkflow];
