import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const trainingWorkflow: DocPage = {
  slug: "training-workflow",
  title: "Training workflow API",
  section: "Training",
  lead: "Build, train, validate, checkpoint, and reload nucleotide and protein models through one public workflow.",
  body: () => (
    <>
      <CodeBlock lang="python" code={`from alignair import TrainingConfig, run_training

request = TrainingConfig.from_genairr(
    "HUMAN_IGH_OGRDB",
    input_space="protein", label_space="protein_classes",
    steps=10, batch_size=16, val_every=5, seed=7, device="cpu",
    model_options={"embed_dim": 8, "filters": 8, "block_out": 32},
)
run = run_training(request, output_dir="runs/protein_classes")
predictor = run.best_aligner(device="cpu")
outputs = predictor.predict_raw(["EVQLVESGGGLVQPGGSLRLSCAAS"])
print(outputs["v_class_logits"].shape)
print(predictor.label_schema["genes"]["v"][0])  # alleles represented by class 0`} />
      <p>
        <code>TrainingConfig</code>, <code>TrainingRun</code>, and <code>run_training</code> are
        owned by <code>alignair.train</code> and exported from the package root. Each supported
        configuration uses the same model class, runner, task interface, optimizer engine, validation
        loop, and checkpoint format. The input and label choices select existing source preparation
        and batch adapters. The short run above exercises the workflow; it does not establish model accuracy.
      </p>
      <H2 id="spaces">Choose input and label spaces</H2>
      <DocTable head={["input_space", "label_space", "Training targets"]} rows={[
        ["nucleotide", "alleles", "Nucleotide positions and source allele sets, with orientation and configured auxiliary tasks."],
        ["protein", "alleles", "Projected residue positions and source nucleotide allele sets; synonymous alleles retain separate labels."],
        ["protein", "protein_classes", "Half-open residue boundaries and V/J protein-equivalence classes; optional D retains source allele labels."],
      ]} />
      <p>
        Defaults remain nucleotide inputs and allele labels. Input windows default to 576 nucleotide
        positions or 192 protein positions. Protein-class batches reserve a terminal boundary slot,
        so a 192-position window accepts at most 191 observed residues. The protein-class preset
        uses the existing translated-reference and conserved-anchor policies; sources must provide
        usable V/J translations. For additional input/label combinations or custom reference views,
        use <DocLink to="model-setup">model setup</DocLink> with explicit adapters and the
        <DocLink to="training-engine">training engine</DocLink>.
      </p>
      <H2 id="request">Configure a run</H2>
      <DocTable head={["Request", "Contract"]} rows={[
        ["dataconfigs / from_genairr", "One or more built-in GenAIRR names or DataConfig objects. The ordered reference derives label and locus counts."],
        ["input_space / label_space", "Explicit model semantics, serialized with the model configuration."],
        ["max_seq_length / has_d", "Optional window and D-head selection. None uses the input/reference defaults; omitting D retains the source reference."],
        ["model_options", "Known model settings, including registered architecture and dimensions. Copied on construction; reference counts and input/label choices cannot be overridden here."],
        ["steps / batch_size / lr / grad_clip", "Update budget, batch size, learning rate and clipping. Invalid requests fail before allocating the model or creating output directories."],
        ["val_every", "Zero disables periodic validation. A positive interval selects the best checkpoint using the shared task metrics on fixed held-out batches."],
        ["seed", "A uint32 seed for model initialization, process RNGs and source construction. Fixed-seed CPU runs are reproducible under the same environment."],
        ["resume_path", "An existing checkpoint. Restore model, task weights, optimizer and process RNG state; configuration and label identity must agree."],
      ]} />
      <p>
        <code>preset="quick"</code>, <code>"desktop"</code>, or <code>"full"</code> selects
        the established step, batch and validation defaults. Explicit fields override the preset.
        Use <code>request.with_(...)</code> to construct a changed request. Model options select
        <DocLink to="model-composition">registered components</DocLink>; custom architectures
        must be registered before constructing or reloading their models.
      </p>
      <p>
        The allele presets use their existing nucleotide/protein adapters. Protein-class training uses
        <code>ProteinSampleBuilder</code> with <code>require_productive=False</code> and
        <code>allow_stops=True</code>. Unusable translations, unknown V/J classes, unavailable
        boundaries, and overlength class samples are skipped; invalid D supervision is masked.
        A prolonged run of rejected class samples raises a training-data error. Presets reject head
        combinations for which their adapters do not supply targets, including protein edit-state
        targets and protein-class auxiliary heads. Custom targets belong in an explicit batch adapter.
      </p>
      <H2 id="observe">Observe successful updates</H2>
      <CodeBlock lang="python" code={`def observe(iteration):
    print(iteration.step, iteration.result.total, iteration.result.parts)

run = run_training(request, output_dir="runs/observed", on_iteration=observe)`} />
      <p>
        The callback receives a <code>TrainingIteration</code> after each successful optimizer update,
        before logging, validation and checkpoint work. It includes detached loss/gradient results
        and source records. No history is accumulated unless the callback chooses to collect it.
        Callback failures propagate. Use shared model
        <DocLink to="model-inference">component observation</DocLink> and the lower-level engine
        when a custom experiment needs direct model access during training.
      </p>
      <H2 id="reload">Checkpoint, resume, and reload</H2>
      <CodeBlock lang="python" code={`continued = run_training(
    request.with_(steps=20, resume_path=run.model_path),
    output_dir="runs/continued",
)
loaded = continued.best_aligner(device="cpu")
raw = loaded.predict_raw(["EVQLVESGGGLVQPGGSLRLSCAAS"])
labels = loaded.label_schema["genes"]["v"]`} />
      <p>
        <code>TrainingRun.model_path</code> is the final checkpoint;
        <code>best_model_path</code> is present when validation selected a best model.
        <code>steps</code> records the actual completed step. Checkpoints retain the safe model
        configuration, weights and source reference, plus trusted Python training state for resume.
        Inference reload uses the safe model/reference sections. Resume restores the process RNGs
        and starts the source under the existing seed-plus-step policy; it does not restore a simulator cursor.
      </p>
      <p>
        The saved <code>label_schema</code> records input space, label space and ordered output
        columns. Allele labels retain source order; protein-class V/J columns store explicit allele
        member lists. Loading validates schema shape and source membership. Resume also checks the
        exact schema, preventing the same-sized heads from being continued with different label IDs.
      </p>
      <H2 id="predictions">Use the appropriate prediction interface</H2>
      <p>
        <code>Aligner.predict_raw</code> returns NumPy arrays for all model heads through the same
        interface for either input space. Keys preserve score semantics: allele probabilities versus
        categorical logits, plus model-frame boundaries and enabled auxiliary outputs. Tokenization
        follows the saved input space. Protein raw input is strict about alphabet and length;
        nucleotide raw input uses the shared nucleotide preparation policy.
      </p>
      <p>
        Nucleotide allele models with residue-boundary heads additionally support
        <code> Aligner.predict</code> and AIRR result assembly. Other input/head profiles use raw
        outputs or their task-specific result processing.
        Raw outputs do not include product calibration, conformal sets or applicability decisions.
        The calibrated protein product loader remains a separate artifact contract; a new training
        checkpoint does not acquire those calibration assets automatically.
      </p>
      <p>
        <code>alignair.train.build.export_bundle</code> validates the selected input/label profile
        and writes an inference artifact without Python training sections. It preserves the label
        schema and raw predictions. Conversion of a self-contained checkpoint preserves that schema
        as well. The existing CLI training command remains the nucleotide bundle workflow; select
        protein training through the Python request above.
      </p>
    </>
  ),
};

export const trainingWorkflowPages = [trainingWorkflow];
