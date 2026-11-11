import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const modelSetup: DocPage = {
  slug: "model-setup",
  title: "Model setup API",
  section: "Design",
  lead: "Bind an ordered nucleotide reference to a model configuration before allocating an AlignAIR model.",
  body: () => (
    <>
      <p>
        <code>alignair.core.prepare_model</code> returns a <code>ModelSetup</code> holding the reference, the
        configuration and the input and label spaces. It derives the allele counts, the D head and the locus count
        from that reference. No parameters are allocated until <code>build_model()</code>, which builds the model and
        binds the setup&apos;s reference to it.
      </p>
      <H2 id="prepare">Prepare and build</H2>
      <CodeBlock lang="python" code={`import GenAIRR as ga
from alignair.core import prepare_model
from alignair.reference import ReferenceSet
from alignair.tasks import build_tasks

reference = ReferenceSet.from_dataconfigs(ga.HUMAN_IGH_OGRDB)
setup = prepare_model(reference)
print(setup.label_counts)          # {'v': 198, 'd': 33, 'j': 7}
print(setup.config.architecture)   # chip-v1

# Run training-request preflight against setup.config and setup.reference here.
model = setup.build_model()        # allocated and bound to setup.reference
tasks = build_tasks(setup.config)  # the chip's objective (ChipObjective)`} />
      <p>
        Every setup builds the same <code>AlignAIR</code> class; the configuration&apos;s architecture name selects its
        components through the <DocLink to="model-composition">model composition API</DocLink>. The factory accepts a
        <code> ReferenceSet</code>. Resolving a simulator name or reading reference files belongs to the provider;
        preparation from an existing reference does not require GenAIRR.
      </p>
      <H2 id="policies">Input and label spaces</H2>
      <DocTable head={["Input / labels", "Defaults", "Batch preparation"]} rows={[
        ["nucleotide / alleles", "6-token vocabulary (pad, A, T, G, C, N), in-model orientation with the bank's k-mer vote, the chip core, anchor heads, and a D presence head when the reference has D; window 576.",
          <><code>NucleotideBatchAdapter</code>, see the <DocLink to="training-batches">batch API</DocLink>.</>],
      ]} />
      <p>
        AlignAIR reads nucleotide sequences and labels them with the reference&apos;s alleles; this is the only
        combination. Any other <code>input_space</code> raises <code>ConfigurationError</code> (&ldquo;input_space must
        be nucleotide: AlignAIR reads nucleotide sequences&rdquo;), and any other <code>label_space</code> raises
        &ldquo;label_space must be alleles&rdquo;. The two fields remain on the configuration and in the saved label
        schema.
      </p>
      <H2 id="labels">Labels and the bound reference</H2>
      <p>
        Allele labels keep the reference&apos;s order and indices. <code>has_d=False</code> omits the D head without
        deleting or reordering the source reference; <code>has_d=True</code> on a reference without D raises
        <code> ConfigurationError</code> (&ldquo;requested D head is unavailable in the source reference&rdquo;). With
        the default <code>has_d=None</code> the reference decides. The locus count comes from the reference&apos;s
        ordered locus schema, and is at least one.
      </p>
      <p>
        The reference is an input, not part of the weights: the chip compiles the bound reference into a bank (one row
        per allele) and scores every row. <code>build_model()</code> binds the setup&apos;s reference; a model can be
        bound to another reference with the same genes later, with <code>model.bind_reference(reference)</code>, and
        its configuration&apos;s allele counts then follow the bound reference. A model whose reference has not been
        bound refuses to run (&ldquo;bank_orientation reads a reference: bind one with model.bind_reference&rdquo;).
        For prediction against a donor genotype or novel alleles, use <code>Aligner.bound_to</code> or the
        <code> reference=</code> argument described in the <DocLink to="python-api">Python API</DocLink>.
      </p>
      <H2 id="options">Configure components</H2>
      <CodeBlock lang="python" code={`setup = prepare_model(
    reference,
    embed_dim=16,         # token/position embedding width (default 32)
    trunk_channels=32,    # the chip's trunk width (default 64, the trained width)
)`} />
      <p>
        Keyword options set <code>AlignAIRConfig</code> fields: <code>architecture</code>, <code>embed_dim</code>,
        <code> trunk_channels</code>, <code>max_seq_length</code>, <code>use_orientation</code>,
        <code> bank_orientation</code>, <code>anchor_heads</code>, <code>d_presence_head</code> and
        <code> reject_empty_sequences</code>. Unknown options raise <code>ConfigurationError</code> (&ldquo;invalid model
        configuration options&rdquo;); attempts to override the reference-derived counts raise &ldquo;reference label
        and locus counts cannot be overridden&rdquo;. The configuration checks its own combinations:
        <code> d_presence_head=True</code> without a D, or <code>bank_orientation=True</code> without
        <code> use_orientation</code>, raises <code>ConfigurationError</code>. A reference whose allele order, sequences
        and index disagree raises <code>DataContractError</code>.
      </p>
      <H2 id="binding">Validate a reference binding</H2>
      <CodeBlock lang="python" code={`from alignair.core import validate_reference_binding

validate_reference_binding(reference, setup.config)

# With a saved label schema, the allele order must match it too.
schema = setup.label_schema()
validate_reference_binding(reference, setup.config, label_schema=schema)`} />
      <p>
        This preflight checks, without allocating parameters, that the configuration&apos;s D selection agrees with the
        reference, that each head&apos;s allele count equals the reference&apos;s (&ldquo;model head size disagrees
        with its reference labels&rdquo;, with the gene and both counts in the context), that a disabled D head has zero
        labels, the locus count, the sequence window and the nucleotide vocabulary. It also checks the
        reference&apos;s names, sequences and index ordering. Preparation, saving and loading share these checks.
      </p>
      <p>
        <code>ModelSetup.label_schema()</code> returns the portable column identity saved with a model:
        <code> {"{'schema_version': 1, 'input_space', 'label_space', 'genes': {gene: allele names}}"}</code>. A schema
        whose genes, lengths or allele order disagree with the reference raises <code>ConfigurationError</code>
        (&ldquo;allele label order disagrees with the source reference&rdquo;). Invalid bindings stored in a model file
        raise <code>ModelIntegrityError</code> at load, with the <code>model_binding</code> stage, the path and the
        original cause; the stored <code>allele_order_sha256</code> and <code>reference_fasta_sha256</code> are
        verified first.
      </p>
      <H2 id="restore">Restore decoded model weights</H2>
      <CodeBlock lang="python" code={`from alignair.core import AlignAIR

# config and state_dict have already been decoded and their bindings checked.
model = AlignAIR.from_state_dict(config, state_dict, device="cpu", source="models/candidate.alignair")
model.bind_reference(reference)    # from_state_dict does not bind a reference`} />
      <p>
        <code>AlignAIR.from_state_dict</code> constructs the configured architecture on CPU, loads its state with
        strict key and shape checks, then moves the model to the requested device and selects evaluation mode.
        Parameters keep their gradients for fine-tuning. Construction uses the components registered for
        <code> config.architecture</code>; the factory does not infer an architecture from weight keys. The bank is not
        part of the weights (its buffers are not saved), so bind a reference before running the model.
      </p>
      <p>
        Missing or unexpected keys and incompatible shapes raise <code>ModelIntegrityError</code> (&ldquo;cannot
        restore model weights&rdquo;) with a <code>model_restoration</code> stage, the architecture and the diagnostic
        source path; the PyTorch error remains the exception&apos;s cause. An unregistered architecture raises
        <code> ConfigurationError</code> (&ldquo;unknown model architecture&rdquo;). For normal inference, use the
        <DocLink to="model-inference"> public loading APIs</DocLink>; this factory accepts decoded state only.
      </p>
      <H2 id="prepared-restoration">Restore through a prepared binding</H2>
      <CodeBlock lang="python" code={`from alignair.core import setup_from_schema

# Bind the saved column order before constructing a restored model.
restored_setup = setup_from_schema(reference, config, saved_schema)
model = restored_setup.restore_model(state_dict, device="cpu", source="models/candidate.alignair")`} />
      <p>
        <code>setup_from_schema</code> checks the saved schema against the reference and configuration and returns a
        <code> ModelSetup</code>. <code>ModelSetup.restore_model</code> revalidates the binding (failures carry the
        <code> model_binding</code> stage), restores a fresh model through <code>AlignAIR.from_state_dict</code> and
        binds the setup&apos;s reference. It accepts decoded state and does not read the source path.
      </p>
      <H2 id="checkpoint-data">Read model files without constructing a model</H2>
      <CodeBlock lang="python" code={`from alignair.model_file import read_model_state

data = read_model_state("model.alignair")
print(data.config.architecture, len(data.model_state))
reference = data.reference
schema = data.metadata["model"]["label_schema"]`} />
      <p>
        <code>read_model_state</code> returns <code>ModelStateData</code> (<code>metadata</code>, <code>config</code>,
        <code> model_state</code>, <code>reference</code>): CPU tensors from the safetensors section, the reference from
        the pickle-free <code>reference_json</code> section, and a verified reference and label binding. It decodes no
        training state, needs no architecture factory and consumes no random state; key and shape checks happen at
        restoration. A file without <code>reference_json</code> was written before AlignAIR 4 and raises
        <code> ModelError</code>.
      </p>
      <CodeBlock lang="python" code={`from alignair.model_file import read_training_state

data = read_training_state("runs/igh/model.alignair", device="cpu")
print(data.step, data.config.architecture)
optimizer_state = data.optimizer_state
scheduler_state = data.scheduler_state`} />
      <p>
        <code>read_training_state</code> returns <code>TrainingStateData</code> with the decoded model tensors (mapped
        to <code>device</code>), reference, configuration, metadata, optimizer, scheduler and process RNG state, the
        completed step and the run&apos;s <code>train_args</code>. It verifies section checksums and the reference and
        label binding without constructing <code>AlignAIR</code>, and does not apply the recorded RNG state. It decodes
        the trusted pickle <code>train_state</code> section, so open only checkpoints you trust; an inference model
        file raises <code>ModelError</code> (&ldquo;no train_state section; this model file cannot resume
        training&rdquo;).
      </p>
      <p>
        Use <code>ModelSetup.restore_model</code> for a fresh reference-bound model, or
        <code> restore_model_state(existing_model, data.model_state, source=...)</code> to load the tensors strictly into
        an existing instance. In-place restoration keeps module modes and parameter objects, including optimizer
        bindings; it follows PyTorch copying semantics rather than rolling back a failed load.
        <code> load_training_state(path, device=...)</code> returns a <code>TrainingState</code> whose model is
        already restored, bound to the checkpoint&apos;s reference and in evaluation mode.
      </p>
      <H2 id="counts">Count-only construction</H2>
      <p>
        When only validated counts are available, <code>build_model_config(v_label_count, j_label_count,
        d_label_count=0, *, has_d=False, num_chain_types=1, max_seq_length=None, ...)</code> selects the same defaults
        and returns an <code>AlignAIRConfig</code>. It cannot verify allele identity or source ordering, so prefer
        <code> prepare_model</code> when a reference is available. <code>ModelSetup</code> is a preparation object, not
        an inference bundle or a resume checkpoint. The <DocLink to="training-engine">training engine</DocLink> and
        runner own optimizer state, record streams, checkpoint timing and resume.
      </p>
    </>
  ),
};

export const setupPages = [modelSetup];
