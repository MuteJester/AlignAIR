import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const modelSetup: DocPage = {
  slug: "model-setup",
  title: "Model setup API",
  section: "Design",
  lead: "Bind an ordered reference to explicit input and label policies before allocating an AlignAIR model.",
  body: () => (
    <>
      <p>
        <code>alignair.core.prepare_model</code> returns a <code>ModelSetup</code> containing the
        reference, configuration, selected input and label spaces, and an optional protein reference
        view. It derives label counts and locus count from that reference. Model parameters are
        allocated only when you call <code>build_model()</code>.
      </p>
      <H2 id="prepare">Prepare and build</H2>
      <CodeBlock lang="python" code={`import GenAIRR as ga
from alignair.core import prepare_model
from alignair.reference import ReferenceSet
from alignair.tasks import build_tasks

reference = ReferenceSet.from_dataconfigs(ga.HUMAN_IGH_OGRDB)
setup = prepare_model(
    reference,
    input_space="protein",
    label_space="protein_classes",
    max_seq_length=192,
    architecture="conv-v1",
)
print(setup.label_counts)  # V/J protein-class counts; D remains an allele count

# Run training-request preflight against setup.config and setup.reference here.
model = setup.build_model()
tasks = build_tasks(setup.config)
weights = tasks.make_weights()`} />
      <p>
        All choices build the same <code>AlignAIR</code> class. Registered architectures select its
        components through the <DocLink to="model-composition">model composition API</DocLink>.
        The factory accepts a <code>ReferenceSet</code> or <code>ProteinReference</code>. Resolving a
        simulator name or reading reference files belongs to the provider; preparation from an existing
        reference does not require GenAIRR.
      </p>
      <H2 id="policies">Input and label policies</H2>
      <DocTable head={["Input / labels", "Established defaults", "Batch preparation"]} rows={[
        ["nucleotide / alleles", "6-token vocabulary, orientation correction, multilabel scores, residue boundary masks, auxiliary heads; window 576.",
          <><code>NucleotideBatchAdapter</code>.</>],
        ["protein / alleles", "23-token vocabulary, orientation disabled, multilabel scores, historical residue boundary masks and auxiliary heads; window 192.",
          <><code>ProteinAlleleBatchAdapter</code>.</>],
        ["protein / protein_classes", "23-token vocabulary, orientation disabled, categorical scores, half-open boundaries, auxiliary heads disabled, empty reads rejected; window 192.",
          <>A <code>ProteinSampleBuilder</code> followed by <code>CategoricalBatchAdapter</code>.</>],
        ["nucleotide / protein_classes", "6-token vocabulary, orientation correction, categorical scores, half-open boundaries, auxiliary heads disabled, empty reads rejected; window 576.",
          "Requires a custom source adapter with nucleotide-coordinate boundaries and reference protein-class labels."],
      ]} />
      <p>
        Input space controls the vocabulary and default orientation policy; label space controls the
        default output-head semantics and label counts. Boundary positions always use the input’s
        coordinate unit. The existing protein-allele defaults remain unchanged. Selecting a different
        boundary or head policy is explicit, through configuration options. See the
        <DocLink to="training-batches"> batch API</DocLink> and
        <DocLink to="training-tasks"> task API</DocLink> for the corresponding target requirements.
        Model construction alone does not supply an adapter for every combination.
      </p>
      <H2 id="labels">Preserve label identity</H2>
      <p>
        Allele labels retain the source order and indices. Protein-class V/J labels come from the
        shared <DocLink to="reference-labels">protein reference view</DocLink>; D retains source
        allele labels. <code>has_d=False</code> omits the D head without deleting or reordering the
        source reference. Enabling D requires a D-bearing source with a nonempty vocabulary.
        Locus count comes from the reference’s ordered locus schema, defaulting to one when absent.
      </p>
      <CodeBlock lang="python" code={`from alignair.reference import build_protein_reference

# Explicit reference policy for a light-chain source with conserved J anchor F.
light_reference = ReferenceSet.from_dataconfigs(ga.HUMAN_IGK_OGRDB)
protein_reference = build_protein_reference(light_reference, anchor_residues={"J": "F"})
setup = prepare_model(
    protein_reference, input_space="protein", label_space="protein_classes",
)
assert setup.reference is protein_reference.source
assert setup.protein_reference is protein_reference`} />
      <p>
        A supplied protein view preserves its anchor and hash-scope policy. Class maps must be valid,
        refer to source alleles, and agree with that view’s translated equivalence classes. A prebuilt
        view keeps its class IDs; validation compares allele groups without imposing the factory’s
        default numbering. Preparation
        checks ordered names, sequences, and allele indices through <code>ReferenceSet.validate()</code>.
        Binding checks run again before model allocation. Keep reference and label maps stable after
        fitting; changing label identity requires a corresponding model and training setup.
      </p>
      <H2 id="options">Configure components and heads</H2>
      <CodeBlock lang="python" code={`setup = prepare_model(
    reference, input_space="protein", label_space="protein_classes",
    embed_dim=16, filters=64, block_out=256,
    use_meta_heads=True,  # supply the corresponding auxiliary targets in your adapter
)`} />
      <p>
        Known <code>AlignAIRConfig</code> options select registered architectures, dimensions, and
        optional heads. Unknown options and attempts to override reference-derived label or locus
        counts raise <code>ConfigurationError</code>. Vocabulary must match the selected input;
        nucleotide orientation correction is unavailable for protein input. Invalid reference ordering
        raises <code>DataContractError</code>. Component-specific shape constraints still apply during
        model construction.
      </p>
      <H2 id="counts">Count-only construction and existing callers</H2>
      <p>
        When only validated counts are available, <code>build_model_config(v_label_count,
        j_label_count, d_label_count=0, ...)</code> selects the same defaults and returns an
        <code> AlignAIRConfig</code>. It cannot verify class identity or source ordering. The historical
        <code> aa_igh_config</code> factory delegates to this implementation. Protein setup helpers,
        public nucleotide training entries, and the reference training plan use the shared setup path.
      </p>
      <p>
        Existing setup tuple layouts, configuration serialization, and weight keys are preserved.
        Protein setup helpers now return the shared <code>AlignAIR</code> model; <code>AAAlignAIR</code>
        remains a compatibility constructor for callers that still use it.
        <code>ModelSetup</code> is a preparation object, not an inference bundle or a resume checkpoint.
        Protein class maps must accompany their weights through the existing label-aware export path.
        The selected <DocLink to="training-engine">training engine</DocLink> and runner continue to own
        optimizer state, source streams, checkpoint timing, and resume behavior.
      </p>
    </>
  ),
};

export const setupPages = [modelSetup];
