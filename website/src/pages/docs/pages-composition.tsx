import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const composition: DocPage = {
  slug: "model-composition",
  title: "Model composition API",
  section: "Design",
  lead: "How the AlignAIR model is assembled from an embedding, an orientation head and the chip core, and how to register a variant.",
  body: () => (
    <>
      <p>
        <code>alignair.core.AlignAIR</code> owns model assembly and the forward pass. Its configuration&apos;s
        <code> architecture</code> name selects a registered <code>ModelComponents</code>; <code>chip-v1</code> is the
        only built-in architecture. The tokenizer fixes the vocabulary and the bound reference fixes the allele
        counts; the model does not infer either from its input. The design and its units are described on the
        <DocLink to="design"> design page</DocLink>.
      </p>
      <H2 id="package-ownership">Package ownership</H2>
      <DocTable head={["Responsibility", "Package"]} rows={[
        ["Model assembly, configuration, component registry, reference-bound setup", "alignair.core"],
        ["The chip's units: trunk, parser, anchor heads, SHM reader, bank compiler, bank scans, orientation, objective", "alignair.network"],
        ["Deterministic refinement at inference: one locus per read, boundaries, D local alignment, V re-score", "alignair.refine"],
        ["The training-task protocol and support-weighted metrics", "alignair.tasks"],
        ["Training adapters, chip targets and inputs, tokenizer", "alignair.data"],
        ["Training engine, runner, checkpoints and record streams", "alignair.train"],
        ["Prediction pipeline and AIRR assembly", "alignair.predict"],
      ]} />
      <H2 id="construction">Construct a model</H2>
      <CodeBlock lang="python" code={`import GenAIRR as ga
from alignair.core import AlignAIR, AlignAIRConfig
from alignair.reference import ReferenceSet

reference = ReferenceSet.from_dataconfigs(ga.HUMAN_IGH_OGRDB)
config = AlignAIRConfig.from_reference(reference)   # counts and has_d from the reference
model = AlignAIR(config)
model.bind_reference(reference)                      # the bank: required before a forward pass`} />
      <p>
        The <DocLink to="model-setup">model setup API</DocLink> does the same through <code>prepare_model</code>, with
        binding checks before parameters are allocated. The model accepts a dictionary containing
        <code> tokenized_sequence</code>: an integer tensor of shape <code>(batch, max_seq_length)</code>, right-padded
        with token 0. Each model owns a deep copy of its construction configuration, so editing the caller&apos;s
        configuration afterwards changes nothing. Treat structural settings as construction inputs: use
        <code> dataclasses.replace</code> and build a new model to change widths or the architecture.
      </p>
      <H2 id="choices">Configuration</H2>
      <DocTable head={["Setting", "Default", "Behavior"]} rows={[
        [<code>architecture</code>, <code>"chip-v1"</code>, "The registered components to build; stored in the model file."],
        [<code>embed_dim</code>, "32", "Width of the learned token-plus-position embedding."],
        [<code>trunk_channels</code>, "64", "The chip trunk's channels per position; chip-v1 is trained at 64."],
        [<code>max_seq_length</code>, "576", "The model window. Position embeddings and every per-position output are this wide."],
        [<code>use_orientation</code>, "True", "Builds the 4-class orientation head and corrects each read in the model."],
        [<code>bank_orientation</code>, "True", "At inference, orients a read by the bound reference: the transform under which most of its 12-mers are germline 12-mers; the orientation head decides when none wins. Requires use_orientation."],
        [<code>anchor_heads</code>, "True", "The V Cys and J Trp/Phe anchor heads; the junction, frame and productivity are derived from them."],
        [<code>d_presence_head</code>, "has_d", "The 3-state D presence head (absent, too short to call, callable). Requires a D."],
        [<code>reject_empty_sequences</code>, "False", "Rejects all-padding rows with an InputError."],
      ]} />
      <H2 id="forward">Forward computation</H2>
      <DocTable head={["Stage", "Computation"]} rows={[
        ["Embedding", <>Learned token and absolute-position embeddings, <code>(batch, window, embed_dim)</code>.</>],
        ["Orientation", <>The head predicts one of 4 transforms (forward, reverse complement, complement, reverse) from
          the initial embedding. With a supplied <code>orientation</code> input (training batches) that transform is
          used; otherwise the argmax, replaced by the bank&apos;s k-mer vote where the vote wins. The tokens are
          transformed and embedded again.</>],
        ["Core (the chip)", <>A dilated convolutional trunk gives every base its features once; the parser reads the
          region path and boundaries; the anchor heads find the V and J codons inside their regions; the SHM reader
          gives each base its mutation and insertion probability and its germline columns; per gene, the bank scan
          scores every allele row of the bound reference that the batch&apos;s genotype allows.</>],
        ["Refinement", <>The V end and J start are refined from the anchors along the bank&apos;s germlines. At
          inference only, each read of a multi-locus bank is restricted to one locus, the junction is aligned to the D
          rows, the D edges are refined on the called D, and the top V rows are re-scored by a banded gapped
          alignment.</>],
      ]} />
      <p>
        Each gene returns start and end logits of shape <code>(batch, window)</code>, their expectations
        <code> {"{gene}_start"}</code> and <code>{"{gene}_end"}</code> of shape <code>(batch, 1)</code>, the allele
        log posterior <code>{"{gene}_log_posterior"}</code> and <code>{"{gene}_allele"}</code>, each allele&apos;s
        likelihood relative to the best (1 for the best, 0 outside the bank), with one column per allele of the bound
        reference. The model also returns <code>orientation_logits</code> and <code>orientation</code>, the anchor
        logits (<code>v_anchor_logits</code>, <code>j_anchor_logits</code>), <code>d_presence_logits</code>,
        <code> mutation_rate</code>, <code>indel_count</code> and the read masks. A batch from the training adapter
        also gets the <code>chip_*</code> tensors the objective reads; a prediction batch does not.
      </p>
      <H2 id="gradient-paths">How the units train together</H2>
      <DocTable head={["Path", "Behavior"]} rows={[
        ["Orientation", "Trained by its own cross entropy. The transform is a hard choice (teacher-forced while training), so the other losses do not reach the orientation head."],
        ["Anchors and parser", "The log of the parser's region marginal is added to each anchor head's position logits, so the anchor and parse losses train each other."],
        ["Scans and parser", <>Each gene&apos;s scan weights bases by the parser&apos;s soft region; its gradient reaches the
          parser through a valve at <code>ROUTE_VALVE</code> (0.1) of its full value.</>],
        ["Teacher forcing", <>While training, each read uses the true regions and columns in place of the
          parser&apos;s and the reader&apos;s with probability <code>P_TEACHER</code> (0.5).</>],
        ["D presence", "Reads detached evidence (how much of the read the parser puts in D, how concentrated the D posterior is), so it does not reshape that evidence."],
      ]} />
      <p>
        The loss terms and their normalisation are listed on the <DocLink to="training-tasks">training task
        API</DocLink> page.
      </p>
      <H2 id="components">Replace components</H2>
      <p>
        <code>ModelComponents</code> holds four factories, each called with the configuration: <code>core</code> and
        <code> objective</code> (required), and <code>embedding</code> and <code>orientation_head</code> (defaulting to
        the token-plus-position embedding and the orientation head). The model calls
        <code> core(emb, mask, end_mask, *, tokens, batch)</code> with the corrected embedding, the read and end masks,
        the corrected tokens and the raw input batch, and adds the returned dictionary to its outputs. A core that reads
        the reference defines <code>bind_reference(reference)</code>. <code>objective(config)</code> returns the
        architecture&apos;s <code>TrainingTasks</code>. This example keeps the chip and removes the position
        embeddings:
      </p>
      <CodeBlock lang="python" code={`from dataclasses import replace
from torch import nn
from alignair.core import prepare_model, register_architecture
from alignair.core.components import architecture_components

class TokenOnlyEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokens = nn.Embedding(config.vocab_size, config.embed_dim)

    def forward(self, tokens):
        return self.tokens(tokens)

register_architecture(
    "chip-token-only-v1",
    replace(architecture_components("chip-v1"), embedding=TokenOnlyEmbedding),
)
model = prepare_model(reference, architecture="chip-token-only-v1").build_model()`} />
      <p>
        Construction checks that each factory returns a PyTorch module and raises <code>ConfigurationError</code>
        with the component name otherwise; it does not run a dummy forward, so test output shapes and gradient paths
        on representative inputs. Registration refuses to overwrite a name (&ldquo;architecture &apos;chip-v1&apos;
        is already registered&rdquo;); use a new versioned name when changing an architecture&apos;s meaning.
      </p>
      <p>
        Model files record the architecture name and never import or execute factory code. A process that loads a
        custom architecture must register it first, or name the modules that register it in the
        <code> ALIGNAIR_ARCHITECTURE_MODULES</code> environment variable (comma-separated), which a subprocess inherits.
        An unknown name raises <code>ConfigurationError</code> (&ldquo;unknown model architecture&rdquo;).
      </p>
      <H2 id="inspection">Inspect components</H2>
      <CodeBlock lang="python" code={`points = model.component_modules()   # {"embedding", "orientation_head", "core"}
trunk = points["core"].trunk
parser = points["core"].parser`} />
      <p>
        Inspection points are the registered modules themselves and add no parameter copies; the orientation head is
        omitted when <code>use_orientation</code> is off. The chip core&apos;s units are attributes of
        <code> model.core</code>: <code>trunk</code>, <code>parser</code>, <code>anchors</code>, <code>reader</code>,
        <code> scans</code>, <code>d_presence</code> and the bound <code>bank</code>.
      </p>
      <H2 id="architecture-development">Develop and verify an architecture variant</H2>
      <p>
        Register stateless factories under a new architecture name, then use
        <DocLink to="model-setup"> reference-bound setup</DocLink> and the
        <DocLink to="training-workflow"> training workflow</DocLink>. For manual optimization,
        <code> build_training_engine</code> checks that the optimizer owns every trainable parameter and that a
        scheduler is bound to it. These tests cover the component contract, registration from outside the package and
        the chip&apos;s registration:
      </p>
      <CodeBlock lang="bash" code={`python -m pytest \\
  tests/alignair/core/test_core_hook.py \\
  tests/alignair/core/test_architecture_modules.py \\
  tests/alignair/network/test_registration.py`} />
      <p>
        Start architecture comparisons from fresh training, with the reference, record stream, seeds and update budget
        matched between baseline and candidate. Resume checks the configuration and ordered label identity; record
        streams restart from seed blocks of their own, so continuation is not an exact replay of an uninterrupted run.
        A warm start (<code>init_from</code>) copies tensors whose names and shapes match and leaves the rest at their
        initialization; it does not establish compatibility between architectures.
      </p>
    </>
  ),
};

export const compositionPages = [composition];
