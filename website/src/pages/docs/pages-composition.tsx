import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

const composition: DocPage = {
  slug: "model-composition",
  title: "Model composition API",
  section: "Design",
  lead: "Construct nucleotide and protein-class AlignAIR models through one configurable model and component interface.",
  body: () => (
    <>
      <p>
        <code>alignair.core.AlignAIR</code> owns model assembly and forward execution for both alphabets.
        Its configuration selects score semantics, boundary slots, and optional heads. The tokenizer and
        reference determine the vocabulary and label counts; the model does not infer them from the input.
      </p>
      <H2 id="construction">Construct a model</H2>
      <p>
        This example uses small label vocabularies to illustrate construction. For training, supply counts
        from the actual nucleotide reference or protein equivalence maps and encode inputs with the matching tokenizer.
      </p>
      <CodeBlock lang="python" code={`from alignair.core import AlignAIR, AlignAIRConfig

nt_config = AlignAIRConfig(
    v_allele_count=3, j_allele_count=2, has_d=False,
    max_seq_length=192,
)
nt_model = AlignAIR(nt_config)

protein_config = AlignAIRConfig(
    v_allele_count=3, j_allele_count=2, has_d=False,
    vocab_size=23, max_seq_length=192,
).with_amino_acid_heads()
protein_model = AlignAIR(protein_config)`} />
      <p>
        Both models accept a dictionary containing <code>tokenized_sequence</code>: an integer tensor of
        shape <code>(batch, max_seq_length)</code> with padding ID zero. The protein preset uses right padding,
        categorical scores, and separate start/end masks. It disables orientation correction, auxiliary
        predictions, and edit-state predictions and rejects empty rows. Existing <code>AAAlignAIR</code> imports
        remain compatibility constructors for this preset.
      </p>
      <H2 id="choices">Configuration and output semantics</H2>
      <DocTable head={["Setting", "Behavior"]} rows={[
        [<code>classification_mode="multilabel"</code>, <><code>{"{gene}_allele"}</code> contains independent sigmoid probabilities.</>],
        [<code>classification_mode="categorical"</code>, <><code>{"{gene}_class_logits"}</code> contains raw logits; apply softmax for class probabilities.</>],
        [<code>boundary_mode="residue"</code>, "Preserves the released nucleotide decoder: both boundary masks select residue positions."],
        [<code>boundary_mode="half_open"</code>, "End slot k is valid when residue k-1 exists. Encoded reads must be shorter than the model window to leave room for the terminal slot."],
        [<code>use_orientation</code>, "Enables nucleotide orientation detection and correction. Leave it disabled for proteins."],
        [<code>use_meta_heads</code>, "Enables the auxiliary tower and its mutation, indel, productivity, and optional locus predictions."],
        [<code>state_head</code>, "Enables per-position edit-state logits independently of auxiliary heads."],
        [<code>reject_empty_sequences</code>, "Rejects all-padding rows with a structured input error."],
      ]} />
      <p>
        Each active gene also returns start/end logits of shape <code>(batch, window)</code> and decoded
        expectations of shape <code>(batch, 1)</code>. <code>config.classification_key(gene)</code> returns
        the appropriate classification key. Construction choices are serialized with the configuration.
      </p>
      <CodeBlock lang="python" code={`from dataclasses import replace

# A categorical model can also expose auxiliary predictions.
config = replace(protein_config, use_meta_heads=True)
model = AlignAIR(config)`} />
      <H2 id="components">Replace components</H2>
      <p>
        <code>ModelComponents</code> supplies factories for embeddings, gene branches, the auxiliary tower
        and heads, orientation, and edit states. Register an architecture under a new versioned name when
        changing its construction. This example removes learned position embeddings for an ablation:
      </p>
      <CodeBlock lang="python" code={`from torch import nn
from alignair.core import ModelComponents, register_architecture

def token_embedding(config):
    return nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)

register_architecture(
    "token-only-v1",
    replace(ModelComponents(), embedding=token_embedding),
)
model = AlignAIR(replace(protein_config, architecture="token-only-v1"))`} />
      <p>
        Embedding factories emit <code>(batch, window, embed_dim)</code>. A gene-branch factory receives
        a <code>GeneSpec</code> and configuration; the branch implements <code>segment</code>,{" "}
        <code>classify</code>, and <code>component_modules</code>. The shared <code>GeneBranch</code> provides
        these methods and a <code>classification_features</code> method for its penultimate representation.
        Its constructor accepts <code>segmentation_tower</code> and <code>classification_tower</code> modules
        so a branch factory can replace feature extraction while retaining boundary and classification heads.
        The segmentation tower emits <code>config.block_out</code> features; the classification tower emits
        the width returned by <code>config.cls_spec(spec)[0]</code>. Both consume embeddings with shape
        <code>(batch, window, embed_dim)</code>. Factories must construct fresh modules for each model.
      </p>
      <p>
        Checkpoints record the architecture name. Register the same factories in a loading process before
        loading that architecture. Unknown names fail with <code>ConfigurationError</code>; model files never
        import or execute factory code. Registration refuses to overwrite an existing name. Use a new name
        when changing the architecture’s meaning.
      </p>
      <H2 id="inspection">Inspect layers and choose training objectives</H2>
      <CodeBlock lang="python" code={`points = model.component_modules()
classification_head = points["v.cls_head"]
segmentation_tower = points["v.seg_tower"]`} />
      <p>
        Inspection points reference the actual registered modules and add no parameter copies. Optional
        components are omitted when disabled. ModelXRay and the assay profiler consume these points;
        custom branches can extend their own inspection dictionary.
      </p>
      <p>
        Use the <DocLink to="training-tasks">training task API</DocLink> to select matching losses and
        metrics from the configured score semantics and enabled heads. Multilabel hierarchical and
        categorical objectives retain their distinct target and reduction contracts. Existing inference
        pipelines also retain their nucleotide AIRR and protein-class output contracts; model composition
        does not make those pipelines interchangeable. See the <DocLink to="python-api">Python API</DocLink>
        for end-to-end prediction.
      </p>
    </>
  ),
};

export const compositionPages = [composition];
