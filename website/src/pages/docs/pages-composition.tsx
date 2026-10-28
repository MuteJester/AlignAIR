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
      <p>
        Shared neural-network components live in <code>alignair.neural</code>, including prediction
        heads and learned uncertainty weights. This package serves both nucleotide and protein models.
        Historical <code>alignair.nn</code> imports delegate to the same components.
      </p>
      <H2 id="construction">Construct a model</H2>
      <p>
        This example uses small label vocabularies to illustrate construction. For training, supply counts
        from the actual nucleotide reference or protein equivalence maps and encode inputs with the matching tokenizer.
        The <DocLink to="model-setup">model setup API</DocLink> derives those counts and binds them to
        an ordered reference before allocating model parameters.
      </p>
      <CodeBlock lang="python" code={`from alignair.core import AlignAIR, build_model_config

nt_config = build_model_config(3, 2, max_seq_length=192)
nt_model = AlignAIR(nt_config)

protein_config = build_model_config(
    3, 2, input_space="protein", label_space="protein_classes",
)
protein_model = AlignAIR(protein_config)`} />
      <p>
        Both models accept a dictionary containing <code>tokenized_sequence</code>: an integer tensor of
        shape <code>(batch, max_seq_length)</code> with padding ID zero. The protein preset uses right padding,
        categorical scores, and separate start/end masks. It disables orientation correction, auxiliary
        predictions, and edit-state predictions and rejects empty rows. Both profiles use the same
        <code> AlignAIR</code> class. Its constructor uses the supplied configuration without replacing
        head choices.
      </p>
      <H2 id="migration">Migrate former protein model imports</H2>
      <p>
        The <code>aa.model</code> and <code>aa.config</code> facades have been removed. Select
        the protein input and label spaces explicitly when preparing a configuration.
      </p>
      <DocTable head={["Former import", "Replacement"]} rows={[
        ["aa.model.AAAlignAIR", "alignair.core.AlignAIR with an explicit protein-class configuration"],
        ["aa.model.AAGeneBranch", "alignair.core.gene_branch.GeneBranch with that configuration"],
        ["aa.model.boundary_masks", "alignair.core.boundaries.boundary_masks"],
        ["aa.config.aa_igh_config", "alignair.core.build_model_config with input_space='protein' and label_space='protein_classes'"],
      ]} />
      <p>
        Product loading retains normalization of historical protein head metadata at the artifact
        boundary. Existing supported state dictionaries, classification columns and decoder outputs
        remain compatible with the shared class.
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
      <H2 id="forward">Forward computation and shared features</H2>
      <DocTable head={["Stage", "Computation and output"]} rows={[
        ["Embedding", <>Learned token and absolute-position embeddings are added to produce
          <code> (batch, window, embed_dim)</code>. This embedding is shared by gene branches; there is
          no shared convolutional encoder before them.</>],
        ["Orientation, when enabled", "Predicts a transform from the initial embedding, transforms the tokens, then embeds them again. Training uses supplied orientation labels; otherwise the model uses a detached argmax."],
        ["Per-gene segmentation", "An independent convolutional tower produces a global feature vector. Two linear heads produce window-sized start/end logits. Masked softmax expectations decode the two boundaries."],
        ["Soft crop", "The decoded boundaries produce a smooth interval mask with a default ramp width of 3 positions. The mask multiplies the shared embedding; it does not extract or resize a sequence."],
        ["Per-gene classification", "A second independent tower consumes the masked embedding. A SiLU hidden layer and final linear head produce categorical logits or sigmoid multilabel scores."],
        ["Optional predictions", "The auxiliary tower and edit-state branch consume the canonical embedding independently of the per-gene crops."],
      ]} />
      <p>
        The default segmentation towers downsample the window by 32; V/J classification towers
        downsample by 128, and the D classification tower by 32. Flattening and a linear projection
        produce each tower’s global vector. Window length therefore affects projection sizes and
        checkpoint shapes. Boundary masks constrain decoded positions and supervised boundary losses;
        they do not zero padding embeddings throughout the convolutional towers. The default embedding
        includes a learned padding-token vector and learned position vectors at padded positions.
      </p>
      <H2 id="gradient-paths">How the objectives interact</H2>
      <DocTable head={["Objective", "Gradient path in the default architecture"]} rows={[
        ["Classification", "Updates the classifier, embedding, and segmentation through the differentiable crop. Classification and segmentation are coupled even when their loss values are reported separately."],
        ["Segmentation", "Updates boundary heads, the segmentation tower, and embedding. It has no path through the classification tower or class head."],
        ["Orientation", "Has its own supervised objective. The hard transform decision blocks classification and boundary losses from training the orientation head."],
        ["Auxiliary and state tasks", "Update their own branches and the shared embedding. They do not use the per-gene classification crops."],
      ]} />
      <p>
        The crop enforces an end of at least start plus one. When this constraint replaces the predicted
        end, classification has no crop gradient to that end estimate; the boundary objective can still
        train it. Record inverted-span frequency alongside boundary error when investigating classification.
        The categorical objective adds mean boundary loss to mean class loss. The hierarchical objective
        uses uncertainty weights and additional span terms. Their numerical loss scales are not directly
        comparable. See the <DocLink to="training-tasks">task contracts</DocLink> before changing weights.
      </p>
      <CodeBlock lang="python" code={`from alignair.xray.dynamics import gradient_conflict

# parts contains graph-bearing tensors from tasks.loss(outputs, targets, weights).
# Call before backward, with gradients enabled.
report = gradient_conflict(
    {name: parts[name] for name in ("segmentation", "classification")},
    list(model.embedding.parameters()),
)
print(report["grad_norm"], report["cosine"])`} />
      <p>
        This existing probe uses autograd without accumulating parameter gradients. A negative cosine
        describes opposing task gradients on this batch; it does not by itself demonstrate an accuracy
        problem. Use a separate loaded probe model in evaluation mode with gradients enabled, and
        repeat over representative batches. Evaluation mode alone does not disable autograd.
      </p>
      <H2 id="components">Replace components</H2>
      <p>
        <code>ModelComponents</code> supplies factories for embeddings, gene branches, the auxiliary tower
        and heads, orientation, and edit states. Register an architecture under a new versioned name when
        changing its construction. This example removes learned position embeddings for an ablation:
      </p>
      <CodeBlock lang="python" code={`from torch import nn
from alignair.core import ModelComponents, register_architecture

def token_embedding(config):
    return nn.Embedding(config.vocab_size, config.embed_dim)

register_architecture(
    "token-only-v1",
    replace(ModelComponents(), embedding=token_embedding),
)
model = AlignAIR(replace(protein_config, architecture="token-only-v1"))`} />
      <p>
        This preserves the default trainable padding-token behavior while removing position embeddings.
        Adding <code>padding_idx=0</code> would change a second factor. Treat padding masking as a
        separate experiment. Classification width scaling is also a parameter-budget heuristic;
        increasing width does not establish feature rank or accuracy.
      </p>
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
        These factories replace stages within the current segmentation-then-classification flow.
        A shared encoder across genes or a joint boundary decoder needs an explicit change to model
        assembly and its contracts. Keep that extension in the shared model interface, with a new
        architecture identity and appropriate tests for both input alphabets.
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
      <H2 id="boundary-experiment">Compare predicted and known boundaries</H2>
      <p>
        Before replacing the feature towers, measure how classification changes when the same trained
        classifier receives known boundaries. Freeze the checkpoint, label schema, cohort, and score
        semantics. Select rows with reliable supervision for both boundaries of the tested gene. Inputs
        and targets must already use the same canonical orientation and padded model coordinates.
        Both conditions below hold orientation fixed, isolating the crop from orientation errors.
      </p>
      <CodeBlock lang="python" code={`import torch
from alignair.core import evaluation_mode

# canonical_inputs and targets describe a fully annotated V cohort batch.
tokens = canonical_inputs["tokenized_sequence"]
inputs = {
    **canonical_inputs,
    "orientation": torch.zeros(len(tokens), device=tokens.device, dtype=torch.long),
}
with evaluation_mode(model):
    predicted = model(inputs)
    embeddings = model.embedding(tokens)
    predicted_scores = predicted[model.cfg.classification_key("v")]
    known_scores = model.branches["v"].classify(
        embeddings,
        targets["v_start"].to(embeddings).reshape(-1, 1),
        targets["v_end"].to(embeddings).reshape(-1, 1),
    )`} />
      <p>
        This recipe uses the default embedding and gene-branch contracts. Adapt it explicitly for an
        architecture with a different canonicalization path. Neither condition updates the weights;
        use the configured multilabel or categorical metrics on the paired scores. Report class accuracy
        and true-label or compatible-set probability changes, stratified by boundary error, read length,
        and crop severity. Keep read IDs and supervised counts identical between conditions.
      </p>
      <p>
        A consistent gain with known boundaries supports investigating localization and crop design.
        A small gap supports investigating the classifier or labels, but does not rule out localization
        problems. Known-boundary scores are a counterfactual diagnostic, not an accuracy upper bound:
        the classifier was trained using predicted crops. Training a detached-crop variant would answer
        a different question about gradient coupling and needs a separate registered architecture,
        matched training runs, and held-out evaluation. Do not use known boundaries in production predictions.
      </p>
    </>
  ),
};

export const compositionPages = [composition];
