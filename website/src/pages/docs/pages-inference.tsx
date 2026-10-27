import { CodeBlock } from "../../components/ui";
import { DocLink, H2, type DocPage } from "./doc-kit";

const modelInference: DocPage = {
  slug: "model-inference",
  title: "Model inference API",
  section: "Design",
  lead: "Prepare sequence inputs, execute shared model batches, and inspect categorical heads or component features through one inference interface.",
  body: () => (
    <>
      <p>
        <code>alignair.predict</code> supplies shared execution mechanics for nucleotide and protein
        models. Tokenizer selection and input policy stay with the caller. The public product APIs
        add their own validation, calibrated sets, coordinate interpretation, and result schemas.
        See <DocLink to="model-setup">model setup</DocLink> for input and label configuration and
        <DocLink to="evaluation-reports">evaluation reports</DocLink> for labelled cohort scoring.
      </p>
      <H2 id="inputs">Bind the tokenizer and input window</H2>
      <CodeBlock lang="python" code={`from alignair.data import SequenceInputAdapter

# Supply the model's tokenizer; a loaded bundle supplies predictor.tokenizer.
adapter = SequenceInputAdapter(
    tokenizer, model.cfg.max_seq_length, device=device,
    vocab_size=model.cfg.vocab_size,
)
inputs = adapter(sequences)  # {"tokenized_sequence": LongTensor[B, L]}`} />
      <p>
        The adapter right-pads encoded sequences to the explicit window. It requires one integer
        CPU token per input position and validates vocabulary bounds when supplied. The default
        rejects empty reads and overlength sequences. <code>overflow="prefix"</code> and
        <code>allow_empty=True</code> are explicit alternatives. Optional <code>alphabet</code>
        restricts raw characters before encoding; normalization otherwise belongs to the supplied
        tokenizer or upstream validator. <code>validate(sequences)</code> checks a complete batch
        before inference when a caller needs eager input validation.
      </p>
      <p>
        Invalid windows, tokenizers, devices, and policies raise <code>ConfigurationError</code>.
        Invalid sequences raise <code>InputError</code>. Malformed, length-changing, or out-of-range
        tokenizer output raises <code>DataContractError</code> with record context. The supplied
        tokenizer object is used directly, preserving verified bundle vocabularies.
      </p>
      <H2 id="forward">Execute one batch or an ordered stream</H2>
      <CodeBlock lang="python" code={`from alignair.predict import forward_batch, prediction_batches

outputs = forward_batch(model, inputs, expected_rows=len(sequences))

for batch in prediction_batches(
    model, source_sequences, adapter, batch_size=128,
    output_keys=("v_class_logits", "j_class_logits"),
):
    print(batch.record_offset, batch.size, batch.outputs.keys())`} />
      <p>
        <code>forward_batch</code> enters evaluation mode, disables gradients, and restores every
        module’s previous flag on success or failure. It returns detached tensors on the model’s
        devices. Outputs must be a nonempty mapping of real tensors with the expected leading batch
        dimension. Missing or malformed requested fields raise <code>InferenceError</code>;
        non-finite requested fields raise <code>NumericalError</code> with field context. Omitted
        <code> output_keys</code> selects every output.
      </p>
      <p>
        <code>prediction_batches</code> is lazy, preserves source order, and includes a final partial
        batch. Its adapter must preserve record order and cardinality. Each result contains the
        zero-based source offset, actual batch size, and detached outputs. Evaluation mode,
        gradient state, and operation context are restored before each result is yielded.
      </p>
      <H2 id="probabilities">Read several categorical heads in one pass</H2>
      <CodeBlock lang="python" code={`from alignair.predict import class_probabilities

probabilities = class_probabilities(
    model, sequences, adapter,
    keys=("v_class_logits", "j_class_logits"), batch_size=128,
)
print(probabilities["v_class_logits"].shape)`} />
      <p>
        This function makes one forward call per batch for every requested head and returns NumPy
        arrays in source order. It expects categorical logit matrices; sigmoid allele outputs use
        the allele evaluation interface. The default <code>probability_device="cpu"</code> preserves
        research calibration arithmetic. <code>probability_device=None</code> performs softmax on
        each output’s existing device. <code>softmax_probabilities</code>, available from
        <code> predict.probabilities</code>, provides the same normalization for an existing matrix.
        NumPy results own CPU snapshots; bfloat16 logits are normalized in float32.
      </p>
      <p>
        <code>predict.probabilities.protein_class_probabilities</code> binds the research protein
        policy: uppercase canonical residues plus X and stop, nonempty reads, and no truncation.
        It validates the input list before model execution. Historical <code>aa.infer.class_probs</code>
        delegates to its single-head interface. Product prediction retains its separate normalized
        input policy, verified tokenizer, single all-head forward, and device-local softmax.
      </p>
      <H2 id="features">Inspect named component inputs</H2>
      <CodeBlock lang="python" code={`from alignair.predict import component_features

print(model.component_modules().keys())
features = component_features(
    model, sequences, adapter,
    components=("v.cls_head", "j.cls_head"),
)`} />
      <p>
        The feature observer concatenates the first positional input of each selected component,
        in the declared order, and returns a NumPy matrix. Every selected component must be a
        distinct registered module, receive a finite two-dimensional batch-by-feature tensor,
        and run exactly once. Missing or repeated invocations raise <code>InferenceError</code>.
        Inputs are copied when observed, and all hooks are removed on every exit path. Model
        parameters and the representation definition remain unchanged by observing those inputs.
      </p>
      <H2 id="ownership">Execution and policy ownership</H2>
      <p>
        Sequence adaptation lives in <code>data.inference</code>; model mode control lives in
        <code>core.runtime</code>. Prediction batching, probabilities, and feature observation live
        in separate modules under <code>predict</code>. Historical training evaluation-context
        imports remain aliases of the shared runtime owner. Nucleotide <code>run_model</code>
        retains uppercase conversion, unknown-to-N encoding, prefix truncation, and empty-read
        behavior through explicit adapter settings. The product feature wrapper retains its
        predeclared V/J representation version. Checkpoint evaluation reuses clean V/J/D head
        probabilities and conformal test probabilities instead of recomputing them.
      </p>
    </>
  ),
};

export const inferencePages = [modelInference];
