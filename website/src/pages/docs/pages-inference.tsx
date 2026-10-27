import { CodeBlock } from "../../components/ui";
import { DocLink, DocTable, H2, type DocPage } from "./doc-kit";

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
      <H2 id="files">Read sequence files before validation</H2>
      <CodeBlock lang="python" code={`from contextlib import closing
from alignair.io import iter_sequence_records

with closing(iter_sequence_records("reads.fa.gz")) as records:
    for record in records:
        print(record.sequence_id, record.sequence)

# Table columns and format overrides are explicit when auto-detection is unsuitable.
with closing(iter_sequence_records(
    "reads.tsv", seq_column="protein", id_column="sample",
)) as records:
    inputs = [record._asdict() for record in records]`} />
      <p>
        <code>iter_sequence_records</code> parses FASTA, four-line FASTQ, CSV/TSV, and nonblank text
        lines. It accepts filesystem paths, gzip files, or <code>"-"</code> for stdin. Auto-detection
        prefers recognized filename suffixes, then examines the first nonblank line. Set
        <code> format</code> to <code>"fasta"</code>, <code>"fastq"</code>, <code>"table"</code>,
        or <code>"txt"</code> to override detection. Table sequence columns default to
        <code> sequence</code>, <code>seq</code>, <code>nucleotide</code>, or <code>read</code>;
        ID columns default to <code>sequence_id</code>, <code>id</code>, or <code>name</code>.
      </p>
      <DocTable head={["RawSequenceRecord", "Contract"]} rows={[
        ["sequence_id", "Header first word or table ID, possibly None; duplicate IDs are retained."],
        ["sequence", "Case and alphabet are preserved. FASTA sequence lines are joined; FASTQ quality is checked and discarded."],
        ["strip_sequence_lines", "False by default. True strips each FASTA/text sequence line before joining or emitting it; interior spaces remain."],
      ]} />
      <p>
        FASTA and FASTQ retain empty and unnamed records for downstream validation and fallback ID
        assignment. Blank text lines are ignored. Sequence text before a FASTA header, malformed
        FASTQ, missing table fields, and invalid CSV quoting raise <code>InputError</code> with path,
        format, and physical line context. These are file-structure failures; per-sequence
        <code> on_invalid="skip"</code> does not suppress them. Decode and compressed-data failures
        raise <code>InputError</code>; filesystem failures raise <code>OperationIOError</code>.
        Translated failures preserve their original exception as the cause.
      </p>
      <p>
        Parsing reads the source once and holds the current record. Owned files close on exhaustion,
        errors, or explicit iterator closure; borrowed stdin remains open. Use <code>closing</code>
        when iteration may stop early. Protein CLI input uses this parser with per-line stripping,
        then the normalizer below; it supports all shared formats and reports empty FASTA entries as
        per-record errors. It still collects the records for one prediction request. Nucleotide
        <code>iter_sequences</code> adds its existing ambiguity cleaning, rejects, ID policy, and
        chunking. Use raw parsing when sequence content must reach a different validation policy.
      </p>
      <H2 id="records">Normalize records without losing source identity</H2>
      <CodeBlock lang="python" code={`from alignair.data import SequenceValidationPolicy, normalize_sequence_inputs

policy = SequenceValidationPolicy(
    alphabet=frozenset("ACDEFGHIKLMNPQRSTVWYX"),
    max_length=192, stop_symbol="*",
)
batch = normalize_sequence_inputs(
    [" acde ", {"sequence_id": "sample", "sequence": "A C"}, "xX"],
    policy, on_invalid="skip",
)
print(batch.valid_indices)  # (0, 2)
print([record.sequence for record in batch.records])  # ["ACDE", "XX"]

# Downstream inference supplies one result per accepted record, in that order.
ordered_results = batch.restore_order(predictions)`} />
      <p>
        <code>normalize_sequence_inputs</code> strips surrounding whitespace, uppercases accepted
        strings, and applies the supplied alphabet and positive length limit. A nucleotide caller
        can supply its own alphabet and window through the same policy type. The normalizer checks
        length after case conversion as well, so Unicode case expansion cannot produce an accepted
        read longer than the model window. Interior whitespace remains invalid. Optional
        <code> stop_symbol</code> identifies one forbidden character for the stop-specific diagnostic.
      </p>
      <p>
        A bare string is one read. An iterable may contain strings or mappings with
        <code> sequence</code> and optional <code>sequence_id</code>. Bare mappings and bytes raise
        <code> TypeError</code> as input-format errors. Missing sequence fields retain the empty-read
        behavior; explicit non-string values produce <code>invalid_type</code> failures. A nonempty
        string ID is preserved exactly, including duplicate IDs. Otherwise the ID is
        <code> seq_&lt;original_index&gt;</code>. IDs do not replace positional identity.
      </p>
      <DocTable head={["NormalizedSequenceBatch", "Contract"]} rows={[
        ["outcomes", "One SequenceRecord or SequenceInputFailure per consumed source item, in original order."],
        ["records / errors", "Tuple views of accepted normalized records and structured failures."],
        ["valid_indices", "Original zero-based positions of accepted records, suitable for positional metadata lookup."],
        ["restore_order(predictions)", "Interleave one downstream result per accepted record with the original failures; reject missing or extra results with DataContractError."],
      ]} />
      <p>
        <code>on_invalid="error"</code> raises <code>InputError</code> at the first invalid record
        and stops consuming the source. Its context includes the original ID, index, and reason
        under <code>sequence_input_validation</code>. <code>on_invalid="skip"</code> retains
        failures in their slots, using <code>empty</code>, <code>interior_whitespace</code>,
        <code> overlength</code>, <code>invalid_character</code>, or <code>invalid_type</code>.
        <code>SequenceInputFailure.to_dict()</code> returns the existing
        <code> sequence_id/error/detail</code> JSON fields. Invalid policies fail before source
        consumption with <code>ConfigurationError</code>.
      </p>
      <p>
        Protein product prediction selects its canonical-residue-plus-X policy through
        <code>aa.product.validate.normalize_input_batch</code>. It normalizes once per request and
        uses the retained positions for calibration-group lists and output restoration. Duplicate IDs
        still share ID-based group lookups while retaining independent positional lookups. The
        historical <code>normalize_inputs</code> split-list interface and <code>Record/ErrorResult</code>
        names remain adapters and aliases of the shared owner. Tokenization and tensor-shape
        validation remain the model-input boundary described below.
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
