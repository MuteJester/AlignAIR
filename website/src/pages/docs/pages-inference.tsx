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
      <H2 id="model-resolution">Resolve and load model artifacts</H2>
      <p>
        <code>Aligner.from_pretrained</code> and <code>load_aa_model</code> use
        <code> alignair.registry.resolve_model</code> for local files, catalog IDs, verified caches,
        and Hugging Face repositories. Each loader then validates its artifact format. Protein
        product bundles retain their class-map, integrity, and calibration-to-model binding checks.
      </p>
      <CodeBlock lang="python" code={`from alignair import Aligner, load_aa_model
from alignair.registry import resolve_model

aligner = Aligner.from_pretrained("models/nucleotide.alignair", device="cpu")
protein = load_aa_model("models/protein.alignair", device="cpu")
results = protein.predict(["EVQLVESGGG"])

# A catalog you maintain can supply either artifact format.
protein = load_aa_model(
    "protein-candidate", registry=["file:///srv/alignair-models"],
    revision="1.0.0", offline=True, device="cpu",
)

# Resolve without decoding when a caller needs the local artifact path.
path = resolve_model("hf://your-org/your-model", revision="release-tag", offline=True)`} />
      <DocTable head={["Option", "Behavior"]} rows={[
        ["registry", "Ordered catalog sources; an empty list disables catalog sources. Without an argument, use ALIGNAIR_REGISTRY, then config.toml registries, then the built-in catalog."],
        ["revision", "Catalog version when no inline @version is present. For Hub repositories, overrides the inline revision. Existing local paths ignore this option."],
        ["offline", "Uses local files and caches. A file:// catalog also works offline; uncached remote artifacts fail."],
        ["token", "Authenticates direct Hub downloads; also accepts HF_TOKEN or HUGGING_FACE_HUB_TOKEN. Do not include credentials in model specifications."],
      ]} />
      <p>
        Pinned catalog artifacts already in the cache can load without contacting a registry;
        recorded checksums are verified before reuse. Hugging Face specifications accept
        <code> hf://org/repo</code>, <code>org/repo</code>, and an optional artifact subpath.
        An explicitly local path that is missing or names a directory raises a model-resolution error.
      </p>
      <p>
        The AA loader retains two historical checkout aliases for
        <code> alignair-aa-igh-human</code> and <code>alignair-aa-igh-human@2.0.0</code>.
        They apply only when the matching local experiment bundle exists and no registry is
        supplied or configured. Other versions use normal catalog resolution. These aliases do
        not establish availability of that model in a public catalog. Use <code>load_aa_model</code>
        for explicit resolution options and reuse the returned predictor for multiple requests.
      </p>
      <CodeBlock lang="bash" code={`alignair predict-aa reads.fa --model protein-candidate \\
  --registry file:///srv/alignair-models --revision 1.0.0 --offline \\
  --device cpu --out predictions.jsonl`} />
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
      <H2 id="source-identity">Join metadata by source identity</H2>
      <CodeBlock lang="python" code={`from contextlib import closing
from alignair.io import (
    build_metadata_index, iter_sequence_batches, lookup_metadata_rows,
)

metadata, columns = build_metadata_index(
    "annotations.csv", id_column="contig_id", normalize_10x=True,
)
with metadata, closing(iter_sequence_batches("reads.fa", chunk_size=20000)) as batches:
    for batch in batches:
        metas = lookup_metadata_rows(metadata, batch.source_ids)
        # Each metadata row corresponds to the same accepted read as batch.ids[i].
        for record, meta in zip(batch.records, metas):
            print(record.sequence_id, record.source_id, record.source_index, meta)`} />
      <p>
        <code>iter_sequence_batches</code> applies nucleotide cleaning and yields
        <code> SequenceReadBatch</code> objects. Each accepted <code>SequenceRead</code> retains
        its unique output <code>sequence_id</code>, cleaned <code>sequence</code>, original
        <code> source_id</code>, and zero-based <code>source_index</code>. A batch exposes
        <code> ids</code>, <code>sequences</code>, <code>source_ids</code>, and its
        <code> n_dropped</code> count. Rejected records retain their original positions in the
        optional rejects output. The historical <code>iter_sequences</code> tuple interface remains
        available; use batches when retaining source provenance or joining metadata.
      </p>
      <p>
        Output IDs are allocated in input order across all chunks. An unoccupied source ID is
        preserved; collisions receive the next available <code>_dupN</code> suffix. Generated names
        are reserved just like real headers. Missing or empty source IDs receive
        <code> seq&lt;source_index&gt;</code> output IDs. A generated output name can occupy a later
        header’s name, so metadata lookup always uses the original source ID:
      </p>
      <DocTable head={["Source header", "Output ID", "Metadata key"]} rows={[
        ["read", "read", "read"],
        ["read", "read_dup1", "read"],
        ["read_dup1", "read_dup1_dup1", "read_dup1"],
      ]} />
      <p>
        <code>lookup_metadata_rows</code> accepts an in-memory mapping or a metadata index and
        returns rows in source order. Repeated source IDs share their unique metadata row; missing
        or empty IDs and unmatched keys receive empty dictionaries. Generated output IDs are never
        invented as metadata keys. With no metadata it returns <code>None</code>. Indexed lookup
        deduplicates source keys and limits each SQL query to 900 parameters. Sequence and metadata
        payloads are processed in chunks; ID reservation memory grows with distinct accepted IDs.
      </p>
      <p>
        <code>load_metadata</code> and <code>build_metadata_index</code>, owned by
        <code> alignair.io.metadata</code>, both reject duplicate join keys with
        <code> DuplicateMetadataId</code>. They parse CSV/TSV, including gzip, through the same
        normalization policy. With <code>normalize_10x=True</code>, nonempty canonical AIRR values
        take precedence over aliases; aliases fill missing values. Duplicate header names and
        colliding output names after <code>rename</code> raise <code>InputError</code>.
        The CLI protects scientific result fields by namespacing conflicting metadata columns.
      </p>
      <p>
        The disk-backed <code>MetadataIndex</code> supports context management and idempotent
        <code> close()</code>. A failed insertion rolls back its entire batch while preserving
        earlier committed batches. Construction or loading failures remove the temporary database.
        Close the returned index after use; lookups and writes after closure raise
        <code> DataContractError</code>. Legacy metadata imports from
        <code> io.sequence_reader</code> remain aliases of the shared owner.
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
        strings, and applies the supplied alphabet and optional positive length limit.
        <code> max_length=None</code> permits unbounded content validation; model-facing presets
        require a finite window. The normalizer checks
        length after case conversion as well, so Unicode case expansion cannot produce an accepted
        read longer than the model window. Interior whitespace is invalid by default. Optional
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
        ["cropped_indices / crop_flags", "Original positions of cropped records, and boolean flags aligned with accepted records, respectively."],
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
        <code>policy.normalize(value)</code> applies the same rules to one value and returns
        <code> SequenceNormalizationResult</code>: <code>sequence</code>, <code>error</code>,
        <code> detail</code>, and <code>was_cropped</code>. A rejected result has no sequence.
        This lets streaming readers validate content without allocating record IDs or batches.
        Batch normalization accepts mapping records by default; <code>allow_records=False</code>
        requires each batch item to be a sequence value instead of interpreting mapping fields.
      </p>
      <DocTable head={["Policy option", "Behavior"]} rows={[
        ["unknown_symbol", "None rejects characters outside the alphabet. An allowed replacement character maps unknown input characters into that alphabet."],
        ["ambiguity_symbols", "Characters outside the alphabet that map to unknown_symbol without counting toward the unusable-character fraction."],
        ["max_invalid_fraction", "Optional finite fraction from 0 to 1. Requires unknown_symbol; reject when the unusable fraction exceeds it."],
        ["overflow", "error rejects overlength input; prefix crops to a required finite window and records crop metadata."],
        ["reject_interior_whitespace", "True by default. False sends interior whitespace through the declared character policy."],
      ]} />
      <p>
        Protein product prediction selects its canonical-residue-plus-X policy through
        <code>aa.product.validate.normalize_input_batch</code>. It normalizes once per request and
        uses the retained positions for calibration-group lists and output restoration. Duplicate IDs
        still share ID-based group lookups while retaining independent positional lookups. The
        historical <code>normalize_inputs</code> split-list interface and <code>Record/ErrorResult</code>
        names remain adapters and aliases of the shared owner. Tokenization and tensor-shape
        validation remain the model-input boundary described below.
      </p>
      <H2 id="nucleotide-policy">Select the nucleotide content policy</H2>
      <CodeBlock lang="python" code={`from alignair.data import (
    nucleotide_file_policy, nucleotide_prediction_policy, prepare_nucleotide_inputs,
)

content = nucleotide_file_policy().normalize(" acgr ")
print(content.sequence)  # ACGN

prepared, cropped = prepare_nucleotide_inputs([" ac?gtn "], max_length=4)
print(prepared, cropped)  # ["ACNG"], [True]

# The same policy can retain source positions and per-record failures.
batch = normalize_sequence_inputs(
    ["ACGTN", "", "N"], nucleotide_prediction_policy(4), on_invalid="skip",
)
print(batch.cropped_indices, batch.crop_flags)  # (0,), (True, False)`} />
      <DocTable head={["Entry point", "Content and window rules"]} rows={[
        ["nucleotide_file_policy(max_length=None)", "Strip surrounding whitespace; uppercase; map IUPAC ambiguity and unknown characters to N. Reject more than 20% non-IUPAC content before checking length. A supplied window rejects overlength input."],
        ["nucleotide_prediction_policy(max_length)", "Strip and uppercase; map every non-ACGTN character to N; crop to the finite model window. It applies no file-quality rejection threshold."],
        ["prepare_nucleotide_inputs(sequences, max_length)", "Require a batch of strings; return prepared sequences and crop flags. Reject bare strings/bytes/mappings, non-string items, empty reads and empty eager batches with InputError."],
      ]} />
      <p>
        The product pipeline, raw nucleotide forward path, genotype repertoire alignment, temperature
        calibration, and assay replay use this shared preparation. Tokenization, orientation,
        germline alignment, and reported coordinates therefore refer to the same normalized read.
        A literal <code>P</code> in submitted nucleotide text becomes <code>N</code>; it cannot
        become an internal padding token. Genotype read lengths reflect normalized, cropped reads.
      </p>
      <p>
        File readers keep their quality-rejection policy; an in-memory prediction does not silently
        drop reads by applying that threshold. Eager preparation validates the complete batch before
        model execution and stops consuming a source at its first invalid value. Streaming prediction
        validates each chunk before forwarding it; an empty stream produces no results.
        The compatibility <code>io.sequence_reader.validate_sequence</code> import delegates to
        <code>data.inputs.validate_nucleotide_sequence</code> and retains its
        <code> (sequence, reason)</code> return shape, including <code>too_long</code> for a rejected
        window. Shared normalization uses <code>overlength</code> for that failure.
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
      <H2 id="protein-batches">Bound protein product forward batches</H2>
      <CodeBlock lang="python" code={`from alignair import load_aa_model, predict_aa

predictor = load_aa_model("models/protein.alignair", device="cpu")
results = predictor.predict(records, batch_size=64, on_invalid="skip")

# The convenience API uses the same request-scoped execution option.
rows = predict_aa(records, model="models/protein.alignair", batch_size=32)`} />
      <CodeBlock lang="bash" code={`alignair predict-aa reads.fa --model models/protein.alignair \\
  --batch-size 64 --out predictions.jsonl`} />
      <p>
        <code>AAPredictor.predict</code>, <code>predict_aa</code>, and <code>predict-aa</code> accept
        a positive integer <code>batch_size</code>, defaulting to 64. It limits accepted records per
        model forward through the same <code>prediction_batches</code> engine used by raw model
        inference. Tokenization uses the bundle’s verified tokenizer, and softmax stays on each
        output’s existing device. Changing batch size does not reload a cached predictor.
      </p>
      <p>
        Normalization runs once for the request. Calibration-group lists retain original input
        positions across skipped records and batch boundaries; ID-based group lookups retain their
        existing behavior for duplicate IDs. Results return in original order, and each forwarded
        batch must produce one prediction per accepted record. All-invalid input skips model execution
        and unused calibration lookup. Coordinate and calibration rules remain unchanged.
      </p>
      <p>
        This bounds model-forward memory. Inputs and returned predictions are still collected for the
        full request. The CLI records batch size in its run metadata. Floating-point scores can vary
        slightly with execution batch size, and nearly tied class rankings can change; keep batch size
        fixed when comparing runs that depend on those rankings.
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
        It validates the input list before model execution. Product prediction retains its separate normalized
        input policy, verified tokenizer, one all-head forward per bounded batch, and device-local softmax.
        The <DocLink to="prediction-calibration">prediction-set calibration API</DocLink> fits and
        applies shared APS/RAPS thresholds to these categorical probabilities.
      </p>
      <H2 id="probability-imports">Migrate former research inference imports</H2>
      <p>
        The <code>alignair.aa.infer</code> wrapper has been removed. Import its functions from the
        shared prediction module; their signatures and CPU softmax default are unchanged.
      </p>
      <DocTable head={["Former import", "Replacement"]} rows={[
        ["aa.infer.class_probs", "alignair.predict.probabilities.class_probs"],
        ["aa.infer.protein_class_probabilities", "alignair.predict.probabilities.protein_class_probabilities"],
      ]} />
      <p>
        <code>class_probs(model, seqs, cfg, device, key, batch=256)</code> returns one NumPy probability
        matrix. Use <code>protein_class_probabilities(model, seqs, cfg, device, keys, batch=256)</code>
        to collect several named heads in one forward per batch.
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
