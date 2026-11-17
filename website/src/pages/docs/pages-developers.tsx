import { CodeBlock, Callout } from "../../components/ui";
import { DocTable, DocLink, H2, H3, type DocPage } from "./doc-kit";

const REPO = "https://github.com/MuteJester/AlignAIR";

function Source({ path, children }: { path: string; children?: string }) {
  return (
    <a href={`${REPO}/blob/main/${path}`} target="_blank" rel="noreferrer">
      <code>{children ?? path}</code>
    </a>
  );
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Architecture                                                                                                      */
/* ---------------------------------------------------------------------------------------------------------------- */

const architecture: DocPage = {
  slug: "architecture",
  title: "Architecture",
  section: "Developers",
  lead: (
    <>
      How AlignAIR's code is laid out: the packages and what each owns, the path from a read to its AIRR row, the model
      file, and where to extend it.
    </>
  ),
  body: () => (
    <>
      <p>
        AlignAIR is one Python package, <code>alignair</code>, in <code>src/alignair</code>. Two development packages sit
        beside it and are left out of the wheel: <code>alignair_assay</code>, which scores AlignAIR and IgBLAST against
        simulated truth and real donors (its <Source path="src/alignair_assay/README.md">README</Source>), and{" "}
        <code>alignair_lab</code>, which runs architecture experiments. This page names the model's internal parts, which
        the user's pages never do; <DocLink to="how-alignair-works">How AlignAIR works</DocLink> tells the same mechanism
        without the code.
      </p>

      <H2 id="packages">The packages</H2>
      <p>Each package owns one concern. Paths are under <code>src/alignair</code>.</p>
      <DocTable
        head={["Package", "What it owns", "Start reading at"]}
        rows={[
          [
            <code>alignair</code>,
            <>
              The supported Python API: <code>Aligner</code> and <code>PredictionResult</code>,{" "}
              <code>read_sequences</code>, <code>iter_sequences</code>, <code>write_airr</code>, <code>AirrWriter</code>,{" "}
              <code>ReferenceSet</code>, <code>TrainingConfig</code>, <code>run_training</code>, <code>TrainingRun</code>,{" "}
              <code>compare_airr</code> and <code>AlignAIRError</code>.
            </>,
            <>
              <code>__init__.py</code>, <code>aligner.py</code>
            </>,
          ],
          [
            <code>cli</code>,
            <>
              The <code>alignair</code> command. <code>main.py</code> builds the parser and registers each subcommand from
              its own module; <code>_options.py</code> defines the options several commands share; <code>_errors.py</code>{" "}
              reports a failure and returns its exit code.
            </>,
            <>
              <code>cli/main.py</code>, <code>cli/predict.py</code>
            </>,
          ],
          [
            <code>core</code>,
            <>
              The model object <code>AlignAIR</code>, its configuration <code>AlignAIRConfig</code>, the architecture
              registry (<code>ModelComponents</code>, <code>register_architecture</code>), binding a configuration to a
              reference (<code>prepare_model</code>, <code>ModelSetup</code>, <code>validate_reference_binding</code>),
              the four orientations, the token vocabulary and the read window.
            </>,
            <>
              <code>core/model.py</code>, <code>core/components.py</code>
            </>,
          ],
          [
            <code>network</code>,
            <>
              The chip, AlignAIR's one built-in architecture (<code>chip-v1</code>): <code>ChipCore</code> and its parts,
              its training objective <code>ChipObjective</code>, and <code>network/refine</code>, the exact steps it runs
              at inference.
            </>,
            <code>network/chip.py</code>,
          ],
          [
            <code>predict</code>,
            <>
              The prediction pipeline, from the model's outputs to AIRR rows, as twelve named stages over one typed
              state. <code>predict/germline</code> places each call in its germline; <code>predict/airr</code> assembles
              the AIRR record; <code>predict/junction.py</code> derives the junction and productivity.
            </>,
            <>
              <code>predict/pipeline.py</code>, <code>predict/stages.py</code>
            </>,
          ],
          [
            <code>reference</code>,
            <>
              Germline references: <code>ReferenceSet</code> (each gene's alleles with their anchors, IMGT-gapped forms,
              loci and provenance), the gene frames, and <code>annotate</code>, which fills a reference's missing anchors
              and gaps from a known one.
            </>,
            <>
              <code>reference/reference_set.py</code>, <code>reference/frames.py</code>
            </>,
          ],
          [
            <code>genotype</code>,
            <>
              Genotype files (<code>load_genotype</code>), a genotype as a subset of the reference (
              <code>genotype_bank</code>), the frequency rule that infers a donor's V genotype, and the AIRR GenotypeSet.
            </>,
            <>
              <code>genotype/bank.py</code>, <code>genotype/infer.py</code>
            </>,
          ],
          [
            <code>io</code>,
            <>
              Reading reads (FASTA, FASTQ, CSV/TSV, gzip, standard input), the metadata join, writing the AIRR TSV, its
              column presets, and the column dictionary's text (<code>io/columns_doc.py</code>).
            </>,
            <>
              <code>io/sequence_reader.py</code>, <code>io/airr.py</code>
            </>,
          ],
          [
            <code>model_file</code>,
            <>
              The <code>.alignair</code> file: its byte layout (<code>container.py</code>), each section's encoding (
              <code>serialize.py</code>), and saving, loading and partial reads (<code>io.py</code>).
            </>,
            <code>model_file/container.py</code>,
          ],
          [
            <code>registry</code>,
            <>
              Model ids: registry sources and their <code>registry.json</code> (<code>sources.py</code>), the verified
              cache and <code>resolve_model</code> (<code>cache.py</code>), Hugging Face repos (<code>hf.py</code>), the
              validator (<code>validate.py</code>), the maintainer's publishing tools (<code>publish.py</code>) and the
              update notice (<code>updates.py</code>).
            </>,
            <code>registry/cache.py</code>,
          ],
          [
            <code>train</code>,
            <>
              Training: <code>TrainingConfig</code>, <code>run_training</code> and <code>TrainingRun</code>, the trainer,
              the engine and loop (internal), checkpoints, the bundle export (<code>bundle.py</code>), warm starts (
              <code>transfer.py</code>), and <code>train/records</code>, the streams of simulated training reads.
            </>,
            <>
              <code>train/workflow.py</code>, <code>train/trainer.py</code>
            </>,
          ],
          [
            <code>data</code>,
            <>
              What the model reads and learns from: read normalisation and tokens (<code>data/inputs.py</code>), the
              training batch adapter (<code>NucleotideBatchAdapter</code>), the chip's per-base targets, library flanks,
              and each training batch's novel alleles.
            </>,
            <>
              <code>data/inputs.py</code>, <code>data/adapters.py</code>
            </>,
          ],
          [
            <code>tasks</code>,
            <>
              The training-task contract (<code>TrainingTasks</code>, <code>build_tasks</code>) and metrics with explicit
              support counts (<code>MetricReport</code>).
            </>,
            <code>tasks/objective.py</code>,
          ],
          [
            <code>genairr</code>,
            <>
              AlignAIR's one adapter to GenAIRR: dataconfigs by name, anchors, the simulated experiment, a custom
              reference from FASTA files, and novel alleles.
            </>,
            <code>genairr/__init__.py</code>,
          ],
          [
            <code>errors</code>,
            <>
              The error contract: <code>AlignAIRError</code> and its classes, each with a code (
              <code>ALIGNAIR_&lt;THING&gt;_&lt;STATE&gt;</code>), a category, an exit code and <code>to_dict()</code>;{" "}
              <code>error_context</code> attaches what an operation was doing.
            </>,
            <code>errors/base.py</code>,
          ],
          [
            <>
              <code>analyze</code>, <code>evaluate</code>
            </>,
            <>
              <code>alignair analyze</code> and <code>alignair compare</code> (<code>compare_airr</code>);{" "}
              <code>alignair benchmark</code> (<code>run_benchmark</code>).
            </>,
            <>
              <code>analyze/compare.py</code>, <code>evaluate/benchmark.py</code>
            </>,
          ],
          [
            <code>parallel.py</code>,
            <>
              <code>predict --threads</code>: prediction on the CPU in worker processes of one thread each.
            </>,
            <code>parallel.py</code>,
          ],
        ]}
      />
      <p>
        Two parts have compiled versions, built from Cython when a wheel is built: the germline CIGAR (
        <code>predict/germline/_derive_cy.pyx</code>) and the V re-score's band (
        <code>network/refine/_vband_cy.pyx</code>). Each has a pure-Python fallback that gives the same result, more
        slowly; <code>alignair doctor</code> reports which one runs (<code>derive_backend</code>,{" "}
        <code>vband_backend</code>).
      </p>

      <H2 id="terms">The model's parts</H2>
      <p>These words name the model's internals. Each is defined here once; the code uses the same words.</p>
      <DocTable
        head={["Term", "Meaning", "Code"]}
        rows={[
          [
            "Weights",
            "The model's learned numbers, fitted in training: the weights section of a model file.",
            <code>AlignAIR.from_state_dict</code>,
          ],
          [
            "Reference",
            <>
              The germline set a model calls from: each gene's (V, D, J) alleles. It is bound when the model loads and is
              not part of the weights.
            </>,
            <code>ReferenceSet</code>,
          ],
          [
            "Frame",
            <>
              A fixed set of columns per gene on which every allele is laid out, so a column is the same germline position
              across alleles and across references: V uses its IMGT-gapped form, J is aligned on its W/F codon, D uses its
              own positions.
            </>,
            <>
              <code>reference/frames.py</code> (<code>FRAME_WIDTHS</code>)
            </>,
          ],
          [
            "Bank",
            <>
              The bound reference compiled for the model: one row per allele (two per D allele: forward and inverted) of
              one-hot bases on its gene's frame. <code>bind_reference</code> rebuilds it; it is never saved. A genotype
              is a subset of its rows.
            </>,
            <>
              <code>network/bank.py</code> (<code>BankCompiler</code>)
            </>,
          ],
          [
            "Architecture",
            "A registered name for a set of model components. The configuration stores it; loading resolves it.",
            <code>core/components.py</code>,
          ],
          [
            "Chip",
            <>
              The one built-in architecture, <code>chip-v1</code>: the core that holds the trunk, parser, anchor heads,
              SHM reader, scans and D presence.
            </>,
            <>
              <code>network/chip.py</code> (<code>ChipCore</code>)
            </>,
          ],
          [
            "Forward pass",
            "One run of the model over a batch of reads.",
            <>
              <code>AlignAIR.forward</code>, <code>predict/forward.py</code>
            </>,
          ],
          [
            "Embedding",
            <>
              Each token of the read (pad, A, T, G, C, N) and its position, as <code>embed_dim</code> learned numbers.
            </>,
            <code>core/embedding.py</code>,
          ],
          [
            "Head",
            "A small output layer that answers one question from the model's features.",
            "",
          ],
          [
            "Orientation head",
            <>
              Says which of four orientations a read is in (forward, reverse complement, complement, reverse). At
              inference, the bank orientation overrides it where one orientation wins a vote of the read's 12-base words
              against the bound reference.
            </>,
            <>
              <code>core/orientation.py</code> (<code>EmbeddingOrientationHead</code>, <code>BankOrientation</code>)
            </>,
          ],
          [
            "Trunk",
            "Dilated convolutions that give every base its features once, for all the parts below to share.",
            <>
              <code>network/trunk.py</code> (<code>DilatedTrunk</code>)
            </>,
          ],
          [
            "Parser",
            <>
              A linear-chain CRF over the regions before V, V, N1, D, N2, J and after, which only moves forward: each
              base's region and each boundary.
            </>,
            <>
              <code>network/parser.py</code> (<code>Parser</code>)
            </>,
          ],
          [
            "Anchor heads",
            "Find the V's conserved Cys codon and the J's Trp/Phe codon, each inside its region.",
            <>
              <code>network/trunk.py</code> (<code>AnchorHead</code>)
            </>,
          ],
          [
            "SHM reader",
            "Per base: germline, substitution or insertion, and the base's column in each gene's frame.",
            <>
              <code>network/reader.py</code> (<code>ShmReader</code>)
            </>,
          ],
          [
            "Scan",
            "Scores every bank row of one gene against the read's bases of that gene, in one matrix product.",
            <>
              <code>network/scan.py</code> (<code>BankScan</code>)
            </>,
          ],
          [
            "D presence",
            "Absent, too short to call, or callable, from the parser's and the D scan's evidence; its inputs are detached, so it does not reshape that evidence.",
            <>
              <code>network/presence.py</code> (<code>DPresenceEvidence</code>)
            </>,
          ],
          [
            "Logits",
            "Scores that a softmax turns into probabilities: of a boundary position, an anchor, an orientation.",
            "",
          ],
          [
            "Refinement",
            <>
              Exact steps without weights, run at inference: one locus per read, the V end and J start along the called
              germlines, the D's local alignment and edges, and the top five V alleles re-scored by a banded gapped
              alignment.
            </>,
            <code>network/refine/</code>,
          ],
          [
            "Candidate set",
            <>
              A gene's alleles scoring at least 0.5 relative to the best, three at most: <code>*_call_set</code> in the
              table.
            </>,
            <code>predict/threshold.py</code>,
          ],
          [
            "Ties",
            <>
              The V alleles that match as many of the read's V bases as the call; listed after it in <code>v_call</code>.
            </>,
            <code>predict/ties.py</code>,
          ],
          [
            "Objective",
            <>
              Each part's own loss against its simulated truth, divided by its chance level, summed with weight 1 (
              <code>LOSS_WEIGHTS</code>).
            </>,
            <>
              <code>network/objective.py</code> (<code>ChipObjective</code>)
            </>,
          ],
          [
            "Teacher forcing",
            <>
              In training, a read uses its true regions and columns in place of the parser's and the SHM reader's, with
              probability <code>P_TEACHER</code> (0.5).
            </>,
            <code>network/chip.py</code>,
          ],
          [
            "Curriculum",
            <>
              Corruption settings that ramp from clean reads (progress 0) to fully corrupted ones (progress 1); a training
              mix draws reads from several points on it.
            </>,
            <code>train/records/curriculum.py</code>,
          ],
          [
            "Record stream",
            "A lazy stream of GenAIRR records for one part of the mix, drawn from its own block of seeds.",
            <code>train/records/streams.py</code>,
          ],
          [
            "Checkpoint",
            "A model file that also holds the trusted pickle sections, from which training resumes.",
            <code>train/checkpoints.py</code>,
          ],
        ]}
      />

      <H2 id="read-to-row">From a read to an AIRR row</H2>
      <p>
        <code>alignair predict</code> and the Python API run the same path. <code>cli/predict.py</code> adds the file
        handling around it: streaming, the metadata join, the genotype pass and the run record.
      </p>

      <H3 id="load">1. Load the model</H3>
      <p>
        <code>Aligner.from_pretrained</code> turns the model spec into a local file with{" "}
        <code>registry.resolve_model</code>: a path is used as it is; a catalog id or <code>id@version</code> is looked up
        in the registries and downloaded once into the cache, checked against the catalog's SHA-256;{" "}
        <code>hf://org/repo</code> comes from that Hugging Face repo. <code>model_file.load_model</code> then restores the
        weights (<code>AlignAIR.from_state_dict</code>, strict on names and shapes), takes the reference from the file's{" "}
        <code>reference_json</code> section, or the <code>--reference</code> FASTA annotated against it, and binds it (
        <code>bind_reference</code>), which compiles the bank. A genotype binds a subset of that reference (
        <code>Aligner.with_genotype</code>, <code>genotype_bank</code>): the bank keeps only its rows.
      </p>

      <H3 id="read">2. Read the input</H3>
      <p>
        <code>io.iter_sequence_batches</code> streams the input in chunks (<code>--chunk-size</code>, 20,000 reads). It
        uppercases each read, turns IUPAC ambiguity codes into N, drops a read with more than 20% other symbols, and
        renames a repeated id <code>_dup1</code>, <code>_dup2</code>. With <code>--metadata</code>,{" "}
        <code>io.build_metadata_index</code> indexes the table in a temporary SQLite file, and each chunk's rows are
        looked up by read id.
      </p>

      <H3 id="pipeline">3. Run the pipeline</H3>
      <p>
        Each chunk goes to <code>Aligner.predict</code>, or, with <code>--threads N</code>, to{" "}
        <code>parallel.ProcessPool</code>, whose single-threaded workers each hold a copy of the aligner.{" "}
        <code>Aligner.predict</code> builds a <code>PredictConfig</code> and calls <code>predict.predict</code>, which runs{" "}
        <code>PIPELINE</code>: twelve stages, each a function of <code>predict/stages.py</code> that takes a{" "}
        <code>PredictionState</code> and returns a copy with its own fields set.
      </p>
      <DocTable
        head={["Stage", "Function", "Sets", "What it does"]}
        rows={[
          [<code>input_policy</code>, <code>prepare_reads</code>, <><code>reads</code>, <code>cropped</code></>, "Cuts each read to the model's window (575 nt) and flags the cut ones."],
          [<code>forward</code>, <code>run_forward</code>, <code>batches</code>, <>The forward pass, batch by batch; each batch's outputs typed as <code>ModelOutputs</code>.</>],
          [<code>clean</code>, <code>decode_outputs</code>, <code>predictions</code>, "Merges the batches; a boundary is the argmax of its position logits."],
          [<code>canonicalize</code>, <code>canonicalize_reads</code>, <code>canonical</code>, "Turns each read to its forward orientation."],
          [<code>segments</code>, <code>correct_spans</code>, <code>segments</code>, "Integer V, D and J spans, inside the read and in order."],
          [<code>alleles</code>, <code>select_calls</code>, <code>calls</code>, "Each gene's candidate set from the allele scores."],
          [<code>germline</code>, <code>align_reads</code>, <code>alignments</code>, "Places each call in its germline: read and germline coordinates, CIGAR."],
          [<code>better_v</code>, <code>promote_better_v</code>, <code>alignments</code>, "A bound V allele the read matches better than the call becomes the call (how a novel allele is called)."],
          [<code>v_presence</code>, <code>find_absent_v</code>, <code>no_v</code>, "Finds the reads that hold no V."],
          [<code>ties</code>, <code>list_v_ties</code>, <code>v_ties</code>, "Lists the V alleles the read cannot tell from its call."],
          [<code>records</code>, <code>build_records</code>, <code>results</code>, <>Each read's typed result (<code>ReadResult</code>).</>],
          [<code>airr</code>, <code>build_rows</code>, <code>rows</code>, <>The AIRR rows (<code>predict.airr.build_airr</code>): junction and productivity from the anchors, IMGT-gapped alignments, regions, identities, flags.</>],
        ]}
      />

      <H3 id="forward-pass">Inside the forward stage</H3>
      <p>
        <code>predict.forward.run_model</code> batches reads of similar width (<code>bucket_batches</code>), runs the model
        in evaluation mode and refuses a non-finite output. On one batch, <code>AlignAIR.forward</code> does:
      </p>
      <CodeBlock
        lang="text"
        title="AlignAIR.forward"
        code={`tokens (batch, 576)          core/tokens.py: pad A T G C N -> 0..5, right-padded
  embedding                  core/embedding.py
  orientation head           core/orientation.py; the bank's 12-base vote wins where it decides
  re-embed the read, turned forward
  core: ChipCore             network/chip.py
    trunk                    every base's features, once
    parser                   region marginals and boundary logits
    anchor heads             the V Cys and J Trp/Phe codon starts
    SHM reader               per base: edit state and frame columns
    scans (V, D, J)          a log posterior over the bank's rows
    D presence               absent / too short to call / callable
    refine (inference only)  locus per read, V end and J start, D local and edges, V re-score
outputs                      {gene}_allele, {gene}_start, {gene}_end, anchor logits, orientation, ...`}
      />
      <p>
        <code>predict.outputs.ModelOutputs.from_network</code> is the one place the pipeline reads the model's output keys:
        a missing required key or an unknown one is an <code>InferenceError</code>.
      </p>

      <H3 id="write">4. Write the table</H3>
      <p>
        <code>io.airr.AirrWriter</code> writes each chunk's rows, with the joined metadata columns, as they come, so memory
        stays flat on any input size. The column presets are <code>io.airr.COLUMN_PRESETS</code>.{" "}
        <code>cli/predict.py</code> then writes <code>&lt;out&gt;.run.json</code>: the model, the bound reference's
        digests, the options and the read counts.
      </p>

      <H3 id="trace">Watch the stages</H3>
      <p>
        <code>run_stages</code> runs the pipeline with hooks. <code>Trace</code>, a hook, keeps each stage's output state
        and its seconds:
      </p>
      <CodeBlock
        lang="python"
        code={`from alignair import Aligner
from alignair.predict import PredictConfig, PredictionState, Trace, run_stages

aligner = Aligner.from_pretrained("alignair4-igh-human")
reads = ["CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGTCCTCGGTGAAGGTC..."]
cfg = PredictConfig.for_model(aligner.model, aligner.reference, batch_size=64)
trace = Trace()
state = run_stages(PredictionState.start(aligner.model, aligner.reference, cfg, aligner.device, reads),
                   hooks=[trace])
print(trace.seconds)                      # each stage's seconds, in order
segments = trace.states["segments"].segments
rows = state.rows                         # what Aligner.predict returns as records`}
      />

      <H2 id="training">Training</H2>
      <p>
        <code>run_training(config)</code> (<code>train/workflow.py</code>) is the one entry point.{" "}
        <code>alignair train</code> builds a <code>TrainingConfig</code>, calls it, and exports a bundle. In order:
      </p>
      <ol>
        <li>
          <strong>Reference.</strong> The config's dataconfigs (GenAIRR's built-in germline sets, or one built from FASTA
          files) resolve through <code>alignair.genairr</code> into a <code>ReferenceSet</code>.
        </li>
        <li>
          <strong>Model.</strong> <code>prepare_model(reference, ..., **config.model_options).build_model()</code>. With{" "}
          <code>init_from</code>, <code>train.transfer.transfer_compatible_weights</code> copies every tensor whose name
          and shape match; the rest start fresh.
        </li>
        <li>
          <strong>Objective.</strong> <code>tasks.build_tasks(config)</code> asks the architecture for its objective (
          <code>ChipObjective</code> for the chip).
        </li>
        <li>
          <strong>Reads.</strong> <code>train.records.training_records</code> draws GenAIRR records from the curriculum's
          streams (<code>CurriculumStreamPolicy</code>), each stream from its own block of seeds (
          <code>train.records.seeds.seed_block</code>); part of them come from synthetic novel V alleles (
          <code>NovelAlleles</code>), many with library flanks (<code>data.flanks.FlankPolicy</code>). With{" "}
          <code>data_workers</code> above 0 the streams run in worker processes (<code>train/records/workers.py</code>,{" "}
          <code>train/assembly.py</code>) and give the same batches as one process does.
        </li>
        <li>
          <strong>Batches.</strong> <code>data.adapters.NucleotideBatchAdapter</code> turns records into tensors and the
          chip's per-base targets (<code>data/chip_inputs.py</code>), each read in one of the four orientations.
        </li>
        <li>
          <strong>Steps.</strong> The engine (<code>train/setup.py</code>, <code>train/engine.py</code>) runs AdamW updates,
          with an optional cosine learning-rate schedule (<code>CosineSchedule</code>); <code>train/guards.py</code> stops
          a run on a non-finite loss or gradient. A run's <code>on_iteration</code> callback sees each step.
        </li>
        <li>
          <strong>Checkpoints.</strong> <code>train/checkpoints.py</code> saves rotating checkpoints and the best one with{" "}
          <code>model_file.save_model</code>, pickle sections included, so a run resumes (<code>resume_path</code>).{" "}
          <code>train.bundle.export_bundle</code> (<code>alignair export</code>) writes the pickle-free model to share.
        </li>
      </ol>
      <p>
        The user's side of this is on <DocLink to="train">Train a model for your reference</DocLink> and{" "}
        <DocLink to="fine-tune">Fine-tune a model</DocLink>.
      </p>

      <H2 id="model-file">The model file</H2>
      <p>
        One <code>.alignair</code> file per model holds everything a run needs: the weights, the configuration and the
        germline reference. Its byte layout (<Source path="src/alignair/model_file/container.py" />), integers
        little-endian:
      </p>
      <DocTable
        head={["Offset", "Size", "Content"]}
        rows={[
          ["0", "7", <><code>b"ALGNAIR"</code>, the format tag</>],
          ["7", "1", <>The major format version (<code>MAJOR_VERSION</code>, 1); a reader refuses a newer one.</>],
          ["8", "8", <>u64: <em>H</em>, the JSON header's length in bytes</>],
          ["16", <em>H</em>, "The JSON header, UTF-8"],
          [<>16 + <em>H</em></>, "...", "The sections' blobs, back to back, each compressed on its own"],
        ]}
      />
      <p>
        The header's <code>sections</code> maps each section's name to its blob: <code>offset</code> (counted from the end
        of the header), <code>compressed_length</code>, <code>payload_length</code>, <code>codec</code> (
        <code>none</code>, <code>zlib</code> or <code>zstd</code>; reading <code>zstd</code> needs the{" "}
        <code>zstandard</code> package), <code>compressed_sha256</code> and <code>payload_sha256</code> (both checked on
        every read), and <code>format</code> (<code>json</code>, <code>safetensors</code>, <code>fasta</code>,{" "}
        <code>python-pickle</code> or <code>bytes</code>). Its other keys describe the model (
        <code>model_file.io.save_model</code> writes them): the format versions, the model's settings, its training, its
        reference's digests (<code>allele_order_sha256</code>, <code>reference_fasta_sha256</code>), its provenance and
        the card's descriptive fields (<code>model_id</code>, <code>model_version</code>, <code>description</code>,{" "}
        <code>license</code>).
      </p>
      <DocTable
        head={["Section", "Format", "Kind", "Holds"]}
        rows={[
          [<code>config</code>, "JSON", "portable", <>The <code>AlignAIRConfig</code>: architecture name, widths, window.</>],
          [<code>weights</code>, "safetensors", "portable", "The weights."],
          [<code>reference</code>, "FASTA", "portable", "The reference's alleles."],
          [<code>reference_json</code>, "JSON", "portable", "The full reference (anchors, gapped forms, loci); a model loads its reference from it."],
          [<code>dataconfig/&lt;i&gt;</code>, "pickle", "trusted", "Each GenAIRR dataconfig the model trained on."],
          [<code>train_state</code>, "pickle", "trusted", "The optimizer, scheduler and random state."],
        ]}
      />
      <p>
        A file is one of two kinds. A <strong>training checkpoint</strong> holds all six sections; pickle runs code when it
        loads, so open only checkpoints you trust. A <strong>publish artifact</strong> (
        <code>save_model(..., include_trusted_pickle=False)</code>, what <code>alignair export</code> writes) holds only
        the four portable sections, loads without running any pickle, and is the only kind the catalog accepts. Loading
        never imports code from a file: the configuration names an architecture, which the loading process must have
        registered. To read a file without loading the model:
      </p>
      <CodeBlock
        lang="python"
        code={`from alignair.model_file import read_metadata, read_model_state

header = read_metadata("model.alignair")      # the JSON header, as alignair info --json prints it
print({name: section["format"] for name, section in header["sections"].items()})
state = read_model_state("model.alignair")    # config, weights as CPU tensors and the reference; no model built`}
      />

      <H2 id="extend">Extension points</H2>

      <H3 id="new-architecture">A new architecture</H3>
      <p>
        The model always embeds the read and corrects its orientation; everything after that is the architecture's core.{" "}
        <code>ModelComponents</code> (<Source path="src/alignair/core/components.py" />) holds four factories, each called
        with the <code>AlignAIRConfig</code>:
      </p>
      <DocTable
        head={["Field", "Required", "Builds"]}
        rows={[
          [
            <code>core</code>,
            "yes",
            <>
              The core. The model calls it as <code>core(emb, mask, end_mask, *, tokens, batch)</code> with the corrected
              embedding, the read and end masks, the corrected tokens and the raw batch, and adds the dict it returns to
              its outputs. A core that reads the reference defines <code>bind_reference(reference)</code>.
            </>,
          ],
          [
            <code>objective</code>,
            "yes",
            <>
              <code>objective(config)</code> returns the architecture's <code>TrainingTasks</code>: its losses and
              metrics.
            </>,
          ],
          [<code>embedding</code>, "no", "Defaults to the token-and-position embedding."],
          [<code>orientation_head</code>, "no", "Defaults to the 4-class orientation head."],
        ]}
      />
      <p>
        For the prediction pipeline to read a core at inference, its outputs must give, for each gene the model calls,{" "}
        <code>{"{gene}_allele"}</code>, <code>{"{gene}_start"}</code> and <code>{"{gene}_end"}</code>, plus{" "}
        <code>mutation_rate</code> and <code>indel_count</code>. The other keys it may give are listed in{" "}
        <code>predict/outputs.py</code>, and any key outside them is refused. This example keeps the chip and drops the
        position embedding:
      </p>
      <CodeBlock
        lang="python"
        code={`from dataclasses import replace

from torch import nn

from alignair.core import prepare_model, register_architecture
from alignair.core.components import architecture_components
from alignair.reference import ReferenceSet


class TokenOnlyEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokens = nn.Embedding(config.vocab_size, config.embed_dim)

    def forward(self, tokens):
        return self.tokens(tokens)


register_architecture("chip-token-only-v1",
                      replace(architecture_components("chip-v1"), embedding=TokenOnlyEmbedding))

reference = ReferenceSet.from_fasta("germline.fasta")
model = prepare_model(reference, architecture="chip-token-only-v1").build_model()
print(sorted(model.component_modules()))   # ['core', 'embedding', 'orientation_head']`}
      />
      <p>
        Construction checks that each factory returns a PyTorch module and raises <code>ConfigurationError</code> naming
        the component otherwise; it runs no trial forward pass, so test output shapes and gradients yourself.{" "}
        <code>register_architecture</code> never overwrites a name: give a changed architecture a new, versioned name. To
        train one, set it in the config: <code>TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", preset="quick",
        model_options={"{"}"architecture": "chip-token-only-v1"{"}"})</code>.
      </p>
      <p>
        Every process that loads such a model must register the architecture first, or name the modules that register it
        in <code>ALIGNAIR_ARCHITECTURE_MODULES</code> (comma-separated), which worker processes inherit; an unknown name
        is a <code>ConfigurationError</code> (&ldquo;unknown model architecture&rdquo;). These tests cover the contract:
      </p>
      <CodeBlock
        code={`python -m pytest tests/alignair/core/test_core_hook.py \\
  tests/alignair/core/test_architecture_modules.py tests/alignair/network/test_registration.py`}
      />

      <H3 id="pipeline-hooks">Pipeline hooks</H3>
      <p>
        <code>run_stages(state, stages=PIPELINE, hooks=())</code> wraps every stage call in each hook,{" "}
        <code>hook(stage, state, run)</code>, the first hook outermost; a hook may replace what a stage returns. A run of
        fewer stages (<code>PIPELINE[:3]</code>) stops early. A stage that leaves a field it sets unset is an{" "}
        <code>InferenceError</code>.
      </p>

      <H3 id="record-source">Another source of training reads</H3>
      <p>
        <code>run_training(config, record_source=...)</code> replaces the GenAIRR provider of every training and
        validation stream. A <code>RecordSource</code> is a callable <code>(dataconfig, parameters, seed)</code> that
        returns fresh record mappings; with <code>data_workers</code> above 0 it runs in worker processes, so it must
        pickle.
      </p>

      <H3 id="registries">Registries</H3>
      <p>
        A model id is looked up in the registries, in this order: <code>--registry</code> (or <code>registry=</code>),
        then <code>ALIGNAIR_REGISTRY</code> (comma-separated), then the <code>registries</code> list of{" "}
        <code>config.toml</code> in AlignAIR's config directory (<code>~/.config/alignair</code> on Linux; the{" "}
        <code>config_dir</code> line of <code>alignair doctor</code>), then the shared catalog. A registry is{" "}
        <code>hf://org/repo</code>, an{" "}
        <code>https://</code> URL, or a <code>file://</code> path: any directory that holds a{" "}
        <code>registry.json</code> and its model files. <DocLink to="publishing">Publishing to the catalog</DocLink> has
        the format.
      </p>

      <H2 id="public-api">What to build on</H2>
      <p>
        The supported API is what <code>alignair</code>, <code>alignair.io</code>, <code>alignair.genotype</code>,{" "}
        <code>alignair.registry</code>, <code>alignair.reference</code> and <code>alignair.train</code> export (
        <DocLink to="python-api">Python API</DocLink>), with the extension points above. The training engine and loop (
        <code>train.engine</code>, <code>train.loop</code>) and the modules whose names start with an underscore are
        internal and change without notice.
      </p>
    </>
  ),
};

/* ---------------------------------------------------------------------------------------------------------------- */
/* Testing                                                                                                           */
/* ---------------------------------------------------------------------------------------------------------------- */

const testing: DocPage = {
  slug: "testing",
  title: "Testing",
  section: "Developers",
  lead: (
    <>
      Run the test suite, choose tests by marker, and pass the gates CI applies: lint, the docs checks, the site's data
      and its plain language; then the website's own tests and build.
    </>
  ),
  body: () => (
    <>
      <H2 id="setup">Set up</H2>
      <p>From a clone of the repository:</p>
      <CodeBlock
        code={`git clone https://github.com/MuteJester/AlignAIR && cd AlignAIR
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build you need
pip install -e ".[dev]"
python -m pytest`}
      />
      <p>
        The <code>test</code> extra adds pytest, the official <code>airr</code> validator, <code>pytest-cov</code>,{" "}
        <code>pytest-timeout</code>, <code>pytest-xdist</code>, the pinned <code>ruff</code>, Jinja2 (the conda recipe
        tests) and SciPy (the architecture lab). <code>dev</code> is <code>cli</code> and <code>test</code> plus{" "}
        <code>build</code>, <code>twine</code> and <code>cibuildwheel</code>; CI installs <code>.[cli,test]</code>.
      </p>
      <p>
        Pytest reads its settings from <code>pyproject.toml</code>: it collects <code>tests/</code>, puts{" "}
        <code>src</code> and the repository root on the path (so <code>alignair_assay</code>, <code>alignair_lab</code> and{" "}
        <code>tests.support</code> import from the checkout), checks markers and settings strictly, and stops any test or
        fixture after 300 seconds.
      </p>

      <H2 id="layout">Where the tests are</H2>
      <DocTable
        head={["Under tests/", "What it covers"]}
        rows={[
          [<code>alignair/</code>, "The package, one directory per subsystem: CLI, I/O, reference, genotype, model files, network, refinement, prediction, tasks, training, registry and errors."],
          [<code>alignair_assay/</code>, "The evaluation harness: observable truth, AIRR scoring, metrics and statistics, suites, runners, reports and its CLI."],
          [<code>alignair_lab/</code>, "The architecture lab: recipes, record providers, screens, noise statistics and replicates."],
          [<code>acceptance/</code>, "Public guarantees: official AIRR validation of golden output, a novel allele given in the reference is called, identical batches from data workers."],
          [<code>robustness/</code>, "Malformed input, and invariants across packages, such as the orientation involution."],
          [<code>packaging/</code>, "Conda recipes, dependency pins, running without the optional packages, the compiled wheel, and the release smoke test."],
          [<code>quality/</code>, "The QA gates themselves: lint, release claims, coverage, isolation, stale text, and that retired names and the development packages stay out of the product."],
          [<code>website/</code>, "The README and the website: links, plain language, the README's shape, and the generated site data."],
          [<code>support/</code>, "Pytest plugins (the policy, environment isolation, model artifacts), the release-claim map and the coverage gate; not part of the package."],
          [<code>fixtures/models/validation.json</code>, "The public models the model-acceptance tests use: ids, versions, sizes and SHA-256 checksums."],
        ]}
      />

      <H2 id="markers">Layers and markers</H2>
      <p>
        Every test declares exactly one layer, usually with <code>pytestmark</code> at the top of its module:{" "}
        <code>component</code> (deterministic contracts, with injected inputs where needed), <code>integration</code> (the
        real simulator, processes, files) or <code>acceptance</code> (public guarantees). Collection fails otherwise
        (&ldquo;declare exactly one QA layer&rdquo;). Four more markers describe what a test needs:
      </p>
      <DocTable
        head={["Marker", "Meaning"]}
        rows={[
          [<code>slow</code>, <>A real model or reference; leave it out of fast runs with <code>-m "not slow"</code>.</>],
          [<code>performance</code>, "Resource behaviour with an explicit budget."],
          [<code>model_validation</code>, "Needs the pre-fetched, checksum-pinned public models."],
          [<code>claim_evidence</code>, "A test that a release claim names; added automatically."],
        ]}
      />

      <H2 id="selection">Run what you need</H2>
      <CodeBlock
        code={`# The whole suite
python -m pytest tests

# Component contracts, without the slow or model-backed tests
python -m pytest tests -m "component and not slow and not model_validation"

# The real simulator, processes and serialization
python -m pytest tests -m integration

# The same contracts in a reproducible shuffled order, and in two concurrent workers (CI runs both)
python -m pytest tests -m "component and not slow and not model_validation" \\
  --test-order=shuffle --order-seed=20261026
python -m pytest tests -m "component and not slow and not model_validation" -n 2

# One subsystem
python -m pytest tests/alignair/predict

# What would run, without running it
python -m pytest --collect-only -q`}
      />
      <p>The suite adds these options (<Source path="tests/support/policy.py" />):</p>
      <DocTable
        head={["Option", "What it does"]}
        rows={[
          [<code>--test-order collected|shuffle|reverse</code>, <>The order tests run in; <code>--order-seed</code> seeds the shuffle.</>],
          [<code>--model-cache DIR</code>, "The directory of pre-fetched, checksum-pinned models the model-validation tests read."],
          [<code>--enforce-claims</code>, "Fails the run when a release claim's test was not selected or did not pass; runs serially, so -n is refused."],
          [<code>--claim-report FILE</code>, "Writes each claim's outcome as JSON."],
        ]}
      />

      <H2 id="gates">Quality gates</H2>
      <p>These are tests in the suite, so <code>python -m pytest</code> runs them all; CI fails on any of them.</p>
      <DocTable
        head={["Gate", "Test", "What fails it"]}
        rows={[
          [
            "Lint",
            <code>tests/quality/test_lint.py</code>,
            <>
              Any ruff finding in <code>src/alignair</code> under <code>pyproject.toml</code>'s{" "}
              <code>[tool.ruff]</code> settings. Run it yourself with <code>ruff check src/alignair</code>. Outside CI it
              skips when ruff is missing or not the pinned version; in CI it fails.
            </>,
          ],
          [
            "Commands in the docs",
            <code>tests/alignair/cli/test_docs_cli_sync.py</code>,
            <>
              An <code>alignair ...</code> command that does not parse against the real CLI, in{" "}
              <code>README.md</code>, an <code>examples/**/README.md</code> (<code>bash</code>, <code>sh</code> and{" "}
              <code>console</code> blocks) or any website code block (<code>code=</code> and <code>cli=</code>).
            </>,
          ],
          [
            "Links",
            <code>tests/website/test_links.py</code>,
            <>
              A <code>DocLink</code>, <code>/docs/</code> or <code>#/docs/</code> link to a page that does not exist, or a{" "}
              <code>?h=</code> heading that page does not have.
            </>,
          ],
          [
            "The README",
            <code>tests/website/test_readme.py</code>,
            "A README too long or too short, the first alignment too far down, a missing model id or section, or sections out of order.",
          ],
          [
            "Site data drift",
            <code>tests/website/test_site_data.py</code>,
            <>
              A file in <code>website/src/data/</code> that differs from what <code>website/scripts/site_data.py</code>{" "}
              writes from the code now.
            </>,
          ],
          [
            "Plain language",
            <code>tests/website/test_plain_language.py</code>,
            <>
              On the README and the user's pages: a word of the model's internals (neural, chip, trunk, heads, bank,
              forward pass, embedding, curriculum, logit), wording that says AlignAIR 4 or its models are not out yet,
              or a raw <code>h2</code> or <code>h3</code> tag in place of <code>H2</code> and <code>H3</code> with an id.
              The developer pages and How AlignAIR works may name the internals.
            </>,
          ],
          [
            "Stale text",
            <code>tests/quality/test_stale_text.py</code>,
            "Docstrings that describe removed behaviour, experiment history in the shipped code, or a public name without a docstring.",
          ],
          [
            "Release claims",
            <code>tests/quality/test_claims.py</code>,
            <>
              A claim in <code>tests.support.claims.CLAIM_TESTS</code> that names a test that does not exist.
            </>,
          ],
        ]}
      />

      <H2 id="coverage">Coverage</H2>
      <CodeBlock
        code={`python -m pytest tests --cov=alignair \\
  --cov-report=json:test-results/coverage.json --junitxml=test-results/junit.xml
python -m tests.support.coverage_gate --coverage test-results/coverage.json`}
      />
      <p>
        <code>tests/coverage-policy.json</code> gives critical files a statement and a branch floor each; the gate fails a
        file under its floor or missing from the report. CI runs it on Python 3.12. Review a floor change together with
        the behaviour it protects.
      </p>

      <H2 id="release-claims">Release claims and the public models</H2>
      <p>
        <code>tests.support.claims.CLAIM_TESTS</code> maps each release claim of the model card to the test that proves
        it. The model-acceptance workflow (<code>.github/workflows/model-validation.yml</code>) fetches the models that{" "}
        <code>tests/fixtures/models/validation.json</code> pins, through AlignAIR's verified downloader, then requires
        every claim's test to pass:
      </p>
      {/* PENDING-V4-PUBLISH: tests/fixtures/models/validation.json is re-pinned to the alignair4-* models at publish. */}
      <CodeBlock
        code={`python -m tests.support.model_artifacts --cache /tmp/alignair-test-models
python -m pytest tests -m "model_validation or acceptance or claim_evidence" \\
  --model-cache=/tmp/alignair-test-models --enforce-claims \\
  --claim-report=test-results/claims.json --junitxml=test-results/acceptance.xml`}
      />
      <p>
        The tests read the cache offline, after rechecking each file's size and checksum; they never fall back to another
        model. These are regression checks on simulated reads; the accuracy claims come from the{" "}
        <DocLink to="comparison">comparison with IgBLAST</DocLink>.
      </p>

      <H2 id="writing-tests">Writing a test</H2>
      <ul>
        <li>
          Use deterministic inputs and assert what can be observed: values, shapes, coordinates, a saved file's contents,
          or the error class and code a bad input raises. Go through public entry points where you can.
        </li>
        <li>
          Declare the layer. Keep small fixtures beside their tests and shared ones in the nearest{" "}
          <code>conftest.py</code>; import helpers from <code>tests/support</code>, not from another test module. The
          root fixture <code>repo_root</code> is the repository's path; write outputs to <code>tmp_path</code>.
        </li>
        <li>
          Each test gets its own AlignAIR, XDG and Hugging Face cache and config directories, and a random seed derived
          from its node id; Python's, NumPy's and PyTorch's random state, the default dtype, gradient mode, deterministic
          settings and the thread count are restored afterwards. Torch runs on 2 threads.
        </li>
        <li>
          A connection or DNS lookup outside the machine fails the test (loopback works). Inject a provider's response
          instead of fetching it.
        </li>
        <li>
          When you move a test that a release claim names, update <code>tests.support.claims.CLAIM_TESTS</code>.
        </li>
      </ul>

      <H2 id="distribution">Distribution checks</H2>
      <p>
        <code>tests/packaging/wheel_smoke.py</code> loads and runs the compiled kernels of each built wheel.{" "}
        <code>tests/packaging/release_smoke.sh</code> runs the first-run path against an installed{" "}
        <code>alignair</code> (a wheel, or the Docker image) with no network: the version, the environment report, a tiny
        train-and-predict demo with its bundle, AIRR validation and a comparison of default and donor-reference output.{" "}
        <code>tests/packaging/installed_contract.py</code> runs outside the checkout and checks the installed package's
        public contracts, and that nothing imports from the source tree. Set <code>ALIGNAIR</code> to choose the command
        the smoke test runs:
      </p>
      <CodeBlock code={`sh tests/packaging/release_smoke.sh /tmp/alignair-release-check`} />
      <p>A release needs both: the source suite and the checks of what was installed.</p>

      <H2 id="website">The website</H2>
      <p>
        The site (Vite, React, TypeScript, Tailwind) has its own unit tests, for navigation, search, heading anchors and
        the CLI/Python tabs. <code>npm run build</code> type-checks every page before it builds, so a page that does not
        compile fails it:
      </p>
      <CodeBlock
        code={`cd website
npm ci
npm test         # the site's unit tests (node --test)
npm run build    # type-check (tsc) and build into website/dist
npm run dev      # a local preview while you edit`}
      />
      <p>
        <code>.github/workflows/website.yml</code> runs <code>npm ci</code>, <code>npm run build</code> and{" "}
        <code>npm test</code> on every pull request that touches <code>website/</code>, and deploys the site to GitHub
        Pages from <code>main</code>.
      </p>

      <H2 id="site-data">Regenerate the site's data</H2>
      <p>
        The option reference, the column dictionary, the error codes, the Python API, the model cards, the comparison and
        the search index render JSON files in <code>website/src/data/</code>, which{" "}
        <Source path="website/scripts/site_data.py" /> writes from the code. Never edit them by hand. From the repository
        root, with AlignAIR importable:
      </p>
      <CodeBlock code={`python website/scripts/site_data.py`} />
      <p>
        It writes <code>cli-options.json</code>, <code>output-columns.json</code>, <code>error-codes.json</code>,{" "}
        <code>python-api.json</code> and <code>search-index.json</code>. Run it after you change a CLI option or its help,
        an output column or its description (<code>io/columns_doc.py</code>), an error class, a public signature or
        docstring, or a docs page's title or headings (the search index lists them); then commit the changed files with
        your change. Two files need inputs from outside the code:
      </p>
      <CodeBlock
        code={`python website/scripts/site_data.py --models DIR                      # models.json, from the seven model files
python website/scripts/site_data.py --comparison A.json B.json --real REAL.json   # comparison.json`}
      />
      <p>
        <code>comparison.json</code> takes only the reports of a locked test-split run of the comparison harness; a
        development-split report is refused.
      </p>
    </>
  ),
};

/* ---------------------------------------------------------------------------------------------------------------- */
/* Publishing                                                                                                        */
/* ---------------------------------------------------------------------------------------------------------------- */

const publishing: DocPage = {
  slug: "publishing",
  title: "Publishing to the catalog",
  section: "Developers",
  lead: (
    <>
      For maintainers: how a trained model becomes a catalog id that anyone can load, and the checks it passes on the
      way.
    </>
  ),
  body: () => (
    <>
      <H2 id="the-catalog">The catalog</H2>
      <p>
        The pretrained models live in one public Hugging Face repo,{" "}
        <a href="https://huggingface.co/AlignAIR/AlignAIR-pretrained" target="_blank" rel="noreferrer">
          AlignAIR/AlignAIR-pretrained
        </a>{" "}
        (<code>DEFAULT_REGISTRY</code>, <code>hf://AlignAIR/AlignAIR-pretrained</code>). It holds a catalog,{" "}
        <code>registry.json</code>, and for each version of each model three files: the pickle-free{" "}
        <code>&lt;id&gt;/&lt;version&gt;.alignair</code>, its <code>.model_card.md</code> and its{" "}
        <code>.reference_manifest.json</code>. <code>--model alignair4-igh-human</code> and{" "}
        <code>Aligner.from_pretrained("alignair4-igh-human")</code> read it.
      </p>
      <ul>
        <li>
          <strong>Anyone reads it</strong>, with no account: <code>alignair models list</code> and{" "}
          <code>--model &lt;id&gt;</code> download anonymously.
        </li>
        <li>
          <strong>Only maintainers write it</strong>, with a Hugging Face write token. No <code>alignair</code> command
          writes to it; the Python calls below do.
        </li>
        <li>
          <strong>No code change is needed for a new version.</strong> AlignAIR reads <code>registry.json</code> when it
          looks a model up: once a new version is uploaded, <code>alignair models list</code> shows it, a run on the plain
          id downloads it, and <code>alignair models update</code> fetches it for the models already in a cache.
        </li>
      </ul>
      {/* PENDING-V4-PUBLISH: the seven alignair4-<locus>-human ids and their first version are uploaded at publish. */}

      <H2 id="registry-format">The registry format</H2>
      <p>
        A registry is a directory (or the same layout on Hugging Face or a web server) with <code>registry.json</code> at
        its root. Its schema is <code>alignair.registry.v1</code>:
      </p>
      <CodeBlock
        lang="json"
        title="registry.json"
        code={`{
  "schema": "alignair.registry.v1",
  "models": {
    "alignair4-igh-human": {
      "description": "Human IGH aligner",
      "latest": "1.0.0",
      "versions": {
        "1.0.0": {
          "file": "alignair4-igh-human/1.0.0.alignair",
          "size": 1287179,
          "artifact_sha256": "<sha256>",
          "allele_order_sha256": "<sha256>",
          "reference_fasta_sha256": "<sha256>",
          "companions": {
            "model_card": {"file": "alignair4-igh-human/1.0.0.model_card.md", "sha256": "<sha256>"},
            "reference_manifest": {"file": "alignair4-igh-human/1.0.0.reference_manifest.json", "sha256": "<sha256>"}
          },
          "created_by_alignair": "4.0.0",
          "model_format_version": 1,
          "min_alignair": null,
          "trained": "2026-11-17T20:10:43Z",
          "metrics": null
        }
      }
    }
  }
}`}
      />
      <DocTable
        head={["Key", "Meaning"]}
        rows={[
          [<code>models.&lt;id&gt;.latest</code>, <>The version a plain id loads; <code>id@version</code> pins another. Must be one of <code>versions</code>.</>],
          [<code>file</code>, "The model file's path, relative to the registry's root."],
          [<><code>size</code>, <code>artifact_sha256</code></>, "The model file's size and SHA-256. The validator checks both; a download is checked against the SHA-256."],
          [<><code>allele_order_sha256</code>, <code>reference_fasta_sha256</code></>, "The embedded reference's two digests: its allele order, and its FASTA."],
          [<code>companions</code>, "The model card and the reference manifest, each with its path and SHA-256."],
          [<><code>created_by_alignair</code>, <code>model_format_version</code>, <code>min_alignair</code>, <code>trained</code>, <code>metrics</code></>, "Copied from the model file's header."],
          [<><code>description</code>, <code>species</code>, <code>receptor</code>, <code>locus</code></>, <>The model's entry, from the first version that gives them; <code>alignair models list</code> shows the description.</>],
        ]}
      />

      <H2 id="export">1. Export the bundle</H2>
      <CodeBlock
        code={`alignair export runs/igh/model.alignair release/igh \\
  --model-id alignair4-igh-human --model-version 1.0.0 --description "Human IGH aligner"
alignair info release/igh/model.alignair`}
      />
      <p>
        <code>alignair export</code> turns a training checkpoint into a bundle: the pickle-free{" "}
        <code>model.alignair</code>, <code>model_card.md</code>, <code>reference_manifest.json</code> and, unless{" "}
        <code>--no-validate</code>, <code>validation_report.json</code> (scores on a fixed held-out stream). The id and
        version go into the model file's card, where the catalog checks them. Never publish a training checkpoint; the
        catalog refuses one. Export from an installed AlignAIR release: the card records the version that wrote it in{" "}
        <code>created_by_alignair</code>, which is <code>0+unknown</code> from a source tree that is not installed. A bundle exported without an id or version can be carded afterwards with{" "}
        <code>alignair.model_file.update_card(src, dst, {"{"}"model_id": ..., "model_version": ...{"}"})</code>, which
        changes descriptive header fields only and carries every section over byte for byte.
      </p>

      <H2 id="publish">2. Add it to a local copy of the catalog</H2>
      <p>
        Start from a full copy of the hub's catalog: <code>publish_bundle</code> validates every version the local{" "}
        <code>registry.json</code> lists, and the upload replaces the hub's <code>registry.json</code> with yours.
      </p>
      <CodeBlock
        lang="python"
        code={`from huggingface_hub import snapshot_download
from alignair.registry.publish import publish_bundle

snapshot_download("AlignAIR/AlignAIR-pretrained", repo_type="model", local_dir="catalog")
problems = publish_bundle("release/igh", "alignair4-igh-human", "1.0.0", "catalog",
                          description="Human IGH aligner")
assert not problems, problems`}
      />
      <p>
        <code>publish_bundle</code> lays the bundle out as the catalog serves it (
        <code>alignair4-igh-human/1.0.0.alignair</code> with its card and manifest beside it), adds the version to{" "}
        <code>registry.json</code> with every checksum, and makes the version <code>latest</code> when it is the highest.
        It is transactional: it stages the files, validates the catalog as it would be, and only then moves the files and
        the new <code>registry.json</code> into place. On a problem it returns the list and writes nothing. Before
        uploading, load it from the local copy as a user would:
      </p>
      <CodeBlock
        code={`alignair models list --registry file://catalog
alignair predict --model alignair4-igh-human@1.0.0 --registry file://catalog \\
  --input reads.fasta --out out.tsv`}
      />

      <H2 id="upload">3. Upload the version</H2>
      <CodeBlock
        lang="python"
        code={`from alignair.registry.publish import upload_version

upload_version("catalog", "alignair4-igh-human", "1.0.0")   # needs HF_TOKEN, a write token
# one commit to AlignAIR/AlignAIR-pretrained: the .alignair, its .model_card.md and
# .reference_manifest.json, and registry.json`}
      />
      <p>
        <code>upload_version</code> validates the version again (<code>ModelNotFoundError</code> when the local catalog
        lacks it, <code>RegistryInvalidError</code> when it fails a check), then sends its three files and{" "}
        <code>registry.json</code> in one commit, so the hub never lists a version without its files. It needs{" "}
        <code>huggingface_hub</code> (<code>pip install "alignair[hub]"</code>). Then check as an anonymous user, on
        another machine or with a fresh <code>ALIGNAIR_CACHE_DIR</code>:
      </p>
      <CodeBlock
        code={`alignair models list
alignair models get alignair4-igh-human@1.0.0
alignair models verify alignair4-igh-human`}
      />

      <H2 id="validation">What is checked</H2>
      <p>
        <code>validate_registry(catalog, artifact_path)</code> (<Source path="src/alignair/registry/validate.py" />) is the
        gate: <code>publish_local</code>, <code>publish_bundle</code> and <code>upload_version</code> all run it. For every
        version of every model it checks that:
      </p>
      <ul>
        <li>the model file exists, and its size and SHA-256 are the catalog's;</li>
        <li>
          it has no pickle section (no <code>dataconfig/*</code>, no <code>train_state</code>) and has a{" "}
          <code>reference_json</code>;
        </li>
        <li>
          both reference digests recompute from that <code>reference_json</code>, and equal the catalog's;
        </li>
        <li>
          its card has <code>model_id</code>, <code>model_version</code>, <code>model_format_version</code> and{" "}
          <code>created_by_alignair</code>, and its id and version are the catalog's;
        </li>
        <li>each companion file exists with the catalog's SHA-256;</li>
        <li>
          <code>latest</code> names a listed version.
        </li>
      </ul>
      <p>It returns a list of problems, empty when all pass. To check a whole catalog directory:</p>
      <CodeBlock
        lang="python"
        code={`from pathlib import Path
from alignair.registry import load_registry, validate_registry

catalog = load_registry("file://catalog")
problems = validate_registry(catalog, lambda relpath: str(Path("catalog") / relpath))
print(problems or "OK")`}
      />
      <p>
        On the user's side, <code>alignair models verify [MODEL]</code> re-hashes the models in the cache against the
        registry and prints <code>OK</code>, <code>MISMATCH</code> or <code>UNKNOWN (not in registry)</code> for each; it
        exits with 1 on a mismatch.
      </p>

      <H2 id="calls">The calls</H2>
      <DocTable
        head={["Call", "What it does"]}
        rows={[
          [
            <code>publish_bundle(bundle_dir, model_id, version, registry_dir, *, description=None)</code>,
            <>
              Publishes an exported bundle with its card and manifest; <code>FileNotFoundError</code>, before anything is
              written, when one of the three files is missing. Returns the problems (<code>[]</code>: published).
            </>,
          ],
          [
            <code>publish_local(artifact_path, model_id, version, registry_dir, *, description=None, entry_extra=None, companions=None)</code>,
            <>
              Publishes one model file, with optional companion files (<code>{"{"}name: path{"}"}</code>) and extra
              entry keys. Returns the problems.
            </>,
          ],
          [
            <code>upload_version(registry_dir, model_id, version, *, repo_id="AlignAIR/AlignAIR-pretrained", token=None, revision=None, api=None)</code>,
            <>
              Uploads one version and <code>registry.json</code> in one commit; <code>token</code> defaults to{" "}
              <code>HF_TOKEN</code>. Returns the uploaded paths.
            </>,
          ],
          [
            <code>validate_registry(catalog, artifact_path)</code>,
            <>
              The problems of a parsed <code>registry.json</code>; <code>artifact_path(relpath)</code> gives a file's
              local path.
            </>,
          ],
          [
            <code>update_card(src_path, dst_path, updates)</code>,
            <>
              In <code>alignair.model_file</code>: a copy of a model file with descriptive card fields changed, every
              section kept byte for byte.
            </>,
          ],
        ]}
      />
      <p>
        The publishing calls are in <code>alignair.registry.publish</code>; <code>validate_registry</code> and{" "}
        <code>load_registry</code> are exported by <code>alignair.registry</code> too.
      </p>

      <H2 id="own-repo">A model in a repo of its own</H2>
      <p>
        A model that is not part of the shared catalog can live in a Hugging Face repo of its own:{" "}
        <code>Aligner.push_to_hub(repo_id)</code> uploads the model file with the <code>model_card.md</code> and{" "}
        <code>reference_manifest.json</code> beside it, in one commit, and users load it with{" "}
        <code>--model hf://org/repo</code>. <DocLink to="share">Share it on Hugging Face</DocLink> walks through it.
      </p>
    </>
  ),
};

/* ---------------------------------------------------------------------------------------------------------------- */
/* Contributing                                                                                                      */
/* ---------------------------------------------------------------------------------------------------------------- */

const contributing: DocPage = {
  slug: "contributing",
  title: "Contributing",
  section: "Developers",
  lead: (
    <>
      How to propose a change to AlignAIR: set up, branch, test, and where the code and the docs live.
    </>
  ),
  body: () => (
    <>
      <p>
        Bug reports, germline sets and models, code and docs are all welcome. This page sums up{" "}
        <a href={`${REPO}/blob/main/CONTRIBUTING.md`} target="_blank" rel="noreferrer">
          CONTRIBUTING.md
        </a>
        , which is the rule; read it before a pull request.
      </p>

      <H2 id="issues">Start with an issue</H2>
      <p>
        Open an issue first for a change that is more than a small fix, so the approach is agreed before you write it. The
        issue templates are a bug report, a feature request, and a model or reference request (a species or locus you
        would like a pretrained model for, or a germline set that fails to parse or train). For a bug, include{" "}
        <code>alignair doctor --json</code>, the command, and, for a prediction problem, a small input file; the{" "}
        <code>--debug</code> traceback and the <code>&lt;out&gt;.run.json</code> beside the table help too. Report a
        security problem privately, as <Source path="SECURITY.md" /> says, not in an issue.
      </p>

      <H2 id="setup">Set up</H2>
      <CodeBlock
        code={`git clone https://github.com/MuteJester/AlignAIR && cd AlignAIR
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
python -m pytest tests
alignair doctor`}
      />
      <p>
        <DocLink to="testing">Testing</DocLink> explains the suite, its markers and its gates, and{" "}
        <DocLink to="architecture">Architecture</DocLink> where each part of the code lives.
      </p>

      <H2 id="pull-requests">Branch, test, open a pull request</H2>
      <ul>
        <li>Work on a branch of your fork, and open the pull request against <code>main</code>.</li>
        <li>Keep it focused on one change, and match the surrounding style; no large unrelated reformatting.</li>
        <li>
          Add tests for new behaviour, each declaring its layer (<code>component</code>, <code>integration</code> or{" "}
          <code>acceptance</code>) and writing its outputs to <code>tmp_path</code>. <code>python -m pytest</code> must
          pass, and so must <code>ruff check src/alignair</code>.
        </li>
        <li>
          When users see the change, update the docs and <code>CHANGELOG.md</code>; when a CLI option, column, error or
          public docstring changes, regenerate the site's data (<code>python website/scripts/site_data.py</code>).
        </li>
        <li>The pull request template has a checklist of the same.</li>
      </ul>
      <p>
        On every pull request, CI runs the suite on Python 3.10, 3.11 and 3.12 with coverage, the component tests in a
        shuffled order and in parallel workers, platform tests on Linux, macOS and Windows, a built wheel through the
        release smoke test, the conda recipe and the model-acceptance tests; the website's build and tests run when{" "}
        <code>website/</code> changes.
      </p>

      <H2 id="where-the-docs-live">Where the docs live</H2>
      <DocTable
        head={["Where", "What", "Checked by"]}
        rows={[
          [
            <code>README.md</code>,
            "The first stop: install, a first alignment, the models, where everything else is.",
            <>
              <code>tests/website/test_readme.py</code>, the command and plain-language checks
            </>,
          ],
          [
            <code>website/src/pages/docs/pages-*.tsx</code>,
            <>
              The docs pages, each a <code>DocPage</code> (slug, title, section, lead, body) registered in{" "}
              <code>website/src/pages/docs/index.ts</code>; <code>nav.ts</code> orders the sections, with the Developers
              pages apart.
            </>,
            <>
              <code>npm run build</code>, the link, command and plain-language checks
            </>,
          ],
          [
            <code>website/src/data/*.json</code>,
            <>
              Generated by <code>website/scripts/site_data.py</code> and rendered by{" "}
              <code>website/src/components/data/</code>: the option reference, column dictionary, error codes, Python API,
              model cards, comparison and search index. Never edited by hand.
            </>,
            <code>tests/website/test_site_data.py</code>,
          ],
          [
            <code>examples/</code>,
            "Sample inputs, each folder with a README whose commands run on them.",
            <code>tests/alignair/cli/test_docs_cli_sync.py</code>,
          ],
          [
            "Docstrings",
            <>
              The Python API page is generated from the public packages' docstrings and signatures.
            </>,
            <code>tests/quality/test_stale_text.py</code>,
          ],
          [
            <>
              <code>CHANGELOG.md</code>, <code>CITATION.cff</code>
            </>,
            "What changed in each release, and how to cite AlignAIR.",
            "Review",
          ],
        ]}
      />
      <p>A docs page follows the rules the tests enforce:</p>
      <ul>
        <li>
          Headings only through <code>H2</code> and <code>H3</code>, each with an <code>id</code>; internal links only
          through <code>DocLink</code>, whose <code>to</code> is a page's slug (or <code>slug?h=heading-id</code>).
        </li>
        <li>
          Every <code>alignair ...</code> command in a code block parses against the real CLI; never show an option the
          CLI does not have.
        </li>
        <li>
          The user's pages and the README speak to a repertoire researcher: reads, alleles, germline sets, genotypes,
          AIRR columns. The model's internal words belong on How AlignAIR works and the Developers pages.
        </li>
        <li>
          Generated facts (options, columns, models, error codes) are rendered from the data files, never retyped.
        </li>
      </ul>
      <p>
        The site deploys from <code>main</code>, through <code>.github/workflows/website.yml</code>, to{" "}
        <a href="https://mutejester.github.io/AlignAIR/" target="_blank" rel="noreferrer">
          mutejester.github.io/AlignAIR
        </a>
        .
      </p>

      <H2 id="licence">Licence</H2>
      <Callout kind="note" title="Your contribution's licence">
        AlignAIR's code is GPL-3.0-or-later, and by contributing you agree that your contribution is licensed the same
        way. Pretrained models and germline data may carry licences of their own: note them in the model card and the
        pull request.
      </Callout>
      <p>
        The project follows the Contributor Covenant, version 2.1 (<Source path="CODE_OF_CONDUCT.md" />): be respectful
        and constructive.
      </p>
    </>
  ),
};

export const developerPages: DocPage[] = [architecture, testing, publishing, contributing];
