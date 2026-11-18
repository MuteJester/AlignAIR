import { CodeBlock } from "../../components/ui";
import CodeTabs from "../../components/CodeTabs";
import OptionTable from "../../components/data/OptionTable";
import ColumnTable, { COLUMNS } from "../../components/data/ColumnTable";
import ModelTable, { MODELS } from "../../components/data/ModelTable";
import ErrorTable from "../../components/data/ErrorTable";
import PythonApi from "../../components/data/PythonApi";
import { DocTable, DocLink, H2, H3, type DocPage } from "./doc-kit";

const SECTION = "Reference";

/* ------------------------------------------------------------------------------------------------------------------ */
/* CLI options                                                                                                        */
/* ------------------------------------------------------------------------------------------------------------------ */
const cli: DocPage = {
  slug: "cli",
  title: "CLI options",
  section: SECTION,
  lead: "Every option of every alignair command, with its default, generated from AlignAIR's own command-line parser.",
  body: () => (
    <>
      <p>
        <code>alignair &lt;command&gt; --help</code> prints the same options in the terminal, and{" "}
        <code>alignair predict --help</code> groups them by purpose, as here. A command that loads a model takes a model
        id (<code>alignair4-igh-human</code>), an id with its version (<code>alignair4-igh-human@&lt;version&gt;</code>
        ), a <code>.alignair</code> file, or a model's own Hugging Face repo (<code>hf://org/repo</code>).
      </p>

      <H2 id="every-command">Options of every command</H2>
      <p>Three options belong to <code>alignair</code> itself:</p>
      <DocTable
        head={["Option", "What it does"]}
        rows={[
          [<code>--version</code>, "Prints AlignAIR's version."],
          [<code>--debug</code>, "On an error, prints the full traceback after the message."],
          [
            <code>--error-format text|json</code>,
            <>
              Prints an error as text (the default) or as one line of JSON for a pipeline to read (
              <DocLink to="error-codes?h=json-errors">JSON errors</DocLink>).
            </>,
          ],
        ]}
      />
      <p>
        <code>--debug</code> and <code>--error-format</code> work before or after the command's name, so the tables
        below list them for every command.
      </p>
      <CodeBlock
        code={`alignair --version
alignair --error-format json predict --model alignair4-igh-human --input reads.fasta --out out.tsv`}
      />

      <H2 id="predict">alignair predict</H2>
      <p>
        Aligns reads and writes the AIRR rearrangement table. The guides:{" "}
        <DocLink to="first-alignment">your first alignment</DocLink>, <DocLink to="prepare-input">input</DocLink>,{" "}
        <DocLink to="donor-genotype">genotypes</DocLink>, <DocLink to="own-reference">your own germline set</DocLink>,{" "}
        <DocLink to="metadata-join">metadata</DocLink>, <DocLink to="output-columns-choice">output columns</DocLink>,{" "}
        <DocLink to="run-at-scale">speed and memory</DocLink> and the{" "}
        <DocLink to="trust-the-calls?h=quality-gates">quality gates</DocLink>.
      </p>
      <CodeBlock code={`alignair predict --model alignair4-igh-human --input reads.fastq.gz --out out.tsv`} />
      <OptionTable command="predict" />

      <H2 id="genotype">alignair genotype</H2>
      <p>
        Infers a donor's V genotype from any tool's AIRR table, by TIgGER's frequency rule, and writes it as an AIRR
        GenotypeSet that <code>predict --genotype</code> reads (
        <DocLink to="donor-genotype?h=genotype-from-a-table">a genotype from an AIRR table</DocLink>).
      </p>
      <CodeBlock code={`alignair genotype out.tsv --out genotype.json`} />
      <OptionTable command="genotype" />

      <H2 id="models">alignair models</H2>
      <p>
        Lists, downloads, updates and checks the pretrained models in the cache (
        <DocLink to="cache-and-environment">Cache, offline use and environment variables</DocLink>). It does nothing
        alone; its subcommands follow.
      </p>
      <OptionTable command="models" />

      <H3 id="models-list">alignair models list</H3>
      <p>
        Lists the models of the registries: each id, its latest version, and whether you have it (
        <code>installed</code>, <code>update available</code> or <code>available</code>).
      </p>
      <CodeBlock code={`alignair models list`} />
      <OptionTable command="models list" />

      <H3 id="models-get">alignair models get</H3>
      <p>Downloads a model into the cache, ahead of an offline run, and prints where it is.</p>
      <CodeBlock code={`alignair models get alignair4-igh-human`} />
      <OptionTable command="models get" />

      <H3 id="models-path">alignair models path</H3>
      <p>
        Prints the path of a model's cached file, downloading it first if needed: the form{" "}
        <code>alignair reference export</code> takes.
      </p>
      <CodeBlock code={`alignair models path alignair4-igh-human`} />
      <OptionTable command="models path" />

      <H3 id="models-update">alignair models update</H3>
      <p>Downloads the latest version of an installed model, or of every installed model.</p>
      <CodeBlock code={`alignair models update`} />
      <OptionTable command="models update" />

      <H3 id="models-verify">alignair models verify</H3>
      <p>
        Checks each cached model file against the checksum its registry lists, and prints <code>OK</code>,{" "}
        <code>MISMATCH</code> or <code>UNKNOWN (not in registry)</code> per version. It exits with status 1 on a
        mismatch.
      </p>
      <CodeBlock code={`alignair models verify`} />
      <OptionTable command="models verify" />

      <H3 id="models-prune">alignair models prune</H3>
      <p>Removes old cached versions, keeping the newest of each model (<code>--keep</code> of them).</p>
      <CodeBlock code={`alignair models prune --keep 1`} />
      <OptionTable command="models prune" />

      <H2 id="info">alignair info</H2>
      <p>
        Prints a model's card without loading the model; given an id, it downloads the model first if it is not cached
        (<DocLink to="model-cards?h=read-a-card">read a model's card</DocLink>).
      </p>
      <CodeBlock code={`alignair info alignair4-igh-human`} />
      <OptionTable command="info" />

      <H2 id="reference">alignair reference</H2>
      <p>Lists the built-in germline sets, or writes a model's germline set as FASTA. Its subcommands follow.</p>
      <OptionTable command="reference" />

      <H3 id="reference-list">alignair reference list</H3>
      <p>
        Lists the germline sets built into GenAIRR, by the name <code>alignair train --dataconfig</code> takes.
      </p>
      <CodeBlock code={`alignair reference list --species Human --chain IGH`} />
      <OptionTable command="reference list" />

      <H3 id="reference-export">alignair reference export</H3>
      <p>
        Writes a model file's germline set as one FASTA of its V, D and J alleles; <code>--gapped</code> writes each V
        IMGT-gapped, as Change-O's <code>CreateGermlines.py</code> needs. It takes a model file, so resolve an id with{" "}
        <code>alignair models path</code> (<DocLink to="own-reference?h=export-the-reference">export a model's germline
        set</DocLink>).
      </p>
      <CodeBlock
        code={`alignair reference export "$(alignair models path alignair4-igh-human)" --fasta igh_gapped.fasta --gapped`}
      />
      <OptionTable command="reference export" />

      <H2 id="doctor">alignair doctor</H2>
      <p>
        Prints what AlignAIR finds in your environment: the versions of Python and its dependencies, the GPU, and the
        cache and settings folders. It never fails (<DocLink to="install?h=check-the-install">check the
        install</DocLink>).
      </p>
      <CodeBlock code={`alignair doctor --json`} />
      <OptionTable command="doctor" />

      <H2 id="compare">alignair compare</H2>
      <p>
        Compares two AIRR tables of the same reads, gene by gene: how often the two tools call the same allele and the
        same gene, and, where they differ, how often the second tool's call is in the first's candidate set. It
        measures agreement, not accuracy (<DocLink to="from-igblast?h=run-both">run both, then compare</DocLink>).
      </p>
      <CodeBlock code={`alignair compare --a alignair.tsv --b igblast.tsv --a-name AlignAIR --b-name IgBLAST --out report.md`} />
      <OptionTable command="compare" />

      <H2 id="analyze">alignair analyze</H2>
      <p>
        Summarizes an AIRR table: the reads, productivity, gene usage, junction lengths, orientation, V identity, how
        complete the fields are, and a structural check.
      </p>
      <CodeBlock code={`alignair analyze out.tsv`} />
      <OptionTable command="analyze" />

      <H2 id="validate-airr">alignair validate-airr</H2>
      <p>
        Checks an AIRR table's structure: the required columns are there, and positions and CIGAR strings stay inside
        the read. It is a quick check, not the AIRR Community's validator, which the <code>airr</code> Python package
        provides.
      </p>
      <CodeBlock code={`alignair validate-airr out.tsv`} />
      <OptionTable command="validate-airr" />

      <H2 id="train">alignair train</H2>
      <p>
        Trains a model for a germline set: a built-in one (<code>--dataconfig</code>) or your own FASTAs (
        <code>--v-fasta</code>, <code>--d-fasta</code>, <code>--j-fasta</code>, <code>--chain-type</code>).{" "}
        <code>--plan</code> checks the set and prints the plan without training; <code>--init-from</code> starts from an
        existing model. At the end it writes a model to share in <code>bundle/</code> under <code>--out</code>. See{" "}
        <DocLink to="do-i-need-to-train">Do I need to train?</DocLink>, <DocLink to="train">Train a model</DocLink> and{" "}
        <DocLink to="fine-tune">Fine-tune a model</DocLink>.
      </p>
      <CodeBlock code={`alignair train --dataconfig HUMAN_IGH_OGRDB --out runs/my_igh --plan`} />
      <OptionTable command="train" />

      <H2 id="export">alignair export</H2>
      <p>
        Turns a training checkpoint into a model to share: a folder holding <code>model.alignair</code>,{" "}
        <code>model_card.md</code>, <code>reference_manifest.json</code> and, unless <code>--no-validate</code>,{" "}
        <code>validation_report.json</code>. <code>alignair train</code> already does this once at the end; use{" "}
        <code>export</code> for another checkpoint, or to record an id and version in the card (
        <DocLink to="share">Share it on Hugging Face</DocLink>).
      </p>
      <CodeBlock
        code={`alignair export runs/my_igh/model.alignair runs/my_igh/release --model-id my-igh-model --model-version 1.0.0`}
      />
      <OptionTable command="export" />

      <H2 id="benchmark">alignair benchmark</H2>
      <p>
        Scores a model on reads GenAIRR simulates with known truth, per kind of read: a quick check of a model, with no
        other tool involved (<DocLink to="evaluate">Evaluate your model</DocLink>).
      </p>
      <CodeBlock code={`alignair benchmark --model alignair4-igh-human --n 200 --json --out benchmark.json`} />
      <OptionTable command="benchmark" />

      <H2 id="demo">alignair demo</H2>
      <p>
        Trains a tiny model for a few steps, offline, and aligns simulated reads with it, with and without a donor
        genotype: it shows every step on your machine. Its calls are not accurate.
      </p>
      <CodeBlock code={`alignair demo -o demo`} />
      <OptionTable command="demo" />
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Python API                                                                                                         */
/* ------------------------------------------------------------------------------------------------------------------ */
const pythonApi: DocPage = {
  slug: "python-api",
  title: "Python API",
  section: SECTION,
  lead: "AlignAIR's Python names, each with its signature and documentation, generated from the code; first, the few you need to start.",
  body: () => (
    <>
      <H2 id="main-entry-points">The main entry points</H2>
      <CodeBlock
        lang="python"
        code={`from alignair import Aligner, read_sequences

aligner = Aligner.from_pretrained("alignair4-igh-human")   # an id, a .alignair file or hf://org/repo
ids, reads, info = read_sequences("reads.fasta")           # FASTA, FASTQ or a table; unusable reads dropped
result = aligner.predict(reads, ids=ids)                   # a PredictionResult
result.write_airr("out.tsv")                               # the table alignair predict writes`}
      />
      <DocTable
        head={["To", "Use"]}
        rows={[
          [
            "Load a model",
            <>
              <code>Aligner.from_pretrained(model, offline=, device=, reference=, registry=, token=, revision=)</code>
            </>,
          ],
          [
            "Align reads",
            <>
              <code>aligner.predict(reads, ids=)</code>; <code>threshold=</code> and <code>cap=</code> set the candidate
              set's rule (0.5 and 3)
            </>,
          ],
          [
            "Write the AIRR table",
            <>
              <code>result.write_airr(path, columns=)</code>, with a preset name or a list of columns
            </>,
          ],
          [
            "Read reads from a file",
            <>
              <code>read_sequences(path, seq_column=, id_column=, collect_rejects=)</code>
            </>,
          ],
          [
            "Align a large file in constant memory",
            <>
              <code>iter_sequences(path, chunk_size=)</code> with <code>AirrWriter</code>, or{" "}
              <code>aligner.predict_iter(reads, ids=)</code> (<DocLink to="run-at-scale?h=memory">example</DocLink>)
            </>,
          ],
          [
            "Call from a donor's genotype",
            <>
              <code>aligner.with_genotype({`{"V": [...], "J": [...]}`})</code>, or with{" "}
              <code>alignair.genotype.load_genotype(path)</code> (<DocLink to="donor-genotype">guide</DocLink>)
            </>,
          ],
          [
            "Call from another germline set",
            <>
              <code>Aligner.from_pretrained(model, reference=ReferenceSet.from_fasta(path))</code>, or{" "}
              <code>aligner.bound_to(...)</code> (<DocLink to="own-reference">guide</DocLink>)
            </>,
          ],
          [
            "Compare two tables",
            <>
              <code>compare_airr(a, b)</code>, each a <code>{`{sequence_id: row}`}</code> dict
            </>,
          ],
          [
            "Train a model",
            <>
              <code>TrainingConfig</code> and <code>run_training</code> (<DocLink to="train">guide</DocLink>)
            </>,
          ],
          [
            "Handle errors",
            <>
              <code>AlignAIRError</code> and the classes of <code>alignair.errors</code> (
              <DocLink to="python-api?h=errors">below</DocLink>)
            </>,
          ],
        ]}
      />
      <p>
        <code>with_genotype</code> and <code>bound_to</code> return a new <code>Aligner</code> and leave the original as
        it was. Two features are on the command line only: <code>--infer-genotype</code>, and worker processes (
        <code>--threads</code>).
      </p>

      <H3 id="records-and-the-table">Records and the table</H3>
      <p>
        <code>result.records</code> holds one dict per read: AlignAIR's own record, not a row of the table. In it the
        read is turned forward, positions count from 0 with the end excluded, true and false are Python{" "}
        <code>True</code> and <code>False</code>, and the candidate set is <code>v_calls</code>, with its scores in{" "}
        <code>v_likelihoods</code>. <code>write_airr</code> turns the records into the AIRR table: positions from 1,{" "}
        <code>T</code> and <code>F</code>, <code>rev_comp</code>, <code>v_call_set</code>. Give tools downstream the
        table, not the records.
      </p>

      <H2 id="errors">Errors</H2>
      <p>
        Every failure raises a subclass of <code>AlignAIRError</code> that carries the same code as the command line (
        <DocLink to="error-codes">Error codes</DocLink>). Branch on the class or on <code>error.code</code>, never on the
        message:
      </p>
      <CodeBlock
        lang="python"
        code={`from alignair import Aligner
from alignair.errors import UnknownAlleleError

aligner = Aligner.from_pretrained("alignair4-igh-human")
try:
    donor = aligner.with_genotype({"V": ["IGHV3-23*01"]})
except UnknownAlleleError as error:
    print(error.code)      # ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND
    print(error.message)   # the genotype names alleles the reference does not hold: IGHV3-23*01`}
      />
      <p>
        Each error also has <code>hint</code>, <code>context</code> (a dict of where it happened),{" "}
        <code>retryable</code> and <code>exit_code</code>; <code>error.to_dict()</code> and <code>error.to_json()</code>{" "}
        give the fields that <code>--error-format json</code> prints.
      </p>

      <H2 id="package-alignair">alignair</H2>
      <p>The names most scripts need: loading, aligning, reading and writing, comparing, training.</p>
      <PythonApi packageName="alignair" />

      <H2 id="package-alignair-io">alignair.io</H2>
      <p>
        Lower-level reading and the metadata join: reads with their place in the file, and a metadata table indexed by
        read id.
      </p>
      <PythonApi packageName="alignair.io" />

      <H2 id="package-alignair-genotype">alignair.genotype</H2>
      <p>Reading a genotype file (YAML, JSON or an AIRR GenotypeSet) and checking it against a germline set.</p>
      <PythonApi packageName="alignair.genotype" />

      <H2 id="package-alignair-reference">alignair.reference</H2>
      <p>
        Germline sets: building one from FASTA or from allele sequences, filling in each allele's IMGT gaps and anchor,
        and reporting which alleles have them.
      </p>
      <PythonApi packageName="alignair.reference" />

      <H2 id="package-alignair-registry">alignair.registry</H2>
      <p>
        Finding and downloading models: registries, the cache and Hugging Face repos (
        <DocLink to="cache-and-environment">Cache, offline use and environment variables</DocLink>).
      </p>
      <PythonApi packageName="alignair.registry" />

      <H2 id="package-alignair-train">alignair.train</H2>
      <p>
        Training: the configuration, the run, its learning-rate schedule and the synthetic novel alleles training adds
        (<DocLink to="train">Train a model</DocLink>).
      </p>
      <PythonApi packageName="alignair.train" />
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Output column dictionary                                                                                           */
/* ------------------------------------------------------------------------------------------------------------------ */
const presetSize = (preset: string) => COLUMNS.filter((column) => column.presets.includes(preset)).length;

const columnDictionary: DocPage = {
  slug: "column-dictionary",
  title: "Output column dictionary",
  section: SECTION,
  lead: "Every column of the AIRR table AlignAIR writes, in file order: its kind and unit, what it means, and the presets that include it.",
  body: () => (
    <>
      <p>
        The table below is generated from AlignAIR's code, so it lists exactly the columns the program writes. What the
        main ones tell you, and how to filter on them, is on{" "}
        <DocLink to="output-columns">Output columns explained</DocLink>.
      </p>

      <H2 id="presets">Presets</H2>
      <p>
        <code>--columns</code> (in Python, <code>write_airr(columns=)</code>) takes a preset or a list of column names (
        <DocLink to="output-columns-choice">Choose output columns</DocLink>).
      </p>
      <DocTable
        head={["Preset", "Columns", "What it holds"]}
        rows={[
          [
            <>
              <code>full</code> (default)
            </>,
            presetSize("full"),
            "Every column below.",
          ],
          [
            <code>core</code>,
            presetSize("core"),
            "The calls, productivity, the junction, each segment's start and end in the read and in its germline, and the CIGAR strings.",
          ],
          [<code>airr</code>, presetSize("airr"), "The fields the AIRR rearrangement schema requires."],
          [
            <code>minimal</code>,
            presetSize("minimal"),
            "The calls and productivity. It skips building the alignments, so it is the fastest.",
          ],
        ]}
      />

      <H2 id="conventions">Conventions</H2>
      <ul>
        <li>
          <strong>True and false</strong> are the AIRR strings <code>T</code> and <code>F</code>. A blank cell means
          unknown, not false; <code>length_cropped</code> is <code>T</code> or blank. In pandas, compare with{" "}
          <code>"T"</code> (<DocLink to="output-columns?h=filter-usable-rows">filters</DocLink>).
        </li>
        <li>
          <strong>Fractions</strong> run from 0 to 1: the <code>*_identity</code> columns and{" "}
          <code>mutation_rate</code>.
        </li>
        <li>
          <strong>Positions in the read</strong> count from 1, both ends included, as AIRR defines them. When{" "}
          <code>rev_comp</code> is <code>T</code>, they count on the reverse complement of <code>sequence</code>.
        </li>
        <li>
          <strong>Lists</strong> are comma-separated: the alleles after the call in <code>v_call</code>, and the
          candidate sets.
        </li>
        <li>
          <strong>AIRR standard or AlignAIR's own.</strong> Every column follows the AIRR rearrangement schema except
          twelve that AlignAIR adds: <code>productive_prediction</code>, <code>v_call_set</code>,{" "}
          <code>d_call_set</code>, <code>j_call_set</code>, <code>mutation_rate</code>, <code>orientation</code>,{" "}
          <code>input_sequence</code>, <code>segmentation_low_quality</code>, <code>length_cropped</code>,{" "}
          <code>airr_assembly_status</code>, <code>airr_assembly_reason</code> and <code>airr_assembly_error</code>.
          Tools that read AIRR tables pass over columns they do not know.
        </li>
        <li>
          A <code>--metadata</code> join adds its own columns to these (
          <DocLink to="metadata-join">Join sample and cell metadata</DocLink>).
        </li>
      </ul>

      <H2 id="all-columns">All columns</H2>
      <ColumnTable />
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Model cards                                                                                                        */
/* ------------------------------------------------------------------------------------------------------------------ */
const modelCards: DocPage = {
  slug: "model-cards",
  title: "Model cards",
  section: SECTION,
  lead: "The seven pretrained human models: the germline set each one calls from, how its alleles are named, how it was trained, and how to read a model's card yourself.",
  body: () => (
    <>
      <ModelTable />
      <p>
        This table is generated from the seven model files. Copy an id with the button beside it;{" "}
        <DocLink to="choose-a-model">Choose a model</DocLink> helps you pick one.
      </p>

      <H2 id="what-a-model-holds">What every model holds</H2>
      <ul>
        <li>One locus of one species: human, and the locus in its id.</li>
        <li>
          Its germline set: every V, D and J allele with its sequence, the IMGT-gapped form of each V, and the position
          of the conserved codons that frame the junction (the V's Cys, the J's Trp or Phe). Every call is an allele of
          this set, unless you give another with <code>--reference</code> (
          <DocLink to="own-reference">Use your own germline reference</DocLink>).
        </li>
        <li>
          What it learned from reads simulated with GenAIRR from that germline set: full-length reads and fragments,
          in all four orientations, with somatic hypermutation on the IG loci and none on the TCR loci. It reads at
          most 575 nt of a read.
        </li>
        <li>Its card: its id and version, the germline set, the allele counts and the training.</li>
      </ul>
      <p>The models are released under GPL-3.0-or-later, as the code is.</p>

      <H2 id="germline-sets">Germline sets and allele names</H2>
      <DocTable
        head={["Models", "Germline sets", "The names you will see"]}
        rows={[
          [
            "IGH",
            "OGRDB's human IGH set (OGRDB V8)",
            <>
              V alleles by their OGRDB labels: in <code>IGHVF6-G21*07</code>, <code>F6</code> is the family and{" "}
              <code>G21</code> the gene group. D and J alleles keep IMGT-style names (<code>IGHD3-10*01</code>,{" "}
              <code>IGHJ6*02</code>).
            </>,
          ],
          [
            "IGK, IGL",
            "OGRDB's human IGK and IGL sets",
            <>
              OGRDB labels of their own, for V and J: <code>IGKV1-5TCS*03</code>, <code>IGKJ2-27FL*01</code>,{" "}
              <code>IGLV6-A5V2*05</code>, <code>IGLJ2/3-HDWH*01</code>.
            </>,
          ],
          [
            "TRA, TRB, TRD, TRG",
            "IMGT's human sets",
            <>
              IMGT names: <code>TRBV20-1*01</code>, <code>TRBD2*01</code>, <code>TRBJ1-2*01</code>.
            </>,
          ],
        ]}
      />
      <p>
        A genotype or a join table written in the other naming does not match:{" "}
        <DocLink to="concepts?h=allele-names">OGRDB and IMGT allele names</DocLink> says how to work with IMGT-named data
        on the IG models.
      </p>

      <H3 id="per-model-notes">Notes per model</H3>
      <ul>
        <li>
          <strong>IGH, TRB and TRD</strong> call V, D and J; the others call V and J. A D is called only when AlignAIR finds at
          least 8 of its germline bases in the read; the TRD D alleles are 8, 9 and 13 nt long (
          <DocLink to="trust-the-calls?h=short-d">D calls on short D segments</DocLink>).
        </li>
        <li>
          <strong>IGH:</strong> two pairs of its D alleles have identical sequences (<code>IGHD4-4*01</code> and{" "}
          <code>IGHD4-11*01</code>, <code>IGHD5-5*01</code> and <code>IGHD5-18*01</code>); <code>d_call</code> names
          one of each pair.
        </li>
        <li>
          <strong>IGK and IGL</strong> are two models: a light-chain library is split by locus and each part aligned
          with its own model (<DocLink to="choose-a-model?h=pick-by-locus">Pick by locus</DocLink>).
        </li>
        <li>
          <strong>The TCR models</strong> were trained without hypermutation, as T cells do not hypermutate.
        </li>
      </ul>

      <H2 id="training">Training</H2>
      <p>Each model was trained on simulated reads drawn fresh at every step, for the number of steps below:</p>
      <DocTable
        head={["Locus", "Chain type", "Species", "Training steps", "Licence"]}
        rows={MODELS.map((model) => [
          model.locus,
          <code>{model.chain}</code>,
          model.species,
          model.training_steps.toLocaleString("en-US"),
          model.licence,
        ])}
      />
      <p>
        How the models compare with IgBLAST, measure by measure, is on{" "}
        <DocLink to="comparison">Comparison with IgBLAST</DocLink>.
      </p>

      <H2 id="read-a-card">Read a model's card: alignair info</H2>
      <p>
        <code>alignair info</code> prints a model's card without loading the model. Given an id, it downloads the model
        first if it is not in the cache:
      </p>
      <CodeBlock
        title="shell (output abridged)"
        code={`$ alignair info alignair4-igh-human
  alleles:   V=198 D=33 J=7   params=314812
  training:  steps=320000 total_sequences_seen=20480000 lr=3e-05
  reference: HUMAN_IGH_OGRDB (BCR_HEAVY, Human)`}
      />
      <DocTable
        head={["Line", "What it says"]}
        rows={[
          [
            "The first line",
            "The model's id and version, the file format, and the AlignAIR version that wrote the file.",
          ],
          [<code>created</code>, "When the file was written."],
          [
            <code>alleles</code>,
            <>
              The V, D and J alleles in its germline set; <code>params</code>, the model's size in learned values.
            </>,
          ],
          [<code>training</code>, "The training steps, the simulated reads seen, and the final learning rate."],
          [<code>reference</code>, "The germline set, its chain type and species."],
          [
            <code>allele order</code>,
            "A fingerprint of the germline set's allele order, checked each time the model loads.",
          ],
          [
            <code>sections</code>,
            <>
              The parts of the file. A training checkpoint from <code>alignair train</code> also lists{" "}
              <code>dataconfig/0</code> and <code>train_state</code>; a model to share does not.
            </>,
          ],
        ]}
      />
      <p>
        <code>alignair info alignair4-igh-human --json</code> prints the whole card as JSON, the full training record
        and the GenAIRR version included.
      </p>

      <H2 id="versions">Versions</H2>
      <p>
        An id alone, <code>alignair4-igh-human</code>, means the latest version in the registry; an id with its
        version, <code>alignair4-igh-human@&lt;version&gt;</code>, pins one. <code>alignair models list</code> shows
        each id's latest version, and <code>alignair models update</code> downloads it. Every download is checked
        against its SHA-256, and every run records the id, version and checksum it used in{" "}
        <code>out.tsv.run.json</code>. Pin the version for a study, so a new release does not change the model halfway
        (<DocLink to="cache-and-environment?h=how-a-model-is-found">how a model is found</DocLink>).
      </p>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Error codes                                                                                                        */
/* ------------------------------------------------------------------------------------------------------------------ */
const errorCodes: DocPage = {
  slug: "error-codes",
  title: "Error codes",
  section: SECTION,
  lead: "Every error AlignAIR reports carries a code: what each code means and what to do, how to read an error, and how a pipeline or a script can catch one.",
  body: () => (
    <>
      <H2 id="read-an-error">Read an error</H2>
      <CodeBlock
        title="shell"
        code={`$ alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv --offline
[ALIGNAIR_MODEL_UNAVAILABLE] offline: model is not available in the local cache or registries
Context: {"command": "predict", "failed_sources": ["hf://AlignAIR/AlignAIR-pretrained"], "model_id": "alignair4-igh-human", "offline": true, "version": null}
Hint: Check registry connectivity, configuration, and offline/cache settings.`}
      />
      <ul>
        <li>
          The first line is the code, in brackets, and the message. Find the code in the{" "}
          <DocLink to="error-codes?h=all-codes">table below</DocLink>: it says what to do.
        </li>
        <li>
          <code>Context</code> says where it happened: the command, the file, the model, the stage.
        </li>
        <li>
          <code>Hint</code>, when there is one, is the first thing to try. Here: download the model while online (
          <code>alignair models get alignair4-igh-human</code>), then run offline.
        </li>
      </ul>
      <p>
        Errors go to standard error, never into the table. <code>--debug</code> adds the full traceback, which a bug
        report needs (<DocLink to="support">Support</DocLink>).
      </p>

      <H2 id="code-scheme">The codes</H2>
      <ul>
        <li>
          Every code starts with <code>ALIGNAIR_</code>, then names what failed and how:{" "}
          <code>ALIGNAIR_MODEL_NOT_FOUND</code>, <code>ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND</code>.{" "}
          <code>ALIGNAIR_ERROR</code> is the general one.
        </li>
        <li>The code is the part meant for scripts: match on it, never on the message, whose wording can change.</li>
        <li>
          In Python, each code belongs to an exception class, shown under the code in the table; two classes can share
          a code (<code>ConfigurationError</code> and <code>TrainingConfigError</code> are both{" "}
          <code>ALIGNAIR_CONFIG_INVALID</code>).
        </li>
        <li>
          An error marked retryable may pass on a second try, such as a network failure (
          <code>ALIGNAIR_REGISTRY_UNAVAILABLE</code>). The JSON form says it for each error.
        </li>
      </ul>

      <H2 id="json-errors">JSON errors, for pipelines</H2>
      <p>
        <code>--error-format json</code> prints the error as one line of JSON on standard error, before or after the
        command's name:
      </p>
      <CodeBlock
        title="shell"
        code={`$ alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv --offline --error-format json
{"category": "registry", "code": "ALIGNAIR_MODEL_UNAVAILABLE", "context": {"command": "predict", "failed_sources": ["hf://AlignAIR/AlignAIR-pretrained"], "model_id": "alignair4-igh-human", "offline": true, "version": null}, "exit_code": 1, "hint": "Check registry connectivity, configuration, and offline/cache settings.", "message": "offline: model is not available in the local cache or registries", "retryable": false, "schema_version": 1, "type": "OfflineError"}`}
      />
      <DocTable
        head={["Field", "What it holds"]}
        rows={[
          [<code>code</code>, "The error code."],
          [<code>message</code>, "The message, for people."],
          [<code>hint</code>, <>What to try, or <code>null</code>.</>],
          [<code>context</code>, "Where it happened, as key and value pairs."],
          [<code>category</code>, <>The kind of failure: <code>input</code>, <code>model</code>, <code>registry</code>, ...</>],
          [<code>exit_code</code>, "The exit status the command returns."],
          [<code>retryable</code>, <><code>true</code> when a second try may pass.</>],
          [<code>type</code>, "The Python exception class."],
          [<code>schema_version</code>, <>The version of this JSON form, <code>1</code>.</>],
        ]}
      />
      <p>
        With <code>--debug</code> as well, the traceback is one more field, <code>traceback</code>, of the same object.
      </p>

      <H2 id="exit-codes">Exit codes</H2>
      <DocTable
        head={["Status", "Meaning"]}
        rows={[
          [<code>0</code>, "Success."],
          [
            <code>1</code>,
            <>
              An AlignAIR error: every code but the two below. A run that breaks a{" "}
              <DocLink to="trust-the-calls?h=quality-gates">quality gate</DocLink> also exits with 1, after writing its
              whole table; <code>alignair models verify</code> exits with 1 on a checksum mismatch.
            </>,
          ],
          [
            <code>2</code>,
            "The command line itself is wrong: an unknown option, a missing argument. The usage message says which; there is no code.",
          ],
          [
            <code>70</code>,
            <>
              An unexpected failure (<code>ALIGNAIR_INTERNAL_ERROR</code>): run it again with <code>--debug</code> and
              report it with the traceback.
            </>,
          ],
          [
            <code>130</code>,
            <>
              Interrupted, with Ctrl-C (<code>ALIGNAIR_INTERRUPTED</code>).
            </>,
          ],
        ]}
      />
      <p>
        A status other than 0 does not always mean nothing was written: check the files the command names in its
        message.
      </p>

      <H2 id="in-python">In Python</H2>
      <p>
        The same errors are exceptions, all subclasses of <code>AlignAIRError</code>, with the same codes:
      </p>
      <CodeBlock
        lang="python"
        code={`from alignair import Aligner, AlignAIRError

try:
    aligner = Aligner.from_pretrained("alignair4-igh-human", offline=True)
except AlignAIRError as error:
    print(error.code)        # ALIGNAIR_MODEL_UNAVAILABLE, when the model is not in the cache
    print(error.to_json())   # the fields --error-format json prints`}
      />
      <p>
        The classes are in <code>alignair.errors</code> (<DocLink to="python-api?h=errors">Python API</DocLink>).
      </p>

      <H2 id="all-codes">All codes</H2>
      <ErrorTable />
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Cache, offline use and environment variables                                                                       */
/* ------------------------------------------------------------------------------------------------------------------ */
const cacheAndEnvironment: DocPage = {
  slug: "cache-and-environment",
  title: "Cache, offline use and environment variables",
  section: SECTION,
  lead: "Where AlignAIR keeps the models it downloads, when it uses the network, how to run without one, and every setting you can change.",
  body: () => (
    <>
      <H2 id="the-cache">Where models are kept</H2>
      <p>
        A model you name by its id is downloaded once, from Hugging Face, into the cache, and read from there afterwards:
      </p>
      <DocTable
        head={["System", "Cache"]}
        rows={[
          [
            "Linux",
            <>
              <code>~/.cache/alignair</code> (under <code>$XDG_CACHE_HOME</code> when it is set)
            </>,
          ],
          ["macOS", <code>~/Library/Caches/alignair</code>],
          [
            "Any",
            <>
              the folder in <code>ALIGNAIR_CACHE_DIR</code>, when it is set
            </>,
          ],
        ]}
      />
      <p>
        <code>alignair doctor</code> prints yours as <code>cache_dir</code>. Inside, each version is one file,{" "}
        <code>models/&lt;id&gt;/&lt;version&gt;.alignair</code>, with its checksum beside it in a{" "}
        <code>.sha256</code> file; <code>alignair models path &lt;id&gt;</code> prints a model's file. A download is
        written to a temporary file, checked against the SHA-256 its registry lists and only then moved into place, under
        a lock: a broken download never leaves a usable file, and jobs that share a cache do not overwrite each
        other's.
      </p>
      <p>
        A model loaded from its own Hugging Face repo (<code>hf://org/repo</code>) is kept in Hugging Face's cache
        instead (<code>~/.cache/huggingface</code> by default; Hugging Face's own <code>HF_HOME</code> moves it).
      </p>

      <H2 id="how-a-model-is-found">How a model is found</H2>
      <DocTable
        head={[<code>--model</code>, "What AlignAIR does"]}
        rows={[
          [<>a <code>.alignair</code> file</>, "Reads it. No network."],
          [
            <>
              <code>hf://org/repo</code>
            </>,
            <>
              Downloads the repo's <code>model.alignair</code> with Hugging Face's library, which{" "}
              <code>pip install "alignair[hub]"</code> installs. <code>--revision</code> pins a branch, tag or commit.
            </>,
          ],
          [
            <>
              an id, <code>alignair4-igh-human</code>
            </>,
            <>
              Looks the id up in the registry to find its latest version (a small file, fetched each run), and
              downloads that version unless it is cached. When the registry cannot be reached, the newest cached version
              is used.
            </>,
          ],
          [
            <>
              an id and version, <code>alignair4-igh-human@&lt;version&gt;</code>
            </>,
            "That version. When it is cached, no network at all.",
          ],
        ]}
      />
      <p>
        An id alone follows the latest version: a new release changes the model a run uses. Pin the version for a
        study. Every run records the id, version and checksum it used in <code>out.tsv.run.json</code>.
      </p>

      <H2 id="offline">Work without a network</H2>
      <p>
        Download the model on a machine with internet access, then run with <code>--offline</code>, which never touches
        the network: an id resolves to the newest cached version (or the version you pin), and a{" "}
        <code>hf://</code> model to Hugging Face's cache.
      </p>
      <CodeTabs
        cli={`alignair models get alignair4-igh-human
alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv --offline`}
        python={`from alignair import Aligner

aligner = Aligner.from_pretrained("alignair4-igh-human", offline=True)   # the cache only`}
      />
      <p>
        On a cluster whose compute nodes have no internet access, point <code>ALIGNAIR_CACHE_DIR</code> at a shared
        folder, run <code>alignair models get</code> once on a node that has access, and run the jobs with{" "}
        <code>--offline</code>. A model that is not in the cache stops the run with{" "}
        <code>ALIGNAIR_MODEL_UNAVAILABLE</code>.
      </p>

      <H3 id="update-notice">The update notice</H3>
      <p>
        <code>alignair predict</code> checks, at most once a day, whether the registry has a newer version of a model
        you have, and prints a line to standard error, such as{" "}
        <code>ℹ alignair4-igh-human &lt;version&gt; available — run: alignair models update alignair4-igh-human</code>.
        The check never delays or fails a run. <code>--quiet</code>, <code>--offline</code>, a pinned{" "}
        <code>id@version</code> or <code>ALIGNAIR_NO_NETWORK</code> turn it off.
      </p>

      <H2 id="registries">Registries</H2>
      <p>
        A registry is the catalog AlignAIR looks a model id up in: a <code>registry.json</code> file listing ids,
        versions and checksums, beside the model files. The default is the AlignAIR models on Hugging Face (
        <code>hf://AlignAIR/AlignAIR-pretrained</code>). A lab can keep its own, and name it in one of three ways:
      </p>
      <ol>
        <li>
          <code>--registry</code>, repeatable, on every command that finds a model (<code>registry=</code> in Python);
        </li>
        <li>
          <code>ALIGNAIR_REGISTRY</code>, several separated by commas;
        </li>
        <li>
          <code>registries</code> in the <DocLink to="cache-and-environment?h=settings-file">settings file</DocLink>.
        </li>
      </ol>
      <p>
        The first of these that is set wins, and it replaces the default: list the default as well to keep the
        pretrained models. A registry is <code>hf://org/repo</code> (a public Hugging Face repo),{" "}
        <code>https://host/path/</code> (a web server) or <code>file:///path</code> (a folder; a plain path works too).
        Registries are read without a token, so keep a private one in a shared folder or on a server your machines can
        reach. <DocLink to="share">Share it on Hugging Face</DocLink> covers publishing a model.
      </p>
      <CodeBlock
        code={`ALIGNAIR_REGISTRY="file:///shared/alignair-registry,hf://AlignAIR/AlignAIR-pretrained" \\
  alignair predict --model lab-igh-mouse --input reads.fasta --out out.tsv`}
      />

      <H2 id="settings-file">The settings file</H2>
      <p>
        AlignAIR reads one optional file, <code>config.toml</code>, in its settings folder:{" "}
        <code>~/.config/alignair</code> on Linux (under <code>$XDG_CONFIG_HOME</code> when it is set),{" "}
        <code>~/Library/Application Support/alignair</code> on macOS, or the folder in <code>ALIGNAIR_CONFIG_DIR</code>.{" "}
        <code>alignair doctor</code> prints it as <code>config_dir</code>. Its one setting is the list of registries:
      </p>
      <CodeBlock
        lang="toml"
        title="config.toml"
        code={`registries = ["file:///shared/alignair-registry", "hf://AlignAIR/AlignAIR-pretrained"]`}
      />

      <H2 id="private-models">Private and gated models</H2>
      <p>
        A model in a private or gated Hugging Face repo needs <code>pip install "alignair[hub]"</code> and a token with
        access to it: <code>--hf-token</code> (<code>token=</code> in Python), else the <code>HF_TOKEN</code>{" "}
        environment variable, else <code>HUGGING_FACE_HUB_TOKEN</code>. For a gated repo, request access on its page
        first.
      </p>
      <CodeBlock code={`HF_TOKEN=hf_xxx alignair predict --model hf://my-lab/igh-model --input reads.fasta --out out.tsv`} />

      <H2 id="environment-variables">Environment variables</H2>
      <DocTable
        head={["Variable", "What it does"]}
        rows={[
          [
            <code>ALIGNAIR_CACHE_DIR</code>,
            <>
              Where downloaded models are kept, instead of the system's cache folder (
              <DocLink to="cache-and-environment?h=the-cache">the cache</DocLink>).
            </>,
          ],
          [
            <code>ALIGNAIR_REGISTRY</code>,
            <>
              The registries to look model ids up in, separated by commas; it replaces the default (
              <DocLink to="cache-and-environment?h=registries">registries</DocLink>). <code>--registry</code> overrides
              it.
            </>,
          ],
          [
            <code>ALIGNAIR_CONFIG_DIR</code>,
            <>
              Where the settings file is read from, instead of the system's settings folder.
            </>,
          ],
          [
            <code>ALIGNAIR_NO_NETWORK</code>,
            <>
              Set to any value, it turns off the daily check for newer model versions. It does not make a run offline:
              use <code>--offline</code> for that.
            </>,
          ],
          [
            <code>HF_TOKEN</code>,
            <>
              The Hugging Face token for a private or gated <code>hf://</code> model, when <code>--hf-token</code> is
              not given.
            </>,
          ],
          [
            <code>HUGGING_FACE_HUB_TOKEN</code>,
            <>
              The same, read when <code>HF_TOKEN</code> is not set.
            </>,
          ],
          [
            <code>HF_HUB_OFFLINE</code>,
            <>
              Hugging Face's own offline switch. AlignAIR honours it for <code>hf://</code> models, as if{" "}
              <code>--offline</code> were given; it does not affect models named by id.
            </>,
          ],
        ]}
      />
    </>
  ),
};

export const referencePages: DocPage[] = [cli, pythonApi, columnDictionary, modelCards, errorCodes, cacheAndEnvironment];
