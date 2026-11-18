import { CodeBlock, Callout } from "../../components/ui";
import CodeTabs from "../../components/CodeTabs";
import { VdjBuilderWidget } from "../../components/widgets/VdjBuilderWidget";
import { DocTable, DocLink, H2, H3, type DocPage } from "./doc-kit";

/* ---------------------------------------------------------------------------------------------------------------- */
/* Install                                                                                                           */
/* ---------------------------------------------------------------------------------------------------------------- */

const install: DocPage = {
  slug: "install",
  title: "Install",
  section: "Get started",
  lead: (
    <>
      Install AlignAIR with pip or run its Docker image, check the install with <code>alignair doctor</code>, and fix
      the usual install problems.
    </>
  ),
  body: () => (
    <>
      <H2 id="pip">Install with pip</H2>
      <CodeBlock code={`pip install alignair`} />
      <p>
        You need Python 3.10 or newer. This one command gives you the <code>alignair</code> command and the Python
        package, with everything they need: PyTorch, and GenAIRR (the repertoire simulator that also ships the built-in
        germline sets). No extra is needed for the command line. To move to a newer release later, run{" "}
        <code>pip install --upgrade alignair</code>.
      </p>
      <p>A fresh environment keeps AlignAIR's dependencies apart from your other tools:</p>
      <CodeBlock
        code={`python3 -m venv alignair-env\nsource alignair-env/bin/activate      # Windows: alignair-env\\Scripts\\activate\npip install alignair`}
      />

      <H3 id="extras">Extras</H3>
      <p>Two optional extras add one package each. Most users need neither.</p>
      <DocTable
        head={["Extra", "Adds", "You need it when"]}
        rows={[
          [
            <code>alignair[hub]</code>,
            <code>huggingface_hub</code>,
            <>
              You load a model straight from its own Hugging Face repo (<code>--model hf://org/repo</code>), for example
              one a colleague shared, or a private or gated one. The seven pretrained models download without it.
            </>,
          ],
          [
            <code>alignair[zstd]</code>,
            <code>zstandard</code>,
            <>
              AlignAIR says <em>Reading this model section requires zstandard</em>: a training file written on a
              machine that had it.
            </>,
          ],
        ]}
      />
      <p>
        Quote the brackets, as some shells (zsh, PowerShell) read them otherwise: <code>pip install "alignair[hub]"</code>.
      </p>

      <H2 id="gpu">GPU or CPU</H2>
      <p>
        A GPU is optional. AlignAIR runs on an NVIDIA GPU (CUDA) or Apple silicon (MPS) when PyTorch finds one, and on
        the CPU otherwise; <code>--device auto</code>, the default, picks in that order. To force one, pass{" "}
        <code>--device cuda</code>, <code>--device mps</code> or <code>--device cpu</code>. On the CPU,{" "}
        <code>--threads N</code> runs N worker processes, which speeds up a large file (
        <DocLink to="run-at-scale">Run at scale</DocLink>).
      </p>
      <p>
        AlignAIR uses whichever PyTorch build pip installs. Two cases call for installing PyTorch yourself, before
        AlignAIR:
      </p>
      <ul>
        <li>
          <strong>Windows with an NVIDIA GPU, or a specific CUDA version:</strong> pick the build in the selector on{" "}
          <a href="https://pytorch.org/get-started/locally/" target="_blank" rel="noreferrer">pytorch.org</a>, install
          it, then <code>pip install alignair</code>.
        </li>
        <li>
          <strong>A Linux machine without a GPU:</strong> the CPU build is a much smaller download.
        </li>
      </ul>
      <CodeBlock code={`pip install torch --index-url https://download.pytorch.org/whl/cpu\npip install alignair`} />

      <H2 id="conda">Conda</H2>
      <p>
        There is no Bioconda package. In a conda environment, install AlignAIR with pip, as above:
      </p>
      <CodeBlock code={`conda create -n alignair python=3.12\nconda activate alignair\npip install alignair`} />

      <H2 id="docker">Docker</H2>
      <p>
        The image <code>alignair/alignair</code> (mirrored at <code>ghcr.io/mutejester/alignair</code>) holds AlignAIR,
        a CPU build of PyTorch and the example reads, in <code>/app/examples</code>. Its entry point is the{" "}
        <code>alignair</code> command, so the arguments are the ones on these pages:
      </p>
      <CodeBlock
        code={`docker run --rm alignair/alignair doctor\n\ndocker run --rm --user "$(id -u):$(id -g)" -e HOME=/data -v "$PWD":/data -w /data \\\n  alignair/alignair predict --model alignair4-igh-human \\\n  --input /app/examples/reads.fasta --out out.tsv`}
      />
      <p>
        The second command mounts your current folder as <code>/data</code> and runs as you, so <code>out.tsv</code>{" "}
        lands in your folder and belongs to you; <code>HOME=/data</code> keeps the downloaded model in{" "}
        <code>.cache/alignair</code> there, so the next run does not download it again. For a pipeline, pin the
        release's version tag (<code>alignair/alignair:&lt;version&gt;</code>) rather than <code>latest</code>. On a
        cluster that runs Apptainer or Singularity instead of Docker, the same image works (
        <DocLink to="run-at-scale?h=apptainer">Apptainer or Singularity</DocLink>).
      </p>

      <H2 id="check-the-install">Check the install</H2>
      <p>
        <code>alignair doctor</code> prints what AlignAIR found in your environment. It never fails; it reports.
      </p>
      <CodeBlock
        code={`$ alignair doctor\nAlignAIR environment\n  python           3.12.3\n  torch            2.11.0\n  cuda_available   False\n  mps_available    False\n  device_auto      cpu\n  cache_dir        /home/you/.cache/alignair\n  config_dir       /home/you/.config/alignair`}
      />
      <p>(Abridged; the full report has a line per dependency.) What to look at:</p>
      <DocTable
        head={["Line", "What it tells you"]}
        rows={[
          [<code>alignair</code>, "The version you installed (first line of the full report)."],
          [<code>python</code>, "Must be 3.10 or newer."],
          [
            <>
              <code>cuda_available</code>, <code>cuda_device</code>, <code>mps_available</code>
            </>,
            "Whether PyTorch sees a GPU, and which.",
          ],
          [
            <code>device_auto</code>,
            <>
              Where predictions run by default: <code>cuda</code>, <code>mps</code> or <code>cpu</code>. If it says{" "}
              <code>cpu</code> on a machine with a GPU, see <DocLink to="install?h=install-problems">install problems</DocLink>.
            </>,
          ],
          [
            <>
              <code>torch</code>, <code>numpy</code>, <code>GenAIRR</code>, <code>safetensors</code>, <code>yaml</code>,{" "}
              <code>platformdirs</code>
            </>,
            <>
              The core dependencies, each with its version. <code>MISSING</code> on any of them means a broken install,
              and the report ends with a <code>WARNING: missing core dependency</code> line.
            </>,
          ],
          [
            <>
              <code>zstandard</code>, <code>huggingface_hub</code>
            </>,
            <>
              The two extras. <code>MISSING</code> is normal unless you need them (<DocLink to="install?h=extras">extras</DocLink>).
            </>,
          ],
          [
            <>
              <code>derive_backend</code>, <code>vband_backend</code>
            </>,
            <>
              <code>cython</code> when AlignAIR's compiled speed-ups are installed; <code>python</code> means they are
              not (pip built AlignAIR from source without a C compiler), and predictions run slower.
            </>,
          ],
          [<code>cache_dir</code>, <>Where downloaded models are kept (set <code>ALIGNAIR_CACHE_DIR</code> to move it).</>],
          [<code>config_dir</code>, "Where AlignAIR reads its optional settings file."],
        ]}
      />
      <p>
        <code>alignair doctor --json</code> prints the same report as JSON, which is the form to paste into a bug report.
      </p>

      <H2 id="models-and-the-cache">Models, the cache and offline use</H2>
      <p>
        Nothing is downloaded at install time. The first time you name a pretrained model, for example{" "}
        <code>--model alignair4-igh-human</code>, AlignAIR downloads it from Hugging Face (no account needed) into the
        cache, and later runs read it from there. On a machine without internet access, such as a cluster's compute
        node, download the model ahead on a machine that has it and run with <code>--offline</code>:
      </p>
      <CodeBlock
        code={`alignair models get alignair4-igh-human\nalignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv \\\n  --offline`}
      />
      <p>
        A lab can share one cache: point <code>ALIGNAIR_CACHE_DIR</code> at a shared folder on every machine.{" "}
        <code>alignair models list</code> lists the models and which ones you have.{" "}
        <DocLink to="cache-and-environment">Cache, offline and environment variables</DocLink> has the details.
      </p>

      <H2 id="install-problems">Install problems</H2>

      <H3 id="python-version">pip finds no matching version, or says AlignAIR requires a different Python</H3>
      <p>
        Your Python is older than 3.10. Check with <code>python3 --version</code>, then make an environment with a newer
        Python (<code>python3.12 -m venv alignair-env</code>, or <code>conda create -n alignair python=3.12</code>) and
        install there.
      </p>

      <H3 id="command-not-found">alignair: command not found</H3>
      <p>
        The environment you installed into is not active in this shell. Activate it (<code>source
        alignair-env/bin/activate</code>, or <code>conda activate alignair</code>) and try again;{" "}
        <code>pip show alignair</code> tells you whether this environment has AlignAIR.
      </p>

      <H3 id="missing-core-dependency">WARNING: missing core dependency, or No module named 'GenAIRR'</H3>
      <p>
        GenAIRR, PyTorch and the others are core dependencies: <code>pip install alignair</code> installs them. If one
        is missing, the install was interrupted or another package removed it. Run{" "}
        <code>pip install alignair</code> again in the same environment; pip installs what is missing. Then run{" "}
        <code>alignair doctor</code> to check.
      </p>

      <H3 id="gpu-not-used">device_auto says cpu on a machine with a GPU</H3>
      <p>
        The PyTorch build in this environment has no GPU support (<code>cuda_available False</code>). Install a CUDA
        build of PyTorch from the selector on{" "}
        <a href="https://pytorch.org/get-started/locally/" target="_blank" rel="noreferrer">pytorch.org</a> into the
        same environment, then check <code>alignair doctor</code> again. The Docker image runs on the CPU only.
      </p>

      <H3 id="hub-missing">loading a model directly from a Hugging Face repo needs 'huggingface_hub'</H3>
      <p>
        The error code is <code>ALIGNAIR_DEPENDENCY_MISSING</code>. You named a model as <code>hf://org/repo</code>:
        install the extra with <code>pip install "alignair[hub]"</code>. A private or gated repo also needs your token,
        with <code>--hf-token</code> or the <code>HF_TOKEN</code> environment variable.
      </p>

      <H3 id="cannot-download">The first run cannot download the model</H3>
      <p>
        <code>ALIGNAIR_REGISTRY_UNAVAILABLE</code> means Hugging Face could not be reached: check the machine's internet
        access or proxy, and try again. <code>ALIGNAIR_MODEL_UNAVAILABLE</code> means you ran with{" "}
        <code>--offline</code> and the model is not in the cache yet: run <code>alignair models get</code> once while
        online (<DocLink to="install?h=models-and-the-cache">above</DocLink>).
      </p>

      <H3 id="not-a-file">model path must be a file</H3>
      <p>
        The code is <code>ALIGNAIR_MODEL_INVALID</code>: <code>--model</code> names a folder. Give the{" "}
        <code>.alignair</code> file inside it (a training run's <code>bundle/</code> folder holds{" "}
        <code>model.alignair</code>).
      </p>

      <Callout kind="tip" title="Every error has a code">
        Every failure prints a code such as <code>ALIGNAIR_MODEL_INVALID</code>. <code>--debug</code> adds the full
        traceback, and <code>--error-format json</code> prints the error as JSON for a pipeline.{" "}
        <DocLink to="error-codes">Error codes</DocLink> lists them all with what to do; for anything else, see{" "}
        <DocLink to="support">Support</DocLink>.
      </Callout>

      <p>
        A problem with a run rather than the install? See <DocLink to="fix-common-problems">Fix common problems</DocLink>.
        Next: <DocLink to="first-alignment">Your first alignment</DocLink>.
      </p>
    </>
  ),
};

/* ---------------------------------------------------------------------------------------------------------------- */
/* Your first alignment                                                                                              */
/* ---------------------------------------------------------------------------------------------------------------- */

const firstAlignment: DocPage = {
  slug: "first-alignment",
  title: "Your first alignment",
  section: "Get started",
  lead: "Align twenty example IGH reads with the human IGH model, read the table you get back, and do the same in Python and on your own reads.",
  body: () => (
    <>
      <p>
        You need AlignAIR installed (<DocLink to="install">Install</DocLink>) and an internet connection for the first
        run, which downloads the model.
      </p>

      <H2 id="get-the-example-reads">Get the example reads</H2>
      <p>
        The example file holds twenty productive human IGH reads, simulated with GenAIRR. Download it into an empty
        folder:
      </p>
      {/* PENDING-V4-PUBLISH: the release tag in this URL is set when the release is cut. */}
      <CodeBlock code={`curl -LO https://raw.githubusercontent.com/MuteJester/AlignAIR/v4.0.0/examples/reads.fasta`} />
      <p>
        With Docker, the same file is already in the image, at <code>/app/examples/reads.fasta</code>.
      </p>

      <H2 id="align-them">Align them</H2>
      <CodeBlock
        code={`$ alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv\naligned 20 reads (0 dropped) -> out.tsv`}
      />
      <ul>
        <li>
          <code>--model</code> names the model: here the human IGH model by its id. The first run downloads it from
          Hugging Face and keeps it in the cache; later runs start at once. It also takes a <code>.alignair</code> file.
        </li>
        <li>
          <code>--input</code> is the reads; <code>--out</code> is the table AlignAIR writes.
        </li>
        <li>
          The line it prints counts the reads aligned and the reads dropped. A read is dropped when more than 20% of
          its characters are unusable: neither a base (A, C, G, T, N) nor an IUPAC ambiguity code such as R or Y, which
          is read as N (<code>--rejects-out rejects.tsv</code> keeps them, with the reason). When
          some rows could be assembled only in part, the line adds how many, for example{" "}
          <code>; 0 failed / 3 partial AIRR assemblies tagged</code>.
        </li>
      </ul>
      <p>
        Beside <code>out.tsv</code> you also get <code>out.tsv.run.json</code>, a record of the run: the model, the
        germline set the calls came from, the options and the counts.
      </p>

      <H2 id="read-the-table">Read the table</H2>
      <p>
        <code>out.tsv</code> is an AIRR rearrangement table: one row per read, tab-separated, with 99 columns. The first
        rows, in the columns you would look at first:
      </p>
      <CodeBlock
        lang="text"
        title="out.tsv"
        code={`sequence_id  v_call                         d_call       j_call    junction_aa                  productive\nread01       IGHVF10-G50*02                 IGHD3-10*01  IGHJ6*02  CATFYYGSGSYYNFVSNYNYYGMDVW   T\nread02       IGHVF10-G42*01,IGHVF10-G42*03  IGHD2-2*02   IGHJ4*02  CAKGTSCNSTSCYTIDYW           T\nread03       IGHVF3-G12*02                  IGHD5-18*02  IGHJ6*03  CARDTLNIVATPISAFAIYYYYYMDVW  T`}
      />
      <p>And a few columns that tell you how far to trust each row:</p>
      <CodeBlock
        lang="text"
        title="out.tsv"
        code={`sequence_id  v_identity          rev_comp  airr_assembly_status\nread01       0.958904109589041   F         complete\nread02       0.9690721649484536  F         complete\nread03       0.9829351535836177  F         complete`}
      />
      <DocTable
        head={["Column", "What it says"]}
        rows={[
          [
            <>
              <code>v_call</code>, <code>d_call</code>, <code>j_call</code>
            </>,
            <>
              The called alleles. When the read cannot tell V alleles apart, <code>v_call</code> is followed by the
              alleles that match it exactly as well, comma-separated, as IgBLAST lists them (read02);{" "}
              <code>d_call</code> and <code>j_call</code> name one allele. The IG models use OGRDB allele
              names (<code>IGHVF10-G50*02</code>); <DocLink to="concepts?h=allele-names">allele names</DocLink> explains
              them.
            </>,
          ],
          [<code>junction_aa</code>, "The junction translated: from the V's conserved Cys to the J's Trp/Phe, both included."],
          [
            <code>productive</code>,
            <>
              <code>T</code> when the rearrangement is in frame with no stop codon, <code>F</code> when it is not, blank
              when it cannot be told.
            </>,
          ],
          [<code>v_identity</code>, "The fraction (0 to 1, not a percentage) of the read's V bases that match the called germline."],
          [
            <code>rev_comp</code>,
            <>
              <code>T</code> when the read came reverse-complemented. AlignAIR turns reads around itself; you never need
              to orient them first.
            </>,
          ],
          [
            <code>airr_assembly_status</code>,
            <>
              <code>complete</code>, <code>partial</code> (with the reason in <code>airr_assembly_reason</code>, such as{" "}
              <code>no_v</code>) or <code>failed</code>. Keep to complete rows for junction and clone work.
            </>,
          ],
        ]}
      />
      <p>
        All 20 example reads come out productive and complete. <DocLink to="output-columns">Output columns
        explained</DocLink> walks through the rest of the table, and the{" "}
        <DocLink to="column-dictionary">column dictionary</DocLink> lists all 99 columns.
      </p>

      <H3 id="in-pandas">In pandas</H3>
      <p>
        The true/false columns hold the strings <code>T</code> and <code>F</code>, as the AIRR standard writes them.
        Compare them with <code>"T"</code>:
      </p>
      <CodeBlock
        lang="python"
        code={`import pandas as pd\n\ntable = pd.read_csv("out.tsv", sep="\\t")\ncomplete = table["airr_assembly_status"] == "complete"\nusable = table[complete & (table["productive"] == "T")]\nprint(len(usable))      # 20`}
      />
      <Callout kind="warning" title="Never .astype(bool) a T/F column">
        Any non-empty string is true in Python, so <code>.astype(bool)</code> turns every <code>"F"</code> into{" "}
        <code>True</code>, and a filter built on it keeps the wrong rows. Compare with <code>== "T"</code>, or read the
        file with <code>pd.read_csv(..., true_values=["T"], false_values=["F"])</code>.
      </Callout>

      <H2 id="the-same-in-python">The same in Python</H2>
      <CodeTabs
        cli={`$ alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv\naligned 20 reads (0 dropped) -> out.tsv`}
        python={`from alignair import Aligner, read_sequences\n\naligner = Aligner.from_pretrained("alignair4-igh-human")\nids, reads, info = read_sequences("reads.fasta")\nresult = aligner.predict(reads, ids=ids)\nresult.write_airr("out.tsv")\nprint(result.records[0]["v_call"], result.records[0]["junction_aa"])\n# IGHVF10-G50*02 CATFYYGSGSYYNFVSNYNYYGMDVW`}
      />
      <ul>
        <li>
          <code>Aligner.from_pretrained</code> takes the same ids and paths as <code>--model</code>, and downloads and
          caches the same way.
        </li>
        <li>
          <code>read_sequences</code> reads FASTA, FASTQ or a table and returns the ids, the reads, and a summary (
          <code>info["n_read"]</code>, <code>info["n_dropped"]</code>); it drops the same reads the command line drops.
        </li>
        <li>
          <code>predict</code> returns a result whose <code>records</code> are one dict per read, in Python's terms:
          positions count from 0 with the end excluded, and true/false fields are <code>True</code> and{" "}
          <code>False</code> (the file has positions from 1, <code>T</code> and <code>F</code>).{" "}
          <code>write_airr</code> writes the AIRR table; give tools downstream the table (
          <DocLink to="python-api?h=records-and-the-table">records and the table</DocLink>).
        </li>
        <li>
          <code>write_airr</code> writes the same table the command line writes; <code>columns="core"</code> (or a list
          of names) writes fewer columns.
        </li>
      </ul>

      <H2 id="your-own-reads">Run it on your own reads</H2>
      <p>Point the input at your file. FASTA, FASTQ and gzipped files are read as they are:</p>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input my_reads.fastq.gz --out my_reads.tsv`}
        python={`from alignair import Aligner, read_sequences\n\naligner = Aligner.from_pretrained("alignair4-igh-human")\nids, reads, info = read_sequences("my_reads.fastq.gz")\naligner.predict(reads, ids=ids).write_airr("my_reads.tsv")`}
      />
      <p>
        A CSV or TSV table works too. AlignAIR finds the sequence column when it is called <code>sequence</code>,{" "}
        <code>seq</code>, <code>nucleotide</code> or <code>read</code>, and the id column when it is called{" "}
        <code>sequence_id</code>, <code>id</code> or <code>name</code>. Name other columns yourself:
      </p>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input reads.csv --out out.tsv \\\n  --sequence-column nt_seq --id-column read_name`}
        python={`ids, reads, info = read_sequences("reads.csv", seq_column="nt_seq", id_column="read_name")\nresult = aligner.predict(reads, ids=ids)`}
      />
      <p>Before a real run, four things are worth knowing:</p>
      <ul>
        <li>
          <strong>Pick the model of your locus.</strong> Each model reads one locus: <code>alignair4-igk-human</code> for
          kappa reads, <code>alignair4-trb-human</code> for TCR beta, and so on.{" "}
          <DocLink to="choose-a-model">Choose a model</DocLink> lists all seven and what each contains.
        </li>
        <li>
          <strong>Long reads are cut.</strong> A read longer than 575 nt is aligned on its first 575 nt (its{" "}
          <code>length_cropped</code> is <code>T</code>). The cut comes before the read is turned around: a forward
          read loses its 3' end, often only constant region but the J and junction when much comes before the V; a
          reverse-complemented one loses the start of its V.{" "}
          <DocLink to="prepare-input?h=long-reads">Reads longer than 575 nt</DocLink> says which rows to check, and
          when to trim.
        </li>
        <li>
          <strong>Ids stay yours.</strong> Each row keeps its read's id; a repeated id gets <code>_dup1</code>,{" "}
          <code>_dup2</code>, ... and every read is kept.
        </li>
        <li>
          <strong>Large files.</strong> The command line streams the input in chunks of 20,000 reads, so memory stays
          flat; in Python, <code>read_sequences</code> reads the whole file at once (see the{" "}
          <DocLink to="python-api">Python API</DocLink> for reading in chunks).
        </li>
      </ul>
      <p>
        <DocLink to="prepare-input">Prepare your input</DocLink> covers input formats in full.
      </p>

      <H2 id="next-steps">Next steps</H2>
      <ul>
        <li>
          <strong>Learn the words.</strong> <DocLink to="concepts">Concepts for repertoire researchers</DocLink>: models,
          germline sets, genotypes and the two allele namings.
        </li>
        <li>
          <strong>Align with a donor's genotype</strong>, given or inferred from the reads:{" "}
          <DocLink to="donor-genotype">Align with a donor genotype</DocLink>.
        </li>
        <li>
          <strong>Use your own germline set</strong>, novel alleles included:{" "}
          <DocLink to="own-reference">Use your own germline reference</DocLink>.
        </li>
        <li>
          <strong>Coming from another tool?</strong> <DocLink to="from-igblast">IgBLAST</DocLink>,{" "}
          <DocLink to="from-mixcr">MiXCR</DocLink>, <DocLink to="from-immcantation">Immcantation</DocLink>.
        </li>
        <li>
          <strong>Judge the calls.</strong> <DocLink to="trust-the-calls">How far to trust the calls</DocLink> and the{" "}
          <DocLink to="comparison">comparison with IgBLAST</DocLink>.
        </li>
        <li>
          <strong>Every option:</strong> the <DocLink to="cli">CLI reference</DocLink>, or{" "}
          <code>alignair predict --help</code>.
        </li>
      </ul>
    </>
  ),
};

/* ---------------------------------------------------------------------------------------------------------------- */
/* Concepts                                                                                                          */
/* ---------------------------------------------------------------------------------------------------------------- */

const concepts: DocPage = {
  slug: "concepts",
  title: "Concepts for repertoire researchers",
  section: "Get started",
  lead: "The words these guides use, in repertoire terms: what a model, a germline set, a reference and a genotype are, how the allele names read, and what a model holds.",
  body: () => (
    <>
      <H2 id="what-alignair-does">What AlignAIR does</H2>
      <p>
        Each B- or T-cell receptor chain is assembled from one V, one D (in IGH, TRB and TRD) and one J germline
        segment. The joins are trimmed and filled with non-templated (N) nucleotides, and B cells add somatic
        hypermutations on top. For each sequenced read, AlignAIR reports which V, D and J alleles it came from, where
        each segment sits in the read, the junction, and whether the rearrangement is productive: one row of an AIRR
        rearrangement table.
      </p>
      <p>Pick segments to see how the choice of V, D and J changes the read and its junction:</p>
      <VdjBuilderWidget />
      <p>
        The widget joins the segments without trimming or N nucleotides and uses IMGT-style names: it shows the idea,
        not a real read. The IGH model calls V alleles by their OGRDB names (<DocLink to="concepts?h=allele-names">below</DocLink>).
      </p>

      <H2 id="glossary">Glossary</H2>

      <H3 id="reads-and-germlines">Reads and germlines</H3>
      <DocTable
        head={["Term", "What it means here"]}
        rows={[
          [
            <strong>Locus</strong>,
            <>
              The chain a read comes from: <code>IGH</code>, <code>IGK</code>, <code>IGL</code> (B cells),{" "}
              <code>TRA</code>, <code>TRB</code>, <code>TRD</code>, <code>TRG</code> (T cells). IGH, TRB and TRD join V,
              D and J; the others join V to J.
            </>,
          ],
          [
            <strong>Gene, allele</strong>,
            <>
              A germline segment (<code>IGHV1-2</code>) and one of its sequence variants (<code>IGHV1-2*02</code>).
              AlignAIR calls alleles.
            </>,
          ],
          [
            <strong>Germline set</strong>,
            <>
              The V, D and J alleles of one locus and species, with their sequences, as IMGT or OGRDB publish them.
              AlignAIR's built-in sets have names such as <code>HUMAN_IGH_OGRDB</code>; <code>alignair reference
              list</code> lists them.
            </>,
          ],
          [
            <strong>GenAIRR</strong>,
            "The repertoire simulator AlignAIR's models are trained on. It also ships the built-in germline sets, and is installed with AlignAIR.",
          ],
          [
            <strong>Dataconfig</strong>,
            <>
              GenAIRR's word for a built-in germline set. <code>alignair train</code> takes one by name, as in{" "}
              <code>--dataconfig HUMAN_IGH_OGRDB</code>.
            </>,
          ],
          [
            <strong>Genotype</strong>,
            <>
              The alleles one donor carries. <code>--genotype</code> limits the calls to them; <code>--infer-genotype</code>{" "}
              infers it from the reads; <code>alignair genotype</code> writes one from any AIRR table. See{" "}
              <DocLink to="donor-genotype">Align with a donor genotype</DocLink>.
            </>,
          ],
          [
            <strong>Novel allele</strong>,
            <>
              An allele the germline set lacks. Give its sequence in a FASTA with <code>--reference</code>; no training
              is needed. See <DocLink to="own-reference">Use your own germline reference</DocLink>.
            </>,
          ],
          [
            <strong>Junction, CDR3</strong>,
            "The junction runs from the V's conserved Cys codon to the J's Trp/Phe codon, both included; the CDR3 is the junction without those two codons.",
          ],
          [
            <strong>SHM</strong>,
            <>
              Somatic hypermutation: the point mutations B cells gain. <code>mutation_rate</code> estimates it per read,
              as a fraction from 0 to 1. T cells do not hypermutate.
            </>,
          ],
          [
            <strong>Orientation</strong>,
            <>
              A read can come forward, reverse-complemented, complemented or reversed. AlignAIR turns it forward itself;{" "}
              <code>orientation</code> names the form it came in, and <code>rev_comp</code> is <code>T</code> for a
              reverse complement.
            </>,
          ],
        ]}
      />

      <H3 id="models">Models</H3>
      <DocTable
        head={["Term", "What it means here"]}
        rows={[
          [
            <strong>Model</strong>,
            <>
              A file (<code>.alignair</code>) that holds a germline set and what was learned from simulated reads of its
              locus. You name it by its id, its path, or <code>hf://org/repo</code> (a Hugging Face repo). See{" "}
              <DocLink to="concepts?h=what-a-model-contains">what a model contains</DocLink>.
            </>,
          ],
          [
            <strong>Model id</strong>,
            <>
              A model's name in the catalog, such as <code>alignair4-igh-human</code>; <code>id@version</code> pins one
              version.
            </>,
          ],
          [
            <strong>Registry, catalog</strong>,
            <>
              The list of model ids and versions AlignAIR looks an id up in: by default, the AlignAIR models on Hugging
              Face. <code>alignair models list</code> shows it; <code>--registry</code> or <code>ALIGNAIR_REGISTRY</code>{" "}
              adds another, such as a lab's own.
            </>,
          ],
          [
            <strong>Cache</strong>,
            <>
              Where downloaded models are kept: <code>~/.cache/alignair</code> on Linux (<code>alignair doctor</code>{" "}
              prints yours, <code>ALIGNAIR_CACHE_DIR</code> moves it). <code>alignair models get</code> fills it ahead;{" "}
              <code>--offline</code> uses only it.
            </>,
          ],
          [
            <strong>Model card</strong>,
            <>
              A model's description: its germline set, allele counts and training. <code>alignair info &lt;id&gt;</code>{" "}
              prints it; <DocLink to="model-cards">Model cards</DocLink> has all seven.
            </>,
          ],
          [
            <strong>Reference</strong>,
            "The germline set a run calls from: the model's own, unless you bind another.",
          ],
          [
            <strong>Bind</strong>,
            <>
              Make a model call from another germline set: <code>--reference germline.fasta</code> on the command line,{" "}
              <code>reference=</code> in Python. The model is not retrained.
            </>,
          ],
          [
            <strong>Train, fine-tune</strong>,
            <>
              Training makes a new model, for a locus, species or germline set no model covers; fine-tuning starts that
              training from an existing model (the <code>--init-from</code> option of <code>alignair train</code>). See{" "}
              <DocLink to="do-i-need-to-train">Do I need to train?</DocLink>
            </>,
          ],
          [
            <strong>Checkpoint</strong>,
            <>
              A training snapshot: the file <code>alignair train</code> saves as it goes. <code>alignair export</code>{" "}
              turns one into a model you can share.
            </>,
          ],
        ]}
      />

      <H3 id="the-output">The output</H3>
      <DocTable
        head={["Term", "What it means here"]}
        rows={[
          [
            <strong>AIRR rearrangement table</strong>,
            <>
              The AIRR Community's standard TSV: one row per read, with columns such as <code>sequence_id</code>,{" "}
              <code>v_call</code> and <code>junction_aa</code>. AlignAIR writes 99 columns by default; Change-O, TIgGER
              and SCOPer read it as it is. See the <DocLink to="column-dictionary">column dictionary</DocLink>.
            </>,
          ],
          [
            <strong>Preset</strong>,
            <>
              A named set of columns for <code>--columns</code>: <code>full</code> (99, the default), <code>core</code>{" "}
              (27), <code>airr</code> (14) or <code>minimal</code> (7).
            </>,
          ],
          [
            <strong>Call and ties</strong>,
            <>
              <code>v_call</code> names the called allele, then the alleles that match the read's V bases exactly as
              well, comma-separated, as IgBLAST lists them, so TIgGER or Change-O can choose among them.
            </>,
          ],
          [
            <strong>Candidate set</strong>,
            <>
              <code>v_call_set</code>, <code>d_call_set</code>, <code>j_call_set</code>: the alleles the model scored at
              least half as likely as its best (three at most), before the read is aligned to the germline. The final
              call can fall outside it.
            </>,
          ],
          [
            <strong>Identity</strong>,
            <>
              <code>v_identity</code>, <code>d_identity</code>, <code>j_identity</code>: the fraction (0 to 1) of the
              segment's bases that match the called germline.
            </>,
          ],
          [
            <strong>Productive</strong>,
            <>
              <code>T</code> when the rearrangement is in frame with no stop codon, <code>F</code> when not, blank when it
              cannot be told.
            </>,
          ],
          [
            <strong>Assembly status</strong>,
            <>
              <code>airr_assembly_status</code>: <code>complete</code>, <code>partial</code> (with a reason in{" "}
              <code>airr_assembly_reason</code>, such as <code>no_v</code>) or <code>failed</code>. When more than 1% of
              rows fail, the command exits with an error status; the table is still written.
            </>,
          ],
          [
            <strong>Dropped read</strong>,
            <>
              A read more than 20% of whose characters are unusable: neither a base (A, C, G, T, N) nor an IUPAC
              ambiguity code (R, Y, S, W, K, M, B, D, H, V, read as N). An empty read is dropped too. It is counted as
              dropped, and written to the <code>--rejects-out</code> file if you give one.
            </>,
          ],
          [
            <strong>575-nt window</strong>,
            <>
              A read longer than 575 nt is aligned on its first 575 nt, and its <code>length_cropped</code> is{" "}
              <code>T</code> (blank for a read that was not cut).
            </>,
          ],
          [
            <strong>Run record</strong>,
            <>
              <code>out.tsv.run.json</code>, written beside the table: the model, the germline set the calls came from,
              the options and the counts. <code>--no-run-metadata</code> leaves it out.
            </>,
          ],
        ]}
      />

      <H2 id="allele-names">OGRDB and IMGT allele names</H2>
      <p>
        The three IG models are trained on OGRDB's human germline sets and call alleles by their OGRDB names. The four
        TCR models are trained on IMGT's sets and use IMGT names. The names you will see:
      </p>
      <DocTable
        head={["Models", "Example names", "How to read them"]}
        rows={[
          [
            "IGH",
            <>
              <code>IGHVF10-G50*02</code>, <code>IGHD3-10*01</code>, <code>IGHJ6*02</code>
            </>,
            <>
              V alleles have OGRDB names: <code>F</code> is the family, <code>G</code> the gene group. D and J alleles
              keep IMGT-style names.
            </>,
          ],
          [
            "IGK, IGL",
            <>
              <code>IGKV1-5TCS*03</code>, <code>IGKJ2-27FL*01</code>, <code>IGLV6-A5V2*05</code>
            </>,
            "OGRDB names for V and J: a number and a short label after the segment.",
          ],
          [
            "TRA, TRB, TRD, TRG",
            <>
              <code>TRBV10-1*01</code>, <code>TRBD2*01</code>, <code>TRBJ1-2*01</code>
            </>,
            "IMGT names.",
          ],
        ]}
      />
      <p>
        This matters downstream. A genotype, a TIgGER result or a gene-usage table written with IMGT names (
        <code>IGHV3-23*01</code>) does not match an IG model's calls. A genotype that names alleles the model's germline
        set does not hold is refused with <code>ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND</code>, rather than matched by a
        guess.
      </p>

      <H3 id="working-with-imgt-named-data">Working with IMGT-named data</H3>
      <p>Two ways:</p>
      <ol>
        <li>
          <strong>Translate the names.</strong> OGRDB's current human sets, at{" "}
          <a href="https://ogrdb.airr-community.org/germline_sets/Homo%20sapiens" target="_blank" rel="noreferrer">
            ogrdb.airr-community.org/germline_sets/Homo sapiens
          </a>
          , name their alleles by their IMGT (IUIS) names and hold the IGH model's sequences, so matching sequences
          maps each label to its IMGT name (<code>IGHVF10-G50*02</code> is <code>IGHV3-11*04</code>).{" "}
          <DocLink to="choose-a-model?h=map-names">Map the labels to IMGT names</DocLink> has the script.
        </li>
        <li>
          <strong>Bind the IMGT germline set itself.</strong> Give the model a FASTA of the IMGT alleles; its calls then
          carry those names. Immcantation's IMGT germline files are read as they are, full IMGT/GENE-DB headers
          included.
        </li>
      </ol>
      <CodeTabs
        cli={`alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv \\\n  --reference imgt_igh.fasta`}
        python={`from alignair import Aligner, ReferenceSet\n\naligner = Aligner.from_pretrained("alignair4-igh-human", reference=ReferenceSet.from_fasta("imgt_igh.fasta"))`}
      />
      <p>
        <DocLink to="choose-a-model?h=bind-imgt">Call with IMGT's set</DocLink> shows the calls this gives and what the
        model's own set lacks; <DocLink to="own-reference">Use your own germline reference</DocLink> says what such a
        FASTA needs.
      </p>

      <H2 id="what-a-model-contains">What a model contains</H2>
      <p>A model file holds three things:</p>
      <ul>
        <li>
          <strong>A germline set.</strong> The alleles it was trained with, with their sequences: the IMGT-gapped V
          sequences and the positions of the conserved codons that frame the junction. This is the reference a run
          calls from unless you bind another.
        </li>
        <li>
          <strong>What it learned.</strong> From millions of simulated reads of its locus, the model learned to find the
          segments in a read, the conserved codons, the mutations, and which germline each base comes from. What it
          learned is about reading a locus, not a list of alleles: the alleles are looked up in the germline set at
          each run.
        </li>
        <li>
          <strong>Its card.</strong> The germline set's name, the allele counts and the training settings, which{" "}
          <code>alignair info</code> prints.
        </li>
      </ul>
      <p>
        Each pretrained model covers one locus of one species: human, and the locus in its id. To see the germline set a
        model calls from, export it as FASTA:
      </p>
      <CodeBlock code={`alignair reference export "$(alignair models path alignair4-igh-human)" --fasta igh_germline.fasta`} />

      <H3 id="narrow-extend-or-train">Narrow it, extend it, or train</H3>
      <p>Because the germline set is looked up at each run, three cases are handled differently:</p>
      <DocTable
        head={["You want to", "Do this", "Training?"]}
        rows={[
          [
            "Call only a donor's alleles",
            <>
              <code>--genotype donor.yaml</code> (a subset of the germline set), or <code>--infer-genotype</code>
            </>,
            "No",
          ],
          [
            "Call alleles the germline set lacks, novel ones included",
            <>
              <code>--reference germline.fasta</code> with those alleles
            </>,
            "No",
          ],
          ["Align another locus, or another species", "Train a model for it", "Yes"],
        ]}
      />
      <p>
        <DocLink to="do-i-need-to-train">Do I need to train?</DocLink> walks through the choice.{" "}
        <DocLink to="how-alignair-works">How AlignAIR works</DocLink> explains, for the curious, how the model uses the
        germline set.
      </p>
    </>
  ),
};

/* ---------------------------------------------------------------------------------------------------------------- */
/* How AlignAIR works (optional): the only user page that names the model's internals, each defined.                 */
/* ---------------------------------------------------------------------------------------------------------------- */

const howItWorks: DocPage = {
  slug: "how-alignair-works",
  title: "How AlignAIR works (optional)",
  section: "Get started",
  lead: "For the curious: what happens between a read and its row in the table, in the model's own terms. You do not need this page to use AlignAIR.",
  body: () => (
    <>
      <H2 id="the-idea">The idea</H2>
      <p>
        AlignAIR learns alignment from reads simulated with GenAIRR, where the truth of every read is known: its alleles,
        the region of every base, its mutations, its conserved codons and its orientation. One model learns all of these
        tasks together. The{" "}
        <a href="https://doi.org/10.1093/nar/gkaf651" target="_blank" rel="noreferrer">
          paper in Nucleic Acids Research
        </a>{" "}
        introduced this approach (multi-task deep learning for adaptive immune receptor alignment); this page describes
        the parts of the current models, as the code in{" "}
        <a href="https://github.com/MuteJester/AlignAIR/tree/main/src/alignair/network" target="_blank" rel="noreferrer">
          src/alignair/network
        </a>{" "}
        builds them.
      </p>
      <p>
        The model is a <strong>neural network</strong>: layers of simple functions whose numbers, the{" "}
        <strong>weights</strong>, were fitted during training. A read goes through it once, in one{" "}
        <strong>forward pass</strong>; exact alignment steps against the germline sequences then check and refine what it
        found, and plain rules build the AIRR row.
      </p>

      <H2 id="from-read-to-row">From a read to a row</H2>
      <CodeBlock
        lang="text"
        title="one read"
        code={`read: bases A C G T N, its first 575 nt
  |
  v  orientation      12-base word vote, else the orientation head; turned forward
  v  embedding        each base becomes a list of learned numbers
  v  trunk            dilated convolutions: features of each base in its context
  +- parser           regions: before V | V | N1 | D | N2 | J | after
  +- anchor heads     the V's Cys codon and the J's Trp/Phe codon
  +- mutation reader  per base: germline, substitution or insertion; its column
  +- bank scans       a score for every allele of the bound germline set
  +- D presence       D absent, too short to call, or callable
  v  refinement       exact alignments: locus, V end, J start, D, top V alleles
  v  calls            candidate set, germline alignment, a better V, ties
  v  AIRR assembly    junction, productivity, gapped alignments, CIGARs, identity
row of the AIRR rearrangement table`}
      />

      <H3 id="orientation">Orientation</H3>
      <p>
        A read may come in any of four forms: forward, reverse complement, complement or reversed. Two signals decide.
        The <strong>k-mer vote</strong> counts, for each form, how many of the read's 12-base words occur in the bound
        germline set; when one form wins outright, it decides. Otherwise the <strong>orientation head</strong> decides: a{" "}
        <strong>head</strong> is a small output layer that answers one question from the model's features, here which of
        the four forms the read is in. The read is turned forward before anything else reads it.
      </p>

      <H3 id="embedding-and-trunk">Embedding and trunk</H3>
      <p>
        Each base becomes an <strong>embedding</strong>: a list of learned numbers that stands for the base and its
        position. The <strong>trunk</strong> then gives every base its features, once, for all the parts below to share.
        It is a stack of <strong>dilated convolutions</strong>: filters that look at a base together with bases 1, 2, 4,
        ... 32 positions away, twice over, so each base's features describe several hundred bases around it.
      </p>

      <H3 id="parser-and-anchors">Parser and anchors</H3>
      <p>
        The <strong>parser</strong> cuts the read into its ordered regions: before V, V, N1, D, N2, J, after. It is a
        linear-chain CRF, a model that scores every allowed path through those regions and picks among them exactly; a
        path can skip a region (a light chain has no D, a fragment no start), but it can never put V after J. Two{" "}
        <strong>anchor heads</strong> find the conserved codons that frame the junction: the V's Cys and the J's
        Trp/Phe.
      </p>

      <H3 id="mutation-reader">Mutation reader</H3>
      <p>
        The <strong>mutation reader</strong> says, for each base, how likely it is a germline base, a substitution or an
        insertion, and which column of the germline it lines up with. It never guesses the germline base itself, which
        would be memorizing the germline set: what the germline is comes only from the bound set.
      </p>

      <H3 id="bank-and-scans">The bank and its scans</H3>
      <p>
        The <strong>bank</strong> is the bound germline set laid out as rows, one per allele, of bases on fixed columns
        (IMGT-gapped for V). It is an input, rebuilt from whatever germline set the run uses, and not part of the
        weights. A <strong>scan</strong> scores every row of one segment's bank against the read: over the read's bases
        in that segment's region, a row gains where it has the read's base at the base's column, pays the cost of a
        mutation where it has another base, and pays a penalty where it has none. The scores become a probability for
        each allele. A donor genotype hides rows; a novel allele is a new row.
      </p>

      <H3 id="d-presence">D presence</H3>
      <p>
        For loci with a D, a small classifier decides from the parser's and the D scan's evidence whether the read shows
        its D: absent, too short to call, or callable.
      </p>

      <H3 id="refinement">Refinement against the germline</H3>
      <p>Then come exact steps, with no weights:</p>
      <ul>
        <li>
          <strong>One locus per read</strong>, when a germline set holds several loci: the locus of the read's most
          likely V and J pair.
        </li>
        <li>
          <strong>V end and J start</strong>: from the anchors, the read is aligned to the called alleles' germlines, and
          each boundary moves to where the read stops following its germline.
        </li>
        <li>
          <strong>The D</strong>: the junction is aligned to every D allele, forward and inverted, and that alignment is
          combined with the D scan; the D's edges are then set on the called D's germline.
        </li>
        <li>
          <strong>The top five V alleles</strong> are re-scored by a banded gapped alignment, which follows an insertion
          or deletion the scan's columns cannot.
        </li>
      </ul>

      <H3 id="calls-and-assembly">Calls and the AIRR row</H3>
      <p>
        Each allele's score is taken relative to the best. The <strong>candidate set</strong> (<code>v_call_set</code>{" "}
        and the others) keeps the alleles at least half as likely as the best, three at most. The read is then aligned
        to the called germline: an allele of the germline set that matches more of the read's V bases than the call
        replaces it, which is how a bound novel allele gets called, and the alleles that match exactly as many are
        listed after the call in <code>v_call</code>. Plain rules then build the row: the junction between the anchors,
        productivity, the IMGT-gapped alignments, CIGARs, identities and the FWR and CDR regions.
      </p>

      <H2 id="how-the-model-learns">How the model learns</H2>
      <p>
        Training draws fresh reads from GenAIRR at every step; there is no fixed data set. Each read comes with its
        truth, and every part above has its own <strong>loss</strong>, a number that measures how wrong that part was on
        the read. Each loss is scaled to start near 1, and they are summed with equal weights, so no task is quietly
        turned down. Half of the time, the parts after the parser and the mutation reader are given the true regions
        and columns instead of the predicted ones (<strong>teacher forcing</strong>), so they learn from correct inputs
        early on.
      </p>
      <p>
        The reads are not all alike. A <strong>curriculum</strong> is a scale of difficulty, from clean reads to fully
        corrupted ones: more somatic hypermutation, sequencing errors and indels. Each training batch mixes reads from
        three points on that scale, reads with heavy hypermutation, and fragments that lost their 5' or 3' end; reads
        come in all four orientations, part of them with library flanks, and most of them productive. The TCR models'
        reads carry no hypermutation, as T cells do not hypermutate. Synthetic novel alleles, drawn by default, teach
        the scans to read an allele from the bound germline set rather than from memory.
      </p>

      <H2 id="why-the-germline-set-is-an-input">Why the germline set is an input</H2>
      <p>
        The weights learned how to read a locus: where its regions are, what a mutation looks like, how a base lines up
        with a germline column. The alleles themselves sit in the bank, which each run builds from its germline set.
        So a donor genotype only hides rows, and a germline FASTA with novel alleles adds rows, both without training.
        A new locus or species is different: its regions and germlines look different from anything the weights
        learned, so it needs a model trained for it.
      </p>

      <H2 id="read-more">Read more</H2>
      <ul>
        <li>
          The paper: Konstantinovsky et al., <em>Enhancing sequence alignment of adaptive immune receptors through
          multi-task deep learning</em>, Nucleic Acids Research 2025,{" "}
          <a href="https://doi.org/10.1093/nar/gkaf651" target="_blank" rel="noreferrer">doi:10.1093/nar/gkaf651</a>.
        </li>
        <li>
          How well this works, measure by measure: the <DocLink to="comparison">comparison with IgBLAST</DocLink>, and
          where it does not: <DocLink to="trust-the-calls">How far to trust the calls</DocLink>.
        </li>
        <li>
          Training your own: <DocLink to="train">Train a model for your reference</DocLink>.
        </li>
      </ul>
    </>
  ),
};

export const startPages: DocPage[] = [install, firstAlignment, concepts, howItWorks];
