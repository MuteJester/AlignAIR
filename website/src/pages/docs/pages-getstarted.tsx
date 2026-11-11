import { CodeBlock, Callout } from "../../components/ui";
import { DocTable, DocLink, type DocPage } from "./doc-kit";

const gettingStarted: DocPage = {
  slug: "getting-started",
  title: "Getting started",
  section: "Get started",
  lead: "Install AlignAIR, check your environment, and align your first reads - including against a donor genotype.",
  body: () => (
    <>
      <Callout kind="note" title="Already using IgBLAST?">
        AlignAIR writes the same AIRR schema, so most of a pipeline keeps working.{" "}
        <DocLink to="migrating-from-igblast">Migrating from IgBLAST</DocLink> maps the commands and output field by
        field, with a real side-by-side.
      </Callout>
      <p>
        The path to your first real prediction is five steps:{" "}
        <strong>install &rarr; see it work &rarr; get a model &rarr; align &rarr; read the output</strong>. You do not
        need the lessons or the training guide to get there.
      </p>
      <h2>1. Install</h2>
      <CodeBlock code={`pip install "AlignAIR[cli]"            # core + CLI (+ huggingface_hub, to load models from Hugging Face)`} />
      <p>
        PyTorch is a dependency; the device backend depends on how torch is installed. AlignAIR auto-detects it: without{" "}
        <code>--device</code> (or with <code>--device auto</code>, the Python API's default), it picks CUDA, then Apple
        MPS, then CPU.
      </p>
      <DocTable
        head={["Platform", "torch install", "AlignAIR", "Backend"]}
        rows={[
          ["Linux / Windows, NVIDIA GPU", <>default <code>pip</code> wheel (CUDA)</>, <code>pip install "AlignAIR[cli]"</code>, "CUDA"],
          ["Linux / Windows, CPU only", <>install CPU torch first (see below)</>, <code>pip install "AlignAIR[cli]"</code>, "CPU"],
          ["macOS, Apple Silicon", <>default <code>pip</code> wheel</>, <code>pip install "AlignAIR[cli]"</code>, "MPS (auto)"],
          ["macOS, Intel", <>default <code>pip</code> wheel</>, <code>pip install "AlignAIR[cli]"</code>, "CPU"],
          ["Docker (CPU)", "included in the image", <code>docker pull alignair/alignair:latest</code>, "CPU"],
        ]}
      />
      <p>For a CPU-only environment, install CPU torch first:</p>
      <CodeBlock code={`pip install torch --index-url https://download.pytorch.org/whl/cpu`} />
      <p>
        On Windows use PowerShell and quote the extra. For Docker, pin a version tag for reproducibility (for example{" "}
        <code>alignair/alignair:3.0.0</code>) rather than only <code>latest</code>; the default image is CPU-only. The same
        image is mirrored at <code>ghcr.io/mutejester/alignair</code> if Docker Hub's anonymous pull limit is a problem.
      </p>
      <p>Verify the install on any platform:</p>
      <CodeBlock code={`alignair doctor        # Python, PyTorch + CUDA/MPS, GenAIRR, cache and config dirs\nalignair demo          # offline end-to-end trial (no download)`} />

      <h3>Offline install and model caching</h3>
{/* PENDING-V4-PUBLISH: alignair4-igh-human is not published yet. */}
      <CodeBlock code={`alignair models get alignair4-igh-human           # cache the model\nalignair predict --input reads.fasta --out out.tsv --model alignair4-igh-human --offline`} />
      <p>
        <code>--offline</code> never touches the network; <code>ALIGNAIR_NO_NETWORK=1</code> turns off the passive
        update check on its own. Catalog models cache in the per-OS user cache directory; <code>alignair doctor</code>{" "}
        prints the exact <code>cache_dir</code>, and <code>ALIGNAIR_CACHE_DIR</code> overrides it. A model pulled
        straight from a Hugging Face repo lives in the Hugging Face cache instead.
      </p>

      <h2>2. See it work in one command</h2>
      <CodeBlock code={`alignair demo`} />
      <p>
        Trains a tiny demo model (not production quality), aligns simulated reads, and aligns them again under a donor
        genotype - the full train, load, predict path with no model download. It writes <code>demo.tsv</code> and{" "}
        <code>demo_donor.tsv</code>.
      </p>

      <h2>3. Get a model</h2>
      <p>
        <code>alignair predict</code> needs a model: a <code>.alignair</code> file, a catalog id (<code>id</code> or{" "}
        <code>id@version</code>), or a Hugging Face repo (<code>hf://org/repo</code>). A remote model is downloaded on
        first use and cached; a public one needs no login. AlignAIR 4 pretrained models come one per locus: IGH first,
        then IGK, IGL, TRA, TRB, TRD and TRG.
      </p>
      {/* PENDING-V4-PUBLISH: alignair4-igh-human and the other alignair4-* ids are not published yet. */}
      <Callout kind="note" title="AlignAIR 4 model ids">
        AlignAIR 4 models are to be published in the shared catalog, one per locus: <code>alignair4-igh-human</code>,{" "}
        <code>alignair4-igk-human</code>, <code>alignair4-igl-human</code>, <code>alignair4-tra-human</code>,{" "}
        <code>alignair4-trb-human</code>, <code>alignair4-trd-human</code> and <code>alignair4-trg-human</code>. None is
        published yet; until then, pass an AlignAIR 4 <code>.alignair</code> path. The hub catalog's current entries (
        <code>alignair-igh-human@1.0.0</code> and the others) are AlignAIR 3 models, which AlignAIR 4 refuses with{" "}
        <em>use alignair&lt;4 to run it</em>.
      </Callout>
      <CodeBlock code={`alignair models list                   # the catalog, fetched live from the hub\nalignair predict --input reads.fasta --out out.tsv --model alignair4-igh-human\n\n# a Hugging Face repo, pinned to a branch, tag or commit (--hf-token or $HF_TOKEN for a private one)\nalignair predict --input reads.fasta --out out.tsv --model hf://<org>/<repo> --revision <commit>\n\nalignair info model.alignair           # a model file's card: reference, alleles, training, sections`} />
      <p>Or train your own for any reference or species:</p>
      <CodeBlock code={`alignair train --dataconfig HUMAN_IGH_OGRDB --out runs/my_igh --preset desktop\n# or your own germline FASTAs\nalignair train --v-fasta v.fa --d-fasta d.fa --j-fasta j.fa --chain-type BCR_HEAVY \\\n  --out runs/custom --preset desktop`} />
      <p>
        See <DocLink to="training">Training a custom model</DocLink> for the full guide.
      </p>

      <h2>4. Align reads</h2>
      <p>Input may be FASTA, FASTQ, CSV/TSV, or TXT (one read per line; optionally <code>.gz</code>, or <code>-</code> for stdin):</p>
      <CodeBlock code={`alignair predict --input reads.fasta --out out.tsv --model runs/my_igh/bundle/model.alignair`} />
      <p>
        <strong>Tabular Data (CSV/TSV):</strong> a table's sequence column is found by name (<code>sequence</code>,{" "}
        <code>seq</code>, <code>nucleotide</code> or <code>read</code>) and its id column likewise (<code>sequence_id</code>,{" "}
        <code>id</code> or <code>name</code>). Name other columns explicitly:
      </p>
      <CodeBlock code={`alignair predict --input reads.csv --out out.tsv --model alignair4-igh-human \\\n  --sequence-column nt_seq --id-column read_name`} />
      <p>
        The output is an AIRR rearrangement TSV with the V/D/J calls, coordinates, junction, productivity, and a per-gene
        candidate-set column (<code>*_call_set</code>). When a read cannot tell V alleles apart, <code>v_call</code> lists
        the call and then the alleles that match the read as well. See <DocLink to="airr-fields">AIRR output fields</DocLink>.
      </p>

      <h2>5. Constrain to a donor genotype</h2>
      <p>
        Supply a genotype as YAML or JSON to restrict calls to a subset of the model's reference, with no retraining.
        Its keys are the segments <code>V</code>, <code>D</code> and <code>J</code>; each lists allele names of the
        model's reference (here OGRDB names, as in a model trained on <code>HUMAN_IGH_OGRDB</code>). A segment you leave
        out keeps all its alleles:
      </p>
      <CodeBlock
        lang="yaml"
        code={`# donor.yaml\nV:\n  - IGHVF1-G3*02\n  - IGHVF6-G21*07\n  - IGHVF6-G21*12\nJ:\n  - IGHJ4*02\n  - IGHJ6*02`}
      />
      <p>Pass the file path to <code>--genotype</code>:</p>
      <CodeBlock code={`alignair predict --input reads.fasta --out out.tsv \\\n  --model runs/my_igh/bundle/model.alignair --genotype donor.yaml`} />
      <p>
        A name the reference does not hold is refused (<code>ALIGNAIR_GENOTYPE_UNKNOWN_ALLELE</code>). A read whose V
        gene is not in the genotype gets no V call, and its row is tagged <code>partial</code> (reason{" "}
        <code>no_v</code>). Or let the reads decide: <code>--infer-genotype</code> infers the donor's V genotype from the
        file's own calls and re-calls the reads it settles (see the <DocLink to="cli">command-line reference</DocLink>).
      </p>
      <Callout kind="note">
        A genotype only narrows the callable set. To call alleles the model's reference lacks - novel alleles, new genes -
        bind a FASTA that holds them with <code>--reference</code>, no retraining needed; the genotype can then name
        them. See the <DocLink to="model-contract">model contract</DocLink> for what a reference swap accepts and refuses.
      </Callout>

      <h2>Common predict options</h2>
      <DocTable
        head={["Flag", "Meaning"]}
        rows={[
          [<code>--genotype FILE</code>, "YAML/JSON genotype for this run (a subset of the bound reference)"],
          [<code>--reference FASTA</code>, "germline alleles (V, D, J) to call from instead of the embedded reference, novel alleles included"],
          [<code>--infer-genotype</code>, "infer the donor's V genotype from the reads (two passes over the file) and re-call the reads it settles"],
          [<code>--metadata FILE</code>, "per-read metadata (e.g. 10x annotations) preserved into output"],
          [<code>--keep-columns LIST</code>, "comma-separated metadata columns to carry through"],
          [<code>--chunk-size N</code>, "reads per streaming chunk (bounded memory; default 20000)"],
          [<code>--columns SPEC</code>, <>a preset (<code>full</code>/<code>core</code>/<code>minimal</code>/<code>airr</code>) or a field list</>],
          [<code>--rejects-out FILE</code>, "write dropped/invalid input records here"],
          [<code>--threads N</code>, "on the CPU, N single-threaded worker processes"],
          [<code>--device cuda|cpu|mps</code>, "force a device (auto if unset)"],
        ]}
      />

      <h2>Next</h2>
      <ul>
        <li><strong>See a full run on real data.</strong> A public human IGH repertoire from download to validated AIRR: <DocLink to="worked-example">Worked example</DocLink>.</li>
        <li><strong>Read the output.</strong> Every column, how it is derived, and how to filter it: <DocLink to="airr-fields">AIRR output fields</DocLink>.</li>
        <li><strong>Coming from another tool.</strong> <DocLink to="migrating-from-igblast">Migrating from IgBLAST</DocLink>.</li>
        <li><strong>Know the limits.</strong> Where AlignAIR is weak and what to use instead: <DocLink to="known-failure-modes">Known failure modes</DocLink>.</li>
        <li><strong>Go deeper.</strong> The <DocLink to="cli">command-line reference</DocLink> and <DocLink to="python-api">Python API</DocLink> for everyday use; <DocLink to="training">Training</DocLink> for a custom reference or species.</li>
      </ul>
    </>
  ),
};

const concepts: DocPage = {
  slug: "concepts",
  title: "Concepts",
  section: "Get started",
  lead: "A short orientation to the terms AlignAIR uses.",
  body: () => (
    <>
      <h2>The problem: V(D)J alignment</h2>
      <p>
        B- and T-cells generate their receptors by recombination - stitching together one V, (one D, for some loci), and
        one J germline gene segment, then adding random junctional nucleotides and (for B-cells) accumulating point
        mutations. Sequencing a repertoire gives you millions of these rearranged reads.
      </p>
      <p>
        Aligning a read means recovering, for each read: which V / D / J alleles it came from, where each segment sits in
        the read, and the junction (CDR3) where they meet. That is what AlignAIR produces, in standard AIRR format.
      </p>

      <h2>Key terms</h2>
      <DocTable
        head={["Term", "Meaning"]}
        rows={[
          [<strong>Locus</strong>, <>A receptor chain's genomic region: <code>IGH</code>, <code>IGK</code>, <code>IGL</code>, <code>TRB</code>, <code>TRA</code>, ... Heavy and beta chains have a D segment; light chains do not.</>],
          [<strong>Gene / allele</strong>, <>A germline segment (<code>IGHV1-2</code>) and its sequence variant (<code>IGHV1-2*02</code>). A read is called to the allele level when possible.</>],
          [<strong>Family</strong>, <>A group of related genes (<code>IGHV1</code>).</>],
          [<strong>Reference / germline catalog</strong>, "The set of alleles a model calls from. A model file embeds the reference it was trained on; another can be bound at predict time."],
          [<strong>Junction / CDR3</strong>, "The hypervariable region spanning the V-(D)-J join; the main determinant of antigen specificity."],
          [<strong>SHM</strong>, "Somatic hypermutation - point mutations B-cells accumulate; heavy SHM makes V alleles harder to call."],
          [<strong>Productive</strong>, "Whether the rearrangement yields a functional receptor (in-frame, no stop codon)."],
          [<strong>Orientation</strong>, "A read may be forward, reverse-complement, complement, or reversed. AlignAIR detects this and aligns in one canonical frame."],
        ]}
      />

      <h2>What AlignAIR gives you</h2>
      <p>For every read, in one pass:</p>
      <ul>
        <li>V / D / J allele calls, plus a candidate set (<code>*_call_set</code>) that widens when the read is genuinely ambiguous; <code>v_call</code> lists the V alleles a read ties with its call.</li>
        <li>Segment coordinates in the read and germline, and per-segment CIGAR / identity.</li>
        <li>Junction / CDR3, <code>productive</code>, mutation rate, and orientation.</li>
      </ul>

      <h2>The reference is an input</h2>
      <p>
        An AlignAIR model scores the alleles of the reference it is bound to; the alleles are an input, not weights. By
        default that is the reference embedded in the model file. At predict time you can narrow it to a donor's
        genotype, or bind another reference with novel alleles or new genes of the model's segments, without retraining.
        A reference missing one of the model's segments is refused, and a new locus or species means training a model.
        This is the <DocLink to="model-contract">model contract</DocLink> - worth reading once.
      </p>
    </>
  ),
};

export const getStartedPages: DocPage[] = [gettingStarted];
export const conceptsPages: DocPage[] = [concepts];
