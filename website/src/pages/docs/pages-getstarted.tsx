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
      <CodeBlock code={`pip install "AlignAIR[cli]"            # core + CLI (+ parasail + AIRR validation)`} />
      <p>
        PyTorch is a dependency; the device backend depends on how torch is installed. AlignAIR auto-detects it at run
        time (<code>--device auto</code> picks CUDA, then Apple MPS, then CPU).
      </p>
      <DocTable
        head={["Platform", "torch install", "AlignAIR", "Backend"]}
        rows={[
          ["Linux / Windows, NVIDIA GPU", <>default <code>pip</code> wheel (CUDA)</>, <code>pip install "AlignAIR[cli]"</code>, "CUDA"],
          ["Linux / Windows, CPU only", <>install CPU torch first (see below)</>, <code>pip install "AlignAIR[cli]"</code>, "CPU"],
          ["macOS, Apple Silicon", <>default <code>pip</code> wheel</>, <code>pip install "AlignAIR[cli]"</code>, "MPS (auto)"],
          ["macOS, Intel", <>default <code>pip</code> wheel</>, <code>pip install "AlignAIR[cli]"</code>, "CPU"],
          ["Docker (CPU)", "included in the image", <code>docker pull ghcr.io/mutejester/alignair:latest</code>, "CPU"],
        ]}
      />
      <p>For a CPU-only environment, install CPU torch first:</p>
      <CodeBlock code={`pip install torch --index-url https://download.pytorch.org/whl/cpu`} />
      <p>
        On Windows use PowerShell and quote the extra. For Docker, pin a version tag for reproducibility (for example{" "}
        <code>ghcr.io/mutejester/alignair:3.0.0</code> once released) rather than only <code>latest</code>; the default image is
        CPU-only.
      </p>
      <p>Verify the install on any platform:</p>
      <CodeBlock code={`alignair doctor        # Python, PyTorch + CUDA/MPS, GenAIRR, parasail\nalignair demo          # offline end-to-end trial (no download)`} />

      <h3>Offline install and model caching</h3>
      <CodeBlock code={`alignair models get alignair-igh-human           # cache the model\nalignair predict --input reads.fasta --out out.tsv --model alignair-igh-human --offline`} />
      <p>
        <code>--offline</code> never touches the network (set <code>ALIGNAIR_NO_NETWORK=1</code> to also disable the
        passive update check). Models cache in the per-OS user cache directory; <code>alignair doctor</code> prints the
        exact <code>cache_dir</code>, and <code>ALIGNAIR_CACHE_DIR</code> overrides it.
      </p>

      <h2>2. See it work in one command</h2>
      <CodeBlock code={`alignair demo`} />
      <p>
        Trains a tiny demo model (not production quality), aligns simulated reads, validates the AIRR output, and runs
        the donor-genotype path - proving the full pipeline with no model download.
      </p>

      <h2>3. Get a model</h2>
      <p>
        <code>alignair predict</code> needs a model: a <code>.alignair</code> file, a catalog id, or a Hugging Face repo
        id. Use a pretrained model (no login, downloaded on first use):
      </p>
      <CodeBlock code={`alignair models list\nalignair predict --input reads.fasta --out out.tsv --model alignair-igh-human`} />
      <p>Or train your own for any reference or species:</p>
      <CodeBlock code={`alignair train --dataconfig HUMAN_IGH_OGRDB --out runs/my_igh --preset desktop\n# or your own germline FASTAs\nalignair train --v-fasta v.fa --d-fasta d.fa --j-fasta j.fa --chain-type BCR_HEAVY \\\n  --out runs/custom --preset desktop`} />
      <p>
        See <DocLink to="training">Training a custom model</DocLink> for the full guide.
      </p>

      <h2>4. Align reads</h2>
      <p>Input may be FASTA, FASTQ, CSV/TSV, or TXT (optionally <code>.gz</code>, or <code>-</code> for stdin):</p>
      <CodeBlock code={`alignair predict --input reads.fasta --out out.tsv --model runs/my_igh/bundle/model.alignair`} />
      <p>
        <strong>Tabular Data (CSV/TSV):</strong> If your input is a table, you must tell the reader which columns hold the sequences and read IDs:
      </p>
      <CodeBlock code={`alignair predict --input reads.csv --out out.tsv --model alignair-igh-human \\\n  --sequence-column sequence --id-column sequence_id`} />
      <p>
        The output is an AIRR rearrangement TSV with the V/D/J calls, coordinates, junction, productivity, and a per-gene
        candidate-set column (<code>*_call_set</code>). See <DocLink to="airr-fields">AIRR output fields</DocLink>.
      </p>

      <h2>5. Constrain to a donor genotype</h2>
      <p>
        Supply a genotype as YAML or JSON to restrict calls to a subset of the model's reference, with no retraining. The file maps gene families/genes to allowed alleles:
      </p>
      <CodeBlock
        lang="yaml"
        code={`# donor.yaml\nIGHV3-23:\n  - IGHV3-23*01\n  - IGHV3-23*04\nIGHV1-2:\n  - IGHV1-2*02`}
      />
      <p>Pass the file path to <code>--genotype</code>:</p>
      <CodeBlock code={`alignair predict --input reads.fasta --out out.tsv \\\n  --model runs/my_igh/bundle/model.alignair --genotype donor.yaml`} />
      <Callout kind="note">
        Alleles the model was not trained on are not callable. A genotype can only narrow the callable set; adding
        alleles requires training a new model. See the <DocLink to="model-contract">model contract</DocLink>.
      </Callout>

      <h2>Common predict options</h2>
      <DocTable
        head={["Flag", "Meaning"]}
        rows={[
          [<code>--genotype FILE</code>, "YAML/JSON genotype for this run (a subset of the model's reference)"],
          [<code>--genotype-method M</code>, <><code>mask</code> (default) / <code>softmax</code> / <code>renormalize</code> / <code>redistribute</code></>],
          [<code>--metadata FILE</code>, "per-read metadata (e.g. 10x annotations) preserved into output"],
          [<code>--keep-columns LIST</code>, "comma-separated metadata columns to carry through"],
          [<code>--chunk-size N</code>, "reads per streaming chunk (bounded memory; default 20000)"],
          [<code>--columns SPEC</code>, <>a preset (<code>full</code>/<code>core</code>/<code>minimal</code>/<code>airr</code>) or a field list</>],
          [<code>--rejects-out FILE</code>, "write dropped/invalid input records here"],
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
          [<strong>Family</strong>, <>A group of related genes (<code>IGHV1</code>). When a read cannot pin down the allele, the answer degrades toward the family.</>],
          [<strong>Reference / germline catalog</strong>, "The set of known alleles a model was trained on. In AlignAIR it is embedded in the model file."],
          [<strong>Junction / CDR3</strong>, "The hypervariable region spanning the V-(D)-J join; the main determinant of antigen specificity."],
          [<strong>SHM</strong>, "Somatic hypermutation - point mutations B-cells accumulate; heavy SHM makes V alleles harder to call."],
          [<strong>Productive</strong>, "Whether the rearrangement yields a functional receptor (in-frame, no stop codon)."],
          [<strong>Orientation</strong>, "A read may be forward, reverse-complement, complement, or reversed. AlignAIR detects this and aligns in one canonical frame."],
        ]}
      />

      <h2>What AlignAIR gives you</h2>
      <p>For every read, in one pass:</p>
      <ul>
        <li>V / D / J allele calls, plus a ranked candidate set (<code>*_call_set</code>) that widens when the read is genuinely ambiguous.</li>
        <li>Segment coordinates in the read and germline, and per-segment CIGAR / identity.</li>
        <li>Junction / CDR3, <code>productive</code>, mutation rate, and orientation.</li>
      </ul>

      <h2>Fixed-reference classifier</h2>
      <p>
        An AlignAIR model can call exactly the alleles in its embedded catalog. You can subset that catalog to a donor's
        genotype at inference, but adding new alleles or a new species means training a model. This is the{" "}
        <DocLink to="model-contract">model contract</DocLink> - worth reading once.
      </p>
    </>
  ),
};

export const getStartedPages: DocPage[] = [gettingStarted];
export const conceptsPages: DocPage[] = [concepts];
