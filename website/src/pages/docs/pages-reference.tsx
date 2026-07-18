import { CodeBlock, Callout } from "../../components/ui";
import { DocTable, DocLink, type DocPage } from "./doc-kit";

const modelContract: DocPage = {
  slug: "model-contract",
  title: "Model contract",
  section: "Reference",
  lead: "An AlignAIR model is a fixed-reference classifier. This one property explains what it can and cannot do.",
  body: () => (
    <>
      <h2>What a model is</h2>
      <p>
        The V/D/J classification heads have a fixed output size whose indices are tied, in order, to the germline allele
        catalog the model was trained on. That catalog travels inside the model file (embedded and hash-verified) and is
        the exact set of alleles the model can call.
      </p>

      <h2>Supported at inference</h2>
      <ul>
        <li><strong>Prediction over the trained catalog.</strong> Every call is one of the embedded reference's alleles.</li>
        <li>
          <strong>Donor-genotype subsetting.</strong> A YAML/JSON genotype that is a subset of the trained reference constrains calls to that donor's alleles (<code>--genotype</code>). Partial genes are fine. Methods (<code>--genotype-method</code>):
          <ul className="list-disc pl-6 space-y-1 my-2">
            <li><code>mask</code> (default): Zeroes out probabilities of forbidden alleles. The relative probabilities of allowed alleles are unchanged. This is the safest default as it does not distort confidence scores.</li>
            <li><code>softmax</code>: Re-runs the softmax function over only the allowed alleles. It concentrates probability mass on the best allele, sharpening confidence.</li>
            <li><code>renormalize</code>: Zeroes out forbidden alleles and rescales the remaining probabilities to sum to 1. Since raw model logits are sigmoidal/multi-label, this rescaling can skew relative confidence.</li>
            <li><code>redistribute</code>: Spread the masked probability mass back across the donor set.</li>
          </ul>
        </li>
        <li><strong>Multi-locus models.</strong> A model trained on several loci attributes each read to a locus and can only call that locus's alleles. Cross-locus calls are impossible.</li>
      </ul>

      <h2>Not supported - requires training a new model</h2>
      <ul>
        <li><strong>Novel alleles.</strong> An allele absent from the trained catalog cannot be called. Supplying one fails immediately with a clear error - it is never silently dropped or mis-indexed.</li>
        <li><strong>Adding alleles / a new species / a new locus.</strong> This changes the allele universe, so it requires training a new model with a new, versioned catalog.</li>
      </ul>

      <h2>Safety guarantees enforced on load</h2>
      <ul>
        <li>A caller-supplied reference whose ordered allele identity does not exactly match the embedded head is rejected - output columns can never be silently mislabelled.</li>
        <li>The embedded reference's allele-order and FASTA fingerprints are verified every time a model loads.</li>
        <li>Genotype constraints validate a non-empty allowed set per constrained gene and reject novel alleles.</li>
      </ul>
    </>
  ),
};

const knownFailureModes: DocPage = {
  slug: "known-failure-modes",
  title: "Known failure modes",
  section: "Reference",
  lead: "Where AlignAIR struggles and what to use instead - because trusting a result means knowing when not to.",
  body: () => (
    <>
      <h2>Where AlignAIR can be wrong</h2>
      <DocTable
        head={["Failure mode", "What happens", "What to do"]}
        rows={[
          ["Junction boundary jitter", "CDR3/junction coordinates can be off by ~1-2 nt (J side worse than V). The junction string is usually fine.", "Treat junction coordinates as approximate; group clones on junction_aa, not single-nt positions."],
          ["Missing anchors -> no junction", "If the reference lacks the conserved anchors for an allele, that read gets calls but an empty junction (honest absence).", "Add anchors to the reference, or accept junction-free rows. Built-in dataconfigs carry anchors."],
          ["Short-read / fragment ambiguity", "A short read carries little signal, so the true allele is genuinely indistinguishable from many others.", "A multi-member *_call_set flags this. It is the model's candidate set (p >= 0.5, capped at 3), not a proof of which alleles are indistinguishable, so do not report the read at allele resolution."],
          ["Full-length, heavily-mutated V", "At high SHM on full-length reads, V allele accuracy is comparable to IgBLAST rather than higher - the hardest regime.", "Cross-check with IgBLAST for SHM-heavy lineage work; *_call_set still narrows candidates."],
          ["Model / reference mismatch", "Running a model on the wrong locus yields plausible-but-meaningless calls.", "Use a model trained for your locus; multi-locus models cannot make cross-locus calls."],
          ["Out-of-scope / contaminant reads", "Non-target sequences still get plausible-looking calls.", "No contaminant classifier runs (is_contaminant is reserved/blank). Screen upstream, or filter on assembly status."],
          ["Tiny/demo models", "alignair demo and the quick preset produce models whose calls are not accurate.", "Use a pretrained model, or train the desktop/full preset."],
        ]}
      />

      <h2>Reserved / not-populated fields</h2>
      <p>
        Framework and CDR regions and per-segment alignments <em>are</em> emitted on complete records. Not populated by
        default: the read-derived <code>c_call</code> (supply the assembler's <code>c_gene</code> via{" "}
        <code>--metadata</code>), <code>is_contaminant</code>, and the calibration extension columns. See{" "}
        <DocLink to="airr-fields">AIRR output fields</DocLink>.
      </p>

      <h2>When to prefer IgBLAST / MiXCR / partis</h2>
      <ul>
        <li><strong>Raw throughput on CPU-only hardware</strong> - without a GPU, AlignAIR runs somewhat below IgBLAST's multi-threaded CPU baseline, so for very large bulk repertoires on CPU-only machines IgBLAST may finish sooner. With a GPU the position reverses substantially. <code>--columns core</code> recovers roughly a tenth of the time by writing a much smaller row; <code>minimal</code> additionally skips the AIRR assembly, at the price of no coordinates and no junction. See <DocLink to="performance">Speed and throughput</DocLink> for the measured figures.</li>
        <li><strong>A long-established, citation-stable standard</strong> - if a pipeline or reviewer expects IgBLAST/IMGT output specifically.</li>
        <li><strong>End-to-end clonotype/repertoire assembly</strong> - MiXCR and Immcantation cover clustering, lineage, and stats AlignAIR does not; AlignAIR's job is the per-read alignment that feeds them.</li>
      </ul>
      <p>
        Switching a pipeline over? <DocLink to="migrating-from-igblast">Migrating from IgBLAST</DocLink> maps the
        commands and output fields and shows a real side-by-side.
      </p>

      <h2>Where AlignAIR fits best</h2>
      <ul>
        <li>A donor/study-specific genotype supplied at predict time with <code>--genotype</code>, no retraining.</li>
        <li>Surfaced ambiguity: a ranked candidate set alongside the top call, rather than a single, possibly wrong, allele.</li>
        <li>Non-human / custom species via <code>alignair train</code> on your own FASTA reference.</li>
        <li>Broad input coverage (fragments, reverse-complement, indels) in one model.</li>
      </ul>
    </>
  ),
};

const troubleshooting: DocPage = {
  slug: "troubleshooting",
  title: "Troubleshooting",
  section: "Reference",
  lead: "Run alignair doctor first - it reports Python, PyTorch + CUDA/MPS, GenAIRR, and parasail.",
  body: () => (
    <>
      <h2>Install</h2>
      <ul>
        <li><strong>torch pulls CUDA / is huge.</strong> For CPU-only, install CPU torch first, then AlignAIR:</li>
      </ul>
      <CodeBlock code={`pip install torch --index-url https://download.pytorch.org/whl/cpu\npip install "AlignAIR[cli]"`} />
      <ul>
        <li><strong>No module named GenAIRR.</strong> Install the CLI extra; GenAIRR is a core dependency on PyPI.</li>
        <li><strong>CUDA not used.</strong> <code>alignair doctor</code> shows <code>CUDA available: False</code> means a CPU torch build; reinstall a CUDA wheel matching your driver.</li>
        <li><strong>Apple Silicon (MPS) acceleration on macOS:</strong> AlignAIR automatically selects Apple Silicon GPU acceleration when available. Run <code>alignair doctor</code> and verify <code>mps_available: True</code>. If it is false or fails to resolve, verify your PyTorch installation.</li>
      </ul>

      <h2>Prediction</h2>
      <ul>
        <li><strong>model not found.</strong> <code>--model</code> takes a pretrained id, a local <code>.alignair</code>/<code>.pt</code> file, or an <code>org/name</code> HF repo id - not a bundle directory; point at the <code>.alignair</code> file inside it.</li>
        <li><strong>no sequence column.</strong> For CSV/TSV, pass <code>--sequence-column</code> (and <code>--id-column</code>).</li>
        <li><strong>Job fails on assembly failure or partial rate gates:</strong> For sequencing runs with high rates of non-target/contaminant sequences, prediction may abort because failures exceed the threshold. Run with <code>--permissive</code> to bypass, or adjust limits with <code>--max-assembly-failures &lt;fraction&gt;</code> and <code>--max-partial-assemblies &lt;fraction&gt;</code>.</li>
      </ul>
      <Callout kind="note" title="Reverse-complement reads">
        Handled automatically. For a reverse-complement read the emitted <code>sequence</code> is the original query and{" "}
        <code>rev_comp=T</code>, so coordinates apply to <code>RC(sequence)</code> (the AIRR / IgBLAST convention). See{" "}
        <DocLink to="airr-fields">the orientation table</DocLink>.
      </Callout>

      <h2>Docker</h2>
      <p>
        Run without installing anything. Mount an input and output directory, and a named volume for the
        model cache so a downloaded <code>--model &lt;id&gt;</code> is not re-fetched on every run:
      </p>
      <CodeBlock code={`docker run --rm \\\n  -v "$PWD:/data" -v alignair-cache:/home/appuser/.cache/alignair \\\n  ghcr.io/mutejester/alignair:latest \\\n  predict --input /data/reads.fasta --out /data/out.tsv --model alignair-igh-human`} />
      <ul>
        <li><strong>Permission denied writing output.</strong> The image runs as a non-root user (uid 1000). Mount a writable output dir; add <code>--user $(id -u):$(id -g)</code> if your host uid differs.</li>
        <li><strong>Model re-downloads every run.</strong> Persist the cache with a volume mounted at <code>/home/appuser/.cache/alignair</code> (as above), or mount a local <code>.alignair</code> file and pass it as <code>--model</code>.</li>
        <li><strong>Reproducibility.</strong> Pin a version tag (<code>ghcr.io/mutejester/alignair:3.0.0</code>) rather than <code>latest</code>. The default image is CPU-only; <code>alignair doctor</code> inside it reports <code>derive_backend: cython</code> (the compiled fast kernel) and <code>parasail</code>.</li>
      </ul>

      <h2>Training</h2>
      <ul>
        <li><strong>V allele has no anchor.</strong> Your custom FASTA has alleles GenAIRR cannot anchor; those reads get calls but no junction. Prefer a curated reference.</li>
        <li><strong>Junction is empty.</strong> Junction needs conserved anchors. Built-in dataconfigs and trained references carry them; plain YAML/JSON genotypes do not.</li>
      </ul>

      <h2>AIRR output</h2>
      <ul>
        <li><strong>Validate any TSV:</strong> <code>alignair validate-airr out.tsv</code> is a fast, dependency-free structural check: required columns present, coordinates in bounds, CIGARs consuming no more query than the emitted sequence, and productivity invariants. It is <em>not</em> the official AIRR validator - for formal schema validation also run <code>airr.validate_rearrangement(path)</code> from the <code>airr</code> library (CI covers this for AlignAIR's own output).</li>
        <li><strong>sequence_alignment / germline_alignment / *_cigar / *_identity</strong> are produced by AlignAIR's own IMGT-gap reconstruction on complete records; they do not require an external aligner. Under <code>--columns minimal</code> the assembly is skipped entirely, so those fields are not emitted at all; <code>core</code> still assembles (it requests the junction) and keeps the exact CIGARs.</li>
      </ul>
    </>
  ),
};

export const referencePages: DocPage[] = [modelContract, knownFailureModes, troubleshooting];
