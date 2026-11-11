import { CodeBlock, Callout } from "../../components/ui";
import { DocTable, DocLink, H2, type DocPage } from "./doc-kit";

const modelContract: DocPage = {
  slug: "model-contract",
  title: "Model contract",
  section: "Reference",
  lead: "An AlignAIR 4 model scores the alleles of the reference it is bound to. This one property explains what it can and cannot do.",
  body: () => (
    <>
      <h2>What a model is</h2>
      <p>
        The network's weights read a read's bases; the germline reference is an input. When a reference is bound, the
        model compiles its alleles into a bank, per segment: each allele's bases on fixed columns (V on its IMGT-gapped
        columns, J lined up on its conserved Trp/Phe codon, D on its own positions). The model scores every allele of
        that bank, and its allele outputs are the bound reference's alleles, in its order. The model file embeds the
        reference it was trained on, fingerprinted (allele order and FASTA SHA-256); that is the bank it binds by
        default.
      </p>

      <h2>Supported at inference, without retraining</h2>
      <ul>
        <li><strong>The embedded reference.</strong> Every call is one of its alleles.</li>
        <li>
          <strong>A donor genotype.</strong> A YAML/JSON genotype (<code>--genotype</code>; Python{" "}
          <code>predict(..., genotype=...)</code>) keyed by segment, <code>V</code>, <code>D</code>, <code>J</code>, each
          listing allele names of the bound reference. The genotype becomes the bank: the calls come only from its
          alleles. A segment it leaves out keeps all its alleles. A read whose V gene is not in the genotype gets no V
          call (row <code>partial</code>, reason <code>no_v</code>).
        </li>
        <li>
          <strong>Another reference.</strong> A FASTA of the model's segments (<code>--reference</code>; Python{" "}
          <code>Aligner.bound_to(ReferenceSet.from_fasta(path))</code> or{" "}
          <code>Aligner.from_pretrained(..., reference=...)</code>): novel alleles of genes the model knows, new genes,
          a curated set. A V allele that matches the read at strictly more bases than the call becomes the call, so a
          novel allele the read supports is called under its own name.
        </li>
        <li>
          <strong>Inferring the genotype.</strong> <code>--infer-genotype</code> infers the donor's V genotype from the
          file's own calls and re-calls the reads it settles (see the <DocLink to="cli">command-line reference</DocLink>).
        </li>
        <li>
          <strong>Multi-locus models.</strong> With a bank of several loci, each read keeps to one locus: that of its
          most probable same-locus V-J pair. Its V, D and J come from that locus, and a read of a locus without D is
          called D-absent.
        </li>
      </ul>

      <h2>How a bound allele gets its gaps and anchor</h2>
      <p>
        Each V and J allele needs an anchor (the first base of the V Cys or J Trp/Phe codon), and each V its IMGT-gapped
        form. A reference bound at predict time is annotated against the model's own (<code>alignair.reference.annotate</code>),
        filling only what the reference does not supply, in this order:
      </p>
      <ol>
        <li>an anchor or gapped form the reference supplies (a FASTA's <code>.</code> or <code>-</code> gaps are kept as the V's gapped form);</li>
        <li>the model's allele with the same sequence: its anchor and gaps;</li>
        <li>for a V, the model's allele of the same length that differs at the fewest bases, when that is at most 10% of them: its gaps and anchor;</li>
        <li>GenAIRR's resolver: a V from its IMGT gaps, a J from a unique F/W-G-X-G motif.</li>
      </ol>
      <p>
        <code>alignair predict --reference</code> names the alleles left without an anchor on stderr, e.g.{" "}
        <code>warning: --reference germline.fasta: V 198/199, J 7/7 anchored; no anchor: IGHV99-1*01 (unresolved)</code>.
        A read called to such an allele keeps the parser's boundary there, without the germline's refinement, and an
        ungapped V sits on its own positions rather than IMGT columns. Give new V alleles IMGT-gapped.
      </p>

      <h2>Refused, with the error</h2>
      <DocTable
        head={["Case", "Error"]}
        rows={[
          [<>A reference without one of the model's segments, e.g. an IGK, IGL, TRA or TRG reference (no D) on an IGH model</>,
           <><code>ALIGNAIR_CONFIG_INVALID</code>: the bound reference has no D alleles for the chip's genes</>],
          [<>On a model of several loci, an allele that belongs to none of them (by its sequence, else its name)</>,
           <><code>ALIGNAIR_INPUT_INVALID</code>: the reference's allele IGHV1-2*02 belongs to none of the model's loci (IGK, IGL). Hint: name it after its locus, or bind it to a model of that locus.</>],
          [<>A genotype naming an allele the bound reference lacks (CLI)</>,
           <><code>ALIGNAIR_GENOTYPE_UNKNOWN_ALLELE</code>: unknown alleles in genotype for v: ['IGHV1-2*99'] (not in the reference). Hint: give their sequences in --reference.</>],
          [<>The same in Python (<code>predict(genotype=...)</code>)</>,
           <><code>ALIGNAIR_INPUT_INVALID</code>: the genotype names alleles the reference does not hold: IGHV1-2*99</>],
          [<>A genotype that leaves a segment with no allele</>,
           <><code>ALIGNAIR_INPUT_INVALID</code>: the genotype leaves gene V with no allele</>],
          [<>An AlignAIR 3 model (a <code>.pt</code>, or a <code>conv-v1</code> <code>.alignair</code>)</>,
           <><code>ALIGNAIR_MODEL_INVALID</code>: this is an AlignAIR 3 model (architecture 'conv-v1'); AlignAIR 4 reads chip models only - use alignair&lt;4 to run it</>],
        ]}
      />
      <Callout kind="warning" title="Genotype keys are segments, not genes">
        A genotype keyed by gene names (<code>IGHV3-23: [...]</code>) is refused as naming alleles the reference does not
        hold, although it does hold them. Key it by <code>V</code>, <code>D</code> and <code>J</code>.
      </Callout>

      <h2>Not checked</h2>
      <ul>
        <li>
          <strong>Species.</strong> A reference of another species with the model's segments binds. How well a model
          calls another species' alleles has not been measured; train a model for that species.
        </li>
        <li>
          <strong>Locus, on a single-locus model.</strong> Every bound allele is placed in the model's one locus. A TRB
          reference (it has D) binds to an IGH model, and its records still say <code>locus</code> IGH. Bind only
          references of the model's locus; for a new locus, train a model.
        </li>
      </ul>

      <h2>Safety guarantees enforced on load</h2>
      <ul>
        <li>The embedded reference's allele-order and FASTA fingerprints are verified every time a model loads; a mismatch is a <code>ModelIntegrityError</code>.</li>
        <li>Loading reads no pickle: the reference comes from the file's JSON section, the weights from safetensors.</li>
        <li>Genotype constraints validate a non-empty allowed set per constrained segment and refuse names the bound reference lacks.</li>
      </ul>
      <p>
        The <DocLink to="reference-identity">reference identity API</DocLink> covers the fingerprints and the binding
        functions from Python.
      </p>
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
      {/* PENDING-V4-NUMBERS: junction coordinate error size and the high-SHM V comparison with IgBLAST. */}
      <Callout kind="note" title="Measurements pending for AlignAIR 4">
        The sizes of these errors, and how AlignAIR 4 compares with IgBLAST, await a fresh head-to-head; the table
        names the failure modes without numbers until then.
      </Callout>
      <DocTable
        head={["Failure mode", "What happens", "What to do"]}
        rows={[
          ["Junction boundary jitter", "CDR3/junction coordinates can be off by a few nucleotides. The junction string is usually fine.", "Treat junction coordinates as approximate; group clones on junction_aa, not single-nt positions."],
          ["No junction", "The junction exists only when both conserved codons (V Cys, J Trp/Phe) lie wholly inside the read; a read clipped before either gets calls but an empty junction (honest absence).", "Use reads that span the junction, or accept junction-free rows."],
          ["Bound alleles without an anchor", "An allele bound with --reference that no rule could anchor keeps the parser's boundary for reads called to it; an ungapped V lies off IMGT columns.", "Give new V alleles IMGT-gapped; read the --reference warning."],
          ["Short-read / fragment ambiguity", "A short read carries little signal, so the true allele is genuinely indistinguishable from many others.", "v_call then lists the V alleles the read matches as well as its call, and a multi-member *_call_set flags it. The call set is the model's candidate set (likelihood at least 0.5 of the best's, at most 3), not a proof of which alleles are indistinguishable, so do not report the read at allele resolution."],
          ["Full-length, heavily-mutated V", "High SHM on full-length reads is the hardest regime for V allele calls.", "Cross-check with IgBLAST for SHM-heavy lineage work; *_call_set still narrows candidates."],
          ["Model / reference mismatch", "Running a model on the wrong locus yields plausible-but-meaningless calls, and a single-locus model does not refuse another locus's reference that has its segments.", "Use a model trained for your locus; a multi-locus model keeps each read to one of its loci."],
          ["Out-of-scope / contaminant reads", "Non-target sequences still get plausible-looking calls.", "No contaminant classifier runs (is_contaminant is reserved/blank). Screen upstream, or filter on assembly status."],
          ["Tiny/demo models", "alignair demo and the quick preset produce models whose calls are not accurate.", "Use a pretrained model, or train the desktop/full preset."],
        ]}
      />

      <h2>Reserved / not-populated fields</h2>
      <p>
        Framework and CDR regions and per-segment alignments <em>are</em> emitted on complete records. Not populated by
        default: the read-derived <code>c_call</code> (supply the assembler's <code>c_gene</code> via{" "}
        <code>--metadata</code>), <code>is_contaminant</code>, and the resolution extension columns (<code>*_resolved_call</code>,{" "}
        <code>*_call_level</code>, <code>*_set_confidence</code>). See{" "}
        <DocLink to="airr-fields">AIRR output fields</DocLink>.
      </p>

      <h2>When to prefer IgBLAST / MiXCR / partis</h2>
      <ul>
        <li><strong>Raw throughput on CPU-only hardware</strong> - for very large bulk repertoires on CPU-only machines, compare on your hardware: the AlignAIR 4 timings against IgBLAST are pending a re-timing on a quiet machine. <code>--threads N</code> runs N worker processes on the CPU. <code>--columns core</code> writes a much smaller row; <code>minimal</code> additionally skips the AIRR assembly, at the price of no coordinates and no junction. See <DocLink to="performance">Speed and throughput</DocLink>.</li>
        <li><strong>A long-established, citation-stable standard</strong> - if a pipeline or reviewer expects IgBLAST/IMGT output specifically.</li>
        <li><strong>End-to-end clonotype/repertoire assembly</strong> - MiXCR and Immcantation cover clustering, lineage, and stats AlignAIR does not; AlignAIR's job is the per-read alignment that feeds them.</li>
      </ul>
      <p>
        Switching a pipeline over? <DocLink to="migrating-from-igblast">Migrating from IgBLAST</DocLink> maps the
        commands and output fields and shows a real side-by-side.
      </p>

      <h2>Where AlignAIR fits best</h2>
      <ul>
        <li>A donor/study-specific genotype supplied at predict time with <code>--genotype</code>, or inferred from the reads with <code>--infer-genotype</code>, no retraining.</li>
        <li>Novel alleles bound at predict time with <code>--reference</code>, no retraining.</li>
        <li>Surfaced ambiguity: tied alleles listed after the call, and a candidate set alongside it, rather than a single, possibly wrong, allele.</li>
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
  lead: "Run alignair doctor first - it reports Python, PyTorch + CUDA/MPS, GenAIRR, the default device, and the cache and config directories.",
  body: () => (
    <>
      <h2>Install</h2>
      <ul>
        <li><strong>torch pulls CUDA / is huge.</strong> For CPU-only, install CPU torch first, then AlignAIR:</li>
      </ul>
      <CodeBlock code={`pip install torch --index-url https://download.pytorch.org/whl/cpu\npip install "AlignAIR[cli]"`} />
      <ul>
        <li><strong>No module named GenAIRR.</strong> Install the CLI extra; GenAIRR is a core dependency on PyPI.</li>
        <li><strong>CUDA not used.</strong> <code>alignair doctor</code> showing <code>cuda_available False</code> means a CPU torch build; reinstall a CUDA wheel matching your driver.</li>
        <li><strong>Apple Silicon (MPS) acceleration on macOS:</strong> the Python API's default <code>device="auto"</code> selects MPS when available; on the command line pass <code>--device auto</code> (or <code>mps</code>), since <code>alignair predict</code> without <code>--device</code> picks CUDA or the CPU. Run <code>alignair doctor</code> and verify <code>mps_available True</code>. If it is false, verify your PyTorch installation.</li>
      </ul>

      <h2>Prediction</h2>
      <ul>
        <li><strong>model not found.</strong> <code>--model</code> takes a catalog id, a local <code>.alignair</code> file, or a Hugging Face repo (<code>hf://org/repo</code> or <code>org/repo</code>) - not a bundle directory; point at the <code>.alignair</code> file inside it.</li>
        <li><strong>"this is an AlignAIR 3 model".</strong> AlignAIR 4 runs chip models only; a <code>.pt</code> or a <code>conv-v1</code> <code>.alignair</code> (including the hub catalog's current <code>alignair-igh-human@1.0.0</code>) runs with <code>pip install 'alignair&lt;4'</code>.</li>
        <li><strong>no sequence column.</strong> For CSV/TSV whose sequence column is not named <code>sequence</code>, <code>seq</code>, <code>nucleotide</code> or <code>read</code>, pass <code>--sequence-column</code> (and <code>--id-column</code>).</li>
        <li><strong>"the bound reference has no D alleles for the chip's genes".</strong> The <code>--reference</code> FASTA lacks a segment the model calls (a light-chain or TRA/TRG reference on an IGH model, or D omitted). Give V, D and J, or use a model of that locus.</li>
        <li><strong>"the genotype names alleles the reference does not hold".</strong> Check the genotype's keys are <code>V</code>, <code>D</code>, <code>J</code> (not gene names) and its names are the bound reference's (<code>alignair reference export model.alignair --fasta ref.fasta</code> lists them).</li>
        <li><strong><code>--genotype</code> with a FASTA fails with an internal error.</strong> <code>--genotype</code> reads YAML or JSON only; pass a FASTA of alleles with <code>--reference</code>.</li>
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
{/* PENDING-V4-PUBLISH: alignair4-igh-human is not published yet. */}
      <CodeBlock code={`docker run --rm \\\n  -v "$PWD:/data" -v alignair-cache:/home/appuser/.cache/alignair \\\n  alignair/alignair:latest \\\n  predict --input /data/reads.fasta --out /data/out.tsv --model alignair4-igh-human`} />
      <ul>
        <li><strong>Permission denied writing output.</strong> The image runs as a non-root user (uid 1000). Mount a writable output dir; add <code>--user $(id -u):$(id -g)</code> if your host uid differs.</li>
        <li><strong>Model re-downloads every run.</strong> Persist the cache with a volume mounted at <code>/home/appuser/.cache/alignair</code> (as above), or mount a local <code>.alignair</code> file and pass it as <code>--model</code>.</li>
        <li><strong>Reproducibility.</strong> Pin a version tag (<code>alignair/alignair:3.0.0</code>) rather than <code>latest</code>; the same tags are mirrored at <code>ghcr.io/mutejester/alignair</code>. The default image is CPU-only; <code>alignair doctor</code> inside it reports <code>derive_backend</code> (<code>cython</code>: the compiled germline-CIGAR kernel).</li>
      </ul>

      <h2>Training</h2>
      <ul>
        <li><strong>Reference build warnings.</strong> <code>alignair train --plan</code> prints GenAIRR's warnings about a custom FASTA (e.g. alleles it cannot annotate). Read them before training; prefer a curated, IMGT-gapped reference.</li>
        <li><strong>Junction is empty.</strong> A junction needs both conserved codons inside the read; a read clipped before the V Cys or after the J Trp/Phe has none.</li>
      </ul>

      <h2>AIRR output</h2>
      <ul>
        <li><strong>Validate any TSV:</strong> <code>alignair validate-airr out.tsv</code> is a fast, dependency-free structural check: required columns present, coordinates in bounds, CIGARs consuming no more query than the emitted sequence, and productivity invariants. It is <em>not</em> the official AIRR validator - for formal schema validation also run <code>airr.validate_rearrangement(path)</code> from the <code>airr</code> library (CI covers this for AlignAIR's own output).</li>
        <li><strong>sequence_alignment / germline_alignment / *_cigar / *_identity</strong> are produced by AlignAIR's own IMGT-gap reconstruction on complete records; they do not require an external aligner. Under <code>--columns minimal</code> the assembly is skipped entirely, so those fields are not emitted at all; <code>core</code> still assembles (it requests the junction) and keeps the exact CIGARs.</li>
      </ul>
    </>
  ),
};

const referenceIdentity: DocPage = {
  slug: "reference-identity",
  title: "Reference identity API",
  section: "Reference",
  lead: "How a model file pins its reference, and how another reference is annotated and bound from Python.",
  body: () => (
    <>
      <H2 id="fingerprints">The embedded reference's fingerprints</H2>
      <CodeBlock lang="python" code={`from alignair.model_file import load_model, read_metadata, serialize

loaded = load_model("model.alignair")              # fingerprints verified here
card = read_metadata("model.alignair")["reference"]
card["allele_order_sha256"], card["reference_fasta_sha256"]
serialize.allele_order_sha256(loaded.reference)  # the same digest, from the decoded reference`} />
      <p>
        A model file records two SHA-256 digests of its embedded reference: the allele order (the output columns of
        each segment, in order) and the reference FASTA. Loading rebuilds the reference from the file's JSON section
        (no pickle) and checks both; a mismatch raises <code>ModelIntegrityError</code>{" "}
        (<code>ALIGNAIR_MODEL_INTEGRITY</code>): <em>allele_order_sha256 mismatch: the embedded reference order does
        not match the model card</em>, or <em>reference_fasta_sha256 mismatch</em>. The reference manifest written
        beside a bundle (<code>reference_manifest.json</code>) carries the same digests, per-segment allele names and
        anchor counts, so the reference can be identified without opening the model file.{" "}
        <code>alignair reference export model.alignair --fasta ref.fasta</code> writes the embedded FASTA.
      </p>
      <H2 id="binding">Annotate and bind another reference</H2>
      <CodeBlock lang="python" code={`from alignair import Aligner
from alignair.reference import ReferenceSet, annotate, anchor_report
from alignair.genotype import genotype_bank

aligner = Aligner.from_pretrained("model.alignair", device="cpu")
candidate = ReferenceSet.from_fasta("germline.fasta")         # V, D, J; the segment from each name

bound = annotate(candidate, known=aligner.reference)          # what bound_to and --reference do
report = anchor_report(bound)
report.complete(), report.summary()                           # "V 199/199, J 7/7 anchored"
donor = aligner.bound_to(candidate)                           # annotates, then binds a copy

subset = genotype_bank(aligner.reference, {"V": ["IGHVF1-G3*02"]})   # a genotype as a bank`} />
      <p>
        <code>annotate(reference, known)</code> returns a copy of <code>reference</code> with every missing V and J
        anchor and V gapped form filled, in the order the <DocLink to="model-contract">model contract</DocLink> lists,
        and records how each anchor was found (<code>reference.gene("V").info[name].anchor_evidence</code>: method{" "}
        <code>supplied</code>, <code>known_allele</code>, <code>nearest_known_allele</code>, or the resolver's, with a
        status). Given a <code>known</code> reference with a locus schema, the result is laid on the model's loci, in
        its order; an allele no locus of the model can hold is refused with <code>InputError</code>.{" "}
        <code>anchor_report</code> counts the anchored alleles and names those without, by status, and the V alleles
        anchored on a codon that is not Cys.
      </p>
      <p>
        Other constructors: <code>ReferenceSet.from_dataconfigs(*dataconfigs)</code> (GenAIRR's built-in references,
        gapped and anchored), <code>ReferenceSet.from_genotype({"{"}"V": {"{"}name: sequence{"}"}, ...{"}"})</code>, and{" "}
        <code>reference.subset({"{"}"V": names{"}"})</code>. <code>genotype_bank(reference, genotype)</code> is the subset a
        genotype binds, in the reference's order; it refuses names the reference lacks and segments left empty.
      </p>
      <H2 id="loading-conversion">Loading and conversion</H2>
      <p>
        <code>load_model(path, reference=...)</code> and <code>Aligner.from_pretrained(..., reference=...)</code>{" "}
        annotate the given reference against the embedded one and bind it; without one, the embedded reference is
        bound. <code>alignair convert</code> rewrites a lab chip model file as <code>chip-v1</code>, carrying the
        weights and the embedded reference over byte for byte, and reads the written file back (reference integrity
        and bindings) before it replaces the destination.
      </p>
    </>
  ),
};

export const referencePages: DocPage[] = [modelContract, referenceIdentity, knownFailureModes, troubleshooting];
