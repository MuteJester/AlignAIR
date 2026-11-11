import { Dna } from "lucide-react";
import { Callout } from "../../components/ui";
import { VdjBuilderWidget } from "../../components/widgets";
import type { Lesson, Track } from "../types";

const whatIsVdj: Lesson = {
  id: "foundations/what-is-vdj",
  slug: "what-is-vdj",
  track: "foundations",
  title: "What AlignAIR does",
  summary: "V(D)J recombination, and what a single forward pass predicts.",
  minutes: 5,
  steps: [
    {
      kind: "explain",
      title: "Immune receptors are assembled from gene segments",
      body: () => (
        <>
          <p>
            B-cell and T-cell receptors are not encoded by single continuous genes. Instead, each receptor chain is stitched together from three types of germline gene segments: <strong>V</strong> (variable), <strong>D</strong> (diversity), and <strong>J</strong> (joining) segments.
          </p>
          <p>
            This process, called <em>V(D)J recombination</em>, is mediated by the <strong>RAG-1/RAG-2</strong> recombinase enzyme complex. During recombination, the DNA is cut at specific RSS signals and joined. In addition to segment recombination, diversity is amplified by <strong>junctional diversity</strong>: exonucleases delete nucleotides from the cut ends, and the enzyme <strong>Terminal Deoxynucleotidyl Transferase (TdT)</strong> inserts random, non-templated nucleotides (N-nucleotides) at the junctions.
          </p>
          <Callout kind="note" title="Locus segment differences">
            Not all receptor chains have a D segment:
            <ul className="list-disc pl-6 space-y-1 my-2">
              <li><strong>V-D-J recombination</strong>: occurs in immunoglobulin heavy chains (<code>IGH</code>), and T-cell receptor beta (<code>TRB</code>) and delta (<code>TRD</code>) chains.</li>
              <li><strong>V-J recombination</strong>: occurs in light chains (<code>IGK</code>, <code>IGL</code>) and T-cell receptor alpha (<code>TRA</code>) and gamma (<code>TRG</code>) chains, which skip D segments entirely.</li>
            </ul>
          </Callout>
          <p>
            To analyze a sequenced read, researchers must <strong>align</strong> it back to the germline catalog to discover: which V, D, and J alleles produced it, where the segment boundaries lie, and the exact sequence of the junction (CDR3).
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "One neural model, many outputs",
      body: () => (
        <>
          <p>
            Classical tools do this in stages (seed, extend, re-align). AlignAIR is a single end-to-end neural network. A shared dilated-convolution trunk gives every base of the read its features once, and specialized units read them:
          </p>
          <ul className="list-disc pl-6 space-y-1.5 my-4">
            <li><strong>Orientation</strong>: detects the read's orientation (forward / reverse-complement / complement / reversed) and turns it forward, in the same pass.</li>
            <li><strong>Parser</strong>: reads the read's path through V, D and J - the segment boundaries.</li>
            <li><strong>Anchor heads</strong>: find the conserved V Cys and J Trp/Phe codons that frame the junction.</li>
            <li><strong>Bank scans</strong>: score every allele of the bound germline reference against the read's bases.</li>
            <li><strong>Mutation reader</strong>: estimates each base's mutation and the read's mutation load.</li>
          </ul>
          <p>
            Deterministic steps then refine the calls and boundaries against the germline, derive the junction and
            productivity, and write a standard, schema-valid AIRR rearrangement TSV.
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Build a receptor from segments",
      body: () => (
        <>
          <p>
            Recombination joins selected segments with non-templated additions. Below, select different segments to watch the assembled sequence and its CDR3 junction change:
          </p>
          <VdjBuilderWidget />
          <p>
            The alleles display in standard AIRR nomenclature: for example, <code>IGHV3-23*01</code> denotes the locus (<code>IGH</code>), segment type (<code>V</code>), family (<code>3</code>), gene (<code>23</code>), and allele (<code>*01</code>).
          </p>
          <Callout kind="note" title="Coordinate indexing convention">
            AlignAIR follows the official AIRR format specifications. All sequence coordinates and segment starts/ends are reported using <strong>1-based closed indexing</strong> relative to the read sequence.
          </Callout>
        </>
      ),
    },
    {
      kind: "mcq",
      prompt: () => (
        <p>
          A read comes off the sequencer reverse-complemented. What does AlignAIR need a separate tool to do first?
        </p>
      ),
      options: [
        "Nothing — it detects orientation itself as one of the model’s outputs",
        "You must reverse-complement the read by hand before aligning",
        "It cannot align reverse-complement reads at all",
      ],
      answer: 0,
      explanation: () => (
        <p>
          Orientation is read by the same network, so reverse-complement and arbitrarily-oriented reads are handled in
          the same forward pass.
        </p>
      ),
    },
  ],
};

const fixedReference: Lesson = {
  id: "foundations/fixed-reference",
  slug: "fixed-reference",
  track: "foundations",
  title: "The reference contract",
  summary: "Where a model's callable alleles come from, and what a donor genotype or another reference changes.",
  minutes: 6,
  steps: [
    {
      kind: "explain",
      title: "The reference is an input",
      body: () => (
        <>
          <p>
            The network scores the alleles of the germline reference it is <strong>bound</strong> to; the alleles are an
            input, not weights. By default that is the reference the model was trained on, which is{" "}
            <strong>embedded inside the model file</strong> (and fingerprinted, so it cannot drift from the weights).
          </p>
          <Callout kind="note" title="One property explains everything">
            Understand that a model scores whichever reference it is bound to, and the rest of its behaviour follows:
            what it can call, how donor genotypes and novel alleles work, and when you must train.
          </Callout>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Donor genotype = a subset, at inference",
      body: () => (
        <>
          <p>
            You can constrain a run to a <strong>donor genotype</strong> — a subset of the model's reference — with no
            retraining. Calls are hard-restricted to that donor's alleles:
          </p>
          <pre tabIndex={0} aria-label="AIRR output example" style={{ margin: "20px 0", padding: "16px", background: "#16151f", borderRadius: "12px", fontFamily: "IBM Plex Mono, monospace", fontSize: "13px", lineHeight: "1.85", color: "#e6e5f0", overflowX: "auto" }}>
            <code>{`alignair predict --input reads.fasta --out out.tsv \\\n  --model alignair4-igh-human --genotype donor.yaml`}</code>
          </pre>
          {/* PENDING-V4-PUBLISH: alignair4-igh-human is not published yet. */}
          <p>
            A genotype only <em>narrows</em> the callable set. To add alleles the model never saw, bind a FASTA that holds
            them with <code>--reference</code> - still no retraining. A new locus or species is a different matter: train
            a model for it.
          </p>
        </>
      ),
    },
    {
      kind: "mcq",
      prompt: () => (
        <p>
          Your donor carries a novel <code>IGHV</code> allele that is not in the model’s embedded reference. What
          happens if you put it in a <code>--genotype</code> file?
        </p>
      ),
      options: [
        "It is silently ignored and the read is called as the nearest known allele",
        "AlignAIR refuses the genotype with a clear error: the allele's sequence belongs in --reference",
        "The model calls it correctly because genotypes add alleles",
      ],
      answer: 1,
      explanation: () => (
        <p>
          A genotype names alleles of the bound reference, so an unknown name is refused up front (
          <code>ALIGNAIR_GENOTYPE_UNKNOWN_ALLELE</code>, hint: give their sequences in <code>--reference</code>) rather
          than silently dropped. Bind a FASTA that holds the novel allele with <code>--reference</code> - no retraining -
          and the genotype can then name it.
        </p>
      ),
    },
    {
      kind: "mcq",
      prompt: () => (
        <p>Why can two labs safely share a <code>.alignair</code> model file and trust the columns match?</p>
      ),
      options: [
        "Because the file name encodes the version",
        "Because the germline reference is embedded and hash-verified on load, so the weights and allele order cannot drift apart",
        "Because AlignAIR re-downloads the reference from IMGT at runtime",
      ],
      answer: 1,
      explanation: () => (
        <p>
          The reference travels inside the file and is fingerprinted (allele order and FASTA SHA-256); loading
          verifies it, so the weights and the default allele order cannot drift apart. A reference you bind yourself
          with <code>--reference</code> is recorded too: the run's <code>.run.json</code> fingerprints the reference the calls
          came from, beside the embedded one.
        </p>
      ),
    },
  ],
};

export const foundationsTrack: Track = {
  slug: "foundations",
  title: "Foundations",
  description: "What V(D)J alignment is, and the one property that defines an AlignAIR model.",
  icon: Dna,
  accent: "from-sky-500 to-blue-600",
  lessons: [whatIsVdj, fixedReference],
};
