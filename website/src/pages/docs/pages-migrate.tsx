import { CodeBlock, Callout } from "../../components/ui";
import { DocTable, DocLink, type DocPage } from "./doc-kit";

const migrating: DocPage = {
  slug: "migrating-from-igblast",
  title: "Migrating from IgBLAST",
  section: "Get started",
  lead: "AlignAIR writes the same AIRR rearrangement schema IgBLAST does, so most of a pipeline keeps working. Here is the field-by-field map, the semantic differences, and a real side-by-side.",
  body: () => (
    <>
      <p>
        IgBLAST with <code>-outfmt 19</code> and <code>alignair predict</code> both emit the AIRR-C rearrangement
        schema, so a downstream reader (Change-O, Immcantation, Scirpy, nf-core/airrflow) written for one reads the
        other. What changes is how you invoke it, how the reference travels, and a handful of AlignAIR extensions. The
        short invocation using the included sample reads is in{" "}
        <a href="https://github.com/MuteJester/AlignAIR/tree/main/examples/igblast-migration" target="_blank" rel="noreferrer">
          examples/igblast-migration
        </a>{" "}
        and its commands are checked against the real CLI in CI.
      </p>

      <h2>Invocation</h2>
      <p>
        IgBLAST needs BLAST databases built from germline FASTAs plus an organism flag. AlignAIR carries its germline
        reference inside the model file, so there is no database to build - you choose a model instead.
      </p>
      <CodeBlock
        title="IgBLAST (your existing command)"
        code={`igblastn -germline_db_V human_gl_V -germline_db_D human_gl_D -germline_db_J human_gl_J \\\n  -auxiliary_data human_gl.aux -organism human -ig_seqtype Ig \\\n  -outfmt 19 -query reads.fasta > igblast.tsv`}
      />
      {/* PENDING-V4-MODEL-ID: <v4-igh-model> stands for the AlignAIR 4 IGH model's id, not yet decided. */}
      <CodeBlock
        title="AlignAIR"
        code={`alignair predict --input reads.fasta --out alignair.tsv --model <v4-igh-model>@<version>`}
      />
      <p>
        Pin the model version (<code>@&lt;version&gt;</code>, or <code>--revision</code> for a Hugging Face repo)
        whenever a result must be reproducible. <code>alignair models list</code> shows what is installed and
        available. <code>{"<v4-igh-model>"}</code> is a placeholder: the AlignAIR 4 IGH model's id is not published yet,
        and any AlignAIR 4 <code>.alignair</code> path works in its place.
      </p>

      <h2>Reference selection</h2>
      <DocTable
        head={["IgBLAST", "AlignAIR"]}
        rows={[
          [<><code>-germline_db_V/D/J</code> point at BLAST databases you build</>, <>The germline reference is embedded in the model and fingerprinted on load; <code>--reference germline.fasta</code> binds another at predict time.</>],
          [<><code>-organism human</code></>, <>Choose the model for your species and locus (AlignAIR 4 models come one per locus).</>],
          ["Add alleles by rebuilding the database", <>Add them to a FASTA (start from <code>alignair reference export</code>) and bind it with <code>--reference</code>, no retraining.</>],
          ["Restrict to a subset by editing the database", <>Restrict to a donor's alleles at predict time: <code>--genotype donor.yaml</code>, or infer them with <code>--infer-genotype</code>, no retraining.</>],
        ]}
      />
      <Callout kind="warning" title="What a reference swap cannot do">
        A bound reference must hold the model's segments (a D-less reference is refused by an IGH model), and it should
        be of the model's locus and species: a single-locus model does not refuse another locus's reference that has
        its segments, but its records keep the model's locus, and how well it calls another locus or species has not
        been measured. A new locus or species is the one case that differs in kind from editing a BLAST database - plan
        an <DocLink to="training">alignair train</DocLink> run. The{" "}
        <DocLink to="model-contract">model contract</DocLink> lists what a swap accepts and refuses.
      </Callout>

      <h2>A real read through both tools</h2>
      {/* PENDING-V4-NUMBERS: re-run this side-by-side with the AlignAIR 4 IGH model (pending head-to-head). */}
      <Callout kind="note" title="Recorded with AlignAIR 3; AlignAIR 4 results pending">
        This side-by-side was recorded with AlignAIR 3 and its <code>alignair-igh-human@1.0.0</code> model. The AlignAIR
        4 comparison awaits a fresh head-to-head; the field mapping below holds for both.
      </Callout>
      <p>
        The same 348 nt human IGH read, aligned by IgBLAST (<code>-outfmt 19</code>) and by AlignAIR 3{" "}
        (<code>alignair predict --model alignair-igh-human@1.0.0</code>). Both rows are genuine tool output, not
        illustrations.
      </p>
      <DocTable
        head={["Field", "IgBLAST", "AlignAIR", "Note"]}
        rows={[
          [<code>v_call</code>, <code>IGHVF10-G37*08</code>, <code>IGHVF10-G37*08</code>, "agree"],
          [<code>j_call</code>, <code>IGHJ6*03</code>, <code>IGHJ6*03</code>, "agree"],
          [<code>d_call</code>, <code>IGHD2-8*02</code>, <code>IGHD1-26*01</code>, "genuine disagreement - D is short and heavily trimmed, the least certain call in both tools"],
          [<>V/D/J <code>_call_set</code></>, <em>not emitted</em>, <><code>IGHVF10-G37*08</code> / <code>IGHD1-26*01</code> / <code>IGHJ6*03</code></>, "AlignAIR extension: the candidate set, capped at 3. Single-member here; widens on ambiguous reads"],
          [<code>junction</code>, <code>TGTGCG...ACTGG</code>, <code>TGTGCG...ACTGG</code>, "agree (nucleotides identical)"],
          [<code>junction_aa</code>, <code>CAKGVILAVTG</code>, <code>CAKGVILAVT.</code>, "AlignAIR 3's J boundary was 2 nt shorter here, so the last codon is incomplete (.)"],
          [<><code>v_sequence_start</code> / <code>_end</code></>, <>1 / 295</>, <>1 / 296</>, "both 1-based; boundaries can differ by 1-2 nt"],
          [<code>v_cigar</code>, <code>295M53S1N</code>, <code>296M</code>, "different CIGAR conventions; both consume the query correctly"],
          [<>productivity / <code>rev_comp</code> / <code>locus</code></>, <>F / F / IGH</>, <>F / F / IGH</>, "agree"],
          [<code>airr_assembly_status</code>, <em>not emitted</em>, <code>complete</code>, "AlignAIR extension: honest per-record quality flag"],
        ]}
      />

      <h2>Output fields</h2>
      <p>
        The shared AIRR columns carry the same meaning, so a reader written for IgBLAST output reads AlignAIR output.
        AlignAIR <strong>adds</strong> columns IgBLAST does not emit:
      </p>
      <ul>
        <li><code>v/d/j_call_set</code> - the candidate set per gene (see below).</li>
        <li><code>airr_assembly_status</code> / <code>airr_assembly_reason</code> - a per-record quality flag; filter to <code>complete</code> before junction or productivity analysis.</li>
        <li><code>mutation_rate</code> - the model's estimate of the read's mutation rate (advisory); <code>productive_prediction</code> repeats <code>productive</code>.</li>
        <li><code>segmentation_low_quality</code>, <code>length_cropped</code>, <code>input_sequence</code> - quality and provenance flags.</li>
      </ul>
      <p>
        Some IgBLAST-specific columns (<code>v_frameshift</code>, <code>d_frame</code>, <code>complete_vdj</code>) are not
        part of AlignAIR's output. The full field contract is in <DocLink to="airr-fields">AIRR output fields</DocLink>.
      </p>

      <h2>Semantic differences worth knowing</h2>
      <ul>
        <li><strong>Ties, as IgBLAST lists them.</strong> When a read cannot tell V alleles apart, <code>v_call</code> lists the call and then the alleles that match the read's V bases as well, as IgBLAST lists equally scoring alleles, so TIgGER and Change-O genotype steps can choose among them. Separately, <code>*_call_set</code> holds each gene's candidate set from the model (alleles scoring at least half the best's likelihood, at most 3). A multi-member set means "do not read this at allele resolution", not "these are the only possibilities".</li>
        <li><strong>Coordinates.</strong> Same 1-based AIRR convention. Boundaries can differ from IgBLAST's by a few nucleotides - group clones on <code>junction_aa</code>, not single-nt positions.</li>
        <li><strong>Junction assembly.</strong> AlignAIR derives the junction from the conserved V Cys and J Trp/Phe codons it finds in the read: from the first base of the Cys codon to the last base of the Trp/Phe codon, as AIRR defines it. A read that does not contain both codons has no junction.</li>
        <li><strong>Confidence.</strong> AlignAIR does not emit a per-call e-value, and the reserved resolution columns (<code>*_set_confidence</code> and the rest) stay blank - do not filter on them.</li>
        <li><strong>Partial predictions.</strong> A row can be <code>airr_assembly_status = partial</code> (valid calls, but the junction or another product could not be assembled). These are emitted with their calls; IgBLAST has no equivalent flag. Filter to <code>complete</code> for junction-level work. See <DocLink to="known-failure-modes">Known failure modes</DocLink>.</li>
      </ul>

      <h2>CPU and GPU</h2>
      <p>
        IgBLAST is CPU-only and multithreaded. AlignAIR runs on either:
      </p>
      <ul>
        <li><strong>GPU</strong> is the fast path: <code>alignair predict</code> uses CUDA when it is available; Apple Silicon MPS takes <code>--device auto</code> or <code>mps</code>.</li>
        <li><strong>CPU</strong> runs scale with <code>--threads N</code> (N single-threaded worker processes).</li>
        <li><strong>CPU-only</strong> works with no change; install the CPU build of PyTorch first to avoid pulling CUDA (see <DocLink to="troubleshooting">Troubleshooting</DocLink>).</li>
      </ul>
      <p>
        AlignAIR 4 throughput figures are pending a re-timing on a quiet machine (see{" "}
        <DocLink to="performance">Speed &amp; throughput</DocLink>) - measure on your own hardware before capacity
        planning.
      </p>

      <h2>Runnable side-by-side</h2>
      <p>
        No ground truth needed. <code>alignair compare</code> reports agreement between two AIRR TSVs on the same reads,
        including set-rescue - how often the other tool's call falls inside AlignAIR's <code>*_call_set</code>, which
        separates shared ambiguity from a real conflict.
      </p>
      <CodeBlock
        code={`# same reads through AlignAIR (igblastn is your existing command, above)\nalignair predict --input reads.fasta --out alignair.tsv --model <v4-igh-model>@<version>\n\n# structural check of the output\nalignair validate-airr alignair.tsv\n\n# where do the two tools agree?\nalignair compare --a alignair.tsv --b igblast.tsv --a-name AlignAIR --b-name IgBLAST`}
      />

      <h2>Migration checklist</h2>
      <ul>
        <li>Pick the model for your locus; pin <code>@version</code> for reproducibility.</li>
        <li>Replace the <code>igblastn ... -outfmt 19</code> call with <code>alignair predict --input ... --out ... --model ...</code>.</li>
        <li>Point your downstream reader at the new TSV - the shared AIRR columns are unchanged.</li>
        <li>Add a filter on <code>airr_assembly_status == complete</code> before junction / productivity analysis.</li>
        <li>Decide how to use <code>*_call_set</code>: treat multi-member sets as below allele resolution.</li>
        <li>If you constrained IgBLAST's database to a donor, pass <code>--genotype donor.yaml</code> instead (or <code>--infer-genotype</code>).</li>
        <li>If you relied on novel or added alleles, bind them with <code>--reference germline.fasta</code>; for a new locus or species, plan an <DocLink to="training">alignair train</DocLink> run.</li>
        <li>Sanity-check a batch with <code>alignair compare</code> before switching the pipeline over.</li>
      </ul>
    </>
  ),
};

export const migratePages: DocPage[] = [migrating];
