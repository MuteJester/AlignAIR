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
        runnable version of this page lives in{" "}
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
      <CodeBlock
        title="AlignAIR"
        code={`alignair predict --input reads.fasta --out alignair.tsv --model alignair-igh-human@1.0.0`}
      />
      <p>
        Pin the model version (<code>@1.0.0</code>) whenever a result must be reproducible. <code>alignair models list</code>{" "}
        shows what is installed and available.
      </p>

      <h2>Reference selection</h2>
      <DocTable
        head={["IgBLAST", "AlignAIR"]}
        rows={[
          [<><code>-germline_db_V/D/J</code> point at BLAST databases you build</>, "The germline reference is embedded in the model and fingerprinted on load."],
          [<><code>-organism human</code></>, <>Choose the model: <code>alignair-igh-human</code>, <code>alignair-igkl-human</code>, <code>alignair-tcrb-human</code>.</>],
          ["Add alleles by rebuilding the database", <>Adding alleles changes the model's fixed output space - train a new model (<DocLink to="training">alignair train</DocLink>).</>],
          ["Restrict to a subset by editing the database", <>Restrict to a donor's alleles at predict time: <code>--genotype donor.yaml</code>, no retraining.</>],
        ]}
      />
      <Callout kind="warning" title="A model's allele set is fixed">
        An AlignAIR model can only call the alleles it was trained on. A genotype <strong>subsets</strong> that
        reference; it cannot add to it. This is the one behaviour that differs in kind from editing a BLAST database - if
        you relied on novel or added alleles in IgBLAST, plan an <DocLink to="training">alignair train</DocLink> run.
      </Callout>

      <h2>A real read through both tools</h2>
      <p>
        The same 348 nt human IGH read, aligned by IgBLAST (<code>-outfmt 19</code>) and by{" "}
        <code>alignair predict --model alignair-igh-human@1.0.0</code>. Both rows are genuine tool output, not
        illustrations.
      </p>
      <DocTable
        head={["Field", "IgBLAST", "AlignAIR", "Note"]}
        rows={[
          [<code>v_call</code>, <code>IGHVF10-G37*08</code>, <code>IGHVF10-G37*08</code>, "agree"],
          [<code>j_call</code>, <code>IGHJ6*03</code>, <code>IGHJ6*03</code>, "agree"],
          [<code>d_call</code>, <code>IGHD2-8*02</code>, <code>IGHD1-26*01</code>, "genuine disagreement - D is short and heavily trimmed, the least certain call in both tools"],
          [<>V/D/J <code>_call_set</code></>, <em>not emitted</em>, <><code>IGHVF10-G37*08</code> / <code>IGHD1-26*01</code> / <code>IGHJ6*03</code></>, "AlignAIR extension: the candidate set (p >= 0.5, capped at 3). Single-member here; widens on ambiguous reads"],
          [<code>junction</code>, <code>TGTGCG...ACTGG</code>, <code>TGTGCG...ACTGG</code>, "agree (nucleotides identical)"],
          [<code>junction_aa</code>, <code>CAKGVILAVTG</code>, <code>CAKGVILAVT.</code>, "AlignAIR's J boundary is 2 nt shorter here, so the last codon is incomplete (.) - see junction assembly below"],
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
        <li><code>mutation_rate</code>, <code>productive_prediction</code> - neural estimates (advisory).</li>
        <li><code>segmentation_low_quality</code>, <code>length_cropped</code>, <code>input_sequence</code> - quality and provenance flags.</li>
      </ul>
      <p>
        Some IgBLAST-specific columns (<code>v_frameshift</code>, <code>d_frame</code>, <code>complete_vdj</code>) are not
        part of AlignAIR's output. The full field contract is in <DocLink to="airr-fields">AIRR output fields</DocLink>.
      </p>

      <h2>Semantic differences worth knowing</h2>
      <ul>
        <li><strong>Call sets, not a single guess.</strong> When a read cannot distinguish alleles, AlignAIR reports every allele it scored at <code>p &gt;= 0.5</code> in <code>*_call_set</code> (ranked, capped at 3) instead of committing to one. A multi-member set means "do not read this at allele resolution", not "these are the only possibilities". IgBLAST reports a single ranked list; there is no direct equivalent.</li>
        <li><strong>Coordinates.</strong> Same 1-based AIRR convention. Boundaries can jitter by 1-2 nt, especially on the J side - group clones on <code>junction_aa</code>, not single-nt positions.</li>
        <li><strong>Junction assembly.</strong> AlignAIR derives the junction from its own coordinates; a boundary that lands mid-codon leaves an incomplete final residue (<code>.</code>), as above. The nucleotide <code>junction</code> is usually exact even when the last <code>junction_aa</code> residue is not.</li>
        <li><strong>Confidence.</strong> AlignAIR does not emit a per-call e-value. Pretrained models carry post-hoc temperature calibration on the allele heads, but the reserved confidence columns stay blank - do not filter on them.</li>
        <li><strong>Partial predictions.</strong> A row can be <code>airr_assembly_status = partial</code> (valid calls, but the junction or another product could not be assembled). These are emitted with their calls; IgBLAST has no equivalent flag. Filter to <code>complete</code> for junction-level work. See <DocLink to="known-failure-modes">Known failure modes</DocLink>.</li>
      </ul>

      <h2>CPU and GPU</h2>
      <p>
        IgBLAST is CPU-only and multithreaded. AlignAIR runs on either:
      </p>
      <ul>
        <li><strong>GPU</strong> (CUDA, or Apple Silicon MPS) is selected automatically when available and is the fast path.</li>
        <li><strong>CPU-only</strong> works with no change; install the CPU build of PyTorch first to avoid pulling CUDA (see <DocLink to="troubleshooting">Troubleshooting</DocLink>).</li>
      </ul>
      <p>
        Measured throughput and the exact hardware it was measured on are on{" "}
        <DocLink to="performance">Speed &amp; throughput</DocLink> - measure on your own hardware before capacity
        planning.
      </p>

      <h2>Runnable side-by-side</h2>
      <p>
        No ground truth needed. <code>alignair compare</code> reports agreement between two AIRR TSVs on the same reads,
        including set-rescue - how often the other tool's call falls inside AlignAIR's <code>*_call_set</code>, which
        separates shared ambiguity from a real conflict.
      </p>
      <CodeBlock
        code={`# same reads through AlignAIR (igblastn is your existing command, above)\nalignair predict --input reads.fasta --out alignair.tsv --model alignair-igh-human@1.0.0\n\n# structural check of the output\nalignair validate-airr alignair.tsv\n\n# where do the two tools agree?\nalignair compare --a alignair.tsv --b igblast.tsv --a-name AlignAIR --b-name IgBLAST`}
      />

      <h2>Migration checklist</h2>
      <ul>
        <li>Pick the model for your locus; pin <code>@version</code> for reproducibility.</li>
        <li>Replace the <code>igblastn ... -outfmt 19</code> call with <code>alignair predict --input ... --out ... --model ...</code>.</li>
        <li>Point your downstream reader at the new TSV - the shared AIRR columns are unchanged.</li>
        <li>Add a filter on <code>airr_assembly_status == complete</code> before junction / productivity analysis.</li>
        <li>Decide how to use <code>*_call_set</code>: treat multi-member sets as below allele resolution.</li>
        <li>If you constrained IgBLAST's database to a donor, pass <code>--genotype donor.yaml</code> instead.</li>
        <li>If you relied on novel or added alleles, plan an <DocLink to="training">alignair train</DocLink> run.</li>
        <li>Sanity-check a batch with <code>alignair compare</code> before switching the pipeline over.</li>
      </ul>
    </>
  ),
};

export const migratePages: DocPage[] = [migrating];
