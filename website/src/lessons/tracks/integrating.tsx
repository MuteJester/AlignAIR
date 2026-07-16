import { Share2 } from "lucide-react";
import { CodeBlock, Callout } from "../../components/ui";
import type { Lesson, Track } from "../types";

const downstream: Lesson = {
  id: "integrating/downstream",
  slug: "downstream",
  track: "integrating",
  title: "Feed AIRR into the ecosystem",
  summary: "What AlignAIR supplies, what you supply, and handing off to Scirpy / Change-O.",
  minutes: 6,
  steps: [
    {
      kind: "explain",
      title: "AlignAIR supplies alignment; you supply sample metadata",
      body: () => (
        <>
          <p>
            AlignAIR produces a standard AIRR rearrangement TSV, so it feeds Scirpy, Change-O/Immcantation, and
            nf-core/airrflow directly. But AlignAIR is <strong>per-read</strong>: single-cell grouping and counts come
            from your metadata, joined at predict time:
          </p>
          <CodeBlock
            code={`alignair predict --input contigs.fasta --out out.tsv --model alignair-igh-human \\
  --metadata filtered_contig_annotations.csv --keep-columns barcode,umis,c_gene`}
          />
          <Callout kind="note" title="Isotype calling limitation">
            Constant region classification (isotype calling) is not resolved from sequence by AlignAIR. If you need isotype information in your final dataset, you must supply the assembler’s constant gene calls (e.g., <code>c_gene</code>) via metadata and carry it over.
          </Callout>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Filter to complete records, then hand off",
      body: () => (
        <>
          <p>Keep the complete records, then read into Scirpy grouped by cell:</p>
          <CodeBlock
            lang="python"
            code={`import scirpy as ir, pandas as pd\n\nt = pd.read_csv("out.tsv", sep="\\t")\nt = t[t["airr_assembly_status"] == "complete"]\nt.to_csv("out.complete.tsv", sep="\\t", index=False)\nadata = ir.io.read_airr("out.complete.tsv")   # cell_id from --metadata`}
          />
        </>
      ),
    },
    {
      kind: "mcq",
      prompt: () => <p>Your single-cell Scirpy import puts every contig in its own cell. What is missing?</p>,
      options: [
        "The junction column",
        "cell_id - AlignAIR is per-read, so you supply the 10x barcode as cell_id via --metadata for true single-cell grouping",
        "Nothing; Scirpy always does that",
      ],
      answer: 1,
      explanation: () => (
        <p>
          AlignAIR does not know which reads share a cell. Provide the barcode as <code>cell_id</code> through{" "}
          <code>--metadata</code> so Scirpy can reconstruct cells. Note that AlignAIR's <code>normalize_10x=True</code> parameter (enabled by default) automatically maps the standard 10x <code>barcode</code> to <code>cell_id</code> and <code>umis</code> to <code>umi_count</code> to perfectly match what Scirpy expects.
        </p>
      ),
    },
  ],
};

const validate: Lesson = {
  id: "integrating/validate",
  slug: "validate",
  track: "integrating",
  title: "Validate the AIRR output",
  summary: "Schema-check the TSV, and choose the right --columns preset.",
  minutes: 5,
  steps: [
    {
      kind: "explain",
      title: "Check the TSV against the AIRR-C schema",
      body: () => (
        <>
          <p>
            Before you hand a TSV to another tool — or archive it — confirm it is valid AIRR. AlignAIR ships a validator that checks the file against the official AIRR-C rearrangement schema:
          </p>
          <CodeBlock code="alignair validate-airr out.tsv" />
          <p>
            A clean exit means every row parses and the required fields are present and well-typed; it reads back through the <code>airr</code> library and Change-O without surprises. This is the fast way to catch a broken hand-off before it fails deep inside a downstream pipeline.
          </p>
          <Callout kind="note" title="Schema-valid is not the same as fully-derived">
            <code>validate-airr</code> checks the <em>shape</em> of the file, not whether every read aligned well. A row can be perfectly schema-valid and still be <code>partial</code>. Use <code>airr_assembly_status</code> for quality, <code>validate-airr</code> for format.
          </Callout>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Choose the right --columns preset",
      body: () => (
        <>
          <p>
            How much AlignAIR writes per row is a choice. <code>--columns</code> trades detail for speed — the lighter presets skip the gapped-alignment assembly:
          </p>
          <div style={{ margin: "20px 0", overflowX: "auto", border: "1px solid #eae9f1", borderRadius: "12px" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13.5px" }}>
              <thead>
                <tr style={{ background: "#f7f6fb", borderBottom: "1px solid #eae9f1" }}>
                  <th style={{ textAlign: "left", padding: "11px 14px", fontWeight: 600 }}>Preset</th>
                  <th style={{ textAlign: "left", padding: "11px 14px", fontWeight: 600 }}>What you get</th>
                  <th style={{ textAlign: "left", padding: "11px 14px", fontWeight: 600 }}>When to use it</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: "1px solid #f0eff5" }}>
                  <td style={{ padding: "11px 14px" }}><code>full</code></td>
                  <td style={{ padding: "11px 14px" }}>Every field: gapped alignments, per-segment CIGAR / identity, regions.</td>
                  <td style={{ padding: "11px 14px" }}>Default. Lineage / mutation work; anything that needs the alignment.</td>
                </tr>
                <tr style={{ borderBottom: "1px solid #f0eff5" }}>
                  <td style={{ padding: "11px 14px" }}><code>core</code></td>
                  <td style={{ padding: "11px 14px" }}>Calls, call sets, coordinates, junction — no gapped alignment.</td>
                  <td style={{ padding: "11px 14px" }}>Clonotype / repertoire summaries at higher throughput.</td>
                </tr>
                <tr style={{ borderBottom: "1px solid #f0eff5" }}>
                  <td style={{ padding: "11px 14px" }}><code>minimal</code></td>
                  <td style={{ padding: "11px 14px" }}>Just the essential calls and coordinates.</td>
                  <td style={{ padding: "11px 14px" }}>Fast triage or huge inputs where you only need the calls.</td>
                </tr>
                <tr style={{ borderBottom: "1px solid #f0eff5" }}>
                  <td style={{ padding: "11px 14px" }}><code>airr</code></td>
                  <td style={{ padding: "11px 14px" }}>The AIRR-required field set.</td>
                  <td style={{ padding: "11px 14px" }}>Maximum interoperability / smallest schema-valid file.</td>
                </tr>
              </tbody>
            </table>
          </div>
          <CodeBlock code={`alignair predict --input reads.fasta --out out.tsv \\\n  --model alignair-igh-human --columns core`} />
          <p>
            You can also pass an explicit comma-separated field list instead of a preset name.
          </p>
        </>
      ),
    },
    {
      kind: "mcq",
      prompt: () => (
        <p>
          You need maximum throughput and only the V/D/J calls and coordinates — no gapped alignments. Which preset?
        </p>
      ),
      options: [
        "full — it is always safest",
        "core or minimal — they skip the gapped-alignment assembly, which is the slow stage",
        "airr — it is the fastest by definition",
      ],
      answer: 1,
      explanation: () => (
        <p>
          The gapped-alignment assembly is the expensive post-processing step. <code>core</code> and <code>minimal</code> skip it, so you keep the calls and coordinates at higher throughput. <code>full</code> reconstructs everything; <code>airr</code> is about schema-completeness, not speed.
        </p>
      ),
    },
  ],
};

export const integratingTrack: Track = {
  slug: "integrating",
  title: "Integrating",
  description: "Hand AlignAIR’s AIRR output to Scirpy, Change-O, and the wider ecosystem.",
  icon: Share2,
  accent: "from-violet-500 to-fuchsia-600",
  lessons: [downstream, validate],
};
