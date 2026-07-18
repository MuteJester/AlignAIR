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
            Before you hand a TSV to another tool — or archive it — check it. AlignAIR ships a fast, dependency-free
            structural validator:
          </p>
          <CodeBlock code="alignair validate-airr out.tsv" />
          <p>
            A clean exit means the required columns are present, the coordinates are in bounds, no per-gene CIGAR
            consumes more query than the emitted <code>sequence</code>, and productivity is self-consistent. It is the
            fast way to catch a broken hand-off before it fails deep inside a downstream pipeline.
          </p>
          <Callout kind="note" title="Two different checks — do not confuse them">
            <p style={{ margin: "0 0 8px" }}>
              <code>validate-airr</code> is AlignAIR's own <strong>structural</strong> check; it does not call the
              official <code>airr</code> library. For formal schema validation, run that too:
            </p>
            <CodeBlock lang="python" code={`import airr\nairr.validate_rearrangement("out.tsv")`} />
            <p style={{ margin: "8px 0 0" }}>
              And neither one tells you a read aligned <em>well</em>: a row can pass both and still be{" "}
              <code>partial</code>. Use <code>airr_assembly_status</code> for quality, these validators for format.
            </p>
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
            How much AlignAIR writes per row is a choice. <code>--columns</code> trades detail for speed. Read the table
            carefully: only <code>minimal</code> actually skips the gapped-alignment assembly, because it is the only
            preset that asks for nothing derived from it:
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
                  <td style={{ padding: "11px 14px" }}>Calls, call sets, coordinates, junction (27 fields). Still assembled — it asks for the junction.</td>
                  <td style={{ padding: "11px 14px" }}>Clonotype / repertoire summaries: a much smaller file, same derived biology.</td>
                </tr>
                <tr style={{ borderBottom: "1px solid #f0eff5" }}>
                  <td style={{ padding: "11px 14px" }}>
                    <code>minimal</code>
                  </td>
                  <td style={{ padding: "11px 14px" }}>
                    Calls + <code>productive</code> only (7 fields). <strong>No coordinates, no junction</strong> — the
                    only preset that skips the assembly.
                  </td>
                  <td style={{ padding: "11px 14px" }}>Fast triage or huge inputs where the allele calls are all you need.</td>
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
          You want the smallest, fastest output that still gives you the junction for clonotype work. Which preset — and
          what does it actually save?
        </p>
      ),
      options: [
        "minimal — it is the fastest, and still gives you the junction",
        "core — a much smaller file than full, but the assembly still runs, because the junction is derived from it",
        "airr — the AIRR-required set is by definition the fastest",
      ],
      answer: 1,
      explanation: () => (
        <p>
          If you need the junction, the assembly has to run — the junction is one of its products. So <code>core</code>{" "}
          buys you a far smaller file (27 fields vs 109), not a skipped assembly; the saving is mostly writing less.{" "}
          <code>minimal</code> is the only preset that truly skips the assembly, and the price is exactly that: no
          junction and no coordinates, just calls and <code>productive</code>. <code>airr</code> is about
          schema-completeness, not speed.
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
