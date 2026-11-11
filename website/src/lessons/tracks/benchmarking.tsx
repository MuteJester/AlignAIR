import { BarChart3 } from "lucide-react";
import { CodeBlock, Callout } from "../../components/ui";
import type { Lesson, Track } from "../types";

const benchmarking: Lesson = {
  id: "benchmarking/evaluate",
  slug: "evaluate",
  track: "benchmarking",
  title: "Evaluate and benchmark",
  summary: "The self-check, the head-to-head, and concordance vs accuracy.",
  minutes: 7,
  steps: [
    {
      kind: "explain",
      title: "Two levels of evaluation",
      body: () => (
        <>
          <p>
            There are two tiers, and they answer different questions. <strong>Tier 1 — the built-in self-check.</strong>{" "}
            Shipped in the CLI, it generates fresh ground-truth reads per stratum and scores one model against that
            truth. Use it to sanity-check any model, including one you just trained:
          </p>
          <CodeBlock code={`alignair benchmark --model alignair4-igh-human --n 200 --seed 0 --format json --out benchmark.json`} />
          <p>
            (<code>alignair4-igh-human</code> is the AlignAIR 4 IGH model's catalog id; it is not published yet, so
            until then pass a <code>.alignair</code> path.) {/* PENDING-V4-PUBLISH: alignair4-igh-human */}
          </p>
          <p>
            <strong>Tier 2 — the head-to-head vs IgBLAST.</strong> Paired comparisons with confidence intervals and
            Holm-corrected verdicts, per-suite diagnostics and full provenance, on GenAIRR-simulated suites scored
            against observable truth and on real donors. This lives in the <code>alignair_assay</code> package, which
            ships in the AlignAIR <strong>source repository</strong>, not the installed PyPI wheel — so you get it from a
            git clone:
          </p>
          <CodeBlock code={`git clone https://github.com/MuteJester/AlignAIR\n# then run alignair_assay from the repo per src/alignair_assay/README.md`} />
          <Callout kind="note" title="Why is benchmarking separate?">
            <code>alignair_assay</code> is a development instrument: it drives IgBLAST and AlignAIR, freezes the simulator
            it builds suites with, and carries statistics only evaluation needs. Keeping it out of the wheel keeps the
            production install small.
          </Callout>
          <Callout kind="note" title="Score both tools on the same germline">
            A fair comparison must give IgBLAST the exact germline the model uses. Export it from the model file, then
            run IgBLAST with <code>-outfmt 19</code>:
            <CodeBlock code={`alignair reference export "$(alignair models path alignair4-igh-human)" --fasta germline.fasta`} />
          </Callout>
        </>
      ),
    },
    {
      kind: "explain",
      title: "What the head-to-head scores",
      body: () => (
        <>
          <p>
            Per gene (V, D, J): whether the top-1 allele is in the read's observable truth set (<code>v.top1_in_set</code>),
            and whether each boundary lies inside its truth interval (<code>*_exact</code>, with the distance in{" "}
            <code>*_err</code>). Global:
          </p>
          <ul className="list-disc pl-6 space-y-1.5 my-4">
            <li>exact junction nucleotide / amino-acid match (<code>junction.nt_exact</code>, <code>junction.aa_exact</code>) and the clone key (<code>clone_key.exact</code>),</li>
            <li>orientation (<code>orientation.correct</code>),</li>
            <li>AIRR validity (<code>airr.valid</code>).</li>
          </ul>
          <p>The full metric catalogue is in the assay's README.</p>
          <p>
            Coverage matters: a tool that emits fewer junctions is scored on the junctions it does emit, so read
            junction accuracy alongside required-field presence.
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Reading benchmark.json",
      body: () => (
        <>
          <p>
            The self-check writes a JSON report <strong>keyed by stratum</strong>, not one flat score — because accuracy
            depends on read quality. Each stratum carries its own metrics (illustrative values, not results):
          </p>
          <CodeBlock
            lang="json"
            code={`{\n  "clean":         { "n": 200, "v_call_acc": 0.99, "d_call_acc": 0.86, "j_call_acc": 0.99,\n                     "junction_nt_exact": 0.95, "junction_len_mae": 0.4,\n                     "v_sequence_start_mae": 0.1, "j_sequence_end_mae": 1.2,\n                     "productive_acc": 0.98 },\n  "moderate":      { "...": "same metrics, corrupted reads" },\n  "high_shm":      { "...": "25% mutation load — the hardest V regime" },\n  "short_janchor": { "...": "J-anchored short fragments" }\n}`}
          />
          <p>
            Read it <strong>per stratum</strong>: a model can look excellent on <code>clean</code> and fall over on{" "}
            <code>high_shm</code> or <code>short_janchor</code>. That spread is the useful signal.
          </p>
          <Callout kind="note" title="Tier 1 and Tier 2 use different metric names">
            The self-check reports <code>v_call_acc</code> (top-1 accuracy), <code>junction_nt_exact</code>,{" "}
            <code>junction_len_mae</code>, <code>*_sequence_start/end_mae</code> and <code>productive_acc</code>. The
            head-to-head package reports the richer set named above (<code>v.top1_in_set</code>, the boundary{" "}
            <code>*_exact</code> / <code>*_err</code>, <code>junction.aa_exact</code>, <code>clone_key.exact</code>,{" "}
            <code>airr.valid</code>) and, comparing two tools, adds a confidence interval plus a Holm-corrected verdict to
            each. Read those, not just the point estimate: a higher number without a verdict is not yet a real
            difference.
          </Callout>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Accuracy is not one number",
      body: () => (
        <>
          <p>
            A single headline accuracy hides everything that matters. The same tool can be strong on one read type and
            weak on another, and two tools can trade places across strata - so a benchmark reports{" "}
            <strong>per-stratum, per-gene</strong> results, not one scoreboard number. Read the shape of it: clean
            full-length reads are near-perfect for good tools, while fragments, heavy SHM, and arbitrary orientation are
            where methods separate.
          </p>
          <p>
            And a visible gap is not automatically a result. Each paired difference carries a simultaneous confidence
            interval and a Holm-corrected verdict, and a difference whose interval still spans zero is not yet real -
            you read the point estimate <em>and</em> its interval together.
          </p>
          <Callout kind="note" title="Record comparison provenance">
            Record the exact model, tool versions, germline reference, input cases, and commands for each comparison.
            This lesson explains how to read comparison results; its examples do not establish comparative accuracy.
          </Callout>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Comparing on your own data",
      body: () => (
        <>
          <p>
            Without ground truth you can still compare two tools with <code>alignair compare</code> — but it measures{" "}
            <strong>concordance, not accuracy</strong>:
          </p>
          <CodeBlock
            code={`alignair compare --a alignair.tsv --b igblast.tsv \\\n  --a-name AlignAIR --b-name IgBLAST --out report.md`}
          />
        </>
      ),
    },
    {
      kind: "mcq",
      prompt: () => (
        <p>
          <code>alignair compare</code> reports 92% agreement and high "set-rescue" between AlignAIR and IgBLAST.
          What can you conclude?
        </p>
      ),
      options: [
        "AlignAIR is 92% accurate",
        "The two tools agree 92% of the time; set-rescue means the other tool's call is inside AlignAIR's set — neither is proven correct without ground truth",
        "IgBLAST is wrong 8% of the time",
      ],
      answer: 1,
      explanation: () => (
        <p>
          With no ground truth, agreement is concordance, not accuracy. Set-rescue only shows the other tool's call is
          present in AlignAIR's candidate set, not which call is right. For accuracy you need the ground-truth
          benchmark.
        </p>
      ),
    },
  ],
};

export const benchmarkingTrack: Track = {
  slug: "benchmarking",
  title: "Benchmarking",
  description: "Evaluate a model, read the metrics, and compare tools honestly.",
  icon: BarChart3,
  accent: "from-amber-500 to-orange-600",
  lessons: [benchmarking],
};
