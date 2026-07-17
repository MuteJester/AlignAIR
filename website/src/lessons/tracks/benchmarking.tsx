import { BarChart3 } from "lucide-react";
import { CodeBlock, Callout } from "../../components/ui";
import { BenchSandboxWidget } from "../../components/widgets";
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
          <CodeBlock code={`alignair benchmark --model alignair-igh-human --n 200 --seed 0 --out benchmark.json`} />
          <p>
            <strong>Tier 2 — the head-to-head vs IgBLAST.</strong> Paired bootstrap confidence intervals, Bonferroni
            correction, per-stratum diagnostics and full provenance. This lives in the <code>alignair_benchmark</code>{" "}
            package, which ships in the AlignAIR <strong>source repository</strong>, not the installed PyPI wheel — so
            you get it from a git clone:
          </p>
          <CodeBlock code={`git clone https://github.com/MuteJester/AlignAIR\n# then install the benchmark package from the repo per its README`} />
          <Callout kind="note" title="Why is benchmarking separate?">
            Keeping <code>alignair_benchmark</code> out of the main PyPI distribution minimizes the production installation footprint. It avoids bringing in heavy statistical analysis and plotting libraries (like SciPy or Matplotlib) that are only required during evaluation.
          </Callout>
          <Callout kind="note" title="Score both tools on the same germline">
            A fair comparison must give IgBLAST the exact germline the model uses. Export it from the model, then run
            IgBLAST with <code>-outfmt 19</code>:
            <CodeBlock code={`alignair reference export alignair-igh-human --fasta germline.fasta`} />
          </Callout>
        </>
      ),
    },
    {
      kind: "explain",
      title: "The 24 metrics",
      body: () => (
        <>
          <p>Per gene (V, D, J): allele top-1-in-set, call-set F1, and MAE of in-read and germline start/end. Global:</p>
          <ul className="list-disc pl-6 space-y-1.5 my-4">
            <li>exact junction nucleotide / amino-acid match,</li>
            <li>productivity and orientation accuracy,</li>
            <li>required-field presence and parseable-AIRR rate.</li>
          </ul>
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
            depends on read quality. Each stratum carries its own metrics:
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
            head-to-head package reports the richer set named above (<code>call_top1_in_set</code>,{" "}
            <code>call_set_f1</code>, <code>ss/se/gs/ge_mae</code>, <code>junction_aa_exact</code>,{" "}
            <code>parseable_airr_rate</code>) and adds a confidence interval plus a Bonferroni-corrected significance
            flag to each. Read those, not just the point estimate: a higher number that is not flagged significant is not
            yet a real difference.
          </Callout>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Where the differences actually are",
      body: () => (
        <>
          <p>
            Accuracy is not one number, and it is not one direction either. Pick a stratum to see how AlignAIR and
            IgBLAST each do on V, D and J allele calling:
          </p>
          <BenchSandboxWidget />
          <p>
            Read the shape of it rather than a scoreboard. On clean full-length reads both tools are essentially
            perfect, so there is nothing to choose between them. AlignAIR's gains concentrate in <strong>D and J</strong>
            , in <strong>degraded and junction-anchored reads</strong>, and in <strong>mixed orientation</strong>, where
            it re-frames the read instead of scoring near chance. IgBLAST is <strong>genuinely better at calling V</strong>{" "}
            on heavily-mutated full-length reads and on 5'-anchored fragments — a real limitation, documented in{" "}
            <em>Known failure modes</em>, not an artifact of the benchmark.
          </p>
          <Callout kind="note" title="Where a gap is not a result">
            Two of these panels show a visible bar difference that the benchmark scores as <em>not significant</em> —
            the Bonferroni-corrected interval still spans zero. That is the previous step's lesson made concrete: the
            point estimate is where you start reading, not where you stop.
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
