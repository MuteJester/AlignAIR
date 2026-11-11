import { CodeBlock, Callout } from "../../components/ui";
import { DocLink, type DocPage } from "./doc-kit";

const benchmarks: DocPage = {
  slug: "benchmarks",
  title: "Benchmarks",
  section: "Evaluation",
  lead: "Evaluate model accuracy on labeled simulated reads and compare tools on shared inputs.",
  body: () => (
    <>
      <p>
        Accuracy evaluation uses simulated reads with known ground truth, covering clean reads, mutation, indels,
        noise, fragments, and orientation changes. Read results by gene and read type, and compare tools on the same
        inputs and compatible germline references. The metrics below describe what the evaluation tools measure.
      </p>

      <Callout kind="note" title="Reproducible comparisons">
        A comparative result needs identified model and tool versions, reference and input fingerprints, and the
        commands used to produce it. This page describes evaluation methods; it does not publish comparative scores.
      </Callout>

      {/* PENDING-V4-NUMBERS: the AlignAIR 4 vs IgBLAST head-to-head has not been run; no scores are published. */}
      <h2>Metrics</h2>
      <p>
        The head-to-head (<code>alignair_assay</code>, in the source repository) scores each read against its{" "}
        <em>observable</em> truth: what a perfect aligner could recover from it. Per gene (V, D, J):
      </p>
      <ul>
        <li><code>v.top1_in_set</code> (and <code>d.</code>, <code>j.</code>): the top call is inside the read's observable truth set (allele accuracy).</li>
        <li>boundary <code>*_exact</code>: the predicted start or end lies inside its truth interval; <code>*_err</code> is the distance to it.</li>
      </ul>
      <p>
        Global: <code>junction.nt_exact</code>, <code>junction.aa_exact</code>, <code>clone_key.exact</code>,{" "}
        <code>orientation.correct</code>, and AIRR validity (<code>airr.valid</code>). The assay's README lists every
        metric.
      </p>

      <p>
        Genuinely ambiguous reads are reported with their ties (<code>v_call</code>) and a set (<code>*_call_set</code>)
        rather than a forced single call, so the scores read the top-1 against the truth set. Where AlignAIR is strong and weak by gene and read
        type is summarized qualitatively in <DocLink to="known-failure-modes">Known failure modes</DocLink>.
      </p>

      <h2>Throughput</h2>
      <p>
        Throughput depends on the device, batch size, and requested output fields. See{" "}
        <DocLink to="performance">Speed and throughput</DocLink> for how throughput is measured; the AlignAIR 4 figures
        are pending a re-timing on a quiet machine.
      </p>
      <Callout kind="note" title="Throughput is not part of the scored benchmark">
        Accuracy and throughput use different measurement boundaries. The linked speed figures describe AlignAIR's
        workload and hardware; a tool-to-tool speed comparison also needs matching inputs and timing boundaries.
      </Callout>

      <h2>Methodology and reproducibility</h2>
      <p>A model self-check, built into the CLI (fresh ground-truth reads per stratum, no IgBLAST):</p>
      <CodeBlock code={`alignair benchmark --model model.alignair --n 200 --seed 0 --format json --out benchmark.json`} />
      <p>
        The built-in CLI <code>alignair benchmark</code> command performs a quick self-evaluation on <strong>4 default strata</strong>:
      </p>
      <ul className="list-disc pl-6 space-y-1 my-2">
        <li><code>clean</code>: error-free synthetic reads.</li>
        <li><code>moderate</code>: moderate sequence corruption.</li>
        <li><code>high_shm</code>: high somatic hypermutation load (25% mutation rate).</li>
        <li><code>short_janchor</code>: J-anchored short fragments.</li>
      </ul>
      <p>
        It outputs simpler self-evaluation metrics: <code>v/d/j_call_acc</code> (top-1 classification accuracy), <code>junction_nt_exact</code> (exact CDR3 match), <code>junction_len_mae</code> (Mean Absolute Error of junction length), <code>*_sequence_start/end_mae</code> (MAE of coordinates), and <code>productive_acc</code>.
      </p>
      <p>
        The full head-to-head vs IgBLAST lives in the <code>alignair_assay</code> package
        (<code>src/alignair_assay</code>), which is excluded from the installed PyPI wheel and runs from a source
        clone (<code>python -m alignair_assay ...</code>; see its README). It builds GenAIRR suites against a frozen
        simulator, runs each tool, scores the AIRR output against observable truth, and compares two tools with
        analytic paired tests (cluster-aware when cases are grouped), Holm-corrected verdicts and simultaneous
        confidence intervals, recording each run's provenance. Run IgBLAST against the same germline (resolve the id
        to a file with <code>alignair models path &lt;id&gt;</code>, then{" "}
        <code>alignair reference export &lt;file&gt; --fasta germline.fasta</code>) with <code>-outfmt 19</code>.
      </p>

      <h2>Compare agreement on your own data</h2>
      <CodeBlock code={`alignair compare --a alignair.tsv --b igblast.tsv \\\n  --a-name AlignAIR --b-name IgBLAST --out report.md`} />
      <p>
        This measures concordance between the two tools, not accuracy: without ground truth, neither call is known to be
        correct, and set-rescue shows the other tool's call is present in AlignAIR's set, not which call is right.
      </p>
    </>
  ),
};

const performance: DocPage = {
  slug: "performance",
  title: "Speed & throughput",
  section: "Evaluation",
  lead: "Measured resource numbers with the workload, backend, and timing boundary stated explicitly.",
  body: () => (
    <>
      {/* PENDING-V4-NUMBERS: prediction throughput and training step time for AlignAIR 4, re-timed on a quiet machine. */}
      <Callout kind="note" title="Results pending for AlignAIR 4">
        The figures this page carried were measured on 2026-07-16 with an earlier AlignAIR release. The AlignAIR 4
        figures are pending a re-timing on a quiet machine; until then, measure on your own hardware.
      </Callout>
      <h2>Prediction throughput</h2>
      <p>
        A throughput figure needs its hardware, software versions (<code>alignair doctor</code>), model, workload
        (read count and length), batch size and timing boundary. Report steady-state numbers as the median of several
        warmed runs, with model loading excluded and neural prediction plus the post-processing the selected output
        needs included. A fresh CLI invocation also pays Python startup and model loading, which dominate small jobs.
      </p>
      <p>The main throughput knobs:</p>
      <ul>
        <li><code>--device cuda</code> - the largest lever; the neural stage is GPU-friendly.</li>
        <li><code>--threads N</code> - on the CPU, N single-threaded worker processes (torch's own threads scale poorly on read batches).</li>
        <li><code>--columns core</code> - a far smaller row (27 fields vs 109). It still runs the AIRR assembly: <code>core</code> asks for the junction, which is one of its products, so the gain is mostly writing less. Only <code>minimal</code> (calls + <code>productive</code>, no coordinates or junction) skips the assembly outright.</li>
        <li><code>--batch-size</code> - larger batches improve GPU utilisation up to memory limits (default 256 on a GPU, 64 on the CPU, 16 per worker with <code>--threads</code>).</li>
        <li><code>--chunk-size</code> - bounds memory for large inputs, not speed.</li>
      </ul>
      <p>Memory stays flat regardless of input size, because prediction streams the input in chunks.</p>

      <h2>Training step time</h2>
      <p>
        A step's time includes GenAIRR record generation, collation, forward/backward, and the optimizer step; a
        preset's wall time is its step count times that, plus periodic validation and checkpoint I/O.{" "}
        <code>--data-workers N</code> generates the records in worker processes (at most one per curriculum stream).{" "}
        <code>--plan</code> reports the resolved configuration, but does not estimate wall time or memory.
      </p>
    </>
  ),
};

const design: DocPage = {
  slug: "design",
  title: "Design & internals",
  section: "Design",
  lead: "A single neural model reads the read against the bound germline reference; deterministic refinement turns it into AIRR.",
  body: () => (
    <>
      <h2>End-to-end pipeline</h2>
      <CodeBlock
        lang="text"
        code={`read (nucleotides)\n  |  tokenize (A/C/G/T/N + pad, fixed window)\n  v\norientation head (+ the bank's k-mer vote) -> re-orient to the forward frame\n  v\ndilated convolutional trunk -> every base's features, once\n  |- parser -> the read's V / D / J regions and boundaries\n  |- anchor heads -> V Cys and J Trp/Phe codon starts\n  |- mutation reader -> per-base mutation / insertion, germline columns\n  '- bank scans (V, D, J) -> a score for every allele of the BOUND reference\n  v\nrefinement (deterministic, against the germline)\n  |- one locus per read (multi-locus banks), D presence\n  |- V re-score (banded gapped alignment), D local alignment, boundaries\n  '- tied V alleles listed after the call\n  v\nAIRR assembly (alignair.predict) -> junction from the anchors, productivity,\n  IMGT-gapped alignments, per-segment CIGAR / identity, np1/np2, flags\n  v\nAIRR rearrangement record`}
      />
      <p>
        The network localises each segment, finds the junction anchors and scores every allele of the bound reference;
        the refinement re-scores the top V alleles by a banded gapped alignment, aligns the junction to the D alleles,
        and moves the boundaries to where the read stops following the called germline. The AIRR assembler
        reconstructs the IMGT-gapped alignments and derives the junction and productivity by rule. The model file
        embeds its training reference, fingerprinted, as the default bank; another can be bound at predict time.
      </p>

      <h2>Key design choices</h2>
      <ul>
        <li><strong>In-model orientation.</strong> Orientation is predicted from the initial embeddings and corrected inside the model, so every downstream unit and the coordinates operate on one canonical frame.</li>
        <li><strong>Parsing + scanning, jointly.</strong> The parser's regions route each base to its gene's bank scan, and the anchor heads read only inside their region - one forward pass yields calls, boundaries and anchors together.</li>
        <li><strong>The reference is an input.</strong> Each allele is a row of the bank on fixed frame columns (IMGT-gapped for V), rebuilt from whichever reference is bound; a donor genotype restricts its rows, and novel alleles are new rows - no retraining.</li>
        <li><strong>Honest ambiguity.</strong> Tied V alleles are listed after the call, and each gene carries a candidate set, so a read that cannot distinguish alleles is visible as such rather than reported as a confident single call.</li>
      </ul>

      <h2>Evaluation</h2>
      <p>
        Evaluation tools score V/D/J allele accuracy, segment coordinates, the junction, productivity, and orientation
        against labeled simulated reads. Tool comparisons should record model, software, reference, and input
        provenance. Evaluation workflows are on the{" "}
        <DocLink to="benchmarks">Benchmarks</DocLink> page.
      </p>

      <h2>Training</h2>
      <p>
        Models train on data streamed from the GenAIRR simulator (no static dataset) over a curriculum that mixes clean
        reads with fragments, SHM, indels, library flanks, and arbitrary orientation. A run embeds the germline
        reference into the resulting <code>.alignair</code> file, so the model is self-contained.
      </p>
    </>
  ),
};

export const evalPages: DocPage[] = [benchmarks, performance, design];
