import { CodeBlock, Callout } from "../../components/ui";
import { DocTable, DocLink, type DocPage } from "./doc-kit";

const benchmarks: DocPage = {
  slug: "benchmarks",
  title: "Benchmarks",
  section: "Evaluation",
  lead: "AlignAIR evaluated against IgBLAST as a reference implementation, on a frozen simulated benchmark.",
  body: () => (
    <>
      <p>
        To quantify accuracy, AlignAIR was evaluated against IgBLAST on a frozen, simulated benchmark with known ground
        truth: 2,600 cases across 13 strata (200 each - clean, moderate and hard full-length, high SHM, high indel,
        noisy/ambiguous, trimmed, five 5'- and 3'-anchored amplicon and short-fragment shapes, and arbitrary
        orientation), scored with paired-case bootstrap confidence intervals and Bonferroni correction. Reference: human
        IGH (OGRDB).
      </p>

      <h2>Summary</h2>
      <p>
        On point estimates, AlignAIR is higher on 18 of the 24 metrics and IgBLAST on 6. Under the Bonferroni-corrected
        paired bootstrap those resolve into statistically supported verdicts: AlignAIR better on 18, IgBLAST better on 4,
        and 2 inconclusive. AlignAIR's gains concentrate in D and J allele calling and in reverse-complement / arbitrary
        orientation. IgBLAST's four supported wins are V-allele calling (a small aggregate edge, 0.856 vs 0.828, from
        short fragments and heavily-mutated full-length reads) and exact junction nt/aa recovery; the two inconclusive
        metrics are the J-segment start/end coordinate MAEs, where IgBLAST's point estimate is higher but the interval
        spans zero.
      </p>
      <Callout kind="note">
        The 18 / 4 / 2 split is the Bonferroni-corrected verdict from the paired bootstrap, not a raw point-estimate
        tally: every &ldquo;better&rdquo; above corresponds to a confidence interval that excludes zero after
        multiple-comparison correction. The two inconclusive metrics are ones where IgBLAST's point estimate is higher
        but its interval spans zero. Full per-metric intervals are in the benchmark report.
      </Callout>

      <h2>Metrics</h2>
      <p>Per gene (V, D, J):</p>
      <ul>
        <li><code>call_top1_in_set</code>: the top call is inside the truth equivalence set (allele accuracy).</li>
        <li><code>call_set_f1</code>: F1 of the predicted equivalence set against the truth set.</li>
        <li><code>ss_mae</code>, <code>se_mae</code>: MAE of the in-read segment start and end.</li>
        <li><code>gs_mae</code>, <code>ge_mae</code>: MAE of the germline start and end.</li>
      </ul>
      <p>Global: junction nt/aa exact match, productivity and orientation accuracy, required-field presence, parseable-AIRR rate.</p>

      <h2>Allele calling (top-1 in truth set)</h2>
      <DocTable
        head={["Gene", "IgBLAST", "AlignAIR"]}
        rows={[
          ["V", <strong>0.856</strong>, "0.828"],
          ["D", "0.534", <strong>0.722</strong>],
          ["J", "0.669", <strong>0.821</strong>],
        ]}
      />
      <p>
        V is the one gene IgBLAST calls better here: the aggregate edge is small and comes from short fragments and
        heavily-mutated full-length reads, where AlignAIR's V is comparable at best. AlignAIR's advantage is in D and J -
        on the short J-anchored fragments (where most of the read lies 3' of V), D recovery is 0.735 vs 0.015 and J is
        0.870 vs 0.085. Genuinely ambiguous reads are reported as a set (<code>*_call_set</code>) rather than a forced
        single call.
      </p>

      <h2>Throughput</h2>
      <p>
        Throughput is device-dependent, so it does not reduce to a single comparison. On CPU-only hardware AlignAIR runs
        somewhat below IgBLAST's multi-threaded CPU baseline; on a GPU it runs several times above it. A lighter{" "}
        <code>--columns core</code> preset recovers roughly a tenth of the time by writing a much smaller row. See{" "}
        <DocLink to="performance">Speed and throughput</DocLink> for the measured per-device figures, stated with their
        hardware, backend and timing boundary.
      </p>
      <Callout kind="note" title="Throughput is not part of the scored benchmark">
        The accuracy comparison above is the frozen, ground-truth head-to-head. The speed figures are measured separately
        (see the linked page) and were not collected as a same-run, same-box race against IgBLAST - so treat the
        direction as reliable and the exact ratio as indicative.
      </Callout>

      <h2>Methodology and reproducibility</h2>
      <p>A model self-check, built into the CLI (fresh ground-truth reads per stratum, no IgBLAST):</p>
      <CodeBlock code={`alignair benchmark --model alignair-igh-human --n 200 --seed 0 --out benchmark.json`} />
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
        The full head-to-head vs IgBLAST lives in the <code>alignair_benchmark</code> package. This package is excluded from the installed PyPI wheel to minimize the runtime install footprint (preventing heavy dev/statistical dependencies in production). It must be installed from a source clone. Its <code>evaluate</code>/<code>compare</code> commands add paired case-bootstrap
        confidence intervals, per-stratum intervals, and Bonferroni-corrected tests, and record provenance (AlignAIR
        version + commit, package versions, CUDA detail, reference/case hashes). Run IgBLAST against the same germline
        (resolve the id to a file with <code>alignair models path &lt;id&gt;</code>, then{" "}
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
      <p>
        Measured 2026-07-16 on an Intel Core i7-10700F (8 cores / 16 threads) and NVIDIA GeForce RTX
        3090 Ti, with Python 3.12.3, PyTorch 2.11.0+cu128, GenAIRR 2.2.0, and the compiled germline-CIGAR
        derivation kernel (<code>alignair doctor</code>: <code>derive_backend=cython</code>). Workload: 400
        GenAIRR human-IGH reads at curriculum progress 0.3 (average 369 nt), the 198-V / 33-D / 7-J
        OGRDB model, and batch size 64.
      </p>
      <h2>Prediction throughput</h2>
      <p>
        Steady-state numbers below are the median of five warmed runs. Model loading is excluded; neural
        prediction and the post-processing required by the selected output are included.
      </p>
      <DocTable
        head={["Device", "Output", "Median reads/s", "Measured peak memory*"]}
        rows={[
          ["CPU", "full (109 fields)", "~112", "~0.75 GB host RSS"],
          ["CPU", "core (27 fields)", "~123", "~0.75 GB host RSS"],
          ["CUDA", "full (109 fields)", "~1,480", "~0.16 GB CUDA allocated"],
          ["CUDA", "core (27 fields)", "~1,720", "~0.16 GB CUDA allocated"],
        ]}
      />
      <p>
        * Host RSS and CUDA allocator peaks are different measurements and should not be compared as if
        they were the same pool. A fresh 400-read CLI invocation, including Python startup, model loading,
        and file writing, took about 5.8 s on CPU and 3.4 s on CUDA; startup therefore dominates small jobs.
      </p>
      <p>The main throughput knobs:</p>
      <ul>
        <li><code>--device cuda</code> - the largest lever; the neural stage is GPU-friendly.</li>
        <li><code>--columns core</code> - a far smaller row (27 fields vs 109). Note both rows above still run the AIRR assembly: <code>core</code> asks for the junction, which is one of its products, so the ~10% gain is mostly writing less. Only <code>minimal</code> (calls + <code>productive</code>, no coordinates or junction) skips the assembly outright; it was not measured here.</li>
        <li><code>--batch-size</code> - larger batches improve GPU utilisation up to memory limits.</li>
        <li><code>--chunk-size</code> - bounds memory for large inputs, not speed.</li>
      </ul>
      <p>Memory stays flat regardless of input size, because prediction streams the input in chunks.</p>

      <h2>Training step time</h2>
      <p>
        These measurements include GenAIRR record generation, collation, forward/backward, and the
        optimizer step. The wall estimate is direct arithmetic from the preset step count; periodic
        validation and checkpoint I/O add extra time.
      </p>
      <DocTable
        head={["Preset", "Batch", "Device", "Median s / step", "Step-only estimate", "Measured peak memory*"]}
        rows={[
          ["desktop (50,000 steps)", "64", "CPU", "~1.69", "~23.5 h", "~3.0 GB host RSS"],
          ["desktop (50,000 steps)", "64", "CUDA", "~0.177", "~2.5 h", "~1.9 GB CUDA allocated"],
          ["full (300,000 steps)", "128", "CUDA", "~0.290", "~24.1 h", "~3.5 GB CUDA allocated"],
        ]}
      />
      <p>
        On this machine CUDA is roughly 9.6x faster per <code>desktop</code> step than CPU; training is not
        purely data-generation-bound. The <code>full</code> preset uses a larger batch and six times as many
        steps, which is why its estimated wall time is close to a day even though each step remains below
        0.3 s. <code>--plan</code> reports the resolved configuration, but does not estimate wall time or memory.
      </p>
    </>
  ),
};

const design: DocPage = {
  slug: "design",
  title: "Design & internals",
  section: "Design",
  lead: "A single neural model produces the full V(D)J alignment; a light post-processing stage turns it into AIRR.",
  body: () => (
    <>
      <h2>End-to-end pipeline</h2>
      <CodeBlock
        lang="text"
        code={`read (nucleotides)\n  |  tokenize (A/C/G/T/N + pad, fixed window)\n  v\nin-model orientation head -> detect orientation, re-orient to the forward frame\n  v\nconvolutional feature encoder (residual conv tower)\n  |- per-gene branches (V, D, J):\n  |    |- segmentation heads -> start / end position\n  |    '- classification head -> per-allele scores (over the embedded catalog)\n  '- meta heads -> mutation rate, indel count, productivity, locus\n  v\npost-processing (alignair.predict)\n  |- allele selection -> top call + candidate set\n  |- germline reader -> refined coordinates, per-segment CIGAR, % identity\n  '- AIRR assembly -> IMGT-gapped alignments, junction / CDR3, np1/np2, flags\n  v\nAIRR rearrangement record`}
      />
      <p>
        The neural network localises each segment and scores alleles; a germline reader (a fast anchored aligner by
        default, with an optional WFA / parasail backend) refines coordinates and produces per-segment CIGARs and
        identity, and the AIRR assembler reconstructs the IMGT-gapped alignments and derives the junction. The germline
        reference is embedded in the model file, so calls are always drawn from a known, fingerprinted catalog.
      </p>

      <h2>Key design choices</h2>
      <ul>
        <li><strong>In-model orientation.</strong> Orientation is predicted from the initial embeddings and corrected inside the model, so every downstream head and the coordinates operate on one canonical frame.</li>
        <li><strong>Segmentation + classification, jointly.</strong> Each gene regresses boundaries and scores alleles from a shared convolutional representation - one forward pass yields calls, coordinates, and quality together.</li>
        <li><strong>Honest ambiguity.</strong> Allele selection emits a ranked candidate set alongside the top call, so a read that cannot distinguish alleles is visible as such rather than reported as a confident single call.</li>
        <li><strong>Fixed-reference classifier.</strong> The heads are tied to the embedded catalog; a donor genotype can subset it at inference, but adding alleles requires training.</li>
      </ul>

      <h2>Evaluation</h2>
      <p>
        In a controlled evaluation against IgBLAST on a frozen 2,600-case benchmark, under a Bonferroni-corrected paired
        bootstrap AlignAIR is statistically better on 18 of 24 metrics (IgBLAST on 4, 2 inconclusive), with the gains
        concentrated in D/J calling and arbitrary orientation; IgBLAST keeps an edge on V and on exact
        junction-nucleotide recovery. Full methodology is on the <DocLink to="benchmarks">Benchmarks</DocLink> page.
      </p>

      <h2>Training</h2>
      <p>
        Models train on data streamed from the GenAIRR simulator (no static dataset) over a curriculum that mixes clean
        reads with fragments, SHM, indels, and arbitrary orientation. A run embeds the germline reference into the
        resulting <code>.alignair</code> file, so the model is self-contained.
      </p>
    </>
  ),
};

export const evalPages: DocPage[] = [benchmarks, performance, design];
