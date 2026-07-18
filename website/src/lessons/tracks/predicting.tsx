import { Terminal } from "lucide-react";
import { CodeBlock, Callout } from "../../components/ui";
import {
  CliWidget,
  AirrFieldsWidget,
  RevcompWidget,
  GenotypeWidget,
  RecFilterWidget,
} from "../../components/widgets";
import type { Lesson, Track } from "../types";

const firstPrediction: Lesson = {
  id: "predicting/first-prediction",
  slug: "first-prediction",
  track: "predicting",
  title: "Your first prediction",
  summary: "Install, grab a model, align reads, and read the AIRR output.",
  minutes: 8,
  steps: [
    {
      kind: "explain",
      title: "Install and check your environment",
      body: () => (
        <>
          <p>Install the CLI, then verify PyTorch, GenAIRR and the device are all wired up:</p>
          <CodeBlock code={`pip install "AlignAIR[cli]"\nalignair doctor`} />
          <p>
            <code>alignair doctor</code> reports the resolved device — <code>auto</code> picks CUDA, then Apple MPS,
            then CPU.
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Predict with a pretrained model",
      body: () => (
        <>
          <p>
            Pass a model <strong>id</strong>; it downloads, hash-verifies and caches on first use (no login). Replay a
            real captured run to see exactly what it prints:
          </p>
          <CliWidget />
          <p>
            Worth noticing in that output: <strong>68 of the 400 reads assembled only partially</strong>, on reads with
            no quality problem beyond ordinary hypermutation. That is not a failure, and the CLI does not treat it as
            one — those rows still carry their V/D/J calls, they just could not have a junction derived. It is the
            honest-absence contract in action, and you will meet it again when you filter output.
          </p>
          <p>
            <code>--input</code> accepts FASTA, FASTQ, CSV/TSV or TXT (optionally <code>.gz</code>, or <code>-</code> for stdin). For a <strong>table</strong>, tell it which columns to read — otherwise it cannot find your sequences:
          </p>
          <CodeBlock code={`alignair predict --input reads.csv --out out.tsv --model alignair-igh-human \\\n  --sequence-column seq --id-column read_id`} />
          <p>
            No reads handy? <code>alignair demo</code> runs the whole pipeline offline on simulated data (the demo model is not accurate — it only proves the pipeline works).
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Read the output: every field, explained",
      body: () => (
        <>
          <p>
            Each aligned read becomes one row of a standard AIRR TSV. Click any field below to see what it means:
          </p>
          <AirrFieldsWidget />
          <p>
            When a <code>*_call_set</code> holds more than one allele, the model did not settle on one — do not report
            that read at allele resolution.
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Orientation and rev_comp",
      body: () => (
        <>
          <p>
            AlignAIR detects orientation itself. For a reverse-complement read the emitted <code>sequence</code> is the <em>original</em> query and <code>rev_comp=T</code>, so the coordinates apply to <code>RC(sequence)</code>. Toggle the strand to see what moves:
          </p>
          <RevcompWidget />
          <Callout kind="warning" title="This is the AIRR / IgBLAST convention">
            A <code>rev_comp=T</code> row does <strong>not</strong> mean the coordinates are wrong or that they apply to the string you see — they apply to its reverse complement. This trips up almost everyone once.
          </Callout>
        </>
      ),
    },
    {
      kind: "mcq",
      prompt: () => (
        <p>
          A row has <code>v_call=IGHV3-23*01</code> but a blank <code>productive</code> and <code>airr_assembly_status=partial</code>. What is the right reading?
        </p>
      ),
      options: [
        "The read is non-productive",
        "Productivity could not be derived for this read; blank means unknown, not False",
        "The prediction failed and the row should be deleted",
      ],
      answer: 1,
      explanation: () => (
        <p>
          <code>productive</code> is a derived fact (in-frame and no stop codon). On a partial record it can be underivable, and AlignAIR leaves it blank — unknown, not <code>F</code>. The neural head’s advisory guess is kept separately in <code>productive_prediction</code>.
        </p>
      ),
    },
  ],
};

const modelsReferences: Lesson = {
  id: "predicting/models-references",
  slug: "models-references",
  track: "predicting",
  title: "Models, references & dataconfigs",
  summary: "Where a model’s callable alleles come from, and what a dataconfig is.",
  minutes: 7,
  steps: [
    {
      kind: "explain",
      title: "A model carries its own reference",
      body: () => (
        <>
          <p>
            Every AlignAIR model has a <strong>fixed reference</strong> baked in: the exact list of germline V/D/J alleles it was trained to recognise. That catalog is embedded in the <code>.alignair</code> file and fingerprinted, so it can never drift from the weights.
          </p>
          <p>
            The V/D/J classification heads have one output slot per allele in that catalog — so the reference literally <em>is</em> the set of alleles the model can call. Almost everything else about a model follows from this one fact.
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "What is a dataconfig?",
      body: () => (
        <>
          <p>
            You don't assemble that reference by hand. AlignAIR uses <a href="https://github.com/MuteJester/GenAIRR" target="_blank" rel="noreferrer" className="text-brand-600 underline">GenAIRR</a> to generate labeled training sequences. A <strong>dataconfig</strong> is a named, built-in germline configuration from GenAIRR — for example <code>HUMAN_IGH_OGRDB</code>.
          </p>
          <p>
            With <code>--dataconfig</code>, you select one of these built-in configs, which defines the germline alleles, conserved junction anchors, and recombination distributions (such as exonucleolytic trimming lengths and N-nucleotide insertion length distributions) used by the simulator.
          </p>
          <p>List the built-in dataconfigs and references your install ships with:</p>
          <CodeBlock code="alignair reference list" />
          <p>
            Alternatively, when training on custom references, you can pass your own sequences with <code>--v-fasta</code>, <code>--d-fasta</code>, and <code>--j-fasta</code>. AlignAIR automatically builds a compatible training reference and discovers the required V/J anchors where possible. Learn more at the <a href="https://mutejester.github.io/GenAIRR/" target="_blank" rel="noreferrer" className="text-brand-600 underline">official GenAIRR documentation</a>.
          </p>
          <Callout kind="note" title="Why anchors matter">
            A dataconfig or custom FASTA provides the necessary anchors to locate the conserved residues. A plain YAML/JSON genotype does <strong>not</strong> carry anchors — which is exactly why a genotype can only <em>subset</em> an existing reference, while a brand-new reference needs a dataconfig or anchored FASTAs.
          </Callout>
        </>
      ),
    },
    {
      kind: "explain",
      title: "How --model resolves",
      body: () => (
        <>
          <p>At predict time, <code>--model</code> accepts three kinds of thing:</p>
          <ul className="list-disc pl-6 space-y-1.5 my-4">
            <li>a <strong>catalog id</strong> like <code>alignair-igh-human</code> — downloaded, hash-verified and cached on first use (no login);</li>
            <li>a local <strong><code>.alignair</code> file</strong>, e.g. <code>runs/igh/bundle/model.alignair</code>;</li>
            <li>an <strong><code>org/name</code> Hugging Face repo id</strong>. Pin an exact build with <code>id@version</code>.</li>
          </ul>
          <CodeBlock code="alignair models list        # the live catalog + install status" />
          <p>
            Pick a model whose reference covers your locus and species. If none does, you train one — see the Reference pages.
          </p>
        </>
      ),
    },
    {
      kind: "mcq",
      prompt: () => (
        <p>
          You work on a species AlignAIR ships no dataconfig or pretrained model for. What is the right move?
        </p>
      ),
      options: [
        "Use the closest human model — it adapts at inference",
        "Train a new model from your own anchored germline FASTAs, producing a new fixed reference",
        "Add the species’ alleles with a --genotype file at predict time",
      ],
      answer: 1,
      explanation: () => (
        <p>
          A model can only call alleles in its embedded reference, and a genotype can only <em>subset</em> that reference. A new species is a new allele universe, so you train a model — from a dataconfig if one exists, otherwise your own anchored FASTAs.
        </p>
      ),
    },
  ],
};

const genotype: Lesson = {
  id: "predicting/genotype",
  slug: "genotype",
  track: "predicting",
  title: "Constrain to a donor genotype",
  summary: "Subset the reference to one donor at inference — no retraining.",
  minutes: 8,
  steps: [
    {
      kind: "explain",
      title: "Why constrain to a genotype?",
      body: () => (
        <>
          <p>
            A model can call any allele in its reference — but a given <strong>donor</strong> only carries a fraction of them. If you know (or have inferred) the donor’s genotype, you can tell AlignAIR to consider only that donor’s alleles.
          </p>
          <p>
            This is a pure <strong>inference-time</strong> switch. It narrows the callable set, which removes impossible calls and tightens ambiguous candidate sets — with <strong>no retraining</strong> and no new model.
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "How to pass a genotype",
      body: () => (
        <>
          <p>
            Supply the genotype as YAML or JSON. Calls are hard-restricted to that donor’s alleles. The file must map gene names to a list of allowed alleles:
          </p>
          <CodeBlock
            lang="yaml"
            code={`# donor.yaml\nIGHV3-23:\n  - IGHV3-23*01\n  - IGHV3-23*04\nIGHV1-2:\n  - IGHV1-2*02`}
          />
          <p>Pass the file path to <code>--genotype</code>:</p>
          <CodeBlock code={`alignair predict --input reads.fasta --out out.tsv \\\n  --model alignair-igh-human --genotype donor.yaml`} />
          <p>Choose how the constraint is applied with <code>--genotype-method</code>:</p>
          <div style={{ margin: "20px 0", overflowX: "auto", border: "1px solid #eae9f1", borderRadius: "12px" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13.5px" }}>
              <thead>
                <tr style={{ background: "#f7f6fb", borderBottom: "1px solid #eae9f1" }}>
                  <th style={{ textAlign: "left", padding: "11px 14px", fontWeight: 600 }}>Method</th>
                  <th style={{ textAlign: "left", padding: "11px 14px", fontWeight: 600 }}>Effect</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: "1px solid #f0eff5" }}>
                  <td style={{ padding: "11px 14px" }}><code>mask</code> (default)</td>
                  <td style={{ padding: "11px 14px" }}>Zero out non-donor alleles, then take the top surviving call. The safest, most literal option.</td>
                </tr>
                <tr style={{ borderBottom: "1px solid #f0eff5" }}>
                  <td style={{ padding: "11px 14px" }}><code>softmax</code></td>
                  <td style={{ padding: "11px 14px" }}>Re-run the softmax over only the donor’s alleles.</td>
                </tr>
                <tr style={{ borderBottom: "1px solid #f0eff5" }}>
                  <td style={{ padding: "11px 14px" }}><code>renormalize</code></td>
                  <td style={{ padding: "11px 14px" }}>Rescale the surviving probabilities so they sum to 1.</td>
                </tr>
                <tr style={{ borderBottom: "1px solid #f0eff5" }}>
                  <td style={{ padding: "11px 14px" }}><code>redistribute</code></td>
                  <td style={{ padding: "11px 14px" }}>Spread the masked probability mass back across the donor set.</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p>
            Partial genotypes are fine — constrain just the V, or only some genes, and leave the rest unconstrained.
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "See it in action",
      body: () => (
        <>
          <p>
            Toggle between the model’s full reference and a donor genotype and watch the callable set shrink. Then try adding an allele the model never learned:
          </p>
          <GenotypeWidget />
        </>
      ),
    },
    {
      kind: "explain",
      title: "A genotype can only subset — never add",
      body: () => (
        <>
          <p>
            This is the rule that matters: a genotype is always a <strong>subset</strong> of the trained reference. An allele the model was never trained on is <strong>not</strong> callable, and supplying one in a genotype fails immediately with a clear error — it is never silently dropped or mis-indexed.
          </p>
          <Callout kind="warning" title="To add alleles, train">
            Adding a novel allele, a new locus, or a new species changes the allele universe, so it requires <strong>training a new model</strong> with a new, versioned reference — not a genotype.
          </Callout>
        </>
      ),
    },
    {
      kind: "mcq",
      prompt: () => (
        <p>
          Constraining to a donor genotype improves your results mainly because it —
        </p>
      ),
      options: [
        "adds the donor’s private alleles to the model",
        "removes alleles the donor cannot carry, so impossible calls disappear and ambiguous *_call_sets shrink",
        "makes the model run faster",
      ],
      answer: 1,
      explanation: () => (
        <p>
          A genotype subsets the callable alleles to those the donor actually has. That eliminates impossible calls and often collapses a multi-allele <code>*_call_set</code> down to a single confident call — at inference, with no retraining. It never adds alleles.
        </p>
      ),
    },
  ],
};

const interpreting: Lesson = {
  id: "predicting/interpreting",
  slug: "interpreting",
  track: "predicting",
  title: "Interpreting the output",
  summary: "Call sets, assembly status, productivity, and the minimum filter.",
  minutes: 9,
  steps: [
    {
      kind: "explain",
      title: "One row per read, honest by design",
      body: () => (
        <>
          <p>
            <code>alignair predict</code> writes a standard <strong>AIRR rearrangement TSV</strong> — one row per read. Its guiding principle is <em>honest absence</em>: a field is left <strong>blank</strong> when it cannot be derived reliably, rather than filled with a guess. A schema-valid row is not the same as a fully-derived one.
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Calls and call sets",
      body: () => (
        <>
          <p>
            Each row has <code>v_call</code> / <code>d_call</code> / <code>j_call</code> — the top allele per gene. But
            when a read genuinely cannot distinguish alleles (a short fragment, say), AlignAIR does not hide it behind a
            single guess: <code>*_call_set</code> carries the model's <strong>candidate set</strong>:
          </p>
          <CodeBlock lang="text" code={`v_call        IGHV3-23*01\nv_call_set    IGHV3-23*01,IGHV3-23*04`} />
          <p>
            Read it for exactly what it is: every allele the model scored at <code>p &gt;= 0.5</code>, ranked, and{" "}
            <strong>capped at three</strong>, falling back to the top-1 call when nothing clears the bar. So a
            multi-member set means the model did not settle on one allele — do not report that read at allele
            resolution. But it is <em>not</em> an exhaustive list of the alleles the read is compatible with (the cap
            can truncate it), and a one-member set is not automatically confident (it may just be the fallback). A donor{" "}
            <code>--genotype</code> is what actually narrows these.
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Every field, explained",
      body: () => (
        <>
          <p>Click any field to see how it is derived and how to read it:</p>
          <AirrFieldsWidget />
        </>
      ),
    },
    {
      kind: "explain",
      title: "Assembly status — complete, partial, failed",
      body: () => (
        <>
          <p><code>airr_assembly_status</code> tells a fully-derived record apart from a thin one:</p>
          <ul className="list-disc pl-6 space-y-1.5 my-4">
            <li><code>complete</code> — calls <strong>and</strong> the assembled products (junction, regions) are all present.</li>
            <li><code>partial</code> — valid calls, but a product such as the junction could not be assembled; <code>airr_assembly_reason</code> says why.</li>
            <li><code>failed</code> — an exception; only the light fields are emitted.</li>
          </ul>
          <p>
            And remember: a blank <code>productive</code> means <strong>unknown</strong>, not <code>False</code>. Productivity is <em>derived</em> from <code>vj_in_frame</code> and <code>stop_codon</code>; on a partial record it can be underivable. The neural head’s advisory guess lives separately in <code>productive_prediction</code>.
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Filter before you analyse",
      body: () => (
        <>
          <p>Before junction or clonotype analysis, keep the complete, high-quality records. Toggle the filter to see what survives:</p>
          <RecFilterWidget />
          <p>A recommended minimum filter in pandas:</p>
          <CodeBlock
            lang="python"
            code={`import pandas as pd\n\ntable = pd.read_csv("out.tsv", sep="\\t")\nusable = table[\n    (table["airr_assembly_status"] == "complete")\n    & ~table["segmentation_low_quality"].fillna(False).astype(bool)\n]`}
          />
        </>
      ),
    },
    {
      kind: "mcq",
      prompt: () => (
        <p>
          A row has <code>v_call=IGHV3-23*01</code>, a <code>v_call_set</code> with three alleles, and <code>airr_assembly_status=partial</code>. How should you use it?
        </p>
      ),
      options: [
        "As a confident single-allele, productive call",
        "Below allele resolution: the allele is ambiguous (3-member set) and the record is only partial — keep it out of junction-level analysis",
        "Delete it — partial rows are errors",
      ],
      answer: 1,
      explanation: () => (
        <p>
          Three alleles in the set means the model did not settle on one, so do not report this read at allele
          resolution. Note that three is the cap, so the true ambiguity may be wider than what you see — and check
          whether the members actually share a gene before you roll the row up to gene or family level; they need not.{" "}
          <code>partial</code> means the junction and other products were not assembled, so it must not feed junction or
          clonotype analysis. Partial and failed rows are still emitted, calls included, so this row is usable — just
          not at allele or junction resolution.
        </p>
      ),
    },
  ],
};

export const predictingTrack: Track = {
  slug: "predicting",
  title: "Predicting",
  description: "Run a pretrained model, constrain it to a donor genotype, and read every field of the AIRR output.",
  icon: Terminal,
  accent: "from-brand-500 to-brand-700",
  lessons: [firstPrediction, modelsReferences, genotype, interpreting],
};
