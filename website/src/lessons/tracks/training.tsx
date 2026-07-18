import { GraduationCap } from "lucide-react";
import { CodeBlock, Callout } from "../../components/ui";
import { CurriculumWidget } from "../../components/widgets";
import type { Lesson, Track } from "../types";

const trainModel: Lesson = {
  id: "training/train-a-model",
  slug: "train-a-model",
  track: "training",
  title: "Train a model for your reference",
  summary: "Built-in dataconfigs vs custom FASTAs, presets, and reading the validation report.",
  minutes: 8,
  steps: [
    {
      kind: "explain",
      title: "Two ways to specify the reference",
      body: () => (
        <>
          <p>Train when you need a reference AlignAIR does not ship. Pick exactly one mode — a built-in dataconfig:</p>
          <CodeBlock code={`alignair train --dataconfig HUMAN_IGH_OGRDB --out runs/igh --preset desktop`} />
          <p>or your own germline FASTAs:</p>
          <CodeBlock
            code={`alignair train --v-fasta v.fasta --j-fasta j.fasta --d-fasta d.fasta \\\n  --chain-type BCR_HEAVY --out runs/my_ref --preset desktop`}
          />
          <p>
            D-bearing chains (<code>BCR_HEAVY</code>, <code>TCR_BETA</code>, <code>TCR_DELTA</code>) require{" "}
            <code>--d-fasta</code>; the others must not be given one.
          </p>
          <Callout kind="note" title="How the model learns: GenAIRR simulation curriculum">
            AlignAIR does not train on a static dataset. Instead, it streams simulated reads on-the-fly from the{" "}
            <strong>GenAIRR simulator</strong>, which synthesizes realistic V(D)J rearrangements and then corrupts them
            along a <strong>curriculum</strong> that ramps from clean, full-length reads to heavily degraded ones. The
            next lesson takes that curriculum apart knob by knob.
          </Callout>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Presets set the scale",
      body: () => (
        <>
          <p>A preset picks steps, batch size and validation interval:</p>
          <CodeBlock
            lang="text"
            code={`quick     300 steps      smoke test / CI only\ndesktop   50,000 steps   a single workstation GPU\nfull      300,000 steps  a production, paper-grade run`}
          />
          <p>
            <strong>Estimating wall-clock:</strong> multiply the preset's step count by the measured seconds-per-step for
            your hardware — see <em>Speed &amp; throughput</em> in the reference for a reproducible measured example. Both
            GenAIRR generation and neural optimisation contribute to each step; on the measured workstation, CUDA was
            about 9.6x faster than CPU for the <code>desktop</code> batch. Hardware and reference size still matter, so
            treat the published figure as a planning baseline rather than a promise. <code>--plan</code> reports the
            resolved reference, model size and schedule without training — but it does <em>not</em> estimate time or
            memory.
          </p>
          <Callout kind="warning" title="quick is not a scientific model">
            The <code>quick</code> preset (and <code>alignair demo</code>) train for a few hundred steps only to prove
            the pipeline runs. Their calls are not accurate. Use <code>desktop</code> or <code>full</code> for a model
            you will make calls with.
          </Callout>
        </>
      ),
    },
    {
      kind: "explain",
      title: "What training writes",
      body: () => (
        <>
          <p>
            You get a self-contained bundle: <code>model.alignair</code> (weights + embedded reference),{" "}
            <code>model_card.md</code>, <code>reference_manifest.json</code> (allele counts, anchor coverage,
            fingerprints), and <code>validation_report.json</code> (per-task metrics on a held-out stream).
          </p>
          <p>
            The headline metrics are <code>v/d/j_allele_top1</code> (top call inside the truth set) and{" "}
            <code>*_seg_mae</code> (coordinate error in nucleotides).
          </p>
          <Callout kind="note" title="Why D accuracy is structurally lower">
            You will notice that <code>d_allele_top1</code> accuracy is always significantly lower than V or J. Biologically, D segments are extremely short (often 10–20 bp) and undergo extensive nucleolytic chewing (terminal deletions) during junction assembly, meaning they carry high inherent ambiguity that even perfect alignment tools cannot fully resolve.
          </Callout>
        </>
      ),
    },
    {
      kind: "mcq",
      prompt: () => (
        <p>
          Your custom run finishes with <code>v_allele_top1 = 0.58</code> on the held-out stream. What is the most
          likely first thing to check?
        </p>
      ),
      options: [
        "Nothing — 0.58 is a great V accuracy",
        "Anchor coverage and step count: a low V top-1 points to too few steps or a reference with poor anchor coverage",
        "The output TSV column order",
      ],
      answer: 1,
      explanation: () => (
        <p>
          V top-1-in-set should be well above 0.9 for a trained <code>desktop</code>/<code>full</code> model. A value that
          low usually means undertraining or a reference the builder could not anchor well — check the reference build
          warnings and consider the <code>full</code> preset.
        </p>
      ),
    },
    {
      kind: "mcq",
      prompt: () => (
        <p>Why is a strong held-out <code>validation_report.json</code> necessary but not sufficient?</p>
      ),
      options: [
        "Because the report is randomly generated",
        "Because it scores the model on the same GenAIRR simulation family it trained on — real experimental reads still need their own evaluation",
        "Because validation always overfits",
      ],
      answer: 1,
      explanation: () => (
        <p>
          The held-out stream measures fit to the simulation distribution. It is a strong signal, but you should still
          benchmark on reads representative of your actual data before trusting a model.
        </p>
      ),
    },
  ],
};

const simulation: Lesson = {
  id: "training/simulation",
  slug: "simulation",
  track: "training",
  title: "How the training data is made",
  summary: "The GenAIRR curriculum, knob by knob — and why TCR loci get zero SHM.",
  minutes: 7,
  steps: [
    {
      kind: "explain",
      title: "No dataset — a simulator",
      body: () => (
        <>
          <p>
            AlignAIR never trains on a fixed corpus. Every batch is generated on the fly by the{" "}
            <strong>GenAIRR simulator</strong>, which recombines real germline segments and then corrupts the result.
            That matters for one reason above all: the simulator <strong>knows the truth</strong> it started from, so
            every read arrives with exact V/D/J identity and coordinates — supervision you could never get from real
            reads, at unlimited volume.
          </p>
          <p>
            Read length and every corruption come from the simulation itself, not from post-hoc cropping, so a read's
            coordinates always stay consistent with the sequence you actually see.
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "The curriculum, knob by knob",
      body: () => (
        <>
          <p>
            Difficulty is not fixed — it <strong>ramps</strong>. A curriculum position drives every corruption knob
            together, from near-clean reads to badly degraded ones. Drag it, and switch the locus:
          </p>
          <CurriculumWidget />
          <p>
            The locus toggle is the part people miss. <strong>T-cells lack AID</strong>, the enzyme behind somatic
            hypermutation, so TCR receptors do not hypermutate. AlignAIR sets the SHM rate to <strong>zero</strong> for a
            TCR locus — not "low", zero — and GenAIRR will in fact refuse a mutation request on TCR reference data.
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Every batch spans read shapes",
      body: () => (
        <>
          <p>
            A second axis runs alongside the corruption ramp: each batch mixes <strong>amplicon shapes</strong>, so the
            model never sees only one kind of read.
          </p>
          <ul className="list-disc pl-6 space-y-1.5 my-4">
            <li><strong>Full-length</strong> reads across the corruption ramp (rehearsal + noise).</li>
            <li><strong>V / framework-anchored</strong> amplicons — keep the 5' end, trim the 3'.</li>
            <li><strong>J-anchored</strong> amplicons — keep the 3'/J end, trim the 5'.</li>
            <li><strong>Both-ends fragments</strong>, bounded so a read is never trimmed to nothing.</li>
            <li>A <strong>heavily-mutated full-length</strong> stream — the hardest V regime (IG only).</li>
          </ul>
          <p>
            This is why the pretrained models handle fragments and arbitrary orientation rather than treating them as
            out-of-distribution: those reads were in <em>every</em> batch. V calling still weakens on the shortest
            fragments - there is only so much V gene in a 3'-anchored read - but D, J, and orientation hold up.
          </p>
        </>
      ),
    },
    {
      kind: "mcq",
      prompt: () => (
        <p>
          You train a TRB (TCR beta) model. What somatic-hypermutation load do its training reads carry?
        </p>
      ),
      options: [
        "The same ramp as IGH — SHM is a property of the simulator, not the locus",
        "Zero — T-cells lack AID, so a TCR locus is capped to 0 regardless of the curriculum position",
        "A low but non-zero rate, to keep the model robust",
      ],
      answer: 1,
      explanation: () => (
        <p>
          SHM is a property of the <em>locus</em>. A TCR locus is always capped to zero, while an IG locus in the same
          run keeps its full ramp — the cap is resolved per reference, not once per run. The model card records the
          effective cap for each locus, so you can always see what a model actually trained on.
        </p>
      ),
    },
  ],
};

const trustModel: Lesson = {
  id: "training/trust",
  slug: "trust",
  track: "training",
  title: "Judge whether to trust your model",
  summary: "Read validation_report.json, check anchor coverage, and know what the numbers cannot tell you.",
  minutes: 8,
  steps: [
    {
      kind: "explain",
      title: "Read validation_report.json",
      body: () => (
        <>
          <p>
            Every bundle ships a report scored on a fixed held-out stream. It is per-task, not one number:
          </p>
          <CodeBlock
            lang="json"
            code={`{\n  "v_allele_top1": 0.96,   "d_allele_top1": 0.71,   "j_allele_top1": 0.98,\n  "v_seg_mae": 1.4,        "d_seg_mae": 3.2,        "j_seg_mae": 1.1,\n  "orientation_acc": 1.0,  "productive_acc": 0.97,\n  "mutation_mae": 0.004,   "indel_mae": 0.12\n}`}
          />
          <ul className="list-disc pl-6 space-y-1.5 my-4">
            <li><code>*_allele_top1</code> — the top call is inside the truth equivalence set. The headline accuracy. Higher is better.</li>
            <li><code>*_seg_mae</code> — segment start/end error in nucleotides. Lower is better.</li>
            <li><code>orientation_acc</code>, <code>productive_acc</code> — head accuracy; <code>chain_type_acc</code> appears for multi-locus models.</li>
            <li><code>mutation_mae</code>, <code>indel_mae</code> — error of the SHM-rate and indel-count estimates.</li>
          </ul>
          <p>
            Best-checkpoint selection uses the mean of <code>v/d/j_allele_top1</code>, so that trio is what training
            optimises toward.
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "Rough acceptance criteria",
      body: () => (
        <>
          <p>Guidelines, not guarantees — the right bar depends on your locus and reference:</p>
          <ul className="list-disc pl-6 space-y-1.5 my-4">
            <li><strong>V and J top-1</strong>: well above 0.9 for a <code>desktop</code>/<code>full</code> run. Much lower means undertraining or a reference with poor anchor coverage.</li>
            <li><strong>D top-1</strong>: expected lower — judge it against the pretrained model for the same locus, not an absolute bar.</li>
            <li><strong>Segmentation MAE</strong>: single-digit nucleotides.</li>
            <li><strong>Orientation accuracy</strong>: near 1.0.</li>
          </ul>
          <Callout kind="note" title="Check anchor coverage before blaming the model">
            <code>reference_manifest.json</code> reports, per gene, how many alleles the builder could anchor. A low
            <code> anchored</code> fraction means many reads will have <strong>no junction at all</strong> — that is a
            reference problem, not a training problem, and no amount of extra steps will fix it.
          </Callout>
        </>
      ),
    },
    {
      kind: "explain",
      title: "The card records what actually happened",
      body: () => (
        <>
          <p>
            A model is only reproducible if it records the distribution it really trained on. The card does — including
            the <strong>effective SHM cap per locus</strong>, which is not the same as what you requested:
          </p>
          <CodeBlock
            lang="text"
            code={`- training steps: 50000\n- effective SHM cap per locus: HUMAN_IGH_OGRDB=uncapped, HUMAN_TCRB_IMGT=0.0\n- heavy-SHM stream (requested): 0.25`}
          />
          <p>
            Read that pair together: the heavy-SHM stream was <em>requested</em> at 0.25, and it applied to IGH — but TRB
            was capped to zero, so that stream carried no mutations for TRB. The resumable checkpoint and the
            distributable bundle carry the identical record, so a published model can always be traced back to its real
            training distribution.
          </p>
        </>
      ),
    },
    {
      kind: "mcq",
      prompt: () => (
        <p>
          Your custom reference trains to <code>v_allele_top1 = 0.94</code>, but most reads come back with an empty{" "}
          <code>junction</code>. Where do you look first?
        </p>
      ),
      options: [
        "Train longer — 0.94 means the model is still underfit",
        "Anchor coverage in reference_manifest.json: without the conserved V/J anchors the junction cannot be placed, however well the model calls alleles",
        "The --columns preset — it is dropping the junction column",
      ],
      answer: 1,
      explanation: () => (
        <p>
          Allele calling and junction extraction are different jobs. A 0.94 V top-1 says the model learned the catalog
          fine; an empty junction means the reference could not be anchored, so there is no Cys/Trp-Phe to measure the
          CDR3 between. That is honest absence, and the fix is the reference, not more steps.
        </p>
      ),
    },
  ],
};

export const trainingTrack: Track = {
  slug: "training",
  title: "Training",
  description: "Train a model for your own reference or species, and judge whether to trust it.",
  icon: GraduationCap,
  accent: "from-emerald-500 to-teal-600",
  lessons: [trainModel, simulation, trustModel],
};
