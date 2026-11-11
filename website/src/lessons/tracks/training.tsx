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
            <strong>GenAIRR simulator</strong>, which synthesizes V(D)J rearrangements and then corrupts them along a{" "}
            <strong>curriculum</strong> that ramps from clean, full-length reads to heavily degraded ones. The next
            lesson takes that curriculum apart knob by knob.
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
            code={`quick     300 steps,     batch 16,  validate every 100     smoke test / CI only\ndesktop   50,000 steps,  batch 64,  validate every 2,000   a single workstation GPU\nfull      300,000 steps, batch 128, validate every 5,000   a production run`}
          />
          {/* PENDING-V4-NUMBERS: AlignAIR 4 training step times (CPU and CUDA) await a re-timing. */}
          <p>
            <strong>Estimating wall-clock:</strong> multiply the preset&apos;s step count by the measured seconds-per-step
            for your hardware. Both GenAIRR generation and neural optimisation contribute to each step, and hardware and
            reference size matter. AlignAIR 4 step timings are pending a re-timing (results pending, see{" "}
            <em>Speed &amp; throughput</em> in the reference), so time a short run on your own machine.{" "}
            <code>--plan</code> reports the resolved reference, model size and schedule without training — but it does{" "}
            <em>not</em> estimate time or memory.
          </p>
          <Callout kind="warning" title="quick is not a scientific model">
            The <code>quick</code> preset (and <code>alignair demo</code>, which trains 5 steps by default) only proves
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
            The run directory holds the resumable checkpoints (<code>model.alignair</code>,{" "}
            <code>model.step&lt;N&gt;.alignair</code> and, when validation ran, <code>model.best.alignair</code>). They
            carry pickle sections, so they are for resuming, not sharing. <code>alignair train</code> then exports the
            best (or last) checkpoint into a self-contained, pickle-free <code>bundle/</code>:{" "}
            <code>model.alignair</code> (weights + embedded reference), <code>model_card.md</code>,{" "}
            <code>reference_manifest.json</code> (allele counts, anchor coverage, fingerprints), and{" "}
            <code>validation_report.json</code> (per-task metrics on a fixed held-out stream).
          </p>
          <p>
            The headline metrics are <code>v/d/j_allele_top1</code> (the top call is inside the truth set),{" "}
            <code>scan_v/d/j_top1</code> (the same for the bank scan&apos;s top allele, the score that picks the best
            checkpoint) and <code>*_seg_mae</code> (segment start/end error in nucleotides).
          </p>
          <Callout kind="note" title="Why D accuracy is structurally lower">
            Expect <code>d_allele_top1</code> to sit below V and J. D segments are short and lose bases to nucleolytic
            trimming at both ends during junction assembly, so many reads carry too little D germline to tell its
            alleles apart — an ambiguity no alignment tool can fully resolve.
          </Callout>
        </>
      ),
    },
    {
      kind: "mcq",
      prompt: () => (
        <p>
          Your custom run finishes with a <code>v_allele_top1</code> far below what a <code>desktop</code> run on a
          built-in reference of the same locus reaches. What is the most likely first thing to check?
        </p>
      ),
      options: [
        "Nothing — custom references always validate low",
        "Anchor coverage and step count: a low V top-1 points to too few steps or a reference with poor anchor coverage",
        "The output TSV column order",
      ],
      answer: 1,
      explanation: () => (
        <>
          {/* PENDING-V4-NUMBERS: AlignAIR 4 reference values for V top-1 after desktop/full training. */}
          <p>
            A V top-1 far below a comparable built-in run usually means undertraining or a reference the builder could
            not anchor well — check the reference build warnings and consider the <code>full</code> preset. (AlignAIR 4
            reference values for a trained model are pending.)
          </p>
        </>
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
            Read length and every corruption come from the simulation itself, and the library flanks added around a
            read move its coordinates with it, so a read&apos;s coordinates always stay consistent with the sequence
            you actually see.
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
            TCR locus — not &ldquo;low&rdquo;, zero — and GenAIRR will in fact refuse a mutation request on TCR
            reference data.
          </p>
        </>
      ),
    },
    {
      kind: "explain",
      title: "The mix spans read shapes",
      body: () => (
        <>
          <p>
            A second axis runs alongside the corruption ramp: the training mix interleaves several{" "}
            <strong>read shapes</strong>, so the model never sees only one kind of read.
          </p>
          <ul className="list-disc pl-6 space-y-1.5 my-4">
            <li><strong>Full-length</strong> reads at three curriculum positions (0.3, 0.6 and 0.9), three quarters of them drawn under GenAIRR&apos;s productive-only selection.</li>
            <li><strong>V-anchored</strong> amplicons — keep the 5' end, lose 150–350 nt of the 3'.</li>
            <li><strong>J-anchored</strong> amplicons — keep the 3'/J end, lose 150–350 nt of the 5'.</li>
            <li><strong>Both-ends fragments</strong>, losing 70–175 nt at each end.</li>
            <li>A <strong>heavily-mutated full-length</strong> stream (SHM rate 0.25) — the hardest V regime (IG only).</li>
          </ul>
          <p>
            On top of every shape, <strong>library flanks</strong> (a leader or a masked primer before V, constant
            region after J) are added to part of the reads, and each read is put in one of four orientations (forward,
            reverse complement, complement, reverse) at random. Fragments, flanked reads and every orientation are part
            of training rather than out-of-distribution surprises — though a short 3'-anchored fragment still holds
            little V sequence, so its V call has less to go on.
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
            Every bundle ships a report scored on a fixed held-out stream. It is per-task, not one number (values
            omitted here):
          </p>
          <CodeBlock
            lang="text"
            code={`v_allele_top1   d_allele_top1   j_allele_top1\nscan_v_top1     scan_d_top1     scan_j_top1\nv_seg_mae       d_seg_mae       j_seg_mae\nparse_v_start_exact ... parse_j_end_exact\nanchor_v_exact  anchor_j_exact  d_presence_acc  orientation_acc\nmutation_mae    indel_mae       shm_brier\ncoord_v_top1 ...                bank_v_size ...`}
          />
          <ul className="list-disc pl-6 space-y-1.5 my-4">
            <li><code>*_allele_top1</code> — the top call is inside the truth set. The headline accuracy. Higher is better.</li>
            <li><code>scan_*_top1</code> — the same for the bank scan&apos;s top allele. Their mean picks the best checkpoint.</li>
            <li><code>*_seg_mae</code> and <code>parse_*_exact</code> — segment boundary error in nucleotides (lower is better) and the share of boundaries placed exactly.</li>
            <li><code>anchor_v_exact</code>, <code>anchor_j_exact</code> — the V Cys and J Trp/Phe codons found exactly; the junction and productivity are derived from them.</li>
            <li><code>d_presence_acc</code>, <code>orientation_acc</code> — accuracy of the D presence and orientation heads.</li>
            <li><code>mutation_mae</code>, <code>indel_mae</code> — error of the SHM-rate and indel-count estimates.</li>
          </ul>
          <p>
            Best-checkpoint selection uses the mean of <code>scan_v/d/j_top1</code>, so that trio is what training
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
          {/* PENDING-V4-NUMBERS: AlignAIR 4 acceptance values (V/J/D top-1, segmentation MAE, orientation accuracy). */}
          <Callout kind="note" title="Results pending for AlignAIR 4">
            Reference values for a trained AlignAIR 4 model are pending a fresh evaluation. Until they are published,
            judge a custom run against a run on a built-in reference of the same locus with the same preset.
          </Callout>
          <p>Guidelines, not guarantees — the right bar depends on your locus and reference:</p>
          <ul className="list-disc pl-6 space-y-1.5 my-4">
            <li><strong>V and J top-1</strong>: much lower than the comparison run means undertraining or a reference with poor anchor coverage.</li>
            <li><strong>D top-1</strong>: expected lower than V and J — judge it against the comparison run, not an absolute bar.</li>
            <li><strong>Anchors</strong>: low <code>anchor_*_exact</code> means junctions will be missing or misplaced.</li>
            <li><strong>Orientation accuracy</strong>: a read put in the wrong orientation is wrong everywhere downstream, so look at it before the others.</li>
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
          Your custom reference trains to a high <code>v_allele_top1</code>, but most reads come back with an empty{" "}
          <code>junction</code>. Where do you look first?
        </p>
      ),
      options: [
        "Train longer — the model must still be underfit",
        "Anchor coverage in reference_manifest.json: without the conserved V/J anchors the junction cannot be placed, however well the model calls alleles",
        "The --columns preset — it is dropping the junction column",
      ],
      answer: 1,
      explanation: () => (
        <p>
          Allele calling and junction extraction are different jobs. A high V top-1 says the model learned the catalog
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
