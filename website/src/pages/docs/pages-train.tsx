import type { ReactNode } from "react";
import { CodeBlock, Callout } from "../../components/ui";
import CodeTabs from "../../components/CodeTabs";
import { DocTable, DocLink, H2, H3, type DocPage } from "./doc-kit";
import OptionTable from "../../components/data/OptionTable";

const SECTION = "Train or fine-tune";

const OGRDB_SETS = "https://ogrdb.airr-community.org/germline_sets/Homo%20sapiens";
const IMGT_VQUEST_SETS = "https://www.imgt.org/vquest/refseqh.html";
const IMGT_GENEDB = "https://www.imgt.org/genedb/";

function Ext({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer" className="text-brand-600 underline">
      {children}
    </a>
  );
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* Do I need to train?                                                                                                */
/* ------------------------------------------------------------------------------------------------------------------ */
const doINeedToTrain: DocPage = {
  slug: "do-i-need-to-train",
  title: "Do I need to train?",
  section: SECTION,
  lead: "Decide whether to give a pretrained model your germline set or to train a model of your own, and what training costs.",
  body: () => (
    <>
      <p>
        Most germline sets need no training. A model reads its alleles from the germline set it is given, so a
        pretrained model calls the alleles of your set as soon as you bind it with <code>--reference</code>. Training
        makes a new model, and it is for what binding cannot do: another species, or a locus no model covers.
      </p>

      <H2 id="bind-or-train">Bind your germline set, or train</H2>
      <DocTable
        head={["Your case", "What to do", "Training?"]}
        rows={[
          [
            "Human reads of IGH, IGK, IGL, TRA, TRB, TRD or TRG, with the model's own germline set",
            <>
              Use the pretrained model (<DocLink to="choose-a-model">Choose a model</DocLink>).
            </>,
            "No",
          ],
          [
            "The same locus and species, but your germline set differs: novel alleles, a curated set, another release, IMGT's set for an IG locus",
            <>
              Bind it: <code>--reference germline.fasta</code> (
              <DocLink to="own-reference">Use your own germline reference</DocLink>).
            </>,
            "No",
          ],
          [
            "Calls only from one donor's alleles",
            <>
              <code>--genotype</code> (<DocLink to="donor-genotype">Align with a donor genotype</DocLink>).
            </>,
            "No",
          ],
          [
            "Another species, a locus no model covers, or a germline set far from the model's",
            <>
              Train a model for it (<DocLink to="train">Train a model for your germline set</DocLink>), or fine-tune a
              pretrained one (<DocLink to="fine-tune">Fine-tune a model</DocLink>).
            </>,
            "Yes",
          ],
        ]}
      />
      <p>
        "Far from the model's" has a concrete meaning. When you bind a FASTA, each V allele needs its IMGT gaps and the
        position of its Cys codon, and each J allele its Trp/Phe codon. AlignAIR takes the ones your FASTA does not give
        from the model's own alleles: the same sequence, or a V of the same length that differs at no more than 10% of
        its bases (<DocLink to="own-reference?h=gaps-and-anchors">how they are filled</DocLink>). A V allele unlike
        any of the model's gets its anchor only from its own IMGT gaps; without them it has none, and its reads get no
        junction. And a set of another species binds when it has the model's segments, but how well a human model calls
        another species' alleles has not been measured. For another species, train.
      </p>

      <H2 id="decide">Decide in four questions</H2>
      <ol>
        <li>
          <strong>Are your reads human, of a locus with a pretrained model?</strong> If not, train: no pretrained model
          has learned your species or locus.
        </li>
        <li>
          <strong>Is the model's germline set the one you want?</strong> If yes, use the model as it is. Check with{" "}
          <code>alignair info alignair4-igh-human</code> (its germline set and allele counts) or export the set with{" "}
          <code>alignair reference export</code>.
        </li>
        <li>
          <strong>Do you only need fewer alleles, one donor's?</strong> Give the genotype with <code>--genotype</code>.
        </li>
        <li>
          <strong>Do you need other or more alleles of the same locus?</strong> Bind your FASTA with{" "}
          <code>--reference</code>. Read the anchor warning it prints before the run: alleles it names have no anchor,
          and many of them mean your set is far from the model's. Give those alleles IMGT-gapped, or train.
        </li>
      </ol>
      <p>
        Not sure whether binding is good enough? Bind your set, align a sample of your reads, and compare the calls
        with those of the tool you use today on the same reads (<code>alignair compare</code>, see{" "}
        <DocLink to="evaluate">Evaluate your model</DocLink>).
      </p>

      <H2 id="what-training-costs">What training costs</H2>
      <p>
        Training needs no sequencing data. It simulates its reads, with their true alleles and boundaries, from your
        germline set as it goes (with GenAIRR, which is installed with AlignAIR). What it needs is time on a machine.
        One <strong>step</strong> trains on one batch of simulated reads; a <strong>preset</strong> sets the number of
        steps, the batch size and how often the model is checked on held-out reads:
      </p>
      <DocTable
        head={["Preset", "Steps", "Reads per step", "Reads in all", "Checked every", "Use"]}
        rows={[
          [
            <code>quick</code>,
            "300",
            "16",
            "4,800",
            "100 steps",
            "Checks that the run works, start to end. Its calls are not usable.",
          ],
          [<code>desktop</code>, "50,000", "64", "3.2 million", "2,000 steps", "The default."],
          [<code>full</code>, "300,000", "128", "38.4 million", "5,000 steps", "A long run, close to the pretrained models' length."],
        ]}
      />
      <p>
        The seven pretrained models trained for 320,000 to 340,000 steps (each one's card gives its count:{" "}
        <DocLink to="model-cards">Model cards</DocLink>). <code>--steps</code>, <code>--batch-size</code> and{" "}
        <code>--val-every</code> override a preset's numbers.
      </p>
      <p>
        How long a step takes depends on your hardware, so AlignAIR gives no time per preset. Train{" "}
        <code>desktop</code> and <code>full</code> on a GPU; on a CPU, keep to <code>--plan</code> and{" "}
        <code>quick</code>. The reads are simulated on the CPU while the model trains: on a GPU machine with spare CPU
        cores, <code>--data-workers N</code> simulates them in N worker processes (at most 28 are used), with the same
        reads as one process. To estimate your own time, time a short run with your preset's batch size and scale it
        by the steps:
      </p>
      <CodeBlock
        code={`time alignair train --dataconfig MOUSE_IGH_IMGT --out runs/timing --preset desktop \\
  --steps 500 --val-every 250 --data-workers 8`}
      />
      <p>
        The time includes the setup and the export at the end, so it overstates a step a little. While it runs, a
        progress line every 50 steps (<code>[50/500] loss …</code>) shows the pace. Disk use is small: a training
        snapshot is a few MB, and a run keeps at most five of them.
      </p>
      <p>
        Next: <DocLink to="train">Train a model for your germline set</DocLink>.
      </p>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Train a model for your germline set                                                                                */
/* ------------------------------------------------------------------------------------------------------------------ */
const train: DocPage = {
  slug: "train",
  title: "Train a model for your germline set",
  section: SECTION,
  lead: "Train a model from a built-in germline set or from your own V, D and J FASTAs, check the plan first, and find the model it writes.",
  body: () => (
    <>
      <p>
        <code>alignair train</code> takes a germline set and an output folder, simulates reads from the set, trains a
        model on them and writes it, ready to use, to <code>bundle/model.alignair</code> in that folder. The set is
        either one of the 106 built into AlignAIR (<code>--dataconfig</code>) or your own FASTA files (
        <code>--v-fasta</code>, <code>--d-fasta</code>, <code>--j-fasta</code> and <code>--chain-type</code>), never
        both.
      </p>

      <H2 id="built-in-set">From a built-in germline set</H2>
      <p>
        <code>alignair reference list</code> lists the built-in sets, with filters by species and by locus or chain
        type:
      </p>
      <CodeBlock
        code={`$ alignair reference list --species Mouse --chain IGH
2 built-in reference(s):
  name                         chain_type             species                has_d
  MOUSE_C57BL6J_IGH_IMGT       BCR_HEAVY              Mouse C57BL/6J         True
  MOUSE_IGH_IMGT               BCR_HEAVY              Mouse                  True`}
      />
      <p>
        The sets come from IMGT for every species, and also from OGRDB for the human IG loci (
        <code>HUMAN_IGH_OGRDB</code>, the pretrained IGH model's set). A model calls alleles by the names of the set it
        trained on: IMGT names from an <code>_IMGT</code> set, OGRDB labels from an <code>_OGRDB</code> set. Pick the
        naming your genotypes and downstream tools use. Then train:
      </p>
      <CodeTabs
        cli={`alignair train --dataconfig MOUSE_IGH_IMGT --out runs/mouse_igh`}
        python={`from alignair import TrainingConfig, run_training

config = TrainingConfig.from_genairr("MOUSE_IGH_IMGT", preset="desktop")
run = run_training(config, output_dir="runs/mouse_igh")
aligner = run.best_aligner()      # the best training snapshot, loaded as an Aligner`}
      />
      <p>
        The command line uses the <code>desktop</code> preset unless you give <code>--preset</code> (
        <DocLink to="do-i-need-to-train?h=what-training-costs">what each preset costs</DocLink>). Several names after{" "}
        <code>--dataconfig</code> train one model of several loci, for example{" "}
        <code>--dataconfig MOUSE_IGK_IMGT MOUSE_IGL_IMGT</code> for both light chains.
      </p>

      <H2 id="plan-first">Check the plan first</H2>
      <p>
        <code>--plan</code> builds the germline set, checks the settings and prints what the run would do, in a few
        seconds, without training: the loci, the alleles per segment, whether there is a D, the longest read the model
        will read (576 positions), the model's size, and the steps and batch size.
      </p>
      <CodeBlock
        title="shell"
        code={`$ alignair train --dataconfig MOUSE_IGH_IMGT --out runs/mouse_igh --plan
training plan:
{
  "loci": [
    "IGH"
  ],
  "alleles": {
    "V": 451,
    "D": 60,
    "J": 8
  },
  "has_d": true,
  "max_seq_length": 576,
  "model_parameters": 314812,
  "steps": 50000,
  "batch_size": 64
}`}
      />
      <p>
        It does not estimate time or memory. A FASTA set that cannot be built is refused here, with the reason, before
        any training. Training also refuses to start when <code>--out</code> already holds a <code>bundle/</code>, so
        a model you have shared is never overwritten; choose a new folder or pass <code>--overwrite</code>.
      </p>

      <H2 id="own-fastas">From your own germline FASTAs</H2>
      <p>
        Give one FASTA per segment and the chain type. The D file is required for a chain with a D gene (
        <code>BCR_HEAVY</code>, <code>TCR_BETA</code>, <code>TCR_DELTA</code>) and refused for the others (
        <code>BCR_LIGHT_KAPPA</code>, <code>BCR_LIGHT_LAMBDA</code>, <code>TCR_ALPHA</code>, <code>TCR_GAMMA</code>).
      </p>
      <CodeTabs
        cli={`alignair train --v-fasta igh_v.fasta --d-fasta igh_d.fasta --j-fasta igh_j.fasta \\
  --chain-type BCR_HEAVY --out runs/c57bl6_igh --plan`}
        python={`from GenAIRR.cartridge_builder import ReferenceCartridgeBuilder
from alignair import TrainingConfig, run_training

dataconfig = ReferenceCartridgeBuilder.from_fasta(
    v_fasta="igh_v.fasta", d_fasta="igh_d.fasta", j_fasta="igh_j.fasta", chain_type="BCR_HEAVY",
).build()
config = TrainingConfig.from_genairr(dataconfig, preset="desktop")
run = run_training(config, output_dir="runs/c57bl6_igh")`}
      />

      <H3 id="where-fastas-come-from">Where germline FASTAs come from</H3>
      <ul>
        <li>
          <strong>OGRDB</strong> publishes germline sets for human, rhesus macaque, mouse (by strain), rainbow trout and{" "}
          <em>Peromyscus leucopus</em> (<Ext href={OGRDB_SETS}>ogrdb.airr-community.org/germline_sets</Ext>, a page per
          species), each as AIRR JSON and gapped or ungapped FASTA. An IGH set holds its V, D and J alleles in one file:
          split it by segment (below). Mouse IGK and IGL sets hold the V alleles of one strain; their J alleles are in
          the sets named "IGKJ (all strains)" and "IGLJ (all strains)".
        </li>
        <li>
          <strong>IMGT</strong>: the IMGT/V-QUEST reference directory (
          <Ext href={IMGT_VQUEST_SETS}>imgt.org/vquest/refseqh.html</Ext>) has one FASTA per species, locus and segment
          (<code>IGHV.fasta</code>, <code>IGHD.fasta</code>, <code>IGHJ.fasta</code>) of its functional, ORF and in-frame
          pseudogene alleles, V alleles IMGT-gapped. The sets are drawn from{" "}
          <Ext href={IMGT_GENEDB}>IMGT/GENE-DB</Ext>.
        </li>
        <li>
          <strong>Your own</strong> set, from a genotyping study or a curated list.
        </li>
      </ul>

      <H3 id="fasta-format">What the FASTAs need</H3>
      <ul>
        <li>
          <strong>Headers.</strong> The allele name is the first word after <code>&gt;</code> (OGRDB's headers are the
          name alone, <code>&gt;IGHV0-24BS*00</code>). An IMGT header, fields joined by <code>|</code>, works as it
          is: the field with a <code>*</code> is taken as the name (<code>IGHV1-11*01</code> in{" "}
          <code>&gt;AC090843|IGHV1-11*01|Mus musculus_C57BL/6|F|V-REGION|…</code>).
        </li>
        <li>
          <strong>Names.</strong> <code>GENE*ALLELE</code>: the gene is the part before the <code>*</code>. A name given
          twice is counted as rejected, and its second sequence is dropped.
        </li>
        <li>
          <strong>Bases.</strong> A, C, G, T and N only, upper or lower case, with <code>.</code> for the IMGT gaps of
          a V. One allele with another letter, such as the ambiguity code Y, gets the whole set refused (IMGT's
          mouse IGHV set has two):
          <code>[ALIGNAIR_CONFIG_INVALID] 216 refdata validation issue(s) (2 fatal, 214 curatable)</code>, then one line
          per fatal issue (<code>[fatal] V allele 394 byte at position 226 is 0x59 ('Y') …</code>). The message numbers
          the allele rather than naming it; this prints the headers of such alleles, to fix or remove:
          <CodeBlock code={`awk '/^>/{name = $0; next} /[^ACGTNacgtn.]/{print name}' IGHV.fasta`} />
        </li>
        <li>
          <strong>Gaps.</strong> Give the V alleles IMGT-gapped when you can: the gapped form gives GenAIRR the most
          to work with when it places each V's anchor.
        </li>
      </ul>

      <p>
        For example, OGRDB's C57BL/6 IGH set: save its "FASTA Gapped" download from the <em>Mus musculus</em> page as{" "}
        <code>igh_gapped.fasta</code>, split it into one file per segment by the letter after <code>IGH</code> in each
        name, and check the plan:
      </p>
      <CodeBlock
        title="shell"
        code={`$ awk '/^>/{out = "igh_" tolower(substr($0, 5, 1)) ".fasta"} {print > out}' igh_gapped.fasta
$ grep -c ">" igh_v.fasta igh_d.fasta igh_j.fasta
igh_v.fasta:102
igh_d.fasta:8
igh_j.fasta:4
$ alignair train --v-fasta igh_v.fasta --d-fasta igh_d.fasta --j-fasta igh_j.fasta --chain-type BCR_HEAVY --out runs/c57bl6_igh --plan
reference build warnings (2): ["infer_identity() was not called — cartridge metadata species remains unknown; reference_set is 'user-supplied'", 'annotation_incomplete: 4 native validation findings; strict compilation will reject this catalogue']
training plan:
{
  "loci": [
    "IGH"
  ],
  "alleles": {
    "V": 102,
    "D": 8,
    "J": 4
  },
  …
}`}
      />
      <p>
        The two warnings are normal for a FASTA set. The first says that the files name no species. The second counts
        findings of GenAIRR's checks of the set, such as a V allele whose Cys codon it could not place; the run goes on.
      </p>

      <H3 id="check-anchors">Check the anchors before you train</H3>
      <p>
        A read gets a junction only when its V and J alleles have an anchor in the model's germline set: the first base
        of the V Cys codon and of the J Trp/Phe codon. Count them from Python before you spend the training time; here,
        IMGT's mouse IGH set from the V-QUEST directory, without its two alleles with a Y:
      </p>
      <CodeBlock
        lang="python"
        code={`from GenAIRR.cartridge_builder import ReferenceCartridgeBuilder
from alignair import ReferenceSet
from alignair.reference import anchor_report

dataconfig = ReferenceCartridgeBuilder.from_fasta(
    v_fasta="IGHV.fasta", d_fasta="IGHD.fasta", j_fasta="IGHJ.fasta", chain_type="BCR_HEAVY",
).build()
report = anchor_report(ReferenceSet.from_dataconfigs(dataconfig))
print(report.anchored, report.total)    # {'V': 610, 'J': 9} {'V': 676, 'J': 9}`}
      />
      <p>
        <code>report.missing</code> names the alleles without an anchor. Reads called to one of them get no junction:
        remove the ones you do not need, or accept it. After training, <code>bundle/reference_manifest.json</code>{" "}
        gives the same counts per segment (<code>n</code> and <code>anchored</code>; D alleles have no anchor). OGRDB's C57BL/6 IGH set above has
        all its V and J alleles anchored.
      </p>

      <H3 id="allele-names">IMGT or OGRDB names</H3>
      <p>
        The model calls alleles by the names in your FASTA files, and keeps them. A genotype or a{" "}
        <code>--reference</code> you give the model later must use the same naming (a genotype in another one is
        refused, <code>ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND</code>). So train on the naming your genotypes, Change-O and
        TIgGER tables and gene-usage plots use. <DocLink to="choose-a-model?h=allele-names">Allele names</DocLink>{" "}
        explains the two schemes.
      </p>

      <H2 id="run-it">Run the training</H2>
      <p>Remove <code>--plan</code> to train. A run prints the plan, then its progress:</p>
      <CodeBlock
        lang="text"
        title="output (abridged; the numbers are left out)"
        code={`$ alignair train --dataconfig MOUSE_IGH_IMGT --out runs/mouse_igh
training plan:
…
novel alleles: MOUSE_IGH_IMGT: 128 of 320 eligible V genes per epoch of 4096 records
[50/50000] loss … gnorm … …
[100/50000] loss … gnorm … …
…
[2000/50000] VAL score …  v_allele_top1=… v_seg_mae=… …
…
trained -> runs/mouse_igh/model.alignair
exported pickle-free bundle -> /home/you/runs/mouse_igh/bundle/model.alignair`}
      />
      <p>
        Every 50 steps, <code>loss</code> measures how wrong the model still is on its training reads; it falls as the
        model learns. At each check (every 2,000 steps with <code>desktop</code>), <code>VAL score</code> scores the
        model on a fixed set of held-out simulated reads, higher is better, and the best-scoring snapshot is kept. At
        the end, the best snapshot is exported as the model to use. <DocLink to="evaluate">Evaluate your model</DocLink>{" "}
        explains the scores.
      </p>
      <Callout kind="warning" title="quick is a test, not a model">
        The <code>quick</code> preset trains 300 steps to show that a run works from start to end. Its calls are not
        usable. Train with <code>desktop</code> or <code>full</code> for a model you will make calls with.
      </Callout>

      <H3 id="novel-alleles">Novel alleles, drawn by default</H3>
      <p>
        With a built-in set of one species, part of the training reads come from novel alleles that AlignAIR makes up
        as it goes: in each block of 4,096 reads, 40% of the V genes that allow it carry a new variant with 1 to 6
        substitutions (the <code>novel alleles</code> line above counts them). This teaches the model to call an allele
        from the germline set it is given rather than from memory, which is what lets a{" "}
        <DocLink to="own-reference?h=novel-alleles">bound set with your novel alleles</DocLink> work without training.
        The model's own germline set stays the one you trained on.
      </p>
      <p>
        A run from your own FASTA files draws none, because they name no species, and says so:{" "}
        <code>novel alleles unavailable: novel alleles train a model of one named species; the reference names none (a
        custom reference); the run trains on the reference's alleles alone</code>. In Python,{" "}
        <code>TrainingConfig(novel_alleles=None)</code> turns them off and a <code>NovelAlleles(...)</code> changes
        them (<DocLink to="python-api">Python API</DocLink>).
      </p>

      <H2 id="python">Train from Python</H2>
      <p>
        <code>TrainingConfig.from_genairr</code> takes built-in set names or GenAIRR germline sets (as built from FASTA
        files above), a <code>preset</code>, and any setting to override (<code>steps</code>, <code>batch_size</code>,{" "}
        <code>val_every</code>, <code>lr</code>, <code>seed</code>, <code>device</code>, <code>data_workers</code>, …);
        it checks every value when it is built. <code>run_training(config, output_dir=...)</code> trains and returns a{" "}
        <code>TrainingRun</code>: <code>model_path</code> (the last snapshot), <code>best_model_path</code> (the best,
        when the model was checked), <code>steps</code> (the steps done), and <code>best_aligner()</code>, the best (or
        last) snapshot loaded as an <code>Aligner</code>, ready to predict.
      </p>
      <p>
        Give a <code>preset</code>, or set <code>steps</code> and <code>val_every</code> yourself: without them, a{" "}
        <code>TrainingConfig</code> trains 100,000 steps and never checks the model (<code>val_every=0</code>), so the
        run keeps no best snapshot. With{" "}
        <code>data_workers</code> above 0, call <code>run_training</code> under an{" "}
        <code>if __name__ == "__main__":</code> guard, as it starts worker processes. <code>run_training</code> writes
        the training snapshots but no bundle; export one with <code>alignair export</code> (
        <DocLink to="share?h=export">Share your model</DocLink>).
      </p>

      <H2 id="options">Every option</H2>
      <OptionTable command="train" />

      <H2 id="resume">Resume an interrupted run (Python)</H2>
      <p>
        <code>alignair train</code> has no option to resume. From Python, <code>resume_path</code> continues a run from
        its latest snapshot: the newest <code>model.step&lt;N&gt;.alignair</code> beside the file you name.
      </p>
      <CodeBlock
        lang="python"
        code={`from alignair import TrainingConfig, run_training

config = TrainingConfig.from_genairr("MOUSE_IGH_IMGT", preset="desktop",
                                     resume_path="runs/mouse_igh/model.alignair")
run = run_training(config, output_dir="runs/mouse_igh_resumed")
print(run.steps, run.best_model_path)`}
      />
      <ul>
        <li>
          <code>steps</code> is the total to reach (here the preset's 50,000), and must be beyond the snapshot's step.
        </li>
        <li>
          The germline set and the model's settings must be the snapshot's; for a run from your own FASTA files, build
          the germline set again from the same files. Otherwise the resume is refused, with a hint to use{" "}
          <code>init_from</code> (<DocLink to="fine-tune">Fine-tune a model</DocLink>).
        </li>
        <li>
          The weights, the optimizer's state and the random state carry on, but the resumed run's reads are not the
          ones the uninterrupted run would have drawn.
        </li>
        <li>
          A new <code>output_dir</code> keeps the first part's best snapshot apart: the resumed run keeps the best of
          its own checks. Export the one you choose with <code>alignair export</code>.
        </li>
      </ul>
      <p>
        <code>resume_path</code> and <code>init_from</code> cannot be given together. A training snapshot holds Python
        pickle sections, which run code when they are loaded: resume only from your own runs or ones you trust.
      </p>

      <H2 id="output">What a run writes</H2>
      <CodeBlock
        title="shell"
        code={`$ ls runs/mouse_igh runs/mouse_igh/bundle
runs/mouse_igh:
bundle  model.alignair  model.best.alignair  model.step46000.alignair  model.step48000.alignair  model.step50000.alignair

runs/mouse_igh/bundle:
model.alignair  model_card.md  reference_manifest.json  validation_report.json`}
      />
      <DocTable
        head={["File", "What it is"]}
        rows={[
          [
            <code>model.step&lt;N&gt;.alignair</code>,
            "A training snapshot, every 2,000 steps and at the last step; the newest three are kept.",
          ],
          [<code>model.alignair</code>, "A copy of the newest snapshot (TrainingRun.model_path)."],
          [
            <code>model.best.alignair</code>,
            "The snapshot that scored best at the checks, when the model was checked (TrainingRun.best_model_path).",
          ],
          [
            <code>bundle/model.alignair</code>,
            <>
              <strong>The model to use and share</strong>: the best (or last) snapshot without its training state. It
              holds the weights, your germline set and its card.
            </>,
          ],
          [
            <code>bundle/model_card.md</code>,
            "The card as text: loci, germline set, allele counts, training steps, intended use, the validation scores and the licence.",
          ],
          [
            <code>bundle/reference_manifest.json</code>,
            "The alleles of each segment with how many were anchored, the germline set's fingerprints, your FASTA files' SHA-256 and the build warnings.",
          ],
          [
            <code>bundle/validation_report.json</code>,
            <>
              The scores on a fixed set of held-out simulated reads (<DocLink to="evaluate?h=validation-report">Evaluate
              your model</DocLink>).
            </>,
          ],
        ]}
      />
      <p>The card starts with what the model is:</p>
      <CodeBlock
        lang="text"
        title="runs/mouse_igh/bundle/model_card.md (start)"
        code={`# AlignAIR model card

- loci: IGH
- reference: MOUSE_IGH_IMGT (IMGT, updated 2026-03-01; Mouse BCR_HEAVY)
- alleles: V=451, D=60, J=8`}
      />

      <H2 id="use-it">Use the model</H2>
      <p>
        Give the bundle's model file wherever a model id goes. Every option of <code>alignair predict</code> works with
        it, <code>--genotype</code> and <code>--reference</code> included:
      </p>
      <CodeTabs
        cli={`alignair predict --model runs/mouse_igh/bundle/model.alignair --input reads.fasta --out out.tsv`}
        python={`from alignair import Aligner, read_sequences

aligner = Aligner.from_pretrained("runs/mouse_igh/bundle/model.alignair")
ids, reads, info = read_sequences("reads.fasta")
aligner.predict(reads, ids=ids).write_airr("out.tsv")`}
      />
      <p>
        Next: <DocLink to="evaluate">Evaluate your model</DocLink>, then{" "}
        <DocLink to="share">Share your model</DocLink>.
      </p>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Fine-tune a model                                                                                                  */
/* ------------------------------------------------------------------------------------------------------------------ */
const fineTune: DocPage = {
  slug: "fine-tune",
  title: "Fine-tune a model",
  section: SECTION,
  lead: "Start a training run from an existing model's weights instead of from scratch, from the command line or from Python.",
  body: () => (
    <>
      <p>
        Fine-tuning is a training run with a warm start: the new model begins with the weights of a model you name, then
        trains on your germline set as any run does. Everything on{" "}
        <DocLink to="train">Train a model for your germline set</DocLink> applies; one option is added.
      </p>

      <H2 id="start-from-a-model">Start from a model</H2>
      <p>
        <code>--init-from</code> takes a model file, a pretrained model's id or a Hugging Face repo (
        <code>hf://org/repo</code>). In Python, <code>init_from</code> takes a model file;{" "}
        <code>resolve_model</code> turns an id into the cached file, downloading it the first time.
      </p>
      <CodeTabs
        cli={`alignair train --dataconfig MOUSE_IGH_IMGT --out runs/mouse_igh_ft --init-from alignair4-igh-human`}
        python={`from alignair import TrainingConfig, run_training
from alignair.registry import resolve_model

config = TrainingConfig.from_genairr("MOUSE_IGH_IMGT", preset="desktop",
                                     init_from=resolve_model("alignair4-igh-human"))
run = run_training(config, output_dir="runs/mouse_igh_ft")`}
      />
      <p>
        <code>--plan</code> works as before and shows the run's steps and batch size. A model from a Hugging Face repo
        needs <code>pip install "alignair[hub]"</code>, and a private one a token in <code>HF_TOKEN</code>.
      </p>

      <H2 id="what-a-warm-start-does">What a warm start does</H2>
      <ul>
        <li>
          Every weight of the new model whose name and shape match one of the source model's is copied; the rest start
          fresh, as in a run from scratch.
        </li>
        <li>
          The germline set is the one you train on. The source's alleles are not carried over, and the new model calls
          only the alleles of your set.
        </li>
        <li>
          The run starts at step 0 and trains for the full number of steps, with a fresh optimizer and learning rate.
          Nothing of the source's training state is used.
        </li>
      </ul>
      <p>
        A model's weights do not depend on the alleles of its germline set, which it is given as an input. So between
        two models AlignAIR builds with its default settings, every weight is copied, whatever the species or locus,
        except the parts for the D gene when the source has none and the new model does (from an IGK model to an IGH
        one, for example): those start fresh. The run says how much it copied before it trains:
      </p>
      <CodeBlock
        lang="text"
        title="output (abridged)"
        code={`$ alignair train --dataconfig MOUSE_C57BL6J_IGH_IMGT --out runs/c57bl6j_ft --init-from runs/mouse_igh/bundle/model.alignair
…
WARM-START from runs/mouse_igh/bundle/model.alignair
backbone transfer: 86/86 tensors (100%)
…`}
      />
      <p>
        A tensor is one block of the model's weights; the lines that follow (left out here) name the parts copied and
        the parts that start fresh.
      </p>

      <H2 id="when-to-try-it">When to try it</H2>
      <ul>
        <li>
          <strong>A related locus or species.</strong> You train a model for a germline set and a model of a related one
          exists: a mouse IGH model started from the human IGH model, or a strain's set started from a model of the
          species' wider set.
        </li>
        <li>
          <strong>A finished run, with other settings.</strong> Resuming continues a run only with its own settings
          and germline set; a warm start lets you train on from your model with another germline set, more steps or
          another learning rate.
        </li>
      </ul>
      <Callout kind="note" title="No recipe">
        AlignAIR does not prescribe how long to fine-tune or with which settings, and whether a warm start gives a
        better model than a run from scratch has not been measured for your case. To find out, train your set both
        ways with the same settings and compare the two models on the same reads (
        <DocLink to="evaluate?h=benchmark">alignair benchmark</DocLink>).
      </Callout>

      <H2 id="resume-or-warm-start">Resume or warm start?</H2>
      <DocTable
        head={["", "Resume (Python resume_path)", "Warm start (--init-from, init_from)"]}
        rows={[
          [
            "What it continues",
            "The same run: its step count, optimizer state and random state",
            "Only the weights; a new run from step 0",
          ],
          ["Germline set", "Must be the snapshot's", "Any"],
          ["Settings", "Must be the snapshot's", "Any"],
          ["From", "A training snapshot of yours", "A model file, a model id, hf://org/repo"],
        ]}
      />
      <p>
        The two cannot be combined. Resuming is described on{" "}
        <DocLink to="train?h=resume">Train a model for your germline set</DocLink>.
      </p>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Evaluate your model                                                                                                */
/* ------------------------------------------------------------------------------------------------------------------ */
const evaluate: DocPage = {
  slug: "evaluate",
  title: "Evaluate your model",
  section: SECTION,
  lead: "Score a trained model on simulated reads with known truth, compare its calls with another tool's on your reads, and check its output.",
  body: () => (
    <>
      <p>
        Three checks, from the cheapest: the validation report the run wrote, <code>alignair benchmark</code> on fresh
        simulated reads with known truth, and, on your own reads, <code>alignair compare</code> against another tool and{" "}
        <code>alignair analyze</code> for quality control.
      </p>

      <H2 id="validation-report">The validation report</H2>
      <p>
        A bundle has a <code>validation_report.json</code> (also listed in its card) unless it was exported with{" "}
        <code>--no-validate</code>: scores on a fixed set of
        simulated reads of the germline set the model trained on, at most 192 reads per set. They score the model's own
        pick, before AlignAIR's germline step refines the call; <code>alignair benchmark</code> scores the final calls.
        The entries to read first:
      </p>
      <DocTable
        head={["Entry", "What it measures", "Better"]}
        rows={[
          [
            <>
              <code>v_allele_top1</code>, <code>d_allele_top1</code>, <code>j_allele_top1</code>
            </>,
            "Share of reads whose top allele is among the read's true alleles (the simulation can list several).",
            "Higher (0-1)",
          ],
          [
            <>
              <code>v_seg_mae</code>, <code>d_seg_mae</code>, <code>j_seg_mae</code>
            </>,
            "Mean error of the segment's start and end in the read, in nucleotides.",
            "Lower",
          ],
          [
            <>
              <code>anchor_v_exact</code>, <code>anchor_j_exact</code>
            </>,
            "Share of reads whose V Cys or J Trp/Phe codon is found at exactly the right base. The junction is built from them.",
            "Higher (0-1)",
          ],
          [<code>orientation_acc</code>, "Share of reads turned to the right orientation.", "Higher (0-1)"],
          [<code>mutation_mae</code>, "Error of the estimated mutation rate.", "Lower"],
        ]}
      />
      <p>
        The other entries score single steps inside the model and matter mostly to developers. Expect D below V and J:
        D segments are short and trimmed at both ends, so many reads hold too little of the D to tell its alleles
        apart.
      </p>
      <Callout kind="note" title="Necessary, not sufficient">
        The validation reads are simulated like the training reads. Good scores say the model learned the simulation;
        your reads still need their own check.
      </Callout>

      <H2 id="benchmark">Score the calls: alignair benchmark</H2>
      <p>
        <code>alignair benchmark</code> simulates fresh reads with known truth from a germline set, aligns them as{" "}
        <code>alignair predict</code> does, and scores the calls, read group by read group:
      </p>
      <CodeBlock
        code={`alignair benchmark --model runs/mouse_igh/bundle/model.alignair
alignair benchmark --model runs/mouse_igh/bundle/model.alignair --json --out benchmark.json`}
      />
      <DocTable
        head={["Group", "The reads"]}
        rows={[
          [<code>clean</code>, "Full length, little hypermutation (mutation rate about 0.02) and few sequencing errors."],
          [<code>moderate</code>, "More of everything: mutation rate about 0.09, more sequencing errors and indels, ends lost."],
          [<code>high_shm</code>, "Heavy hypermutation (mutation rate 0.25). On a TCR locus the reads carry none, as T cells do not hypermutate."],
          [<code>short_janchor</code>, "Short reads that keep the J end and lose 150 to 350 nt of the 5' end."],
        ]}
      />
      <p>Each group gets a row of the text report; <code>--json</code> writes the same numbers under these names:</p>
      <DocTable
        head={["Column (JSON name)", "What it measures"]}
        rows={[
          [
            <>
              <code>v</code>, <code>d</code>, <code>j</code> (<code>v_call_acc</code>, …)
            </>,
            "Share of reads whose call (its first allele) is one of the read's true alleles.",
          ],
          [<><code>junc_nt</code> (<code>junction_nt_exact</code>)</>, "Share of reads whose junction is exactly right, base for base."],
          [<><code>junc_lenMAE</code> (<code>junction_len_mae</code>)</>, "Mean error of the junction's length, in nucleotides."],
          [
            <>
              <code>v_start_mae</code> … <code>j_end_mae</code> (<code>v_sequence_start_mae</code>, …)
            </>,
            "Mean error of the V and J start and end in the read, in nucleotides.",
          ],
          [<code>productive_acc</code>, "Share of reads whose productive call matches the truth."],
        ]}
      />
      <p>
        The reads depend only on the germline set, <code>--n</code> and <code>--seed</code>, so two models run with the
        same three are scored on the same reads: the way to compare a fine-tuned model with one trained from scratch.
        By default the reads are simulated from the first germline set the model's card names. A model trained from
        your own FASTA files has no built-in set to simulate from: give <code>--dataconfig</code> a built-in set whose
        allele names your model uses, or rely on its validation report.
      </p>
      <OptionTable command="benchmark" />

      <H2 id="compare">Compare with another tool: alignair compare</H2>
      <p>
        On your own reads there is no truth, but there is agreement. Align the same reads with your model and with the
        tool you trust today, as AIRR tables, and compare them read by read:
      </p>
      <CodeBlock
        title="shell (output shortened; two small example tables)"
        code={`$ alignair compare --a alignair.tsv --b igblast.tsv --a-name AlignAIR --b-name IgBLAST
# AIRR agreement: AlignAIR vs IgBLAST

Matched **4** sequences by \`sequence_id\` (AlignAIR: 4, IgBLAST: 4; AlignAIR-only 0, IgBLAST-only 0).

## Call agreement (rows where both tools called the gene)

| gene | n | allele agreement | gene agreement | set-rescue of disagreements |
| --- | --- | --- | --- | --- |
| V | 4 | 75.0% | 100.0% | 100.0% |
| D | 4 | 75.0% | 75.0% | 0.0% |
| J | 4 | 100.0% | 100.0% | n/a |

- junction (nt) agreement: 100.0%
- productivity agreement: 100.0%`}
      />
      <p>
        Set-rescue is the share of disagreements where the second table's allele is in the first table's candidate set
        (its <code>v_call_set</code>, <code>d_call_set</code>, <code>j_call_set</code>): shared ambiguity, not a
        conflict. So give AlignAIR's table as <code>--a</code>. The report ends with examples of
        the reads that disagree. Agreement is not accuracy: where the tools differ, either may be right.{" "}
        <code>--json</code> gives the numbers as JSON, <code>--out</code> writes the report to a file. Both tables must
        use the same allele names, so align with the same germline set in both tools.
      </p>

      <H2 id="analyze">Check the output: alignair analyze</H2>
      <p>
        <code>alignair analyze out.tsv</code> summarises a table your model wrote, in three parts. What to look for in a
        new model's output on your reads:
      </p>
      <DocTable
        head={["Part", "Lines", "What to look for"]}
        rows={[
          [
            "REPERTOIRE",
            "reads, productive, in-frame, stop codons, unique junctions, the top V, D and J alleles, CDR3 lengths",
            "A productive share and allele usage that fit what you know of the sample.",
          ],
          [
            "QC",
            <>
              orientation, indel-flagged reads, mean V identity, and completeness: the share of rows with{" "}
              <code>v_call</code>, <code>j_call</code>, <code>junction</code>, <code>cdr3</code>
            </>,
            <>
              A <code>junction</code> share well below the <code>v_call</code> share points to alleles without an
              anchor (see <code>reference_manifest.json</code>); a low V identity to reads far from your germline set.
            </>,
          ],
          ["VALIDATION", "required AIRR columns, coordinate and junction-length checks", <>Ends with <code>=&gt; VALID</code>.</>],
        ]}
      />
      <p>
        <code>--json</code> gives the report as JSON. For how far the pretrained models' calls can be trusted, and how
        they were measured, see <DocLink to="trust-the-calls">How far to trust the calls</DocLink> and the{" "}
        <DocLink to="comparison">comparison with IgBLAST</DocLink>.
      </p>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Share your model                                                                                                   */
/* ------------------------------------------------------------------------------------------------------------------ */
const share: DocPage = {
  slug: "share",
  title: "Share your model",
  section: SECTION,
  lead: "Turn a training run into one model file others can run, put it on Hugging Face, and load it back by name.",
  body: () => (
    <>
      <p>
        A trained model is one file, <code>model.alignair</code>, with its germline set and card inside. Anyone with
        AlignAIR runs it with <code>--model path/to/model.alignair</code>; no other file is needed.
      </p>

      <H2 id="export">Export the model: alignair export</H2>
      <p>
        <code>alignair train</code> already exported its best snapshot to <code>bundle/</code>. Run{" "}
        <code>alignair export</code> yourself when you trained from Python (<code>run_training</code> writes no
        bundle), to export another snapshot, or to record a name and version in the card:
      </p>
      <CodeBlock
        title="shell"
        code={`$ alignair export runs/mouse_igh/model.best.alignair runs/mouse_igh/release \\
    --model-id mouse-igh-lab --model-version 1.0.0 --description "Mouse IGH, IMGT set"
exported pickle-free bundle -> /home/you/runs/mouse_igh/release/model.alignair
$ head -4 runs/mouse_igh/release/model_card.md
# AlignAIR model card

- model: mouse-igh-lab 1.0.0
- loci: IGH`}
      />
      <p>
        The folder gets the same four files as a run's <code>bundle/</code>: <code>model.alignair</code>,{" "}
        <code>model_card.md</code>, <code>reference_manifest.json</code> and <code>validation_report.json</code>. The
        export scores the model on the validation reads (<code>--no-validate</code> skips that; the card then says the
        scores are pending), checks that the new file loads, and refuses a folder that exists unless you pass{" "}
        <code>--overwrite</code>.
      </p>
      <Callout kind="warning" title="Share the exported model, never a snapshot">
        A training snapshot (<code>model.alignair</code>, <code>model.step&lt;N&gt;.alignair</code> or{" "}
        <code>model.best.alignair</code> in the run folder) holds its training state as Python pickle sections, which run
        code when they are loaded. The exported model has none and loads without running any code. For a run from your
        own FASTA files, the export reads the germline set from the snapshot's pickle section, and says so: export only
        snapshots you trust.
      </Callout>
      <OptionTable command="export" />

      <H2 id="hugging-face">Put it on Hugging Face</H2>
      <p>
        <code>Aligner.push_to_hub</code> uploads the model file, with the <code>model_card.md</code> and{" "}
        <code>reference_manifest.json</code> beside it, to a Hugging Face model repo in one commit. It needs{" "}
        <code>pip install "alignair[hub]"</code> and a Hugging Face token with write access, given as{" "}
        <code>token=</code> or in <code>HF_TOKEN</code>. It is Python only.
      </p>
      <CodeBlock
        lang="python"
        code={`from alignair import Aligner

aligner = Aligner.from_pretrained("runs/mouse_igh/release/model.alignair")
aligner.push_to_hub("your-org/alignair-mouse-igh")                 # creates a private repo
# aligner.push_to_hub("your-org/alignair-mouse-igh", private=False)  # or a public one`}
      />
      <ul>
        <li>
          <strong>Private by default.</strong> A repo it creates is private unless you pass <code>private=False</code>.
          A repo that exists keeps its visibility; change it on Hugging Face.
        </li>
        <li>
          <strong>It refuses</strong> a training snapshot (the message names <code>alignair export</code>), and an{" "}
          <code>Aligner</code> made by <code>with_genotype</code> or <code>bound_to</code>, which is not a file.
        </li>
        <li>
          <code>filename=</code> names the file in the repo (<code>model.alignair</code> by default), and{" "}
          <code>revision=</code> commits to a branch. <code>aligner.save_pretrained("folder")</code> copies the model
          file to a folder instead.
        </li>
      </ul>

      <H2 id="load-it-back">Load it back</H2>
      <p>
        <code>hf://org/repo</code> works wherever a model id does. It needs <code>alignair[hub]</code>; a private repo
        also needs a token (<code>--hf-token</code>, <code>token=</code>, or <code>HF_TOKEN</code>).{" "}
        <code>--revision</code>, or <code>@revision</code> after the name, pins a branch, tag or commit.
      </p>
      <CodeTabs
        cli={`alignair predict --model hf://your-org/alignair-mouse-igh --input reads.fasta --out out.tsv`}
        python={`from alignair import Aligner, read_sequences

aligner = Aligner.from_pretrained("hf://your-org/alignair-mouse-igh")   # token="hf_..." for a private repo
ids, reads, info = read_sequences("reads.fasta")
aligner.predict(reads, ids=ids).write_airr("out.tsv")`}
      />
      <p>
        <code>alignair info hf://your-org/alignair-mouse-igh</code> prints its card, and{" "}
        <code>--init-from hf://your-org/alignair-mouse-igh</code> fine-tunes from it. The download is kept in Hugging
        Face's own cache (<code>~/.cache/huggingface</code> by default), not AlignAIR's, and <code>--offline</code> reads
        it from there (<DocLink to="cache-and-environment">Cache, offline and environment variables</DocLink>).
      </p>

      <H2 id="licence">The model's licence</H2>
      <p>
        AlignAIR and its pretrained models are released under GPL-3.0-or-later, and the card AlignAIR writes for a
        model you train says so, with how to cite AlignAIR. A model also holds its germline set, which keeps the terms
        of its source (OGRDB, IMGT, or yours): read them before you share a model, and add them to{" "}
        <code>model_card.md</code>.
      </p>

      <H2 id="catalog">The pretrained catalog</H2>
      <p>
        The models named by id, such as <code>alignair4-igh-human</code>, live in AlignAIR's shared catalog, which the
        maintainers publish (<DocLink to="publishing">Publishing to the catalog</DocLink>). To propose a model for it,
        open a "Model or reference request" issue (<DocLink to="support">Support</DocLink>).
      </p>
    </>
  ),
};

export const trainPages: DocPage[] = [doINeedToTrain, train, fineTune, evaluate, share];
