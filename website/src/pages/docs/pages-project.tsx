import type { ReactNode } from "react";
import { CodeBlock, Callout } from "../../components/ui";
import { DocTable, DocLink, H2, type DocPage } from "./doc-kit";

const SECTION = "Project";

const REPO = "https://github.com/MuteJester/AlignAIR";
const ISSUES = `${REPO}/issues`;
const PAPER = "https://doi.org/10.1093/nar/gkaf651";

function Ext({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer" className="text-brand-600 underline">
      {children}
    </a>
  );
}

const BIBTEX = `@article{konstantinovsky2025alignair,
  title   = {Enhancing sequence alignment of adaptive immune receptors through multi-task deep learning},
  author  = {Konstantinovsky, Thomas and Peres, Ayelet and Eisenberg, Ran and Polak, Pazit and
             Lindenbaum, Ofir and Yaari, Gur},
  journal = {Nucleic Acids Research},
  volume  = {53},
  number  = {13},
  pages   = {gkaf651},
  year    = {2025},
  doi     = {10.1093/nar/gkaf651}
}`;

/* ------------------------------------------------------------------------------------------------------------------ */
/* Cite                                                                                                               */
/* ------------------------------------------------------------------------------------------------------------------ */
const cite: DocPage = {
  slug: "cite",
  title: "Cite",
  section: SECTION,
  lead: "The reference to cite when AlignAIR contributes to your work, as text and BibTeX.",
  body: () => (
    <>
      <H2 id="the-paper">The paper</H2>
      <p>If you use AlignAIR, please cite:</p>
      <blockquote>
        Thomas Konstantinovsky, Ayelet Peres, Ran Eisenberg, Pazit Polak, Ofir Lindenbaum, Gur Yaari.{" "}
        <strong>Enhancing sequence alignment of adaptive immune receptors through multi-task deep learning.</strong>{" "}
        <em>Nucleic Acids Research</em>, Volume 53, Issue 13, 22 July 2025, gkaf651.{" "}
        <Ext href={PAPER}>https://doi.org/10.1093/nar/gkaf651</Ext>
      </blockquote>

      <H2 id="bibtex">BibTeX</H2>
      <CodeBlock lang="bibtex" title="BibTeX" code={BIBTEX} />
      <p>
        The repository's <Ext href={`${REPO}/blob/main/CITATION.cff`}>CITATION.cff</Ext> holds the same reference, so
        GitHub's "Cite this repository" button gives it too, as BibTeX or APA.
      </p>

      <H2 id="methods">What to report in your methods</H2>
      <ul>
        <li>
          <strong>The AlignAIR version</strong>: <code>alignair --version</code>.
        </li>
        <li>
          <strong>The model and its version</strong>, and the germline set it called from. Each run records them in{" "}
          <code>out.tsv.run.json</code>, beside the table: the model id, version and checksum, and the germline set (with
          a <code>--reference</code> or <code>--genotype</code> file, its path and SHA-256).
        </li>
        <li>
          <strong>The germline set's source.</strong> Cite OGRDB or IMGT, or the study your set comes from, as its
          provider asks. The IG models call from OGRDB's human sets, the TCR models from IMGT's (
          <DocLink to="model-cards">Model cards</DocLink>).
        </li>
      </ul>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Support                                                                                                            */
/* ------------------------------------------------------------------------------------------------------------------ */
const support: DocPage = {
  slug: "support",
  title: "Support",
  section: SECTION,
  lead: "Where to ask a question or report a bug, and what to put in the report so it can be acted on.",
  body: () => (
    <>
      <p>
        First look up the error's code on <DocLink to="error-codes">Error codes</DocLink> and your symptom on{" "}
        <DocLink to="fix-common-problems">Fix common problems</DocLink>, then search the{" "}
        <Ext href={ISSUES}>existing issues</Ext>.
      </p>

      <H2 id="where-to-ask">Where to ask</H2>
      <DocTable
        head={["You have", "Open"]}
        rows={[
          [
            "A bug: a crash, a wrong result, a refusal you do not understand",
            <>
              A <Ext href={`${ISSUES}/new?template=bug_report.md`}>Bug report</Ext> issue.
            </>,
          ],
          [
            "A question on how to do something",
            <>
              An <Ext href={ISSUES}>issue</Ext>. A question that shows a gap in these pages gets the pages fixed.
            </>,
          ],
          [
            "A model for a species or locus AlignAIR does not cover, or a germline set that fails to build or train",
            <>
              A <Ext href={`${ISSUES}/new?template=model_or_reference_request.md`}>Model or reference request</Ext>{" "}
              issue.
            </>,
          ],
          [
            "An idea for a feature",
            <>
              A <Ext href={`${ISSUES}/new?template=feature_request.md`}>Feature request</Ext> issue.
            </>,
          ],
          [
            "A question about the paper's methods",
            <>
              The corresponding authors of the <Ext href={PAPER}>paper</Ext>.
            </>,
          ],
          [
            "A security problem",
            <>
              No public issue: email <a href="mailto:thomaskon90@gmail.com">thomaskon90@gmail.com</a> (
              <Ext href={`${REPO}/blob/main/SECURITY.md`}>security policy</Ext>).
            </>,
          ],
        ]}
      />

      <H2 id="what-to-include">What to put in a bug report</H2>
      <ul>
        <li>
          <strong>Your environment</strong>: the full output of <code>alignair doctor</code> (Python, PyTorch and the
          GPU it sees, the dependencies, the cache folder). <code>alignair doctor --json</code> gives it as JSON.
        </li>
        <li>
          <strong>The exact command</strong>, and its full output run again with <code>--debug</code>, which adds the
          traceback to the error.
        </li>
        <li>
          <strong>The model</strong>: its id and version, or the path of your <code>.alignair</code> file. For a
          prediction, attach the run's <code>out.tsv.run.json</code>, which records the model, the germline set and the
          options.
        </li>
        <li>
          <strong>A small input that shows the problem</strong>: a few reads is best. If your reads are sensitive, a
          made-up read that gives the same error is as good.
        </li>
        <li>
          <strong>For a training problem</strong>: the output of the same command with <code>--plan</code>, which
          includes the germline set's build warnings.
        </li>
      </ul>
      <CodeBlock
        code={`alignair --version
alignair doctor
alignair predict --model alignair4-igh-human --input few_reads.fasta --out out.tsv --debug`}
      />
      <p>
        Every failure has a code (<code>ALIGNAIR_&lt;THING&gt;_&lt;STATE&gt;</code>). <code>--error-format json</code>{" "}
        prints the error as one JSON object, with its code, message, context and hint, which is the form to paste when
        a pipeline reports it:
      </p>
      <CodeBlock
        title="shell"
        code={`$ alignair predict --model my_model.alignair --input reads.fasta --out out.tsv --error-format json
{"category": "model", "code": "ALIGNAIR_MODEL_NOT_FOUND", "context": {"command": "predict", "model_id": "my_model.alignair", "path": "my_model.alignair", "stage": "model_resolution"}, "exit_code": 1, "hint": null, "message": "model file does not exist", "retryable": false, "schema_version": 1, "type": "ModelNotFoundError"}`}
      />

      <H2 id="contributing">Contributing</H2>
      <p>
        Bug fixes, germline sets, models and documentation are welcome. Open an issue first for a large change.{" "}
        <DocLink to="contributing">Contributing</DocLink> explains how to set up the code and run the tests, and the
        repository's <Ext href={`${REPO}/blob/main/CONTRIBUTING.md`}>CONTRIBUTING.md</Ext> has the rules for pull
        requests.
      </p>
    </>
  ),
};

/* ------------------------------------------------------------------------------------------------------------------ */
/* Changelog                                                                                                          */
/* ------------------------------------------------------------------------------------------------------------------ */
const changelog: DocPage = {
  slug: "changelog",
  title: "Changelog",
  section: SECTION,
  lead: "What AlignAIR 4 brings, in short; every change is in the changelog on GitHub.",
  body: () => (
    <>
      <p>
        The full list of changes, release by release, is in{" "}
        <Ext href={`${REPO}/blob/main/CHANGELOG.md`}>CHANGELOG.md</Ext> on GitHub. The main changes of AlignAIR 4:
      </p>

      <H2 id="models">Models</H2>
      <ul>
        <li>
          Seven pretrained human models, one per locus: IGH, IGK, IGL, TRA, TRB, TRD and TRG. They download by id on
          first use and are cached (<DocLink to="choose-a-model">Choose a model</DocLink>).
        </li>
        <li>
          A model file is self-contained: the weights, the germline set and a card, loaded without running any code.{" "}
          <code>alignair info</code> prints a model's card.
        </li>
      </ul>

      <H2 id="germline-sets-and-genotypes">Germline sets and genotypes</H2>
      <ul>
        <li>
          A model calls from the germline set it is given. <code>--reference germline.fasta</code> binds your own set,
          novel alleles and new genes included, with no training; missing IMGT gaps and anchors are filled from the
          model's own alleles (<DocLink to="own-reference">Use your own germline reference</DocLink>).
        </li>
        <li>
          <code>--genotype</code> limits the calls to a donor's alleles; <code>--infer-genotype</code> infers the
          donor's V genotype from the reads in the same run, by a frequency rule like TIgGER's; <code>alignair genotype</code>{" "}
          writes one from any AIRR table as an AIRR GenotypeSet (<DocLink to="donor-genotype">Align with a donor
          genotype</DocLink>).
        </li>
        <li>
          <code>alignair reference export --gapped</code> writes a model's germline set with IMGT-gapped V alleles, as
          Change-O's <code>CreateGermlines.py</code> needs.
        </li>
      </ul>

      <H2 id="calls-and-output">Calls and output</H2>
      <ul>
        <li>
          <code>v_call</code> lists, after the call, the alleles that match the read equally well, as IgBLAST does; an
          allele of the germline set that matches the read at more bases than the model's call becomes the call.
        </li>
        <li>
          The junction and productivity follow by rule from the V Cys and J Trp/Phe codons found in the read.
        </li>
        <li>
          The output is an AIRR rearrangement table whose positions and CIGARs follow the AIRR schema and which
          Change-O's <code>CreateGermlines.py</code> reads as written; a read cut inside its J is marked partial. It
          has 99 columns, or a preset with <code>--columns</code>, and a{" "}
          <code>.run.json</code> record of the model, germline set and options beside it (
          <DocLink to="output-columns">Output columns explained</DocLink>).
        </li>
      </ul>

      <H2 id="running">Running</H2>
      <ul>
        <li>
          <code>--threads N</code> aligns on the CPU in N worker processes; the batch size follows the device (
          <DocLink to="run-at-scale">Run at scale</DocLink>).
        </li>
        <li>
          Every failure has a stable code, <code>ALIGNAIR_&lt;THING&gt;_&lt;STATE&gt;</code>;{" "}
          <code>--error-format json</code> prints it as JSON and <code>--debug</code> adds the traceback (
          <DocLink to="error-codes">Error codes</DocLink>).
        </li>
      </ul>

      <H2 id="training">Training</H2>
      <ul>
        <li>
          Training reads are simulated with library flanks and, for a built-in germline set, novel alleles, so a model
          calls alleles it never saw from the set it is given.
        </li>
        <li>
          <code>alignair train --init-from MODEL</code> fine-tunes from a model (
          <DocLink to="fine-tune">Fine-tune a model</DocLink>); <code>--data-workers</code> simulates the training
          reads in parallel.
        </li>
        <li>
          <code>alignair export</code> turns a training snapshot into a model to share, with its card, and{" "}
          <code>Aligner.push_to_hub</code> uploads it with its card (<DocLink to="share">Share your model</DocLink>).
        </li>
      </ul>
      <Callout kind="note" title="Python">
        <code>Aligner</code> is the one way to predict, and <code>run_training(TrainingConfig(...))</code> the one way to
        train. The supported names are exported from the <code>alignair</code> package (
        <DocLink to="python-api">Python API</DocLink>).
      </Callout>
    </>
  ),
};

export const projectPages: DocPage[] = [cite, support, changelog];
