import { CodeBlock, Callout } from "../../components/ui";
import { DocTable, type DocPage } from "./doc-kit";

const BIBTEX = `@article{Konstantinovsky2025AlignAIR,
  author  = {Konstantinovsky, Thomas and Peres, Ayelet and Eisenberg, Ran
             and Polak, Pazit and Lindenbaum, Ofir and Yaari, Gur},
  title   = {Enhancing sequence alignment of adaptive immune receptors
             through multi-task deep learning},
  journal = {Nucleic Acids Research},
  year    = {2025},
  month   = jul,
  volume  = {53},
  number  = {13},
  pages   = {gkaf651},
  doi     = {10.1093/nar/gkaf651},
  url     = {https://doi.org/10.1093/nar/gkaf651}
}`;

const citationSupport: DocPage = {
  slug: "citation-support",
  title: "Citation & support",
  section: "Community",
  lead: "How to cite AlignAIR, and where to send bugs, questions, feature requests, and security reports.",
  body: () => (
    <>
      <h2>Citing AlignAIR</h2>
      <p>
        If AlignAIR contributes to your work, please cite the paper:
      </p>
      <p>
        Konstantinovsky T, Peres A, Eisenberg R, Polak P, Lindenbaum O, Yaari G.{" "}
        <strong>Enhancing sequence alignment of adaptive immune receptors through multi-task deep learning.</strong>{" "}
        <em>Nucleic Acids Research</em>, Volume 53, Issue 13, 22 July 2025, gkaf651.{" "}
        <a href="https://doi.org/10.1093/nar/gkaf651" target="_blank" rel="noopener noreferrer">
          https://doi.org/10.1093/nar/gkaf651
        </a>
      </p>
      <Callout kind="tip" title="Cite this repository">
        The repository ships a{" "}
        <a href="https://github.com/MuteJester/AlignAIR/blob/main/CITATION.cff" target="_blank" rel="noreferrer">
          <code>CITATION.cff</code>
        </a>
        , so GitHub shows a &ldquo;Cite this repository&rdquo; button that exports the reference as BibTeX or APA
        automatically.
      </Callout>
      <h3>BibTeX</h3>
      <CodeBlock lang="bibtex" title="BibTeX" code={BIBTEX} />

      <h2>Getting help</h2>
      <p>
        Search the{" "}
        <a href="https://github.com/MuteJester/AlignAIR/issues" target="_blank" rel="noreferrer">
          existing issues
        </a>{" "}
        first, then pick the route that fits:
      </p>
      <DocTable
        head={["What you have", "Where it goes"]}
        rows={[
          [
            <>A bug or unexpected behaviour</>,
            <>
              GitHub Issues,{" "}
              <a href="https://github.com/MuteJester/AlignAIR/issues/new?template=bug_report.md" target="_blank" rel="noreferrer">
                Bug report
              </a>{" "}
              template (see the checklist below).
            </>,
          ],
          [
            <>A feature request</>,
            <>
              GitHub Issues,{" "}
              <a href="https://github.com/MuteJester/AlignAIR/issues/new?template=feature_request.md" target="_blank" rel="noreferrer">
                Feature request
              </a>{" "}
              template.
            </>,
          ],
          [
            <>A pretrained model for a species / locus we do not cover</>,
            <>
              GitHub Issues,{" "}
              <a href="https://github.com/MuteJester/AlignAIR/issues/new?template=model_or_reference_request.md" target="_blank" rel="noreferrer">
                Model or reference request
              </a>{" "}
              template.
            </>,
          ],
          [
            <>A usage or &ldquo;how do I&rdquo; question</>,
            <>
              Open a GitHub issue after searching - a question that turns out to be a bug or a docs gap becomes a fix
              for everyone.
            </>,
          ],
          [
            <>A scientific or methods question about the paper</>,
            <>Contact the corresponding authors listed on the paper.</>,
          ],
          [
            <>A security-sensitive report</>,
            <>
              <strong>Do not open a public issue.</strong> Email{" "}
              <a href="mailto:thomaskon90@gmail.com">thomaskon90@gmail.com</a> privately - see the{" "}
              <a href="https://github.com/MuteJester/AlignAIR/blob/main/SECURITY.md" target="_blank" rel="noreferrer">
                security policy
              </a>
              .
            </>,
          ],
        ]}
      />

      <h2>What a useful bug report contains</h2>
      <p>
        The five things that turn a report we cannot act on into one we can fix quickly:
      </p>
      <ul>
        <li><strong>AlignAIR version</strong> - <code>alignair --version</code> (for example, <code>alignair 3.0.0</code>).</li>
        <li><strong>Model and version</strong> - the id you passed, pinned: e.g. <code>alignair-igh-human@1.0.0</code>, or the path to your <code>.alignair</code> file.</li>
        <li><strong>Environment</strong> - the full output of <code>alignair doctor</code> (Python, PyTorch + CUDA/MPS, GenAIRR, parasail, and the cache directory).</li>
        <li><strong>The exact command</strong> you ran, verbatim.</li>
        <li><strong>A minimal reproducer</strong> - the smallest input that triggers it (a handful of reads is ideal), plus what you expected versus what happened.</li>
      </ul>
      <CodeBlock
        code={`alignair --version        # AlignAIR version\nalignair doctor           # environment: Python, torch + device, GenAIRR, parasail, cache dir`}
      />
      <Callout kind="note" title="A tiny reproducer beats a big one">
        A three-read FASTA that reproduces the problem is worth more than a whole run. If the input is sensitive, a
        synthetic sequence that triggers the same error is perfect.
      </Callout>
    </>
  ),
};

export const communityPages: DocPage[] = [citationSupport];
