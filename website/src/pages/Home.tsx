import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { ArrowLeftRight, ArrowRight, BookOpen, FlaskConical, Play, Quote, Scale, TriangleAlert } from "lucide-react";
import { CodeBlock } from "../components/ui";
import ModelTable from "../components/data/ModelTable";
import { H2 } from "./docs/doc-kit";

// The first rows of the real output of the command below (the human IGH model on examples/reads.fasta).
const HEADER = ["sequence_id", "v_call", "d_call", "j_call", "junction_aa", "productive"];
const ROWS = [
  ["read01", "IGHVF10-G50*02", "IGHD3-10*01", "IGHJ6*02", "CATFYYGSGSYYNFVSNYNYYGMDVW", "T"],
  ["read02", "IGHVF10-G42*01,IGHVF10-G42*03", "IGHD2-2*02", "IGHJ4*02", "CAKGTSCNSTSCYTIDYW", "T"],
  ["read03", "IGHVF3-G12*02", "IGHD5-18*02", "IGHJ6*03", "CARDTLNIVATPISAFAIYYYYYMDVW", "T"],
];

const DOORS = [
  {
    to: "/docs/first-alignment",
    icon: Play,
    title: "Run on your data",
    body: "Your first alignment, then input formats, donor genotypes, your own germline set and output columns.",
  },
  {
    to: "/docs/from-igblast",
    icon: ArrowLeftRight,
    title: "Coming from IgBLAST, MiXCR or Immcantation",
    body: "Your commands, germline databases and columns, mapped to AlignAIR's.",
    links: [
      { to: "/docs/from-igblast", label: "IgBLAST" },
      { to: "/docs/from-mixcr", label: "MiXCR" },
      { to: "/docs/from-immcantation", label: "Immcantation" },
    ],
  },
  {
    to: "/docs/do-i-need-to-train",
    icon: FlaskConical,
    title: "Train or fine-tune",
    body: "A model for another species, locus or germline set, and when you do not need one.",
  },
];

const EVIDENCE = [
  { to: "/docs/comparison", icon: Scale, title: "Comparison with IgBLAST", body: "Every measure on every locus, ties included, and the method." },
  { to: "/docs/trust-the-calls", icon: TriangleAlert, title: "Where it fails", body: "Known failure modes, and what to do about each." },
  { href: "https://doi.org/10.1093/nar/gkaf651", icon: BookOpen, title: "The paper", body: "Konstantinovsky et al., Nucleic Acids Research, 2025." },
  { to: "/docs/cite", icon: Quote, title: "Cite AlignAIR", body: "The reference and its BibTeX." },
];

function SectionTitle({ id, children }: { id: string; children: ReactNode }) {
  return (
    <div className="font-display text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
      <H2 id={id}>{children}</H2>
    </div>
  );
}

const card =
  "group flex h-full flex-col rounded-xl border border-slate-200 bg-white p-5 transition-colors hover:border-brand-400 " +
  "dark:border-slate-800 dark:bg-slate-900 dark:hover:border-brand-500/60";

export default function Home() {
  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6">
      {/* One-liner */}
      <section className="pt-14 sm:pt-20">
        <h1 className="font-display text-4xl font-bold leading-tight tracking-tight text-slate-900 sm:text-5xl dark:text-white">
          V(D)J allele assignment for IG and TCR repertoires
        </h1>
        <p className="mt-5 max-w-3xl text-lg leading-relaxed text-slate-600 dark:text-slate-300">
          Reads in (FASTA, FASTQ or a table), an AIRR rearrangement table out, ready for Change-O, TIgGER and SCOPer.
          Pretrained human models for IGH, IGK, IGL, TRA, TRB, TRD and TRG.
        </p>
      </section>

      {/* Install, one command, and what you get back */}
      <section aria-label="Quick start" className="mt-8">
        {/* PENDING-V4-PUBLISH: the release tag in this URL is set when the release is cut. */}
        <CodeBlock
          title="shell"
          code={`$ pip install alignair\n$ curl -LO https://raw.githubusercontent.com/MuteJester/AlignAIR/v4.0.0/examples/reads.fasta\n$ alignair predict --model alignair4-igh-human --input reads.fasta --out out.tsv\naligned 20 reads (0 dropped) -> out.tsv`}
        />
        <div className="overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
          <p className="border-b border-slate-200 px-4 py-2 text-xs font-semibold text-brand-700 dark:border-slate-800 dark:text-brand-300">
            out.tsv, an AIRR rearrangement table (first rows and columns)
          </p>
          <div tabIndex={0} role="region" aria-label="First rows of out.tsv" className="aa-scrollbar overflow-x-auto">
            <table className="w-full border-collapse text-left font-mono text-[0.8rem]">
              <thead>
                <tr className="text-slate-500 dark:text-slate-400">
                  {HEADER.map((name) => (
                    <th key={name} className="whitespace-nowrap px-4 py-2 font-medium">
                      {name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ROWS.map((row) => (
                  <tr key={row[0]} className="border-t border-slate-100 text-slate-800 dark:border-slate-800/60 dark:text-slate-200">
                    {row.map((value, i) => (
                      <td key={HEADER[i]} className="whitespace-nowrap px-4 py-2">
                        {value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <p className="mt-4 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
          The first run downloads the model and caches it. A GPU is optional. When a read cannot tell V alleles apart,{" "}
          <code>v_call</code> lists them all (read02), as IgBLAST does.{" "}
          <Link to="/docs/install" className="font-semibold text-brand-700 underline underline-offset-2 dark:text-brand-300">
            Install options
          </Link>{" "}
          ·{" "}
          <Link to="/docs/first-alignment" className="font-semibold text-brand-700 underline underline-offset-2 dark:text-brand-300">
            Read the table
          </Link>
        </p>
      </section>

      {/* Three doors */}
      <section aria-label="Where to go next" className="mt-12 grid gap-4 sm:grid-cols-3">
        {DOORS.map(({ to, icon: Icon, title, body, links }) =>
          links ? (
            <div key={to} className={card}>
              <Icon aria-hidden="true" className="h-5 w-5 text-brand-600 dark:text-brand-400" />
              <span className="mt-3 font-semibold text-slate-900 dark:text-white">{title}</span>
              <span className="mt-1.5 flex-1 text-sm leading-relaxed text-slate-600 dark:text-slate-400">{body}</span>
              <span className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-sm font-semibold">
                {links.map((link) => (
                  <Link key={link.to} to={link.to} className="text-brand-700 underline underline-offset-2 dark:text-brand-300">
                    {link.label}
                  </Link>
                ))}
              </span>
            </div>
          ) : (
          <Link key={to} to={to} className={card}>
            <Icon aria-hidden="true" className="h-5 w-5 text-brand-600 dark:text-brand-400" />
            <span className="mt-3 font-semibold text-slate-900 dark:text-white">{title}</span>
            <span className="mt-1.5 flex-1 text-sm leading-relaxed text-slate-600 dark:text-slate-400">{body}</span>
            <ArrowRight
              aria-hidden="true"
              className="mt-3 h-4 w-4 text-slate-400 transition-transform group-hover:translate-x-0.5 group-hover:text-brand-600 dark:group-hover:text-brand-400"
            />
          </Link>
          ),
        )}
      </section>

      {/* The seven models */}
      <section className="mt-16">
        <SectionTitle id="pretrained-models">Pretrained models</SectionTitle>
        <p className="mt-2 text-slate-600 dark:text-slate-400">
          One model per locus. Copy an id into <code className="font-mono text-[0.9em]">--model</code>.
        </p>
        <ModelTable />
        <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-400">
          The IG models name alleles as OGRDB does, the TCR models as IMGT does. Another germline set, or novel
          alleles? Bind the germline FASTA with{" "}
          <Link to="/docs/own-reference" className="whitespace-nowrap font-semibold text-brand-700 underline underline-offset-2 dark:text-brand-300">
            --reference
          </Link>
          . Another species?{" "}
          <Link to="/docs/do-i-need-to-train" className="font-semibold text-brand-700 underline underline-offset-2 dark:text-brand-300">
            Fine-tune or train a model
          </Link>
          .
        </p>
      </section>

      {/* Evidence strip */}
      <section className="mt-16 border-t border-slate-200 pt-10 pb-4 dark:border-slate-800">
        <SectionTitle id="how-far-to-trust-it">How far to trust it</SectionTitle>
        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {EVIDENCE.map(({ to, href, icon: Icon, title, body }) => {
            const inner = (
              <>
                <Icon aria-hidden="true" className="h-5 w-5 text-brand-600 dark:text-brand-400" />
                <span className="mt-3 font-semibold text-slate-900 dark:text-white">{title}</span>
                <span className="mt-1.5 text-sm leading-relaxed text-slate-600 dark:text-slate-400">{body}</span>
              </>
            );
            return to ? (
              <Link key={title} to={to} className={card}>
                {inner}
              </Link>
            ) : (
              <a key={title} href={href} target="_blank" rel="noreferrer" className={card}>
                {inner}
              </a>
            );
          })}
        </div>
      </section>
    </div>
  );
}
