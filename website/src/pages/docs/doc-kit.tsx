import type { ReactNode } from "react";
import { Link } from "react-router-dom";

/** Internal link to another reference page. */
export function DocLink({ to, children }: { to: string; children: ReactNode }) {
  return <Link to={`/docs/${to}`}>{children}</Link>;
}

/** Internal link to a lesson track/overview. */
export function LearnLink({ to, children }: { to: string; children: ReactNode }) {
  return <Link to={to}>{children}</Link>;
}

/** A reference page authored in TSX. */
export interface DocPage {
  slug: string;
  title: string;
  section: string;
  lead?: ReactNode;
  body: () => ReactNode;
}

/** Shared building blocks so reference pages stay compact (no markdown). */

export function PageHeader({ title, lead }: { title: string; lead?: ReactNode }) {
  return (
    <header className="mb-8 border-b border-slate-200 pb-6 dark:border-slate-800">
      <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white sm:text-4xl">{title}</h1>
      {lead && <p className="mt-3 text-lg text-slate-500 dark:text-slate-400">{lead}</p>}
    </header>
  );
}

/** A styled table from a head row + body rows (cells may be strings or JSX). */
export function DocTable({ head, rows }: { head: ReactNode[]; rows: ReactNode[][] }) {
  return (
    <div
      tabIndex={0}
      role="region"
      aria-label="Scrollable data table"
      className="aa-scrollbar my-6 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800"
    >
      <table className="w-full border-collapse text-left text-sm">
        <thead>
          <tr className="border-b border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-900/60">
            {head.map((h, i) => (
              <th key={i} className="px-4 py-3 font-semibold text-slate-700 dark:text-slate-200">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr
              key={ri}
              className="border-b border-slate-100 last:border-0 dark:border-slate-800/60 hover:bg-slate-50/60 dark:hover:bg-slate-900/30"
            >
              {row.map((cell, ci) => (
                <td
                  key={ci}
                  className="px-4 py-3 align-top text-slate-600 dark:text-slate-300 [&_code]:rounded [&_code]:bg-slate-100 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.85em] [&_code]:text-brand-700 dark:[&_code]:bg-slate-800 dark:[&_code]:text-brand-300"
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Section heading with a stable id for anchor links. */
export function H2({ id, children }: { id?: string; children: ReactNode }) {
  return <h2 id={id}>{children}</h2>;
}

export function H3({ id, children }: { id?: string; children: ReactNode }) {
  return <h3 id={id}>{children}</h3>;
}
