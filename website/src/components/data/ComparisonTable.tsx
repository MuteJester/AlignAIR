import comparisonData from "../../data/comparison.json";
import { formatDifference, formatMeasure } from "../../lib/measure";

interface Row {
  measure: string;
  igblast: number;
  alignair: number;
  delta: number;
  ci95: [number, number];
  verdict: string;
}
interface Comparison {
  status: "pending" | "ready";
  loci?: Record<string, { reference: string; suites: { name: string; title: string; cases: number; rows: Row[] }[] }>;
}

export const COMPARISON = comparisonData as unknown as Comparison;

function verdictLabel(verdict: string): string {
  if (/b_better|alignair/i.test(verdict) && !/igblast/i.test(verdict)) return "AlignAIR better";
  if (/a_better|igblast/i.test(verdict)) return "IgBLAST better";
  return "tie";
}

/** The comparison with IgBLAST on the locked test split (generated from the comparison run), every measure shown. */
export default function ComparisonTable({ locus }: { locus: string }) {
  const entry = COMPARISON.status === "ready" ? COMPARISON.loci?.[locus] : undefined;
  if (!entry) {
    return <p className="my-4 text-sm text-slate-500">Results of the locked test run appear here.</p>;
  }
  return (
    <div>
      {entry.suites.map((suite) => (
        <div key={suite.name} className="my-6">
          <p className="mb-2 text-sm font-semibold">{suite.title} <span className="font-normal text-slate-500">({suite.cases} reads)</span></p>
          <div className="aa-scrollbar overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
            <table className="w-full border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-900/60">
                  <th className="px-4 py-2 font-semibold">Measure</th>
                  <th className="px-4 py-2 font-semibold">IgBLAST</th>
                  <th className="px-4 py-2 font-semibold">AlignAIR</th>
                  <th className="px-4 py-2 font-semibold">Difference (95% CI)</th>
                  <th className="px-4 py-2 font-semibold">Verdict</th>
                </tr>
              </thead>
              <tbody>
                {suite.rows.map((row) => (
                  <tr key={row.measure} className="border-b border-slate-100 last:border-0 dark:border-slate-800/60">
                    <td className="px-4 py-2 font-mono text-[0.85em]">{row.measure}</td>
                    <td className="px-4 py-2">{formatMeasure(row.measure, row.igblast)}</td>
                    <td className="px-4 py-2">{formatMeasure(row.measure, row.alignair)}</td>
                    <td className="px-4 py-2">{formatDifference(row.measure, row.delta, row.ci95)}</td>
                    <td className="px-4 py-2">{verdictLabel(row.verdict)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
}
