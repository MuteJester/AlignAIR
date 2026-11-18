import { COMPARISON } from "./ComparisonTable";

interface RealMeasure { igblast: number; alignair: number; delta: number; ci95: [number, number]; verdict: string;
  paired: number; clonotypes: number }
interface RealScope { donors: string[]; measures: Record<string, RealMeasure> }
interface Real { donors: string[]; scopes: Record<string, RealScope> }

const SCOPES: [string, string][] = [["race", "5' RACE"], ["multiplex", "Multiplex PCR"], ["pooled", "All four donors"]];
const pct = (value: number) => `${(100 * value).toFixed(1)}%`;
const verdict = (v: string) => (v === "b_better" ? "AlignAIR better" : v === "a_better" ? "IgBLAST better" : "no difference");

function Cell({ m }: { m?: RealMeasure }) {
  if (!m) return <td className="block px-4 py-1 sm:table-cell sm:py-3">–</td>;
  return (
    <td className="block px-4 py-1 sm:table-cell sm:py-3">
      {pct(m.igblast)} → {pct(m.alignair)} <span className="text-slate-500">({verdict(m.verdict)})</span>
    </td>
  );
}

/** The locked real-data run (IGH donors with genomic truth): its declared measure per protocol, generated. */
export default function RealComparison() {
  const real = (COMPARISON as unknown as { real?: Real }).real;
  if (COMPARISON.status !== "ready" || !real) {
    return <p className="my-4 text-sm text-slate-500">Results of the locked test run appear here.</p>;
  }
  return (
    <div className="aa-scrollbar my-6 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
      <table className="w-full border-collapse text-left text-sm">
        <thead className="hidden sm:table-header-group">
          <tr className="border-b border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-900/60">
            <th className="px-4 py-3 font-semibold">Library protocol (donors)</th>
            <th className="px-4 py-3 font-semibold">V calls the donor's genome lacks (lower is better): IgBLAST → AlignAIR</th>
            <th className="px-4 py-3 font-semibold">Reads with a known answer called right: IgBLAST → AlignAIR</th>
          </tr>
        </thead>
        <tbody>
          {SCOPES.filter(([key]) => real.scopes[key]).map(([key, label]) => (
            <tr key={key} className="block border-b border-slate-100 py-2 last:border-0 sm:table-row sm:py-0 dark:border-slate-800/60">
              <td className="block px-4 py-1 font-semibold sm:table-cell sm:py-3">
                {label} <span className="font-normal text-slate-500">({real.scopes[key].donors.join(", ")})</span>
              </td>
              <Cell m={real.scopes[key].measures["real.v.unsupported"]} />
              <Cell m={real.scopes[key].measures["real.v.known_top1"]} />
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
