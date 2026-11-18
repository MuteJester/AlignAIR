import { COMPARISON } from "./ComparisonTable";
import { formatMeasure } from "../../lib/measure";
import { winsByFamily, type ComparisonLocus } from "../../lib/wins";

/** Where AlignAIR beats IgBLAST in the locked comparison, by family of measures (generated from comparison.json). */
export default function ComparisonWins() {
  if (COMPARISON.status !== "ready" || !COMPARISON.loci) {
    return <p className="my-4 text-sm text-slate-500">Results of the locked test run appear here.</p>;
  }
  const families = winsByFamily(COMPARISON.loci as Record<string, ComparisonLocus>);
  return (
    <div className="aa-scrollbar my-6 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
      <table className="w-full border-collapse text-left text-sm">
        <thead className="hidden sm:table-header-group">
          <tr className="border-b border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-900/60">
            <th className="px-4 py-3 font-semibold">Measure</th>
            <th className="px-4 py-3 font-semibold">AlignAIR better in</th>
            <th className="px-4 py-3 font-semibold">Loci</th>
            <th className="px-4 py-3 font-semibold">Largest gain: IgBLAST → AlignAIR</th>
          </tr>
        </thead>
        <tbody>
          {families.map((family) => (
            <tr key={family.label} className="block border-b border-slate-100 py-2 last:border-0 sm:table-row sm:py-0 dark:border-slate-800/60">
              <td className="block px-4 py-1 font-semibold sm:table-cell sm:py-3">{family.label}</td>
              <td className="block px-4 py-1 sm:table-cell sm:py-3">
                {family.wins} of {family.comparisons} comparisons
              </td>
              <td className="block px-4 py-1 sm:table-cell sm:py-3">{family.loci.join(", ")}</td>
              <td className="block px-4 py-1 text-slate-600 sm:table-cell sm:py-3 dark:text-slate-300">
                {formatMeasure(family.example.measure, family.example.igblast)} →{" "}
                {formatMeasure(family.example.measure, family.example.alignair)}{" "}
                <span className="text-slate-500">({family.example.locus}, {family.example.suite})</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
