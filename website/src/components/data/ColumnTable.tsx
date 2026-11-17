import columnsData from "../../data/output-columns.json";

interface Column {
  name: string;
  kind: string;
  unit: string;
  meaning: string;
  presets: string[];
}

export const COLUMNS = columnsData as Column[];

/** Every column AlignAIR writes (generated from the writer): kind, unit, meaning and the presets that include it. */
export default function ColumnTable({ names }: { names?: string[] }) {
  const rows = names ? COLUMNS.filter((column) => names.includes(column.name)) : COLUMNS;
  return (
    <div className="aa-scrollbar my-6 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
      <table className="w-full border-collapse text-left text-sm">
        <thead>
          <tr className="border-b border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-900/60">
            <th className="px-4 py-3 font-semibold">Column</th>
            <th className="px-4 py-3 font-semibold">Kind</th>
            <th className="px-4 py-3 font-semibold">Meaning</th>
            <th className="px-4 py-3 font-semibold">In presets</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((column) => (
            <tr key={column.name} id={column.name} className="scroll-mt-24 border-b border-slate-100 align-top last:border-0 dark:border-slate-800/60">
              <td className="whitespace-nowrap px-4 py-3 font-mono text-[0.85em] font-semibold text-brand-700 dark:text-brand-300">{column.name}</td>
              <td className="whitespace-nowrap px-4 py-3 text-slate-500">{column.kind}{column.unit ? ` (${column.unit})` : ""}</td>
              <td className="px-4 py-3 text-slate-600 dark:text-slate-300">{column.meaning}</td>
              <td className="px-4 py-3 text-xs text-slate-500">{column.presets.join(", ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
