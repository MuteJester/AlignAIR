import errorsData from "../../data/error-codes.json";

interface ErrorCode {
  code: string;
  class: string;
  category: string;
  exit_code: number;
  retryable: boolean;
  meaning: string;
  action: string;
  anchor: string;
}

const ERRORS = errorsData as ErrorCode[];

/** Every error code (generated from alignair.errors): what it means and what to do. */
export default function ErrorTable() {
  return (
    <div className="aa-scrollbar my-6 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
      <table className="w-full border-collapse text-left text-sm">
        <thead>
          <tr className="border-b border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-900/60">
            <th className="px-4 py-3 font-semibold">Code</th>
            <th className="px-4 py-3 font-semibold">Meaning</th>
            <th className="px-4 py-3 font-semibold">What to do</th>
          </tr>
        </thead>
        <tbody>
          {ERRORS.map((error) => (
            <tr key={`${error.code}-${error.class}`} id={error.anchor} className="scroll-mt-24 border-b border-slate-100 align-top last:border-0 dark:border-slate-800/60">
              <td className="px-4 py-3">
                <span className="font-mono text-[0.85em] font-semibold text-brand-700 dark:text-brand-300">{error.code}</span>
                <span className="block text-xs text-slate-500">{error.class} · exit {error.exit_code}{error.retryable ? " · retryable" : ""}</span>
              </td>
              <td className="px-4 py-3 text-slate-600 dark:text-slate-300">{error.meaning}</td>
              <td className="px-4 py-3 text-slate-600 dark:text-slate-300">{error.action}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
