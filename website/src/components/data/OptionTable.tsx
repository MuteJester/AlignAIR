import optionsData from "../../data/cli-options.json";

interface Option {
  flags: string[];
  metavar: string | null;
  kind: string;
  default: string | number | boolean | null;
  choices: (string | number)[] | null;
  required: boolean;
  repeatable: boolean;
  help: string;
  anchor: string;
}
interface Command {
  help: string;
  groups: { title: string; options: Option[] }[];
}

const OPTIONS = optionsData as Record<string, Command>;

/** Every option of one `alignair` command (generated from the CLI), one table per option group. */
export default function OptionTable({ command }: { command: string }) {
  const entry = OPTIONS[command];
  if (!entry) return null;
  return (
    <div>
      {entry.groups.map((group) => (
        <div key={group.title} className="my-6">
          <p className="mb-2 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">{group.title}</p>
          <div className="aa-scrollbar overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
            <table className="w-full border-collapse text-left text-sm">
              <tbody>
                {group.options.map((option) => (
                  <tr key={option.anchor} id={option.anchor} className="scroll-mt-24 border-b border-slate-100 align-top last:border-0 dark:border-slate-800/60">
                    <td className="whitespace-nowrap px-4 py-3 font-mono text-[0.85em] font-semibold text-brand-700 dark:text-brand-300">
                      {option.flags.join(", ")}
                      {option.metavar && option.kind !== "flag" ? <span className="text-slate-500"> {option.metavar}</span> : null}
                      {option.required ? <span className="ml-2 rounded bg-amber-100 px-1.5 text-[10px] font-semibold uppercase text-amber-800 dark:bg-amber-500/10 dark:text-amber-300">required</span> : null}
                    </td>
                    <td className="px-4 py-3 text-slate-600 dark:text-slate-300">
                      {option.help}
                      {option.choices ? <span className="block text-xs text-slate-500">One of: {option.choices.join(", ")}</span> : null}
                      {option.default !== null && option.default !== "" ? (
                        <span className="block text-xs text-slate-500">Default: <code>{String(option.default)}</code></span>
                      ) : null}
                      {option.repeatable ? <span className="block text-xs text-slate-500">Repeatable.</span> : null}
                    </td>
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
