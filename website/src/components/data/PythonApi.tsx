import apiData from "../../data/python-api.json";

interface Member {
  name: string;
  kind: string;
  signature: string;
  doc: string;
}
interface Entry {
  name: string;
  kind: string;
  signature: string;
  doc: string;
  members: Member[];
  anchor: string;
}

const API = apiData as Entry[];

/** The public Python API of one package (generated from the code): each name, its signature and its docstring. */
export default function PythonApi({ packageName }: { packageName: string }) {
  const entries = API.filter((entry) => entry.name.slice(0, entry.name.lastIndexOf(".")) === packageName);
  return (
    <div className="space-y-8">
      {entries.map((entry) => (
        <section key={entry.name} id={entry.anchor} className="scroll-mt-24">
          <p className="font-mono text-sm">
            <span className="text-xs uppercase text-slate-500">{entry.kind} </span>
            <span className="font-semibold text-brand-700 dark:text-brand-300">{entry.name}</span>
            <span className="text-slate-600 dark:text-slate-300">{entry.signature}</span>
          </p>
          {entry.doc && <p className="mt-2 whitespace-pre-line text-sm text-slate-600 dark:text-slate-300">{entry.doc}</p>}
          {entry.members.length > 0 && (
            <ul className="mt-3 space-y-3 border-l border-slate-200 pl-4 dark:border-slate-800">
              {entry.members.map((member) => (
                <li key={member.name}>
                  <p className="font-mono text-[0.85em]"><span className="font-semibold">.{member.name}</span>{member.signature}</p>
                  {member.doc && <p className="mt-1 whitespace-pre-line text-sm text-slate-600 dark:text-slate-300">{member.doc}</p>}
                </li>
              ))}
            </ul>
          )}
        </section>
      ))}
    </div>
  );
}
