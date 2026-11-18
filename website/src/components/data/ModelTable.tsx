import { useState } from "react";
import { Check, Copy } from "lucide-react";
import modelsData from "../../data/models.json";

interface Model {
  id: string;
  locus: string;
  chain: string;
  species: string;
  germline_set: string;
  naming: string;
  alleles: { V: number; D: number; J: number };
  training_steps: number;
  licence: string;
  anchor: string;
}

export const MODELS = modelsData as Model[];

function CopyId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      aria-label={`Copy ${id}`}
      onClick={() => navigator.clipboard?.writeText(id).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); })}
      className="ml-1 inline-flex rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-slate-800 dark:hover:text-slate-200"
    >
      {copied ? <Check className="h-3.5 w-3.5" aria-hidden="true" /> : <Copy className="h-3.5 w-3.5" aria-hidden="true" />}
    </button>
  );
}

/** The seven pretrained models (generated from the model files): locus, copyable id, germline set and allele counts. */
export default function ModelTable() {
  return (
    <div className="aa-scrollbar my-6 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
      <table className="w-full border-collapse text-left text-sm">
        <thead>
          <tr className="border-b border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-900/60">
            <th className="px-4 py-3 font-semibold">Locus</th>
            <th className="px-4 py-3 font-semibold">Model id</th>
            <th className="px-4 py-3 font-semibold">Germline set</th>
            <th className="px-4 py-3 font-semibold">Allele names</th>
            <th className="px-4 py-3 font-semibold">Alleles V / D / J</th>
          </tr>
        </thead>
        <tbody>
          {MODELS.map((model) => (
            <tr key={model.id} id={model.anchor} className="scroll-mt-24 border-b border-slate-100 last:border-0 dark:border-slate-800/60">
              <td className="px-4 py-3 font-semibold">{model.locus}</td>
              <td className="whitespace-nowrap px-4 py-3 font-mono text-[0.85em] text-brand-700 dark:text-brand-300">{model.id}<CopyId id={model.id} /></td>
              <td className="px-4 py-3 text-slate-600 dark:text-slate-300">{model.germline_set}</td>
              <td className="px-4 py-3 text-slate-600 dark:text-slate-300">{model.naming}</td>
              <td className="px-4 py-3 text-slate-600 dark:text-slate-300">{model.alleles.V} / {model.alleles.D || "–"} / {model.alleles.J}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
