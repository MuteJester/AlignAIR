import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search } from "lucide-react";
import indexData from "../data/search-index.json";
import { rank, type SearchEntry } from "../lib/search";

const INDEX = indexData as SearchEntry[];
const KIND_LABEL: Record<SearchEntry["kind"], string> = {
  page: "Page", heading: "Section", option: "Option", column: "Column", model: "Model", error: "Error code",
};

/** The search button of the top bar and its dialog: Ctrl/Cmd-K or "/" opens it, arrows move, Enter opens, Esc closes. */
export default function SearchPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const results = useMemo(() => rank(INDEX, query), [query]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const typing = event.target instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName);
      if ((event.key === "k" && (event.metaKey || event.ctrlKey)) || (event.key === "/" && !typing)) {
        event.preventDefault();
        setOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (open) setTimeout(() => input.current?.focus(), 0);
    else setQuery("");
  }, [open]);

  useEffect(() => setActive(0), [query]);

  const go = (entry: SearchEntry) => {
    setOpen(false);
    navigate(entry.href.replace(/^#/, ""));
  };

  const onInputKey = (event: React.KeyboardEvent) => {
    if (event.key === "Escape") setOpen(false);
    else if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive((i) => Math.min(i + 1, results.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (event.key === "Enter" && results[active]) go(results[active]);
  };

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-500 transition-colors hover:border-brand-400 hover:text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-400 dark:hover:text-slate-200"
        aria-label="Search the docs"
      >
        <Search className="h-4 w-4" aria-hidden="true" />
        <span className="hidden lg:inline">Search flags, columns, models…</span>
        <kbd className="hidden rounded border border-slate-200 px-1.5 text-[11px] font-medium text-slate-400 lg:inline dark:border-slate-700">Ctrl K</kbd>
      </button>
      {open && (
        <div
          className="fixed inset-0 z-[60] flex items-start justify-center bg-slate-900/40 px-4 pt-[12vh] backdrop-blur-sm"
          onMouseDown={(event) => event.target === event.currentTarget && setOpen(false)}
        >
          <div role="dialog" aria-modal="true" aria-label="Search the docs" className="w-full max-w-xl overflow-hidden rounded-xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-slate-900">
            <div className="flex items-center gap-2 border-b border-slate-200 px-4 dark:border-slate-700">
              <Search className="h-4 w-4 text-slate-400" aria-hidden="true" />
              <input
                ref={input}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={onInputKey}
                placeholder="Search pages, options (--genotype), columns (v_call), models, error codes"
                aria-label="Search"
                className="w-full bg-transparent py-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:text-slate-100"
              />
              <kbd className="rounded border border-slate-200 px-1.5 text-[11px] text-slate-400 dark:border-slate-700">Esc</kbd>
            </div>
            <ul role="listbox" className="aa-scrollbar max-h-[60vh] overflow-y-auto py-2">
              {query.trim() && results.length === 0 && (
                <li className="px-4 py-6 text-center text-sm text-slate-500">No results for “{query.trim()}”</li>
              )}
              {results.map((entry, i) => (
                <li key={`${entry.href}-${entry.title}`} role="option" aria-selected={i === active}>
                  <button
                    type="button"
                    onMouseEnter={() => setActive(i)}
                    onClick={() => go(entry)}
                    className={`flex w-full flex-col items-start px-4 py-2 text-left ${i === active ? "bg-brand-50 dark:bg-brand-500/10" : ""}`}
                  >
                    <span className="flex items-center gap-2">
                      <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-slate-500 dark:bg-slate-800 dark:text-slate-400">{KIND_LABEL[entry.kind]}</span>
                      <span className="font-mono text-sm font-semibold text-slate-900 dark:text-slate-100">{entry.title}</span>
                    </span>
                    <span className="mt-0.5 line-clamp-1 text-xs text-slate-500 dark:text-slate-400">{entry.text}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </>
  );
}
