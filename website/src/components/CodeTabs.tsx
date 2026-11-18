import { useEffect, useState } from "react";
import { CodeBlock } from "./ui";
import { readTab, writeTab, type Tab } from "../lib/tabs";

const EVENT = "alignair:tab";

function store(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

/**
 * One example in its CLI and Python forms, with a tab each. Every CodeTabs on every page switches together, and the
 * choice is remembered.
 */
export default function CodeTabs({ cli, python, cliTitle = "shell", pythonTitle = "python" }: {
  cli: string;
  python: string;
  cliTitle?: string;
  pythonTitle?: string;
}) {
  const [tab, setTab] = useState<Tab>(() => readTab(store()));
  useEffect(() => {
    const onChange = (event: Event) => setTab((event as CustomEvent<Tab>).detail);
    window.addEventListener(EVENT, onChange);
    return () => window.removeEventListener(EVENT, onChange);
  }, []);
  const choose = (next: Tab) => {
    writeTab(store(), next);
    window.dispatchEvent(new CustomEvent<Tab>(EVENT, { detail: next }));
  };
  const button = (value: Tab, label: string) => (
    <button
      type="button"
      role="tab"
      aria-selected={tab === value}
      onClick={() => choose(value)}
      className={
        tab === value
          ? "border-b-2 border-brand-500 px-3 py-1.5 text-sm font-semibold text-slate-900 dark:text-white"
          : "px-3 py-1.5 text-sm text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200"
      }
    >
      {label}
    </button>
  );
  return (
    <div className="my-5">
      <div role="tablist" aria-label="Example form" className="flex gap-1 border-b border-slate-200 dark:border-slate-800">
        {button("cli", "CLI")}
        {button("python", "Python")}
      </div>
      {tab === "cli" ? <CodeBlock code={cli} title={cliTitle} /> : <CodeBlock code={python} lang="python" title={pythonTitle} />}
    </div>
  );
}
