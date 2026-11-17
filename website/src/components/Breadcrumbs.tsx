import { Link } from "react-router-dom";
import { ChevronRight } from "lucide-react";

/** Docs › section › page, above a docs page's title. */
export default function Breadcrumbs({ section, title }: { section: string; title: string }) {
  return (
    <nav aria-label="Breadcrumb" className="mb-3 flex items-center gap-1 text-xs text-slate-500 dark:text-slate-400">
      <Link to="/docs" className="hover:text-slate-900 dark:hover:text-white">Docs</Link>
      <ChevronRight className="h-3 w-3" aria-hidden="true" />
      <span>{section}</span>
      <ChevronRight className="h-3 w-3" aria-hidden="true" />
      <span aria-current="page" className="text-slate-700 dark:text-slate-200">{title}</span>
    </nav>
  );
}
