import { useEffect } from "react";
import { Navigate, useParams, Link, NavLink, useNavigate } from "react-router-dom";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { DOC_SECTIONS, getDoc, firstDocSlug, adjacentDocs } from "./docs";
import { PageHeader } from "./docs/doc-kit";
import { cn } from "../lib/util";

function Sidebar({ current }: { current: string }) {
  return (
    <nav className="space-y-6">
      {DOC_SECTIONS.map((section) => (
        <div key={section.title}>
          <p className="mb-2 px-3 text-xs font-bold uppercase tracking-wider text-slate-400">{section.title}</p>
          <ul className="space-y-0.5">
            {section.pages.map((p) => (
              <li key={p.slug}>
                <NavLink
                  to={`/docs/${p.slug}`}
                  className={cn(
                    "block rounded-lg px-3 py-1.5 text-sm transition-colors",
                    p.slug === current
                      ? "bg-brand-50 font-semibold text-brand-700 dark:bg-brand-500/10 dark:text-brand-300"
                      : "text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-white",
                  )}
                >
                  {p.title}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}

export default function DocsPage() {
  const { slug } = useParams<{ slug: string }>();
  const doc = getDoc(slug);
  const navigate = useNavigate();

  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [slug]);

  if (!slug) return <Navigate to={`/docs/${firstDocSlug()}`} replace />;
  if (!doc) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-24 text-center">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Page not found</h1>
        <p className="mt-2 text-slate-500">
          <Link to={`/docs/${firstDocSlug()}`} className="text-brand-600 underline">
            Back to the docs
          </Link>
        </p>
      </div>
    );
  }

  const { prev, next } = adjacentDocs(doc.slug);

  const handleMobileNavChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    navigate(`/docs/${e.target.value}`);
  };

  return (
    <div className="mx-auto flex max-w-7xl gap-8 px-4 py-8 sm:px-6">
      <aside className="hidden w-56 shrink-0 lg:block">
        <div className="sticky top-24 aa-scrollbar max-h-[calc(100vh-7rem)] overflow-y-auto pb-8">
          <Sidebar current={doc.slug} />
        </div>
      </aside>

      <article className="min-w-0 flex-1 max-w-3xl">
        <div className="mb-6 lg:hidden">
          <label htmlFor="docs-mobile-select" className="sr-only">Select documentation page</label>
          <select
            id="docs-mobile-select"
            value={doc.slug}
            onChange={handleMobileNavChange}
            className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-800 shadow-sm focus:border-brand-500 focus:outline-none dark:border-slate-800 dark:bg-slate-900 dark:text-slate-100"
          >
            {DOC_SECTIONS.map((section) => (
              <optgroup key={section.title} label={section.title}>
                {section.pages.map((p) => (
                  <option key={p.slug} value={p.slug}>
                    {p.title}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>

        <PageHeader title={doc.title} lead={doc.lead} />
        <div className="prose-aa max-w-none">{doc.body()}</div>

        <nav className="mt-16 grid grid-cols-1 gap-4 border-t border-slate-200 pt-8 sm:grid-cols-2 dark:border-slate-800">
          {prev ? (
            <Link
              to={`/docs/${prev.slug}`}
              className="group flex flex-col rounded-xl border border-slate-200 p-4 transition-colors hover:border-brand-400 dark:border-slate-800 dark:hover:border-brand-500/50"
            >
              <span className="inline-flex items-center gap-1 text-xs text-slate-400">
                <ArrowLeft className="h-3.5 w-3.5" /> Previous
              </span>
              <span className="mt-1 font-semibold text-slate-800 group-hover:text-brand-600 dark:text-slate-100">
                {prev.title}
              </span>
            </Link>
          ) : (
            <div />
          )}
          {next && (
            <Link
              to={`/docs/${next.slug}`}
              className="group flex flex-col items-end rounded-xl border border-slate-200 p-4 text-right transition-colors hover:border-brand-400 dark:border-slate-800 dark:hover:border-brand-500/50"
            >
              <span className="inline-flex items-center gap-1 text-xs text-slate-400">
                Next <ArrowRight className="h-3.5 w-3.5" />
              </span>
              <span className="mt-1 font-semibold text-slate-800 group-hover:text-brand-600 dark:text-slate-100">
                {next.title}
              </span>
            </Link>
          )}
        </nav>
      </article>
    </div>
  );
}
