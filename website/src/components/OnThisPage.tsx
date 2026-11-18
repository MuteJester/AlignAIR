import { useEffect, useState, type RefObject } from "react";
import { Link } from "react-router-dom";
import { headingPath } from "../lib/anchors";

interface Item {
  id: string;
  text: string;
  level: number;
}

/** "On this page": the page's headings (h2, h3 with an id), the one in view highlighted; a link keeps the route. */
export default function OnThisPage({ container, slug }: { container: RefObject<HTMLElement>; slug: string }) {
  const [items, setItems] = useState<Item[]>([]);
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    const root = container.current;
    if (!root) return;
    const headings = Array.from(root.querySelectorAll<HTMLElement>("h2[id], h3[id]"));
    setItems(headings.map((h) => ({ id: h.id, text: h.textContent ?? "", level: h.tagName === "H3" ? 3 : 2 })));
    if (typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting);
        if (visible.length) setActive(visible[0].target.id);
      },
      { rootMargin: "-80px 0px -70% 0px" },
    );
    headings.forEach((h) => observer.observe(h));
    return () => observer.disconnect();
  }, [container, slug]);

  if (items.length < 2) return null;
  return (
    <nav aria-label="On this page" className="text-sm">
      <p className="mb-2 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">On this page</p>
      <ul className="space-y-1.5 border-l border-slate-200 dark:border-slate-800">
        {items.map((item) => (
          <li key={item.id} className={item.level === 3 ? "pl-6" : "pl-3"}>
            <Link
              to={headingPath(slug, item.id)}
              className={
                item.id === active
                  ? "font-semibold text-brand-700 dark:text-brand-300"
                  : "text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white"
              }
            >
              {item.text}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}
