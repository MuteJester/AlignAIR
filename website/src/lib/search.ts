/** The search box's ranking over the generated index (`src/data/search-index.json`). Pure, so `npm test` checks it. */
export interface SearchEntry {
  kind: "page" | "heading" | "option" | "column" | "model" | "error";
  title: string;
  text: string;
  href: string;
  /** A page's lead or a section's prose: matched after every title and text. */
  body?: string;
}

const KIND_ORDER: SearchEntry["kind"][] = ["option", "column", "model", "error", "page", "heading"];

/**
 * The entries matching `query`, best first: a title equal to the query, then one with a word equal to it (words split
 * at spaces, dashes and underscores too, so "TRA" finds alignair4-tra-human), then one starting with it, then one with
 * a word starting with it, then one containing it, then a match in the text, then in the body (a section's prose).
 * Case-insensitive; a flag's dashes count.
 */
export function rank(index: SearchEntry[], query: string, limit = 20): SearchEntry[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const scored: { entry: SearchEntry; score: number }[] = [];
  for (const entry of index) {
    const title = entry.title.toLowerCase();
    const text = entry.text.toLowerCase();
    const words = title.split(/[\s,/·_-]+/);
    let score = -1;
    if (title === q || title.split(", ").includes(q)) score = 0;
    else if (words.includes(q)) score = 1;
    else if (title.startsWith(q)) score = 2;
    else if (title.split(/[\s,/·]+/).some((word) => word.startsWith(q))) score = 3;
    else if (title.includes(q)) score = 4;
    else if (text.includes(q)) score = 5;
    else if (entry.body?.toLowerCase().includes(q)) score = 6;
    if (score >= 0) scored.push({ entry, score });
  }
  return scored
    .sort((a, b) => a.score - b.score || KIND_ORDER.indexOf(a.entry.kind) - KIND_ORDER.indexOf(b.entry.kind)
      || a.entry.title.length - b.entry.title.length)
    .slice(0, limit)
    .map((item) => item.entry);
}
