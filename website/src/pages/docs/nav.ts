/**
 * The docs' navigation: which sections a reader sees, in what order, and where Previous/Next lead. Pure (no React),
 * so `npm test` checks it. User pages and developer pages are two separate journeys: the sidebar shows the current
 * one, and Previous/Next never cross from one to the other.
 */
export const USER_SECTIONS = ["Get started", "Coming from…", "Use on your data", "Understand the results",
  "Reference", "Train or fine-tune", "Project"];
export const DEVELOPER_SECTIONS = ["Developers"];
// Sections of pages written before the restructure, shown to developers until those pages are retired.
const LEGACY_DEVELOPER_SECTIONS = ["Design", "Maintainer"];

export type Audience = "user" | "developer";

export interface NavPage {
  slug: string;
  title: string;
  section: string;
}

export function audienceOf(section: string): Audience {
  return DEVELOPER_SECTIONS.includes(section) || LEGACY_DEVELOPER_SECTIONS.includes(section) ? "developer" : "user";
}

/** The audience's sections in reading order (a section the order does not name comes after, as first seen). */
export function sectionsFor<T extends NavPage>(pages: T[], audience: Audience): { title: string; pages: T[] }[] {
  const order = audience === "user" ? USER_SECTIONS : DEVELOPER_SECTIONS;
  const titles: string[] = [];
  for (const page of pages) {
    if (audienceOf(page.section) === audience && !titles.includes(page.section)) titles.push(page.section);
  }
  const rank = (title: string) => (order.includes(title) ? order.indexOf(title) : order.length + titles.indexOf(title));
  return titles
    .sort((a, b) => rank(a) - rank(b))
    .map((title) => ({ title, pages: pages.filter((page) => page.section === title) }));
}

/** The pages before and after `slug` in its audience's reading order. */
export function adjacent<T extends NavPage>(pages: T[], slug: string): { prev?: T; next?: T } {
  const current = pages.find((page) => page.slug === slug);
  if (!current) return {};
  const ordered = sectionsFor(pages, audienceOf(current.section)).flatMap((section) => section.pages);
  const i = ordered.findIndex((page) => page.slug === slug);
  return { prev: i > 0 ? ordered[i - 1] : undefined, next: i + 1 < ordered.length ? ordered[i + 1] : undefined };
}
