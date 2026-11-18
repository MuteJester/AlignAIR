/**
 * Links to a heading of a docs page. The router owns the URL fragment (`#/docs/<slug>`), so a heading is named in the
 * route's query (`?h=<id>`), not in a second `#`. Pure, so `npm test` checks it.
 */
export function headingPath(slug: string, id: string): string {
  return `/docs/${slug}?h=${encodeURIComponent(id)}`;
}

export function headingHref(slug: string, id: string): string {
  return `#${headingPath(slug, id)}`;
}

/** The heading id a route's query names, or null. */
export function parseHeading(search: string): string | null {
  return new URLSearchParams(search).get("h");
}
