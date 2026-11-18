/**
 * Text written for a terminal or a docstring (CLI help, error meanings), shown on a page: a ``code`` or `code` span
 * becomes code, and *emphasis* loses its asterisks (an asterisk inside a word, as in IGHV1-2*02, stays). Pure, so
 * `npm test` checks it.
 */
export interface InlinePart {
  code: boolean;
  text: string;
}

const EMPHASIS = /(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s.,;:)]|$)/g;

export function inlineParts(text: string): InlinePart[] {
  const parts: InlinePart[] = [];
  const pattern = /``(.+?)``|`([^`]+)`/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) parts.push({ code: false, text: text.slice(last, match.index) });
    parts.push({ code: true, text: match[1] ?? match[2] });
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push({ code: false, text: text.slice(last) });
  return parts.map((part) => (part.code ? part : { code: false, text: part.text.replace(EMPHASIS, "$1$2") }));
}
