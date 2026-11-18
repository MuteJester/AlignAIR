/**
 * How the comparison shows a measure, by the assay's rule: a measure whose name has `_err` is a mean in its own unit
 * (nucleotides, or a count), shown as a number; every other measure is a rate, shown as a percentage, and its
 * difference in percentage points. Pure, so `npm test` checks it.
 */
const isError = (measure: string) => measure.includes("_err");

const signed = (value: number, digits: number) =>
  `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(value).toFixed(digits)}`;
const bare = (value: number, digits: number) => `${value < 0 ? "−" : ""}${Math.abs(value).toFixed(digits)}`;

export function formatMeasure(measure: string, value: number): string {
  return isError(measure) ? bare(value, 2) : `${(100 * value).toFixed(1)}%`;
}

export function formatDifference(measure: string, delta: number, ci95: [number, number]): string {
  if (isError(measure)) return `${signed(delta, 2)} [${bare(ci95[0], 2)}, ${bare(ci95[1], 2)}]`;
  return `${signed(100 * delta, 1)} pp [${bare(100 * ci95[0], 1)}, ${bare(100 * ci95[1], 1)}]`;
}
