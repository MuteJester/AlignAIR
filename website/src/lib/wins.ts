/**
 * Where AlignAIR beats IgBLAST in the locked comparison, by family of measures a repertoire researcher reads: for each
 * family, the comparisons (suite x locus x measure) AlignAIR wins out of all of them, the loci where it wins, and its
 * largest gain as an example. Productivity is not a family here: AlignAIR does not lead on it. Pure, so `npm test`
 * checks it.
 */
export interface ComparisonRow {
  measure: string;
  igblast: number;
  alignair: number;
  delta: number;
  ci95: [number, number];
  verdict: string;
}
export interface ComparisonLocus {
  reference: string;
  suites: { name: string; title: string; cases: number; rows: ComparisonRow[] }[];
}
export interface WinFamily {
  label: string;
  wins: number;
  comparisons: number;
  loci: string[];
  example: { locus: string; suite: string; measure: string; igblast: number; alignair: number };
}

const FAMILIES: { label: string; measures: string[] }[] = [
  { label: "V mutation count (mean error, lower is better)", measures: ["v.mutation_count_err"] },
  { label: "J allele", measures: ["j.top1_in_set"] },
  { label: "D allele", measures: ["d.top1_in_set"] },
  { label: "V allele", measures: ["v.top1_in_set"] },
  { label: "Exact junction", measures: ["junction.nt_exact", "junction.length_exact"] },
  { label: "Clone key", measures: ["clone_key.exact"] },
  { label: "Segment boundaries", measures: ["v.seq_start_exact", "v.seq_end_exact", "d.seq_start_exact",
    "d.seq_end_exact", "j.seq_start_exact", "j.seq_end_exact", "v.germ_start_err", "j.germ_end_err"] },
  { label: "No V call on a read too short to show one", measures: ["v_short.abstain_ok"] },
  { label: "V called where the read has none (lower is better)", measures: ["v.spurious_call"] },
  { label: "V indels found", measures: ["v.indel_found"] },
];

const WIN = "b_better";

export function winsByFamily(loci: Record<string, ComparisonLocus>): WinFamily[] {
  const out: WinFamily[] = [];
  for (const family of FAMILIES) {
    let wins = 0;
    let comparisons = 0;
    const won = new Set<string>();
    let best: WinFamily["example"] | null = null;
    let bestGain = -1;
    for (const [locus, entry] of Object.entries(loci)) {
      for (const suite of entry.suites) {
        for (const row of suite.rows) {
          if (!family.measures.includes(row.measure)) continue;
          comparisons += 1;
          if (row.verdict !== WIN) continue;
          wins += 1;
          won.add(locus);
          const gain = Math.abs(row.delta);
          if (gain > bestGain) {
            bestGain = gain;
            best = { locus, suite: suite.title || suite.name, measure: row.measure, igblast: row.igblast,
              alignair: row.alignair };
          }
        }
      }
    }
    if (wins > 0 && best) out.push({ label: family.label, wins, comparisons, loci: [...won].sort(), example: best });
  }
  return out.sort((a, b) => b.wins - a.wins);
}
