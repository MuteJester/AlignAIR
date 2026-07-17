import { useState } from "react";

// Every number below is a measured point estimate from the frozen 2,600-case human-IGH
// benchmark (13 strata x 200 cases), metric `genes.<gene>.call_top1_in_set`, compared with
// paired-case bootstrap and a Bonferroni-corrected 95% CI. `verdict` is the benchmark's own
// decision, not a hand-reading of the point estimates. Nothing here is interpolated: a stratum
// is a discrete generated regime, not a point on a length axis.

type Verdict = "alignair" | "igblast" | "tie" | "unresolved";

type Gene = { gene: string; igblast: number; alignair: number; verdict: Verdict };

type Stratum = {
  key: string;
  label: string;
  blurb: string;
  genes: Gene[];
};

const STRATA: Stratum[] = [
  {
    key: "clean_full",
    label: "clean full-length",
    blurb: "Full-length reads, no SHM, no indels. The easy case - and a solved one for both tools.",
    genes: [
      { gene: "V", igblast: 1.0, alignair: 1.0, verdict: "tie" },
      { gene: "D", igblast: 0.99, alignair: 0.995, verdict: "unresolved" },
      { gene: "J", igblast: 1.0, alignair: 0.995, verdict: "unresolved" },
    ],
  },
  {
    key: "high_shm",
    label: "high SHM",
    blurb: "Full-length, heavily hypermutated. IgBLAST holds a real edge on V here; AlignAIR leads on D and J.",
    genes: [
      { gene: "V", igblast: 0.99, alignair: 0.895, verdict: "igblast" },
      { gene: "D", igblast: 0.495, alignair: 0.69, verdict: "alignair" },
      { gene: "J", igblast: 0.755, alignair: 0.9, verdict: "alignair" },
    ],
  },
  {
    key: "hard_full",
    label: "hard full-length",
    blurb: "Full-length with combined SHM, indels and noise. V is a wash; D and J separate the tools.",
    genes: [
      { gene: "V", igblast: 0.99, alignair: 0.98, verdict: "unresolved" },
      { gene: "D", igblast: 0.55, alignair: 0.825, verdict: "alignair" },
      { gene: "J", igblast: 0.775, alignair: 0.91, verdict: "alignair" },
    ],
  },
  {
    key: "amplicon_fr2",
    label: "FR2 amplicon",
    blurb: "A 5'-anchored fragment that keeps V but cuts away most of D and J. IgBLAST calls V better here; neither tool can recover D or J from signal that is not in the read.",
    genes: [
      { gene: "V", igblast: 0.995, alignair: 0.82, verdict: "igblast" },
      { gene: "D", igblast: 0.0, alignair: 0.035, verdict: "alignair" },
      { gene: "J", igblast: 0.0, alignair: 0.145, verdict: "alignair" },
    ],
  },
  {
    key: "amplicon_jxshort",
    label: "short J-anchored",
    blurb: "A short 3'-anchored read: almost no V left, but the junction is intact. The widest gap in the benchmark - and V stays near zero for both.",
    genes: [
      { gene: "V", igblast: 0.01, alignair: 0.02, verdict: "unresolved" },
      { gene: "D", igblast: 0.015, alignair: 0.735, verdict: "alignair" },
      { gene: "J", igblast: 0.085, alignair: 0.87, verdict: "alignair" },
    ],
  },
  {
    key: "orientation",
    label: "mixed orientation",
    blurb: "Reads presented in an arbitrary orientation. AlignAIR's orientation head re-frames the read before alignment; the ~0.5 baseline is what chance looks like.",
    genes: [
      { gene: "V", igblast: 0.515, alignair: 1.0, verdict: "alignair" },
      { gene: "D", igblast: 0.49, alignair: 0.995, verdict: "alignair" },
      { gene: "J", igblast: 0.5, alignair: 0.995, verdict: "alignair" },
    ],
  },
];

const VERDICT_CHIP: Record<Verdict, { text: string; fg: string; bg: string }> = {
  alignair: { text: "AlignAIR better", fg: "#3d34b8", bg: "#eeecfb" },
  igblast: { text: "IgBLAST better", fg: "#8a5a17", bg: "#fbf1e2" },
  tie: { text: "tie", fg: "#6b6980", bg: "#f1f0f5" },
  unresolved: { text: "not significant", fg: "#6b6980", bg: "#f1f0f5" },
};

const MONO = "IBM Plex Mono, monospace";

function Bar({ label, value, color, track, valueColor }: { label: string; value: number; color: string; track: string; valueColor: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "10px", marginTop: "5px" }}>
      <span style={{ width: "52px", fontFamily: MONO, fontSize: "10.5px", color: "#6f6d85", flexShrink: 0 }}>{label}</span>
      <div aria-hidden="true" style={{ flex: 1, height: "13px", borderRadius: "7px", background: track, overflow: "hidden" }}>
        <div
          style={{
            height: "100%",
            borderRadius: "7px",
            background: color,
            transition: "width 0.25s ease-out",
            width: `${Math.round(value * 100)}%`,
          }}
        ></div>
      </div>
      <span style={{ width: "40px", textAlign: "right", fontFamily: MONO, fontSize: "12px", fontWeight: 600, color: valueColor }}>
        {value.toFixed(3)}
      </span>
    </div>
  );
}

export function BenchSandboxWidget() {
  const [active, setActive] = useState(STRATA[1].key);
  const stratum = STRATA.find((s) => s.key === active) ?? STRATA[0];

  return (
    <div style={{ margin: "22px 0", border: "1px solid #eae9f1", borderRadius: "14px", background: "#fff", padding: "18px" }}>
      <span id="bench-stratum-label" style={{ fontFamily: MONO, fontSize: "11px", fontWeight: 600, letterSpacing: "0.08em", color: "#6f6d85" }}>
        BENCHMARK STRATUM
      </span>
      <div role="group" aria-labelledby="bench-stratum-label" style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginTop: "10px" }}>
        {STRATA.map((s) => {
          const on = s.key === active;
          return (
            <button
              key={s.key}
              type="button"
              aria-pressed={on}
              onClick={() => setActive(s.key)}
              style={{
                fontFamily: MONO,
                fontSize: "11.5px",
                fontWeight: 600,
                padding: "6px 11px",
                borderRadius: "7px",
                cursor: "pointer",
                border: `1px solid ${on ? "#574fd6" : "#eae9f1"}`,
                background: on ? "#574fd6" : "#fff",
                color: on ? "#fff" : "#6b6980",
              }}
            >
              {s.label}
            </button>
          );
        })}
      </div>

      <p style={{ margin: "14px 0 0", fontSize: "13px", lineHeight: 1.6, color: "#57556a" }}>{stratum.blurb}</p>

      <div style={{ marginTop: "16px", display: "flex", flexDirection: "column", gap: "15px" }}>
        {stratum.genes.map((g) => {
          const chip = VERDICT_CHIP[g.verdict];
          return (
            <div key={g.gene}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "10px" }}>
                <span style={{ fontFamily: "Space Grotesk, sans-serif", fontWeight: 600, fontSize: "13px", color: "#16151f" }}>
                  {g.gene} allele - top-1 in truth set
                </span>
                <span
                  style={{
                    fontFamily: MONO,
                    fontSize: "10.5px",
                    fontWeight: 600,
                    padding: "3px 8px",
                    borderRadius: "20px",
                    color: chip.fg,
                    background: chip.bg,
                  }}
                >
                  {chip.text}
                </span>
              </div>
              <Bar label="AlignAIR" value={g.alignair} color="#574fd6" track="#f1f0f8" valueColor="#4238c4" />
              <Bar label="IgBLAST" value={g.igblast} color="#9795a8" track="#f4f3f7" valueColor="#6f6d85" />
            </div>
          );
        })}
      </div>

      <p
        style={{
          margin: "16px 0 0",
          paddingTop: "12px",
          borderTop: "1px dashed #eae9f1",
          fontFamily: MONO,
          fontSize: "10.5px",
          lineHeight: 1.65,
          color: "#6b6980",
        }}
      >
        Measured point estimates from a frozen 2,600-case human-IGH benchmark, 200 cases per stratum. Verdicts come from
        a paired-case bootstrap with a Bonferroni-corrected 95% interval - &ldquo;not significant&rdquo; means the
        interval spans zero, so the visible gap is not yet a real difference. Strata are discrete generated regimes, not
        points on a length axis; nothing is interpolated between them. Reproduce with{" "}
        <code style={{ color: "#4238c4" }}>alignair benchmark</code>.
      </p>
    </div>
  );
}
