import { useState } from "react";

export function BenchSandboxWidget() {
  const [flen, setFlen] = useState(312);

  const t = Math.max(0, Math.min(1, (flen - 80) / 232));
  const mk = (short: number, full: number) => {
    const v = short + (full - short) * t;
    return {
      pct: Math.round(v * 100) + "%",
      w: Math.round(v * 100) + "%",
    };
  };

  const rows = [
    { name: "V", shortA: 0.74, fullA: 0.776, shortI: 0.60, fullI: 0.745 },
    { name: "D", shortA: 0.71, fullA: 0.694, shortI: 0.34, fullI: 0.538 },
    { name: "J", shortA: 0.88, fullA: 0.842, shortI: 0.47, fullI: 0.713 },
  ].map((r) => {
    const a = mk(r.shortA, r.fullA);
    const i = mk(r.shortI, r.fullI);
    return {
      gene: r.name,
      aPct: a.pct,
      aW: a.w,
      iPct: i.pct,
      iW: i.w,
    };
  });

  return (
    <div style={{ margin: "22px 0", border: "1px solid #eae9f1", borderRadius: "14px", background: "#fff", padding: "20px 18px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "11px", fontWeight: 600, letterSpacing: "0.08em", color: "#8b899d" }}>FRAGMENT LENGTH</span>
        <span style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "12.5px", fontWeight: 600, color: "#4238c4" }}>
          {flen >= 312 ? "full-length · 312 nt" : `${flen} nt fragment`}
        </span>
      </div>
      <input
        type="range"
        min="40"
        max="312"
        step="4"
        value={flen}
        onChange={(e) => setFlen(Number(e.target.value))}
        style={{ width: "100%", marginTop: "12px", accentColor: "#574fd6", cursor: "pointer" }}
      />
      <div style={{ display: "flex", gap: "18px", marginTop: "6px", fontFamily: "IBM Plex Mono, monospace", fontSize: "11px", color: "#8b899d" }}>
        <span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
          <span style={{ width: "10px", height: "10px", borderRadius: "3px", background: "#574fd6" }}></span>
          AlignAIR
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
          <span style={{ width: "10px", height: "10px", borderRadius: "3px", background: "#c9c7d6" }}></span>
          IgBLAST
        </span>
      </div>
      <div style={{ marginTop: "16px", display: "flex", flexDirection: "column", gap: "16px" }}>
        {rows.map((m) => (
          <div key={m.gene}>
            <span style={{ fontFamily: "Space Grotesk, sans-serif", fontWeight: 600, fontSize: "13px", color: "#16151f" }}>{m.gene} allele — top-1 in set</span>
            <div style={{ marginTop: "8px", display: "flex", alignItems: "center", gap: "10px" }}>
              <div style={{ flex: 1, height: "14px", borderRadius: "7px", background: "#f1f0f8", overflow: "hidden" }}>
                <div style={{ height: "100%", borderRadius: "7px", background: "#574fd6", transition: "width 0.2s", width: m.aW }}></div>
              </div>
              <span style={{ width: "44px", fontFamily: "IBM Plex Mono, monospace", fontSize: "12px", fontWeight: 600, color: "#4238c4" }}>{m.aPct}</span>
            </div>
            <div style={{ marginTop: "6px", display: "flex", alignItems: "center", gap: "10px" }}>
              <div style={{ flex: 1, height: "14px", borderRadius: "7px", background: "#f1f0f8", overflow: "hidden" }}>
                <div style={{ height: "100%", borderRadius: "7px", background: "#c9c7d6", transition: "width 0.2s", width: m.iW }}></div>
              </div>
              <span style={{ width: "44px", fontFamily: "IBM Plex Mono, monospace", fontSize: "12px", fontWeight: 600, color: "#8b899d" }}>{m.iPct}</span>
            </div>
          </div>
        ))}
      </div>
      <p style={{ margin: "16px 0 0", paddingTop: "12px", borderTop: "1px dashed #eae9f1", fontFamily: "IBM Plex Mono, monospace", fontSize: "11px", lineHeight: 1.6, color: "#a09eb2" }}>
        Illustrative, interpolated from the frozen human-IGH benchmark. Drag to see AlignAIR’s edge grow on short fragments.
      </p>
    </div>
  );
}
