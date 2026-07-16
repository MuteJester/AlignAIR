import { useState } from "react";

export function CliWidget() {
  const [ran, setRan] = useState(false);
  const cliOut = [
    { s: "resolved device: cuda:0", c: "#7fd8b0", prefix: "✔ " },
    { s: "loaded alignair-igh-human  (reference fingerprint verified)", c: "#7fd8b0", prefix: "✔ " },
    { s: "aligning 1,000 reads ................ done (2.4s)", c: "#a9a7ba", prefix: "✔ " },
    { s: "wrote out.tsv  ·  982 complete · 14 partial · 4 failed", c: "#7fd8b0", prefix: "✔ " },
  ];

  return (
    <div style={{ margin: "22px 0", borderRadius: "14px", overflow: "hidden", border: "1px solid #24222f", boxShadow: "0 18px 40px -26px rgba(30,28,52,0.5)" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", background: "#211f2c", borderBottom: "1px solid #2f2c3d" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "7px" }}>
          <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#3a3746" }}></span>
          <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#3a3746" }}></span>
          <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#3a3746" }}></span>
          <span style={{ marginLeft: "6px", fontFamily: "IBM Plex Mono, monospace", fontSize: "10.5px", color: "#6f6c80", letterSpacing: "0.08em" }}>bash — alignair</span>
        </div>
        {ran && (
          <button
            type="button"
            onClick={() => setRan(false)}
            style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "11px", color: "#a9a7ba", background: "#2f2c3d", border: "none", borderRadius: "6px", padding: "4px 10px", cursor: "pointer" }}
          >
            reset
          </button>
        )}
      </div>
      <div style={{ padding: "16px", background: "#16151f", fontFamily: "IBM Plex Mono, monospace", fontSize: "13px", lineHeight: "1.9", color: "#e6e5f0" }}>
        <div>
          <span style={{ color: "#6f6c80" }}>$</span> alignair predict --input reads.fasta --out out.tsv{" "}
          <span style={{ color: "#b7f3d8" }}>--model alignair-igh-human</span>
        </div>
        {ran && (
          <div style={{ marginTop: "6px" }}>
            {cliOut.map((ln, idx) => (
              <div key={idx} style={{ animation: "aa-rise 0.3s ease-out both" }}>
                <span style={{ color: "#7fd8b0" }}>{ln.prefix}</span>
                <span style={{ color: ln.c }}>{ln.s}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div style={{ padding: "12px 16px", background: "#1b1a24", borderTop: "1px solid #2f2c3d", display: "flex", justifyContent: "flex-end" }}>
        <button
          type="button"
          onClick={() => setRan(true)}
          style={{ display: "inline-flex", alignItems: "center", gap: "8px", background: "#574fd6", color: "#fff", border: "none", borderRadius: "8px", padding: "8px 18px", fontFamily: "inherit", fontSize: "13px", fontWeight: 600, cursor: "pointer" }}
        >
          ▶ Run prediction
        </button>
      </div>
    </div>
  );
}
