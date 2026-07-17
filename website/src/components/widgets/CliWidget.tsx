import { useState } from "react";

// The output line below is captured verbatim from a real run of this exact command against the
// pretrained alignair-igh-human model, on 400 GenAIRR-simulated human IGH reads (clean and
// moderate-SHM full-length). It matches the CLI's real format string in cli/predict.py: one line,
// no progress spinner, no timing, no device banner. Do not embellish it - if the CLI's output
// changes, re-capture rather than hand-edit.

const COMMAND = "alignair predict --input reads.fasta --out out.tsv --model alignair-igh-human";
const OUTPUT = "aligned 400 reads (0 dropped) -> out.tsv; 0 failed / 68 partial AIRR assemblies tagged";

const MONO = "IBM Plex Mono, monospace";

export function CliWidget() {
  const [ran, setRan] = useState(false);

  return (
    <div style={{ margin: "22px 0" }}>
      <div
        style={{
          borderRadius: "14px",
          overflow: "hidden",
          border: "1px solid #24222f",
          boxShadow: "0 18px 40px -26px rgba(30,28,52,0.5)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "10px 14px",
            background: "#211f2c",
            borderBottom: "1px solid #2f2c3d",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "7px" }}>
            <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#3a3746" }}></span>
            <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#3a3746" }}></span>
            <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#3a3746" }}></span>
            <span style={{ marginLeft: "6px", fontFamily: MONO, fontSize: "10.5px", color: "#6f6c80", letterSpacing: "0.08em" }}>
              bash — alignair
            </span>
          </div>
          {ran && (
            <button
              type="button"
              onClick={() => setRan(false)}
              style={{
                fontFamily: MONO,
                fontSize: "11px",
                color: "#a9a7ba",
                background: "#2f2c3d",
                border: "none",
                borderRadius: "6px",
                padding: "4px 10px",
                cursor: "pointer",
              }}
            >
              reset
            </button>
          )}
        </div>

        <div
          style={{
            padding: "16px",
            background: "#16151f",
            fontFamily: MONO,
            fontSize: "13px",
            lineHeight: "1.9",
            color: "#e6e5f0",
            overflowX: "auto",
          }}
        >
          <div style={{ whiteSpace: "nowrap" }}>
            <span style={{ color: "#6f6c80" }}>$</span> alignair predict --input reads.fasta --out out.tsv{" "}
            <span style={{ color: "#b7f3d8" }}>--model alignair-igh-human</span>
          </div>
          {ran && (
            <div style={{ marginTop: "6px", whiteSpace: "nowrap", color: "#a9a7ba", animation: "aa-rise 0.3s ease-out both" }}>
              {OUTPUT}
            </div>
          )}
        </div>

        <div
          style={{
            padding: "12px 16px",
            background: "#1b1a24",
            borderTop: "1px solid #2f2c3d",
            display: "flex",
            justifyContent: "flex-end",
          }}
        >
          <button
            type="button"
            onClick={() => setRan(true)}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              background: "#574fd6",
              color: "#fff",
              border: "none",
              borderRadius: "8px",
              padding: "8px 18px",
              fontFamily: "inherit",
              fontSize: "13px",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            ▶ Replay captured run
          </button>
        </div>
      </div>

      <p style={{ margin: "10px 2px 0", fontFamily: MONO, fontSize: "10.5px", lineHeight: 1.65, color: "#6b6980" }}>
        Captured from a real run of <code style={{ color: "#4238c4" }}>{COMMAND}</code> on 400 simulated human IGH reads
        (clean and moderate-SHM, full-length). That single line is the whole of it: the CLI reports no timing, no device
        banner and no progress bar, so what you see here is what you get. Measured throughput lives on the{" "}
        <em>Speed and throughput</em> page, where the hardware it was measured on is stated.
      </p>
    </div>
  );
}
