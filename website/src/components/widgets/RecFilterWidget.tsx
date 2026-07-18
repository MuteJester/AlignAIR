import { useState } from "react";

export function RecFilterWidget() {
  const REC_ROWS = [
    { id: "read_0417", v: "IGHV3-23*01", status: "complete" },
    { id: "read_0418", v: "IGHV1-2*02", status: "complete" },
    { id: "read_0419", v: "IGHV4-34*01", status: "partial" },
    { id: "read_0420", v: "(none)", status: "failed" },
  ];

  const [filt, setFilt] = useState(false);
  const kept = REC_ROWS.filter((r) => !filt || r.status === "complete").length;

  return (
    <div style={{ margin: "22px 0", border: "1px solid #eae9f1", borderRadius: "14px", background: "#fff", padding: "20px 18px" }}>
      <div role="group" aria-label="Record filter" style={{ display: "inline-flex", padding: "4px", background: "#f1f0f8", borderRadius: "10px", gap: "4px" }}>
        <button
          type="button"
          aria-pressed={!filt}
          onClick={() => setFilt(false)}
          style={{
            fontFamily: "inherit",
            fontSize: "13px",
            fontWeight: 600,
            cursor: "pointer",
            border: "none",
            borderRadius: "7px",
            padding: "7px 16px",
            background: !filt ? "#574fd6" : "#ffffff",
            color: !filt ? "#ffffff" : "#56546a",
            transition: "all 0.15s",
          }}
        >
          All records
        </button>
        <button
          type="button"
          aria-pressed={filt}
          onClick={() => setFilt(true)}
          style={{
            fontFamily: "inherit",
            fontSize: "13px",
            fontWeight: 600,
            cursor: "pointer",
            border: "none",
            borderRadius: "7px",
            padding: "7px 16px",
            background: filt ? "#574fd6" : "#ffffff",
            color: filt ? "#ffffff" : "#56546a",
            transition: "all 0.15s",
          }}
        >
          Complete only
        </button>
      </div>
      <div style={{ marginTop: "16px", display: "flex", flexDirection: "column", gap: "8px" }}>
        {REC_ROWS.map((r) => {
          const dropped = filt && r.status !== "complete";
          const sc = r.status === "complete" ? "#0f6b4e" : r.status === "partial" ? "#8a5600" : "#a12b3f";
          const sb = r.status === "complete" ? "#eef7f2" : r.status === "partial" ? "#fdf6e9" : "#fdeef0";
          return (
            <div
              key={r.id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                border: "1px solid #eae9f1",
                borderRadius: "10px",
                padding: "11px 14px",
                opacity: dropped ? "0.55" : "1",
                transition: "opacity 0.2s",
              }}
            >
              <span style={{ flex: "0 0 96px", fontFamily: "IBM Plex Mono, monospace", fontSize: "12.5px", color: "#16151f" }}>{r.id}</span>
              <span style={{ flex: 1, fontFamily: "IBM Plex Mono, monospace", fontSize: "12.5px", color: "#4238c4" }}>{r.v}</span>
              <span style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "11px", fontWeight: 600, borderRadius: "6px", padding: "3px 9px", background: sb, color: sc }}>
                {r.status}
              </span>
              {dropped && <span className="sr-only">filtered out</span>}
            </div>
          );
        })}
      </div>
      <p aria-live="polite" style={{ margin: "14px 0 0", fontFamily: "IBM Plex Mono, monospace", fontSize: "12px", color: "#6f6d85" }}>
        {kept} of {REC_ROWS.length} records kept
      </p>
    </div>
  );
}
