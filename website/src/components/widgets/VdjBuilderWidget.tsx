import { useState } from "react";

export function VdjBuilderWidget() {
  const VDJ_V = [
    { call: "IGHV3-23*01", seq: "GCCTGGGGGGTCCCTGAGACTCTCCTGT" },
    { call: "IGHV1-2*02", seq: "GCCTGGGTGAAGCAGACTCCTCACCTGT" },
    { call: "IGHV4-34*01", seq: "GCCTGCGGGGTCTATTCCAGGAACCGCC" },
  ];
  const VDJ_D = [
    { call: "IGHD3-10*01", seq: "GTATTACTATGGTTCGGGGAGTTAT", motif: "DYYGSGSY" },
    { call: "IGHD2-2*01", seq: "AGGATATTGTAGTAGTACCAGCTGC", motif: "DIVVVPAA" },
    { call: "IGHD6-19*01", seq: "GGGTATAGCAGTGGCTGGTAC", motif: "GYSSGW" },
  ];
  const VDJ_J = [
    { call: "IGHJ4*02", seq: "ACTACTTTGACTACTGGGGCCAG", end: "FDYW" },
    { call: "IGHJ6*02", seq: "ATTACTACTACTACGGTATGGACGTC", end: "YYYYGMDVW" },
    { call: "IGHJ3*02", seq: "TGATGCTTTTGATATCTGGGGCCAA", end: "AFDIW" },
  ];

  const [vi, setVi] = useState(0);
  const [di, setDi] = useState(0);
  const [ji, setJi] = useState(0);

  return (
    <div style={{ margin: "22px 0", border: "1px solid #eae9f1", borderRadius: "14px", background: "#fff", padding: "20px 18px" }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(140px, 100%), 1fr))", gap: "14px" }}>
        <div role="group" aria-labelledby="vdj-v-label">
          <p id="vdj-v-label" style={{ margin: "0 0 8px", fontFamily: "IBM Plex Mono, monospace", fontSize: "11px", fontWeight: 600, letterSpacing: "0.08em", color: "#4238c4" }}>V SEGMENT</p>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {VDJ_V.map((o, i) => (
              <button
                key={o.call}
                type="button"
                aria-pressed={i === vi}
                onClick={() => setVi(i)}
                style={{
                  textAlign: "left",
                  cursor: "pointer",
                  fontFamily: "IBM Plex Mono, monospace",
                  fontSize: "12px",
                  borderRadius: "8px",
                  padding: "7px 10px",
                  transition: "all 0.12s",
                  background: i === vi ? "#eef0ff" : "#ffffff",
                  border: `1px solid ${i === vi ? "#574fd6" : "#eae9f1"}`,
                  color: i === vi ? "#4238c4" : "#56546a",
                }}
              >
                {o.call}
              </button>
            ))}
          </div>
        </div>
        <div role="group" aria-labelledby="vdj-d-label">
          <p id="vdj-d-label" style={{ margin: "0 0 8px", fontFamily: "IBM Plex Mono, monospace", fontSize: "11px", fontWeight: 600, letterSpacing: "0.08em", color: "#7a2fb0" }}>D SEGMENT</p>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {VDJ_D.map((o, i) => (
              <button
                key={o.call}
                type="button"
                aria-pressed={i === di}
                onClick={() => setDi(i)}
                style={{
                  textAlign: "left",
                  cursor: "pointer",
                  fontFamily: "IBM Plex Mono, monospace",
                  fontSize: "12px",
                  borderRadius: "8px",
                  padding: "7px 10px",
                  transition: "all 0.12s",
                  background: i === di ? "#f5edfb" : "#ffffff",
                  border: `1px solid ${i === di ? "#9b4bd6" : "#eae9f1"}`,
                  color: i === di ? "#7a2fb0" : "#56546a",
                }}
              >
                {o.call}
              </button>
            ))}
          </div>
        </div>
        <div role="group" aria-labelledby="vdj-j-label">
          <p id="vdj-j-label" style={{ margin: "0 0 8px", fontFamily: "IBM Plex Mono, monospace", fontSize: "11px", fontWeight: 600, letterSpacing: "0.08em", color: "#2b62ad" }}>J SEGMENT</p>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {VDJ_J.map((o, i) => (
              <button
                key={o.call}
                type="button"
                aria-pressed={i === ji}
                onClick={() => setJi(i)}
                style={{
                  textAlign: "left",
                  cursor: "pointer",
                  fontFamily: "IBM Plex Mono, monospace",
                  fontSize: "12px",
                  borderRadius: "8px",
                  padding: "7px 10px",
                  transition: "all 0.12s",
                  background: i === ji ? "#eaf1fb" : "#ffffff",
                  border: `1px solid ${i === ji ? "#3f7fd6" : "#eae9f1"}`,
                  color: i === ji ? "#2b62ad" : "#56546a",
                }}
              >
                {o.call}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div aria-hidden="true" style={{ marginTop: "18px", display: "flex", alignItems: "stretch", gap: "4px", height: "38px" }}>
        <div style={{ flex: 3, borderRadius: "8px 4px 4px 8px", background: "#574fd6", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "IBM Plex Mono, monospace", fontSize: "12px", fontWeight: 600, color: "#fff" }}>V</div>
        <div style={{ flex: 0.4, background: "#c6c3d6", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "IBM Plex Mono, monospace", fontSize: "9px", color: "#413f52" }}>N</div>
        <div style={{ flex: 1, background: "#9b4bd6", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "IBM Plex Mono, monospace", fontSize: "12px", fontWeight: 600, color: "#fff" }}>D</div>
        <div style={{ flex: 0.4, background: "#c6c3d6", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "IBM Plex Mono, monospace", fontSize: "9px", color: "#413f52" }}>N</div>
        <div style={{ flex: 1.8, borderRadius: "4px 8px 8px 4px", background: "#3f7fd6", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "IBM Plex Mono, monospace", fontSize: "12px", fontWeight: 600, color: "#fff" }}>J</div>
      </div>
      <p style={{ marginTop: "12px", marginBottom: "0", fontFamily: "IBM Plex Mono, monospace", fontSize: "11.5px", lineHeight: "1.7", color: "#6f6d85", wordBreak: "break-all" }}>
        {VDJ_V[vi].seq} · {VDJ_D[di].seq} · {VDJ_J[ji].seq}
      </p>
      <div aria-live="polite" style={{ marginTop: "14px", paddingTop: "14px", borderTop: "1px dashed #eae9f1", display: "grid", gap: "7px" }}>
        <div style={{ display: "flex", gap: "10px", fontFamily: "IBM Plex Mono, monospace", fontSize: "12.5px" }}>
          <span style={{ width: "88px", color: "#6f6d85" }}>v_call</span>
          <span style={{ color: "#4238c4", fontWeight: 600 }}>{VDJ_V[vi].call}</span>
        </div>
        <div style={{ display: "flex", gap: "10px", fontFamily: "IBM Plex Mono, monospace", fontSize: "12.5px" }}>
          <span style={{ width: "88px", color: "#6f6d85" }}>d_call</span>
          <span style={{ color: "#7a2fb0", fontWeight: 600 }}>{VDJ_D[di].call}</span>
        </div>
        <div style={{ display: "flex", gap: "10px", fontFamily: "IBM Plex Mono, monospace", fontSize: "12.5px" }}>
          <span style={{ width: "88px", color: "#6f6d85" }}>j_call</span>
          <span style={{ color: "#2b62ad", fontWeight: 600 }}>{VDJ_J[ji].call}</span>
        </div>
        <div style={{ display: "flex", gap: "10px", fontFamily: "IBM Plex Mono, monospace", fontSize: "12.5px" }}>
          <span style={{ width: "88px", color: "#6f6d85" }}>junction_aa</span>
          <span style={{ color: "#0f6b4e", fontWeight: 600 }}>{"CAR" + VDJ_D[di].motif + VDJ_J[ji].end}</span>
        </div>
      </div>
    </div>
  );
}
