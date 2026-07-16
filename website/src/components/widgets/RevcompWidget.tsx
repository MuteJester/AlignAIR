import { useState } from "react";

export function RevcompWidget() {
  const [rc, setRc] = useState(false);

  return (
    <div style={{ margin: "22px 0", border: "1px solid #eae9f1", borderRadius: "14px", background: "#fff", padding: "20px 18px" }}>
      <div style={{ display: "inline-flex", padding: "4px", background: "#f1f0f8", borderRadius: "10px", gap: "4px" }}>
        <button
          type="button"
          onClick={() => setRc(false)}
          style={{
            fontFamily: "inherit",
            fontSize: "13px",
            fontWeight: 600,
            cursor: "pointer",
            border: "none",
            borderRadius: "7px",
            padding: "7px 16px",
            background: !rc ? "#574fd6" : "#ffffff",
            color: !rc ? "#ffffff" : "#56546a",
            transition: "all 0.15s",
          }}
        >
          Forward
        </button>
        <button
          type="button"
          onClick={() => setRc(true)}
          style={{
            fontFamily: "inherit",
            fontSize: "13px",
            fontWeight: 600,
            cursor: "pointer",
            border: "none",
            borderRadius: "7px",
            padding: "7px 16px",
            background: rc ? "#574fd6" : "#ffffff",
            color: rc ? "#ffffff" : "#56546a",
            transition: "all 0.15s",
          }}
        >
          Reverse-complement
        </button>
      </div>
      <div style={{ marginTop: "18px", display: "grid", gap: "10px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", fontFamily: "IBM Plex Mono, monospace", fontSize: "13px" }}>
          <span style={{ width: "130px", color: "#8b899d" }}>rev_comp</span>
          <span style={{ fontWeight: 600, color: rc ? "#a66a00" : "#12805c" }}>{rc ? "T" : "F"}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", fontFamily: "IBM Plex Mono, monospace", fontSize: "13px" }}>
          <span style={{ width: "130px", color: "#8b899d" }}>v_sequence_start/end</span>
          <span style={{ fontWeight: 600, color: "#574fd6", transition: "color 0.15s" }}>{rc ? "235–312" : "1–78"}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", fontFamily: "IBM Plex Mono, monospace", fontSize: "13px" }}>
          <span style={{ width: "130px", color: "#8b899d" }}>j_sequence_start/end</span>
          <span style={{ fontWeight: 600, color: "#3f7fd6", transition: "color 0.15s" }}>{rc ? "1–99" : "214–312"}</span>
        </div>
      </div>
      <p style={{ margin: "16px 0 0", paddingTop: "14px", borderTop: "1px dashed #eae9f1", fontFamily: "IBM Plex Mono, monospace", fontSize: "12px", lineHeight: 1.6, color: "#8b899d" }}>
        {rc ? "sequence = original query · coordinates apply to RC(sequence)" : "sequence = query as aligned · coordinates apply directly"}
      </p>
    </div>
  );
}
