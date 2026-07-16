import { useState } from "react";

export function GenotypeWidget() {
  const GENO_ALLELES = [
    { name: "IGHV3-23*01", donor: true },
    { name: "IGHV3-23*04", donor: false },
    { name: "IGHV1-2*02", donor: true },
    { name: "IGHV1-46*01", donor: false },
    { name: "IGHV4-34*01", donor: true },
    { name: "IGHV4-59*01", donor: false },
    { name: "IGHV5-51*01", donor: false },
    { name: "IGHV3-30*03", donor: true },
    { name: "IGHV3-7*01", donor: false },
    { name: "IGHV1-69*01", donor: false },
  ];

  const [mode, setMode] = useState<"full" | "donor">("full");
  const [novel, setNovel] = useState(false);

  const callable = GENO_ALLELES.filter((a) => mode === "full" || a.donor).length;

  return (
    <div style={{ margin: "22px 0", border: "1px solid #eae9f1", borderRadius: "14px", background: "#fff", padding: "20px 18px" }}>
      <div style={{ display: "inline-flex", padding: "4px", background: "#f1f0f8", borderRadius: "10px", gap: "4px" }}>
        <button
          type="button"
          onClick={() => setMode("full")}
          style={{
            fontFamily: "inherit",
            fontSize: "13px",
            fontWeight: 600,
            cursor: "pointer",
            border: "none",
            borderRadius: "7px",
            padding: "7px 16px",
            background: mode === "full" ? "#574fd6" : "#ffffff",
            color: mode === "full" ? "#ffffff" : "#56546a",
            transition: "all 0.15s",
          }}
        >
          Model reference
        </button>
        <button
          type="button"
          onClick={() => setMode("donor")}
          style={{
            fontFamily: "inherit",
            fontSize: "13px",
            fontWeight: 600,
            cursor: "pointer",
            border: "none",
            borderRadius: "7px",
            padding: "7px 16px",
            background: mode === "donor" ? "#574fd6" : "#ffffff",
            color: mode === "donor" ? "#ffffff" : "#56546a",
            transition: "all 0.15s",
          }}
        >
          Donor genotype
        </button>
      </div>
      <div style={{ marginTop: "16px", display: "flex", flexWrap: "wrap", gap: "8px" }}>
        {GENO_ALLELES.map((a) => {
          const c = mode === "full" || a.donor;
          return (
            <span
              key={a.name}
              style={{
                fontFamily: "IBM Plex Mono, monospace",
                fontSize: "12px",
                borderRadius: "8px",
                padding: "6px 10px",
                transition: "all 0.15s",
                background: c ? "#eef0ff" : "#f6f6fa",
                border: `1px solid ${c ? "#574fd6" : "#eae9f1"}`,
                color: c ? "#4238c4" : "#b7b5c6",
                textDecoration: c ? "none" : "line-through",
              }}
            >
              {a.name}
            </span>
          );
        })}
      </div>
      <div style={{ marginTop: "14px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: "12px", flexWrap: "wrap" }}>
        <span style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "12px", fontWeight: 600, color: "#4238c4" }}>
          {callable} of 10 alleles callable
        </span>
        <button
          type="button"
          onClick={() => setNovel((n) => !n)}
          style={{ fontFamily: "inherit", fontSize: "12.5px", fontWeight: 600, cursor: "pointer", border: "1px solid #e9a9b4", background: "#fff", color: "#a12b3f", borderRadius: "8px", padding: "6px 12px" }}
        >
          {novel ? "Remove novel allele" : "+ Try a novel allele"}
        </button>
      </div>
      {novel && (
        <div style={{ marginTop: "12px", display: "flex", gap: "11px", alignItems: "flex-start", border: "1px solid #e9a9b4", background: "#fdeef0", borderRadius: "10px", padding: "12px 14px" }}>
          <span style={{ fontFamily: "IBM Plex Mono, monospace", fontWeight: 700, color: "#c0344a" }}>✕</span>
          <div>
            <p style={{ margin: 0, fontFamily: "IBM Plex Mono, monospace", fontSize: "12.5px", fontWeight: 600, color: "#a12b3f" }}>
              IGHV7-81*99 — rejected
            </p>
            <p style={{ margin: "4px 0 0", fontSize: "13px", lineHeight: 1.6, color: "#7a2230" }}>
              Not in the model’s embedded reference. A genotype can only subset what the model already knows — to call a novel allele you must train a new model.
            </p>
          </div>
        </div>
      )}
      <p style={{ margin: "14px 0 0", paddingTop: "12px", borderTop: "1px dashed #eae9f1", fontSize: "13.5px", lineHeight: 1.65, color: "#6f6d85" }}>
        {mode === "full"
          ? "The full embedded reference — every allele the model was trained to call."
          : "A donor genotype hard-restricts calls to this donor’s alleles. Impossible calls vanish and ambiguous sets shrink — with no retraining."}
      </p>
    </div>
  );
}
