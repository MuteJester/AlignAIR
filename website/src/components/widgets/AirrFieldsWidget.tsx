import { useState } from "react";

export function AirrFieldsWidget() {
  const AIRR_FIELDS = [
    { key: "sequence_id", val: "read_0417", desc: "The identifier of the input read, carried through unchanged so every row traces back to its source." },
    { key: "v_call", val: "IGHV3-23*01", color: "#574fd6", desc: "The V allele call. When a read genuinely cannot disambiguate alleles, the full equivalence set is reported in `v_call_set` — treat a multi-member set as a family-level call, not a confident single answer." },
    { key: "d_call", val: "IGHD3-10*01", color: "#9b4bd6", desc: "The D allele call. D segments are short and mutated, so `d_call` is the least certain of the three — read it alongside `d_call_set`." },
    { key: "j_call", val: "IGHJ4*02", color: "#3f7fd6", desc: "The J allele call. Like V and D it carries a `j_call_set` when several alleles remain consistent with the read." },
    { key: "junction", val: "CARDYYGSGSYYFDYW", color: "#12805c", desc: "The CDR3 junction amino-acid sequence, when it can be derived. This is the region that defines clonal identity downstream." },
    { key: "rev_comp", val: "F", desc: "Whether the read was reverse-complemented. `T` means the emitted `sequence` is the original query and coordinates apply to its reverse complement." },
    { key: "productive", val: "", color: "#8b899d", desc: "A derived fact: in-frame with no stop codon. On a partial record it can be underivable, and AlignAIR leaves it **blank** — unknown, not `F`." },
    { key: "airr_assembly_status", val: "complete", desc: "Honest quality flag: `complete`, `partial`, or `failed`. Filter to `complete` before junction or productivity analysis." },
  ];

  const [sel, setSel] = useState("sequence_id");
  const active = AIRR_FIELDS.find((f) => f.key === sel) || AIRR_FIELDS[0];

  return (
    <div style={{ margin: "22px 0", border: "1px solid #eae9f1", borderRadius: "14px", background: "#fff", overflow: "hidden" }}>
      <div style={{ padding: "14px 16px 4px", display: "flex", flexWrap: "wrap", gap: "8px", background: "#faf9fd", borderBottom: "1px solid #f0eff5", paddingBottom: "14px" }}>
        {AIRR_FIELDS.map((f) => (
          <button
            key={f.key}
            type="button"
            onClick={() => setSel(f.key)}
            style={{
              fontFamily: "IBM Plex Mono, monospace",
              fontSize: "12px",
              fontWeight: 500,
              cursor: "pointer",
              borderRadius: "8px",
              padding: "7px 11px",
              transition: "all 0.12s",
              border: `1px solid ${f.key === sel ? "#574fd6" : "#eae9f1"}`,
              background: f.key === sel ? "#eef0ff" : "#ffffff",
              color: f.key === sel ? "#4238c4" : "#56546a",
            }}
          >
            {f.key}
          </button>
        ))}
      </div>
      <div style={{ padding: "20px 18px", animation: "aa-rise 0.25s ease-out both" }} key={sel}>
        <div style={{ display: "flex", alignItems: "baseline", gap: "12px", flexWrap: "wrap" }}>
          <span style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "13px", color: "#8b899d" }}>{sel}</span>
          <span style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "14px", fontWeight: 600, color: active.color || "#16151f" }}>
            {active.val === "" ? "(blank — unknown)" : active.val}
          </span>
        </div>
        <p style={{ margin: "12px 0 0", fontSize: "14.5px", lineHeight: 1.7, color: "#3a3849" }}>
          {active.desc}
        </p>
      </div>
    </div>
  );
}
