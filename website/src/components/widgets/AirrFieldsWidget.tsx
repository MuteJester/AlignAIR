import { useState } from "react";

export function AirrFieldsWidget() {
  const AIRR_FIELDS = [
    { key: "sequence_id", val: "read_0417", desc: "The identifier of the input read, carried through unchanged so every row traces back to its source." },
    { key: "v_call", val: "IGHV3-23*01", color: "#574fd6", desc: "The top V allele call. Alongside it, v_call_set lists the alleles the model scored as more-likely-present-than-not, ordered by probability and capped at three. Keep the set as reported: it is the model's candidate set, not a proof that the read cannot distinguish those alleles, and it is not exhaustive." },
    { key: "d_call", val: "IGHD3-10*01", color: "#9b4bd6", desc: "The D allele call. D segments are short and heavily trimmed during junction assembly, so this is the least certain of the three - read it alongside d_call_set, and expect a set or a blank more often than for V or J." },
    { key: "j_call", val: "IGHJ4*02", color: "#3f7fd6", desc: "The J allele call. Like V and D it carries a j_call_set when several alleles clear the threshold." },
    { key: "junction", val: "TGTGCGAGAGATTACTATGGTTCGGGGAGTTATTATTTTGACTACTGG", color: "#12805c", desc: "The CDR3 junction as NUCLEOTIDES, including the conserved Cys and Trp/Phe codons. This is the region that defines clonal identity downstream. Blank when the anchors cannot be placed - honest absence, not a guess." },
    { key: "junction_aa", val: "CARDYYGSGSYYFDYW", color: "#12805c", desc: "The same junction translated to amino acids. Clonal grouping is usually more robust on junction_aa than on single-nucleotide positions, because the junction coordinates can jitter by a nucleotide or two." },
    { key: "rev_comp", val: "F", desc: "Whether the alignment used the reverse complement. T means the emitted sequence is the ORIGINAL query and the coordinates apply to its reverse complement - the AIRR and IgBLAST convention." },
    { key: "productive", val: "", color: "#8b899d", desc: "A derived fact: in-frame with no stop codon. On a partial record it can be underivable, and AlignAIR leaves it blank - unknown, not false. The neural head's advisory guess lives separately in productive_prediction." },
    { key: "airr_assembly_status", val: "complete", desc: "Honest quality flag: complete, partial, or failed. Filter to complete before junction or productivity analysis." },
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
