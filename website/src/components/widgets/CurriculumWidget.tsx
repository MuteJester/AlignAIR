import { useState } from "react";

/**
 * The GenAIRR training curriculum, knob by knob. Values mirror `Curriculum(indel_form="weighted").params(p)` in
 * `alignair/train/gym/curriculum/base.py` (linear ramps over the curriculum position p, with Python's
 * round-half-to-even for the integer ranges), plus the per-locus SHM rule from
 * `alignair.train.gym.streams.mutation_cap_for`: a TCR locus is always capped to zero. Orientation is not a
 * curriculum knob: the batch adapter draws one of 4 orientations per read, uniformly, at every position.
 */
const lerp = (a: number, b: number, p: number) => a + (b - a) * p;
const roundHalfEven = (x: number) => {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
};
const SEQUENCING_ERROR_SCALE = 0.757;

export function CurriculumWidget() {
  const [p, setP] = useState(0.3);
  const [tcr, setTcr] = useState(false);

  const shm = tcr ? 0 : lerp(0.005, 0.15, p);
  const indelShare = lerp(0.05, 0.25, p);
  const seqError = lerp(0, 0.02, p) * SEQUENCING_ERROR_SCALE;
  const knobs = [
    {
      label: "Somatic hypermutation (S5F)",
      value: tcr ? "0 — T-cells lack AID" : `${(shm * 100).toFixed(1)}% per base`,
      frac: tcr ? 0 : shm / 0.15,
      muted: tcr,
    },
    { label: "Reads with polymerase indels", value: `${Math.round(indelShare * 100)}% of reads, 1–3 each`, frac: indelShare / 0.25, muted: false },
    { label: "Sequencing error", value: `${(seqError * 100).toFixed(2)}% per base`, frac: p, muted: false },
    { label: "Ambiguous (N) bases", value: `0 – ${roundHalfEven(lerp(0, 5, p))}`, frac: p, muted: false },
    { label: "End-loss (read shortening)", value: `0 – ${roundHalfEven(lerp(0, 40, p))} nt per end`, frac: p, muted: false },
    { label: "Non-forward orientation (fixed)", value: "75% of reads, at every position", frac: 0.75, muted: true },
  ];

  const stage = p < 0.25 ? "early — mostly clean" : p < 0.7 ? "mid — realistic noise" : "late — heavily degraded";

  return (
    <div style={{ margin: "22px 0", border: "1px solid #eae9f1", borderRadius: "14px", background: "#fff", padding: "20px 18px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "10px" }}>
        <label htmlFor="curriculum-position" style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "11px", fontWeight: 600, letterSpacing: "0.08em", color: "#6f6d85" }}>
          CURRICULUM POSITION
        </label>
        <span aria-hidden="true" style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "12.5px", fontWeight: 600, color: "#4238c4" }}>
          progress {p.toFixed(2)} · {stage}
        </span>
      </div>
      <input
        id="curriculum-position"
        type="range"
        min="0"
        max="1"
        step="0.05"
        value={p}
        onChange={(e) => setP(Number(e.target.value))}
        aria-valuetext={`position ${p.toFixed(2)} of 1.00, ${stage}`}
        aria-describedby="curriculum-note"
        style={{ width: "100%", marginTop: "12px", accentColor: "#574fd6", cursor: "pointer" }}
      />

      <div role="group" aria-label="Locus" style={{ marginTop: "14px", display: "flex", gap: "8px" }}>
        {[
          { k: false, label: "IG locus (IGH / IGK / IGL)" },
          { k: true, label: "TCR locus (TRA / TRB)" },
        ].map((o) => (
          <button
            key={String(o.k)}
            type="button"
            aria-pressed={tcr === o.k}
            onClick={() => setTcr(o.k)}
            style={{
              flex: 1,
              padding: "8px 10px",
              borderRadius: "9px",
              cursor: "pointer",
              fontFamily: "IBM Plex Mono, monospace",
              fontSize: "11.5px",
              fontWeight: 600,
              border: `1px solid ${tcr === o.k ? "#574fd6" : "#eae9f1"}`,
              background: tcr === o.k ? "#f2f1fb" : "#fff",
              color: tcr === o.k ? "#4238c4" : "#56546a",
            }}
          >
            {o.label}
          </button>
        ))}
      </div>

      <div style={{ marginTop: "18px", display: "flex", flexDirection: "column", gap: "13px" }}>
        {knobs.map((k) => (
          <div key={k.label}>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: "10px" }}>
              <span style={{ fontFamily: "Space Grotesk, sans-serif", fontWeight: 600, fontSize: "13px", color: k.muted ? "#6f6d85" : "#16151f" }}>
                {k.label}
              </span>
              <span style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: "12px", fontWeight: 600, color: k.muted ? "#6f6d85" : "#4238c4" }}>
                {k.value}
              </span>
            </div>
            <div aria-hidden="true" style={{ marginTop: "6px", height: "10px", borderRadius: "5px", background: "#f1f0f8", overflow: "hidden" }}>
              <div
                style={{
                  height: "100%",
                  borderRadius: "5px",
                  background: k.muted ? "#dcdbe8" : "#574fd6",
                  transition: "width 0.2s",
                  width: `${Math.max(0, Math.min(1, k.frac)) * 100}%`,
                }}
              ></div>
            </div>
          </div>
        ))}
      </div>

      <p id="curriculum-note" style={{ margin: "16px 0 0", paddingTop: "12px", borderTop: "1px dashed #eae9f1", fontFamily: "IBM Plex Mono, monospace", fontSize: "11px", lineHeight: 1.6, color: "#6f6d85" }}>
        Ramps mirror <code>Curriculum(indel_form=&quot;weighted&quot;).params(p)</code>. The training mix also
        interleaves amplicon shapes (V-anchored, J-anchored, both-ends fragments), a heavy-SHM stream and library
        flanks — switch to a TCR locus to see SHM drop to zero.
      </p>
    </div>
  );
}
