import { useState } from "react";
import { Link } from "react-router-dom";
import { tracks } from "../lessons/content";

export default function Home() {
  const [copied, setCopied] = useState(false);
  const [alignType, setAlignType] = useState<"igh" | "igkl" | "tcrb" | "other">("igh");

  const copyQuick = () => {
    const code = 'pip install "AlignAIR[cli]"\nalignair doctor\nalignair demo';
    navigator.clipboard?.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    });
  };

  const features = [
    { num: '01', title: 'End-to-end neural model', body: 'One network detects orientation, localizes V/D/J, and calls alleles from a shared representation — no multi-stage heuristic pipeline.' },
    { num: '02', title: 'Self-contained models', body: 'Each model embeds a fingerprinted germline reference and loads without executing any pickle. Pretrained IGH, IGK+IGL and TRB are a command away.' },
    { num: '03', title: 'Donor-genotype constraint', body: 'Restrict calls to a subset of the model’s reference at inference — no retraining required.' },
    { num: '04', title: 'Standard AIRR output', body: 'Schema-valid AIRR rearrangement TSV with honest quality flags, ready for Scirpy, Change-O, Immcantation and nf-core/airrflow.' },
  ];

  return (
    <div style={{ background: "#fbfbfd", color: "#16151f", fontFamily: "'IBM Plex Sans', system-ui, -apple-system, sans-serif" }}>
      {/* Hero */}
      <section style={{ position: "relative", overflow: "hidden", borderBottom: "1px solid #eae9f1" }}>
        <div aria-hidden="true" style={{ position: "absolute", inset: 0, pointerEvents: "none", background: "radial-gradient(52% 46% at 68% -6%, rgba(87,79,214,0.13) 0%, rgba(87,79,214,0) 72%)" }}></div>
        <div aria-hidden="true" style={{ position: "absolute", inset: 0, pointerEvents: "none", opacity: 0.5, backgroundImage: "linear-gradient(#efeef6 1px, transparent 1px), linear-gradient(90deg, #efeef6 1px, transparent 1px)", backgroundSize: "46px 46px", WebkitMaskImage: "radial-gradient(60% 55% at 50% 0%, #000 0%, transparent 78%)", maskImage: "radial-gradient(60% 55% at 50% 0%, #000 0%, transparent 78%)" }}></div>
        <div className="mx-auto" style={{ position: "relative", maxWidth: "1200px", padding: "92px 28px 84px", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(320px, 100%), 1fr))", gap: "56px", alignItems: "center" }}>
          <div style={{ animation: "aa-rise 0.5s ease-out both" }}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: "8px", fontFamily: "'IBM Plex Mono', monospace", fontSize: "11.5px", fontWeight: 500, letterSpacing: "0.14em", textTransform: "uppercase", color: "#4a41c9", background: "#efeefc", border: "1px solid #e0ddfa", padding: "6px 12px", borderRadius: "999px" }}>
              <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#574fd6" }}></span>
              End-to-end neural V(D)J aligner
            </span>
            <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: "52px", lineHeight: 1.05, letterSpacing: "-0.03em", margin: "22px 0 0", color: "#16151f" }}>
              Align immune repertoires with one neural model.
            </h1>
            <p style={{ margin: "24px 0 0", fontSize: "18px", lineHeight: 1.6, color: "#56546a", maxWidth: "34em" }}>
              A single forward pass reads orientation, V/D/J segmentation and allele identity from one shared
              representation — no multi-stage heuristic search. Deterministic post-processing then turns those
              predictions into coordinates, CIGARs, the junction and a standard AIRR record. Learn it by doing, then
              reach for the docs.
            </p>
            <div style={{ marginTop: "34px", display: "flex", flexWrap: "wrap", gap: "16px", alignItems: "center" }}>
              <Link to="/docs/getting-started" style={{ display: "inline-flex", alignItems: "center", gap: "9px", padding: "13px 24px", borderRadius: "11px", background: "#574fd6", color: "#fff", fontSize: "15px", fontWeight: 600, boxShadow: "0 6px 20px rgba(87,79,214,0.28)", transition: "all 0.2s" }} className="hover:opacity-90">
                Install &amp; align sequences <span style={{ fontFamily: "'IBM Plex Mono', monospace" }}>→</span>
              </Link>
              <Link to="/learn" style={{ display: "inline-flex", alignItems: "center", gap: "9px", padding: "13px 24px", borderRadius: "11px", background: "#fff", color: "#2a2836", fontSize: "15px", fontWeight: 600, border: "1px solid #dcdbe8", transition: "all 0.2s" }} className="hover:bg-slate-50">
                Learn how AlignAIR works
              </Link>
              <Link to="/docs/training" style={{ fontSize: "14.5px", fontWeight: 600, color: "#4a41c9", textDecoration: "underline", whiteSpace: "nowrap" }}>
                Train custom model
              </Link>
            </div>
            <div style={{ marginTop: "28px", display: "inline-flex", alignItems: "center", gap: "12px", fontFamily: "'IBM Plex Mono', monospace", fontSize: "13.5px", color: "#56546a", background: "#fff", border: "1px solid #eae9f1", borderRadius: "10px", padding: "11px 15px" }}>
              <span aria-hidden="true" style={{ color: "#6f6d85" }}>$</span>
              <span>pip install <span style={{ color: "#16151f" }}>"AlignAIR[cli]"</span></span>
            </div>

            {/* Evidence strip: the method is peer-reviewed, the numbers are reproducible, and the
                limits are documented. All three should be reachable without scrolling. */}
            <div style={{ marginTop: "26px", paddingTop: "20px", borderTop: "1px solid #eae9f1", maxWidth: "34em" }}>
              <p style={{ margin: 0, fontSize: "14px", lineHeight: 1.6, color: "#56546a" }}>
                Published in{" "}
                <em>Nucleic Acids Research</em>:{" "}
                <a
                  href="https://doi.org/10.1093/nar/gkaf651"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: "#4a41c9", fontWeight: 600, textDecoration: "underline" }}
                >
                  Enhancing sequence alignment of adaptive immune receptors through multi-task deep learning
                </a>{" "}
                <span style={{ color: "#6f6d85" }}>(Konstantinovsky et al., 2025, gkaf651)</span>.
              </p>
              <div style={{ marginTop: "12px", display: "flex", flexWrap: "wrap", gap: "8px 18px", fontSize: "14px", fontWeight: 600 }}>
                <Link to="/docs/benchmarks" style={{ color: "#4a41c9", textDecoration: "underline" }}>
                  Benchmarks &amp; how to reproduce them
                </Link>
                <Link to="/docs/known-failure-modes" style={{ color: "#4a41c9", textDecoration: "underline" }}>
                  Where it fails
                </Link>
              </div>
            </div>
          </div>

          {/* Signature alignment visual — a decorative illustration of one aligned read. Exposed to
              assistive tech as a single labelled image rather than a soup of low-contrast spans. */}
          <div
            role="img"
            aria-label="Illustration of a 312 bp human IGH read aligned to its V, D and J segments, producing the calls IGHV3-23*01, IGHD3-10*01 and IGHJ4*02 with junction_aa CARDYYGSGSYYFDYW."
            style={{ animation: "aa-rise 0.6s ease-out both" }}
          >
            <div style={{ background: "#fff", border: "1px solid #e7e6ef", borderRadius: "16px", boxShadow: "0 24px 60px -30px rgba(30,28,52,0.32), 0 6px 18px -12px rgba(30,28,52,0.14)", overflow: "hidden" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "13px 18px", borderBottom: "1px solid #f0eff5", background: "#faf9fd" }}>
                <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "12px", color: "#6f6d85" }}>read_0417 · 312 bp</span>
                <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "11px", fontWeight: 600, color: "#4a41c9", background: "#efeefc", padding: "3px 9px", borderRadius: "6px" }}>IGH · human</span>
              </div>
              <div style={{ padding: "20px 18px 22px" }}>
                <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "12.5px", letterSpacing: "0.04em", color: "#b9b7c7", whiteSpace: "nowrap", overflow: "hidden" }}>
                  …GCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCTTT…
                </div>
                <div style={{ position: "relative", height: "40px", marginTop: "14px" }}>
                  <div style={{ position: "absolute", top: 0, left: "2%", width: "60%", height: "40px", borderRadius: "8px", background: "#574fd6", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "inset 0 -2px 0 rgba(0,0,0,0.12)" }}>
                    <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "12px", fontWeight: 600, color: "#fff" }}>V</span>
                  </div>
                  <div style={{ position: "absolute", top: 0, left: "63.5%", width: "8%", height: "40px", borderRadius: "8px", background: "#9b4bd6", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "inset 0 -2px 0 rgba(0,0,0,0.12)" }}>
                    <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "12px", fontWeight: 600, color: "#fff" }}>D</span>
                  </div>
                  <div style={{ position: "absolute", top: 0, left: "73%", width: "25%", height: "40px", borderRadius: "8px", background: "#3f7fd6", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "inset 0 -2px 0 rgba(0,0,0,0.12)" }}>
                    <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "12px", fontWeight: 600, color: "#fff" }}>J</span>
                  </div>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "'IBM Plex Mono', monospace", fontSize: "10.5px", color: "#b9b7c7", marginTop: "7px" }}>
                  <span>1</span>
                  <span>104</span>
                  <span>208</span>
                  <span>312</span>
                </div>

                <div style={{ marginTop: "18px", borderTop: "1px dashed #eae9f1", paddingTop: "16px", display: "grid", gap: "8px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", fontFamily: "'IBM Plex Mono', monospace", fontSize: "12.5px" }}>
                    <span style={{ width: "92px", flexShrink: 0, color: "#6f6d85" }}>v_call</span>
                    <span style={{ color: "#16151f", fontWeight: 500 }}>IGHV3-23*01</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", fontFamily: "'IBM Plex Mono', monospace", fontSize: "12.5px" }}>
                    <span style={{ width: "92px", flexShrink: 0, color: "#6f6d85" }}>d_call</span>
                    <span style={{ color: "#16151f", fontWeight: 500 }}>IGHD3-10*01</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", fontFamily: "'IBM Plex Mono', monospace", fontSize: "12.5px" }}>
                    <span style={{ width: "92px", flexShrink: 0, color: "#6f6d85" }}>j_call</span>
                    <span style={{ color: "#16151f", fontWeight: 500 }}>IGHJ4*02</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", fontFamily: "'IBM Plex Mono', monospace", fontSize: "12.5px" }}>
                    <span style={{ width: "92px", flexShrink: 0, color: "#6f6d85" }}>junction_aa</span>
                    <span style={{ color: "#12805c", fontWeight: 500 }}>CARDYYGSGSYYFDYW</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="mx-auto" style={{ maxWidth: "1200px", padding: "84px 28px 0" }}>
        <h2 className="sr-only">Why use AlignAIR</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(240px, 100%), 1fr))", gap: "18px" }}>
          {features.map((f) => (
            <article key={f.num} style={{ background: "#fff", border: "1px solid #eae9f1", borderRadius: "14px", padding: "24px 22px 26px" }}>
              <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "12px", fontWeight: 600, color: "#574fd6", letterSpacing: "0.06em" }}>
                {f.num}
              </span>
              <h3 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, fontSize: "17px", letterSpacing: "-0.01em", margin: "12px 0 0", color: "#16151f" }}>
                {f.title}
              </h3>
              <p style={{ margin: "9px 0 0", fontSize: "13.5px", lineHeight: 1.6, color: "#6f6d85" }}>
                {f.body}
              </p>
            </article>
          ))}
        </div>
      </section>

      {/* Model Finder */}
      <section className="mx-auto" style={{ maxWidth: "840px", padding: "92px 28px 0" }}>
        <div style={{ textAlign: "center" }}>
          <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "11.5px", fontWeight: 500, letterSpacing: "0.14em", textTransform: "uppercase", color: "#6f6d85" }}>
            Model Finder
          </span>
          <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: "34px", letterSpacing: "-0.02em", margin: "12px 0 0", color: "#16151f" }}>
            What are you aligning?
          </h2>
          <p style={{ margin: "12px 0 0", fontSize: "16px", color: "#56546a" }}>
            Select your receptor locus to see the correct alignment model and command.
          </p>
        </div>

        <div role="group" aria-label="Choose your receptor locus" style={{ marginTop: "32px", display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "10px" }}>
          {[
            { id: "igh", label: "Human IGH (Heavy)" },
            { id: "igkl", label: "Human IGK + IGL (Light)" },
            { id: "tcrb", label: "Human TRB (TCR Beta)" },
            { id: "other", label: "Another reference / species" },
          ].map((opt) => (
            <button
              key={opt.id}
              type="button"
              aria-pressed={alignType === opt.id}
              aria-controls="model-finder-panel"
              onClick={() => setAlignType(opt.id as "igh" | "igkl" | "tcrb" | "other")}
              style={{
                fontFamily: "'Space Grotesk', sans-serif",
                fontSize: "14px",
                fontWeight: 600,
                padding: "10px 18px",
                borderRadius: "10px",
                border: alignType === opt.id ? "1px solid #574fd6" : "1px solid #dcdbe8",
                background: alignType === opt.id ? "#574fd6" : "#fff",
                color: alignType === opt.id ? "#fff" : "#2a2836",
                cursor: "pointer",
                boxShadow: alignType === opt.id ? "0 4px 12px rgba(87,79,214,0.2)" : "none",
                transition: "all 0.15s ease",
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>

        <div id="model-finder-panel" role="region" aria-live="polite" aria-label="Recommended model" style={{ marginTop: "24px", background: "#fff", border: "1px solid #eae9f1", borderRadius: "16px", padding: "28px", boxShadow: "0 10px 30px -15px rgba(30,28,52,0.1)" }}>
          {alignType === "igh" && (
            <div>
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
                <span style={{ fontSize: "12px", fontWeight: 600, color: "#fff", background: "#574fd6", padding: "4px 10px", borderRadius: "6px" }}>Pretrained</span>
                <span style={{ fontSize: "13px", fontWeight: 500, color: "#56546a" }}>Model ID: <strong>alignair-igh-human</strong></span>
                <span style={{ fontSize: "13px", fontWeight: 500, color: "#56546a" }}>Source: <strong>OGRDB (V198/D33/J7)</strong></span>
              </div>
              <p style={{ margin: 0, fontSize: "14.5px", color: "#56546a", lineHeight: 1.6 }}>
                Aligns human immunoglobulin heavy chains. Supports somatic hypermutations (SHM) and full-length V(D)J assembly. Limitations: junctions can jitter ~1-2 nt on heavy SHM; D-allele calls carry inherent biological ambiguity.
              </p>
              <pre tabIndex={0} aria-label="Human IGH prediction command" style={{ marginTop: "16px", padding: "14px 16px", background: "#16151f", borderRadius: "10px", fontFamily: "'IBM Plex Mono', monospace", fontSize: "13px", color: "#e6e5f0", overflowX: "auto" }}>
                <span style={{ color: "#9d9bb0" }}>$</span> alignair predict --input reads.fasta --out out.tsv --model alignair-igh-human
              </pre>
            </div>
          )}
          {alignType === "igkl" && (
            <div>
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
                <span style={{ fontSize: "12px", fontWeight: 600, color: "#fff", background: "#574fd6", padding: "4px 10px", borderRadius: "6px" }}>Pretrained</span>
                <span style={{ fontSize: "13px", fontWeight: 500, color: "#56546a" }}>Model ID: <strong>alignair-igkl-human</strong></span>
                <span style={{ fontSize: "13px", fontWeight: 500, color: "#56546a" }}>Source: <strong>OGRDB (V349/J18)</strong></span>
              </div>
              <p style={{ margin: 0, fontSize: "14.5px", color: "#56546a", lineHeight: 1.6 }}>
                Multi-locus model for human IGK (Kappa) and IGL (Lambda) light chains. Automatically attributes reads to the correct locus. Limitations: no D segment is present (empty by design).
              </p>
              <pre tabIndex={0} aria-label="Human IGK and IGL prediction command" style={{ marginTop: "16px", padding: "14px 16px", background: "#16151f", borderRadius: "10px", fontFamily: "'IBM Plex Mono', monospace", fontSize: "13px", color: "#e6e5f0", overflowX: "auto" }}>
                <span style={{ color: "#9d9bb0" }}>$</span> alignair predict --input reads.fasta --out out.tsv --model alignair-igkl-human
              </pre>
            </div>
          )}
          {alignType === "tcrb" && (
            <div>
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
                <span style={{ fontSize: "12px", fontWeight: 600, color: "#fff", background: "#574fd6", padding: "4px 10px", borderRadius: "6px" }}>Pretrained</span>
                <span style={{ fontSize: "13px", fontWeight: 500, color: "#56546a" }}>Model ID: <strong>alignair-tcrb-human</strong></span>
                <span style={{ fontSize: "13px", fontWeight: 500, color: "#56546a" }}>Source: <strong>IMGT (V98/D3/J16)</strong></span>
              </div>
              <p style={{ margin: 0, fontSize: "14.5px", color: "#56546a", lineHeight: 1.6 }}>
                Aligns human T-cell receptor beta chains. Trained with somatic hypermutation set to zero, since T cells do not hypermutate. Scope: this checkpoint covers the TRB locus only. Alpha, gamma and delta are not in the pretrained registry — they are not excluded by the architecture, so you can train a model for them from a TRA/TRG/TRD reference with <strong>alignair train</strong>.
              </p>
              <pre tabIndex={0} aria-label="Human TRB prediction command" style={{ marginTop: "16px", padding: "14px 16px", background: "#16151f", borderRadius: "10px", fontFamily: "'IBM Plex Mono', monospace", fontSize: "13px", color: "#e6e5f0", overflowX: "auto" }}>
                <span style={{ color: "#9d9bb0" }}>$</span> alignair predict --input reads.fasta --out out.tsv --model alignair-tcrb-human
              </pre>
            </div>
          )}
          {alignType === "other" && (
            <div>
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
                <span style={{ fontSize: "12px", fontWeight: 600, color: "#fff", background: "#df382c", padding: "4px 10px", borderRadius: "6px" }}>Custom Train</span>
                <span style={{ fontSize: "13px", fontWeight: 500, color: "#56546a" }}>Any Locus or Species</span>
              </div>
              <p style={{ margin: 0, fontSize: "14.5px", color: "#56546a", lineHeight: 1.6 }}>
                There is no pretrained model for other loci (like TCR Alpha/Gamma/Delta or IG Heavy/Light chains of non-human species). You must train a custom model using a built-in GenAIRR DataConfig or your own germline FASTA files.
              </p>
              <pre tabIndex={0} aria-label="Custom reference training command" style={{ marginTop: "16px", padding: "14px 16px", background: "#16151f", borderRadius: "10px", fontFamily: "'IBM Plex Mono', monospace", fontSize: "13px", color: "#e6e5f0", overflowX: "auto" }}>
                <span style={{ color: "#9d9bb0" }}>$</span> alignair train --v-fasta v.fasta --d-fasta d.fasta --j-fasta j.fasta --chain-type BCR_HEAVY --out runs/my_ref
              </pre>
              <p style={{ margin: "12px 0 0 0", fontSize: "13.5px" }}>
                Read the <Link to="/docs/training" style={{ color: "#4a41c9", textDecoration: "underline", fontWeight: 600 }}>Custom Reference Training guide</Link> to get started.
              </p>
            </div>
          )}
        </div>
      </section>

      {/* Quick start */}
      <section className="mx-auto" style={{ maxWidth: "760px", padding: "92px 28px 0", textAlign: "center" }}>
        <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "11.5px", fontWeight: 500, letterSpacing: "0.14em", textTransform: "uppercase", color: "#6f6d85" }}>
          Quick start
        </span>
        <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: "34px", letterSpacing: "-0.02em", margin: "12px 0 0", color: "#16151f" }}>
          Up and running in three lines
        </h2>
        <p style={{ margin: "12px 0 0", fontSize: "16px", color: "#56546a" }}>
          Install the CLI, run the doctor checks, and launch the offline demo immediately.
        </p>
        <div style={{ marginTop: "30px", textAlign: "left", borderRadius: "14px", overflow: "hidden", border: "1px solid #24222f", boxShadow: "0 22px 50px -28px rgba(30,28,52,0.5)" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "11px 16px", background: "#211f2c", borderBottom: "1px solid #2f2c3d" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ width: "11px", height: "11px", borderRadius: "50%", background: "#3a3746" }}></span>
              <span style={{ width: "11px", height: "11px", borderRadius: "50%", background: "#3a3746" }}></span>
              <span style={{ width: "11px", height: "11px", borderRadius: "50%", background: "#3a3746" }}></span>
              <span style={{ marginLeft: "8px", fontFamily: "'IBM Plex Mono', monospace", fontSize: "11px", color: "#9d9bb0", letterSpacing: "0.08em" }}>bash</span>
            </div>
            <button
              type="button"
              onClick={copyQuick}
              aria-label={copied ? "Copied to clipboard" : "Copy install commands"}
              style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "11px", color: "#a9a7ba", background: "#2f2c3d", border: "none", borderRadius: "6px", padding: "4px 10px", cursor: "pointer" }}
            >
              <span aria-hidden="true">{copied ? "Copied ✓" : "Copy"}</span>
            </button>
          </div>
          <pre tabIndex={0} aria-label="Quick start commands" style={{ margin: 0, padding: "20px 18px", background: "#16151f", fontFamily: "'IBM Plex Mono', monospace", fontSize: "13.5px", lineHeight: 2, color: "#e6e5f0", overflowX: "auto" }}>
            <span style={{ color: "#9d9bb0" }}>$</span> pip install <span style={{ color: "#b7f3d8" }}>"AlignAIR[cli]"</span>{"\n"}
            <span style={{ color: "#9d9bb0" }}>$</span> alignair doctor{"\n"}
            <span style={{ color: "#9d9bb0" }}>$</span> alignair demo
          </pre>
        </div>

        {/* Real data workflow */}
        <div style={{ marginTop: "32px", textAlign: "left" }}>
          <h3 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: "20px", color: "#16151f" }}>
            Aligning your own sequences
          </h3>
          <p style={{ margin: "8px 0 0", fontSize: "14.5px", color: "#56546a", lineHeight: 1.5 }}>
            To run alignment on your own data, view available models and pass your sequence file. You can download sample files from the repository's <a href="https://github.com/MuteJester/AlignAIR/tree/main/examples" target="_blank" rel="noreferrer" style={{ color: "#4a41c9", textDecoration: "underline", fontWeight: 600 }}>examples/</a> folder:
          </p>
          <pre tabIndex={0} aria-label="Real sequence prediction commands" style={{ marginTop: "14px", padding: "16px 18px", background: "#16151f", border: "1px solid #24222f", borderRadius: "12px", fontFamily: "'IBM Plex Mono', monospace", fontSize: "13px", lineHeight: 1.8, color: "#e6e5f0", overflowX: "auto" }}>
            <span style={{ color: "#9d9bb0" }}>$</span> alignair models list{"\n"}
            <span style={{ color: "#9d9bb0" }}>$</span> alignair predict --input reads.fasta --out out.tsv --model alignair-igh-human
          </pre>
        </div>
      </section>

      {/* Train section */}
      <section className="mx-auto" style={{ maxWidth: "1200px", padding: "92px 28px 0" }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(320px, 100%), 1fr))", gap: "48px", alignItems: "center", background: "linear-gradient(135deg, #1e1c34 0%, #11101d 100%)", borderRadius: "24px", padding: "48px", border: "1px solid #2f2c3d", boxShadow: "0 20px 40px -15px rgba(0,0,0,0.3)" }}>
          <div>
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "11px", fontWeight: 600, letterSpacing: "0.14em", textTransform: "uppercase", color: "#9b8bf4" }}>
              Custom References
            </span>
            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: "32px", letterSpacing: "-0.02em", margin: "14px 0 0", color: "#fff" }}>
              Train on your own reference or species
            </h2>
            <p style={{ margin: "16px 0 0", fontSize: "16px", lineHeight: 1.6, color: "#a9a7ba" }}>
              No pretrained model for your species or locus? AlignAIR is fully customizable. Train the model using a built-in GenAIRR DataConfig or supply your own V, D, and J germline FASTAs to generate a self-contained, fingerprinted model package.
            </p>
            <div style={{ marginTop: "24px" }}>
              <Link to="/docs/training" style={{ display: "inline-flex", alignItems: "center", gap: "8px", padding: "12px 20px", borderRadius: "10px", background: "#574fd6", color: "#fff", fontSize: "14px", fontWeight: 600, transition: "opacity 0.2s" }} className="hover:opacity-90">
                Read the training guide <span style={{ fontFamily: "'IBM Plex Mono', monospace" }}>→</span>
              </Link>
            </div>
          </div>
          <div style={{ background: "#16151f", borderRadius: "16px", border: "1px solid #24222f", padding: "24px", overflow: "hidden" }}>
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "11px", color: "#9d9bb0", display: "block", marginBottom: "12px" }}># Train model on a custom germline reference</span>
            <pre tabIndex={0} aria-label="Custom germline reference training command" style={{ margin: 0, fontFamily: "'IBM Plex Mono', monospace", fontSize: "13px", lineHeight: 1.8, color: "#e6e5f0", overflowX: "auto" }}>
              <span style={{ color: "#cfcde3" }}>alignair train \</span>{"\n"}
              <span style={{ color: "#cfcde3" }}>  --v-fasta v.fasta \</span>{"\n"}
              <span style={{ color: "#cfcde3" }}>  --d-fasta d.fasta \</span>{"\n"}
              <span style={{ color: "#cfcde3" }}>  --j-fasta j.fasta \</span>{"\n"}
              <span style={{ color: "#cfcde3" }}>  --chain-type BCR_HEAVY \</span>{"\n"}
              <span style={{ color: "#cfcde3" }}>  --out runs/my_reference \</span>{"\n"}
              <span style={{ color: "#cfcde3" }}>  --preset desktop</span>
            </pre>
          </div>
        </div>
      </section>

      {/* Learning tracks */}
      <section className="mx-auto" style={{ maxWidth: "1200px", padding: "92px 28px 0" }}>
        <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: "24px", marginBottom: "30px" }}>
          <div>
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "11.5px", fontWeight: 500, letterSpacing: "0.14em", textTransform: "uppercase", color: "#6f6d85" }}>
              Interactive lessons
            </span>
            <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: "34px", letterSpacing: "-0.02em", margin: "10px 0 0", color: "#16151f" }}>
              Learn AlignAIR by doing
            </h2>
            <p style={{ margin: "10px 0 0", fontSize: "16px", color: "#56546a", maxWidth: "40em" }}>
              Short, hands-on lessons — from first principles to inference, training and benchmarking. Progress saves in your browser.
            </p>
          </div>
          <Link to="/learn" style={{ flexShrink: 0, display: "inline-flex", alignItems: "center", gap: 7, fontSize: "14px", fontWeight: 600, color: "#4a41c9" }}>
            All tracks <span style={{ fontFamily: "'IBM Plex Mono', monospace" }}>→</span>
          </Link>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(240px, 100%), 1fr))", gap: "18px" }}>
          {tracks.map((t) => (
            <Link
              key={t.slug}
              to={`/learn/${t.slug}/${t.lessons[0].slug}`}
              style={{ display: "flex", flexDirection: "column", background: "#fff", border: "1px solid #eae9f1", borderRadius: "14px", padding: "22px", color: "#16151f", transition: "all 0.15s" }}
              className="hover:border-indigo-200 hover:-translate-y-0.5 hover:shadow-sm"
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: "34px", height: "34px", borderRadius: "9px", background: "#f2f1fb", color: "#574fd6", fontFamily: "'IBM Plex Mono', monospace", fontSize: "13px", fontWeight: 600 }}>
                  {t.lessons[0].id.split("/")[0] === "foundations" && t.slug === "foundations" ? "01" :
                   t.slug === "predicting" ? "02" :
                   t.slug === "training" ? "03" :
                   t.slug === "benchmarking" ? "04" : "05"}
                </span>
                <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "11px", color: "#6f6d85" }}>
                  {t.lessons.length} {t.lessons.length === 1 ? "lesson" : "lessons"}
                </span>
              </div>
              <h3 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, fontSize: "18px", letterSpacing: "-0.01em", margin: "16px 0 0" }}>
                {t.title}
              </h3>
              <p style={{ margin: "8px 0 0", fontSize: "13.5px", lineHeight: 1.6, color: "#6f6d85" }}>
                {t.description}
              </p>
            </Link>
          ))}
        </div>
      </section>

      {/* Trust */}
      <section className="mx-auto" style={{ maxWidth: "900px", padding: "92px 28px 84px" }}>
        <div style={{ border: "1px solid #eae9f1", borderRadius: "18px", background: "linear-gradient(180deg, #ffffff, #faf9fd)", padding: "44px", textAlign: "center" }}>
          <span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: "46px", height: "46px", borderRadius: "12px", background: "#f2f1fb", color: "#574fd6", fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600, fontSize: "15px" }}>
            {`{ }`}
          </span>
          <p style={{ margin: "20px auto 0", maxWidth: "42em", fontSize: "17px", lineHeight: 1.6, color: "#3a3849" }}>
            AlignAIR is open source under <strong style={{ fontWeight: 600, color: "#16151f" }}>GPL-3.0</strong>. Models are self-contained and safe to load, carrying a fingerprinted germline reference that cannot drift from the weights.
          </p>
          <div style={{ marginTop: "26px", display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "12px" }}>
            <a href="https://github.com/MuteJester/AlignAIR" target="_blank" rel="noreferrer" style={{ padding: "10px 20px", borderRadius: "10px", background: "#fff", border: "1px solid #dcdbe8", color: "#2a2836", fontSize: "14px", fontWeight: 600, transition: "all 0.15s" }} className="hover:bg-slate-50">
              GitHub
            </a>
            <a href="https://huggingface.co/AlignAIR/AlignAIR-pretrained" target="_blank" rel="noreferrer" style={{ padding: "10px 20px", borderRadius: "10px", background: "#fff", border: "1px solid #dcdbe8", color: "#2a2836", fontSize: "14px", fontWeight: 600, transition: "all 0.15s" }} className="hover:bg-slate-50">
              Model hub
            </a>
          </div>
        </div>
      </section>
    </div>
  );
}
