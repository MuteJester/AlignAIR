import { Link } from "react-router-dom";
import { tracks } from "../lessons/content";
import { isLessonComplete } from "../lib/util";

export default function Learn() {
  return (
    <div style={{ maxWidth: "900px", margin: "0 auto", padding: "56px 28px 84px", fontFamily: "'IBM Plex Sans', system-ui, sans-serif", color: "#16151f" }}>
      <header style={{ marginBottom: "44px" }}>
        <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "11.5px", fontWeight: 500, letterSpacing: "0.14em", textTransform: "uppercase", color: "#8b899d" }}>
          Interactive lessons
        </span>
        <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: "42px", letterSpacing: "-0.025em", margin: "12px 0 0", color: "#16151f" }}>
          Learn AlignAIR by doing
        </h1>
        <p style={{ margin: "14px 0 0", fontSize: "17px", lineHeight: 1.6, color: "#56546a", maxWidth: "44em" }}>
          Five tracks take you from what V(D)J alignment is, all the way to inference, training and honest benchmarking. Each lesson is short and hands-on — your progress is saved in this browser.
        </p>
      </header>

      <div style={{ display: "flex", flexDirection: "column", gap: "40px" }}>
        {tracks.map((t, idx) => {
          const done = t.lessons.filter((l) => isLessonComplete(l.id)).length;
          const total = t.lessons.length;
          const pct = Math.round((done / total) * 100);

          return (
            <section key={t.slug}>
              <div style={{ display: "flex", alignItems: "flex-start", gap: "16px", marginBottom: "16px" }}>
                <span style={{ flexShrink: 0, display: "inline-flex", alignItems: "center", justifyContent: "center", width: "42px", height: "42px", borderRadius: "11px", background: "#f2f1fb", color: "#574fd6", fontFamily: "'IBM Plex Mono', monospace", fontSize: "15px", fontWeight: 600 }}>
                  {String(idx + 1).padStart(2, "0")}
                </span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: "12px" }}>
                    <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, fontSize: "22px", letterSpacing: "-0.015em", margin: 0, color: "#16151f" }}>
                      {t.title}
                    </h2>
                    <span style={{ flexShrink: 0, fontFamily: "'IBM Plex Mono', monospace", fontSize: "12px", color: "#9b99ac" }}>
                      {done}/{total} done
                    </span>
                  </div>
                  <p style={{ margin: "4px 0 0", fontSize: "14px", color: "#6f6d85" }}>
                    {t.description}
                  </p>
                  <div style={{ marginTop: "12px", height: "6px", borderRadius: "999px", background: "#edecf4", overflow: "hidden" }}>
                    <div style={{ height: "100%", borderRadius: "999px", background: "#574fd6", transition: "width 0.5s", width: `${pct}%` }}></div>
                  </div>
                </div>
              </div>

              <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: "8px" }} className="sm:pl-[58px]">
                {t.lessons.map((l) => {
                  const d = isLessonComplete(l.id);
                  const dotColor = d ? "#12805c" : "#d3d1e0";
                  const dotFill = d ? "#12805c" : "transparent";

                  return (
                    <li key={l.id}>
                      <Link
                        to={`/learn/${t.slug}/${l.slug}`}
                        style={{ display: "flex", alignItems: "center", gap: "14px", border: "1px solid #eae9f1", background: "#fff", borderRadius: "12px", padding: "15px 18px", color: "#16151f", transition: "all 0.15s" }}
                        className="hover:border-indigo-200 hover:shadow-sm"
                      >
                        <span
                          style={{
                            flexShrink: 0,
                            width: "20px",
                            height: "20px",
                            borderRadius: "50%",
                            border: `2px solid ${dotColor}`,
                            background: dotFill,
                            display: "inline-flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: "11px",
                            color: "#fff",
                            fontWeight: 700,
                          }}
                        >
                          {d ? "✓" : ""}
                        </span>
                        <span style={{ flex: 1, minWidth: 0 }}>
                          <span style={{ display: "block", fontWeight: 600, fontSize: "15px" }}>
                            {l.title}
                          </span>
                          <span style={{ display: "block", fontSize: "13px", color: "#8b899d", marginTop: "2px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {l.summary}
                          </span>
                        </span>
                        <span style={{ flexShrink: 0, fontFamily: "'IBM Plex Mono', monospace", fontSize: "12px", color: "#b9b7c7" }}>
                          {l.minutes} min
                        </span>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </section>
          );
        })}
      </div>
    </div>
  );
}
