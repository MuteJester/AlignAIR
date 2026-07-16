import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import type { Lesson, Step } from "./types";
import { getLessonProgress, saveLessonProgress } from "../lib/util";
import { nextLessonOf } from "./content";

function McqStep({
  step,
  onCorrect,
  lessonTitle,
  lessonBackHref,
  stepNum,
  stepTotal,
  stepPct,
  onPrev,
  canPrev,
}: {
  step: Extract<Step, { kind: "mcq" }>;
  onCorrect: () => void;
  lessonTitle: string;
  lessonBackHref: string;
  stepNum: number;
  stepTotal: number;
  stepPct: number;
  onPrev: () => void;
  canPrev: boolean;
}) {
  const [picked, setPicked] = useState<number | null>(null);
  const answered = picked !== null;
  const correct = picked === step.answer;

  const tryAgain = () => {
    setPicked(null);
  };

  return (
    <div>
      <div style={{ marginBottom: "30px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
          <Link to={lessonBackHref} style={{ display: "inline-flex", alignItems: "center", gap: "7px", fontFamily: "'IBM Plex Mono', monospace", fontSize: "12px", color: "#8b899d" }}>
            <span>←</span> {lessonTitle}
          </Link>
          <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "12px", color: "#b9b7c7" }}>
            Step {stepNum} / {stepTotal}
          </span>
        </div>
        <div style={{ height: "5px", borderRadius: "999px", background: "#edecf4", overflow: "hidden" }}>
          <div style={{ height: "100%", borderRadius: "999px", background: "#574fd6", transition: "width 0.4s", width: `${stepPct}%` }}></div>
        </div>
      </div>

      <div style={{ animation: "aa-rise 0.35s ease-out both" }}>
        <span style={{ display: "inline-block", fontFamily: "'IBM Plex Mono', monospace", fontSize: "11px", fontWeight: 500, letterSpacing: "0.12em", textTransform: "uppercase", color: "#574fd6", background: "#f2f1fb", padding: "4px 10px", borderRadius: "6px" }}>
          Check your understanding
        </span>
        <div style={{ fontSize: "19px", lineHeight: "1.55", fontWeight: 500, color: "#16151f", marginTop: "16px" }}>
          {step.prompt()}
        </div>

        <div style={{ marginTop: "22px", display: "flex", flexDirection: "column", gap: "10px" }}>
          {step.options.map((opt, i) => {
            const isPicked = picked === i;
            const right = answered && i === step.answer;
            const wrong = answered && isPicked && i !== step.answer;

            let optBorder = "#eae9f1";
            let optBg = "#ffffff";
            let optColor = "#2a2836";
            let mark = "";
            let markColor = "";

            if (right) {
              optBorder = "#8fcfb4";
              optBg = "#eef7f2";
              optColor = "#0f6b4e";
              mark = "✓";
              markColor = "#12805c";
            } else if (wrong) {
              optBorder = "#e9a9b4";
              optBg = "#fdeef0";
              optColor = "#a12b3f";
              mark = "✕";
              markColor = "#c0344a";
            }

            const opacity = answered && !right && !wrong ? "0.55" : "1";

            return (
              <button
                key={i}
                type="button"
                disabled={answered}
                onClick={() => setPicked(i)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: "12px",
                  width: "100%",
                  textAlign: "left",
                  border: `1px solid ${optBorder}`,
                  background: optBg,
                  color: optColor,
                  opacity: opacity,
                  borderRadius: "12px",
                  padding: "15px 17px",
                  fontFamily: "inherit",
                  fontSize: "15px",
                  fontWeight: 500,
                  cursor: answered ? "default" : "pointer",
                  transition: "border-color 0.15s, background 0.15s",
                }}
              >
                <span>{opt}</span>
                {mark && (
                  <span style={{ flexShrink: 0, fontWeight: 700, fontSize: "16px", color: markColor }}>
                    {mark}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {answered && (
          <div
            style={{
              marginTop: "18px",
              border: `1px solid ${correct ? "#9ad3bb" : "#f0d59a"}`,
              background: correct ? "#eef7f2" : "#fdf4e6",
              borderRadius: "12px",
              padding: "17px 18px",
              animation: "aa-rise 0.3s ease-out both",
            }}
          >
            <p style={{ margin: "0 0 7px", fontWeight: 600, fontSize: "15px", color: "#16151f" }}>
              {correct ? "Correct" : "Not quite"}
            </p>
            <div style={{ margin: 0, fontSize: "14.5px", lineHeight: 1.7, color: "#3a3849" }}>
              {step.explanation()}
            </div>
            {correct ? (
              <div style={{ marginTop: "16px" }}>
                <button
                  type="button"
                  onClick={onCorrect}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "9px",
                    padding: "11px 22px",
                    borderRadius: "10px",
                    background: "#574fd6",
                    color: "#fff",
                    fontFamily: "inherit",
                    fontSize: "14px",
                    fontWeight: 600,
                    border: "none",
                    cursor: "pointer",
                  }}
                >
                  Continue <span style={{ fontFamily: "'IBM Plex Mono', monospace" }}>→</span>
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={tryAgain}
                style={{
                  marginTop: "14px",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  fontFamily: "inherit",
                  fontSize: "13.5px",
                  fontWeight: 600,
                  color: "#a66a00",
                  textDecoration: "underline",
                  textUnderlineOffset: "3px",
                }}
              >
                Try again
              </button>
            )}
          </div>
        )}

        {canPrev && (
          <div style={{ marginTop: "26px" }}>
            <button
              type="button"
              onClick={onPrev}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                fontFamily: "inherit",
                fontSize: "14px",
                color: "#8b899d",
              }}
            >
              ← Previous
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default function LessonPlayer({ lesson }: { lesson: Lesson }) {
  const [index, setIndex] = useState(() => Math.min(getLessonProgress(lesson.id).step, lesson.steps.length - 1));
  const [finished, setFinished] = useState(() => getLessonProgress(lesson.id).completed);
  const total = lesson.steps.length;
  const step = lesson.steps[index];
  const next = useMemo(() => nextLessonOf(lesson.id), [lesson.id]);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [index, finished]);

  const advance = () => {
    if (index + 1 >= total) {
      setFinished(true);
      saveLessonProgress(lesson.id, total - 1, true);
    } else {
      const ni = index + 1;
      setIndex(ni);
      saveLessonProgress(lesson.id, ni, false);
    }
  };

  const prevStep = () => {
    setIndex((i) => Math.max(0, i - 1));
  };

  const restart = () => {
    saveLessonProgress(lesson.id, 0, false);
    setIndex(0);
    setFinished(false);
  };

  const stepPct = finished ? 100 : Math.round((index / total) * 100);

  if (finished) {
    const nextHref = next ? `/learn/${next.track}/${next.slug}` : "/learn";
    const nextLabel = next ? `Next: ${next.title}` : "Back to all lessons";

    return (
      <div style={{ maxWidth: "760px", margin: "0 auto", padding: "40px 28px 0" }}>
        <div style={{ textAlign: "center", padding: "64px 0 0", animation: "aa-rise 0.4s ease-out both" }}>
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              width: "64px",
              height: "64px",
              borderRadius: "18px",
              background: "#12805c",
              color: "#fff",
              fontSize: "30px",
              boxShadow: "0 12px 30px -12px rgba(18,128,92,0.6)",
            }}
          >
            ✓
          </span>
          <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: "32px", letterSpacing: "-0.02em", margin: "22px 0 0", color: "#16151f" }}>
            Lesson complete
          </h1>
          <p style={{ margin: "10px 0 0", fontSize: "16px", color: "#56546a" }}>
            You finished <strong style={{ color: "#2a2836", fontWeight: 600 }}>{lesson.title}</strong>.
          </p>
          <div style={{ marginTop: "30px", display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "12px" }}>
            <Link
              to={nextHref}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "9px",
                padding: "12px 22px",
                borderRadius: "11px",
                background: "#574fd6",
                color: "#fff",
                fontSize: "15px",
                fontWeight: 600,
                boxShadow: "0 6px 20px rgba(87,79,214,0.28)",
              }}
            >
              {nextLabel} <span style={{ fontFamily: "'IBM Plex Mono', monospace" }}>→</span>
            </Link>
            <Link
              to="/learn"
              style={{
                padding: "12px 22px",
                borderRadius: "11px",
                background: "#fff",
                border: "1px solid #dcdbe8",
                color: "#2a2836",
                fontSize: "15px",
                fontWeight: 600,
              }}
            >
              All lessons
            </Link>
          </div>
          <button
            type="button"
            onClick={restart}
            style={{
              marginTop: "22px",
              background: "none",
              border: "none",
              cursor: "pointer",
              fontFamily: "inherit",
              fontSize: "13.5px",
              color: "#8b899d",
              textDecoration: "underline",
              textUnderlineOffset: "3px",
            }}
          >
            Review this lesson again
          </button>
        </div>
      </div>
    );
  }

  if (step.kind === "mcq") {
    return (
      <div style={{ maxWidth: "760px", margin: "0 auto", padding: "40px 28px 84px" }}>
        <McqStep
          step={step}
          onCorrect={advance}
          lessonTitle={lesson.title}
          lessonBackHref="/learn"
          stepNum={index + 1}
          stepTotal={total}
          stepPct={stepPct}
          onPrev={prevStep}
          canPrev={index > 0}
        />
      </div>
    );
  }

  return (
    <div style={{ maxWidth: "760px", margin: "0 auto", padding: "40px 28px 84px", fontFamily: "'IBM Plex Sans', system-ui, sans-serif", color: "#16151f" }}>
      <div style={{ marginBottom: "30px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
          <Link to="/learn" style={{ display: "inline-flex", alignItems: "center", gap: "7px", fontFamily: "'IBM Plex Mono', monospace", fontSize: "12px", color: "#8b899d" }}>
            <span>←</span> {lesson.title}
          </Link>
          <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "12px", color: "#b9b7c7" }}>
            Step {index + 1} / {total}
          </span>
        </div>
        <div style={{ height: "5px", borderRadius: "999px", background: "#edecf4", overflow: "hidden" }}>
          <div style={{ height: "100%", borderRadius: "999px", background: "#574fd6", transition: "width 0.4s", width: `${stepPct}%` }}></div>
        </div>
      </div>

      <div style={{ animation: "aa-rise 0.35s ease-out both" }}>
        {step.title && (
          <h1 style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: "27px", letterSpacing: "-0.02em", margin: "0 0 4px", color: "#16151f" }}>
            {step.title}
          </h1>
        )}

        <div style={{ marginTop: "18px" }}>
          {step.body()}
        </div>

        <div style={{ marginTop: "32px", display: "flex", alignItems: "center", gap: "14px" }}>
          <button
            type="button"
            onClick={advance}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "9px",
              padding: "13px 26px",
              borderRadius: "11px",
              background: "#574fd6",
              color: "#fff",
              fontFamily: "inherit",
              fontSize: "15px",
              fontWeight: 600,
              border: "none",
              cursor: "pointer",
              boxShadow: "0 6px 20px rgba(87,79,214,0.28)",
            }}
          >
            Continue <span style={{ fontFamily: "'IBM Plex Mono', monospace" }}>→</span>
          </button>
          {index > 0 && (
            <button
              type="button"
              onClick={prevStep}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                fontFamily: "inherit",
                fontSize: "14px",
                color: "#8b899d",
              }}
            >
              ← Previous
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
