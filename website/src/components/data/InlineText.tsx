import { inlineParts } from "../../lib/inline";

/** Help or docstring text with its backtick spans shown as code (``inlineParts``). */
export default function InlineText({ text }: { text: string }) {
  return (
    <>
      {inlineParts(text).map((part, i) =>
        part.code ? (
          <code key={i} className="rounded bg-slate-100 px-1 font-mono text-[0.9em] [overflow-wrap:anywhere] dark:bg-slate-800">
            {part.text}
          </code>
        ) : (
          <span key={i}>{part.text}</span>
        ),
      )}
    </>
  );
}
