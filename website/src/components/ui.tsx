import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { Check, Copy, Info, AlertTriangle, Lightbulb } from "lucide-react";
import { cn } from "../lib/util";

/* -------------------------------------------------------------------------- */
/* Button                                                                     */
/* -------------------------------------------------------------------------- */
type ButtonProps = {
  children: ReactNode;
  to?: string;
  href?: string;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
  className?: string;
  disabled?: boolean;
};

export function Button({
  children,
  to,
  href,
  onClick,
  variant = "primary",
  size = "md",
  className,
  disabled,
}: ButtonProps) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-xl font-semibold transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500/60 disabled:opacity-50 disabled:cursor-not-allowed";
  const variants = {
    primary:
      "bg-brand-600 text-white shadow-lg shadow-brand-600/20 hover:bg-brand-500 hover:shadow-brand-500/30",
    secondary:
      "bg-slate-100 text-slate-800 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-700",
    ghost:
      "text-slate-600 hover:text-brand-600 dark:text-slate-300 dark:hover:text-brand-400 hover:bg-slate-100 dark:hover:bg-slate-800",
  };
  const sizes = { sm: "px-3 py-1.5 text-sm", md: "px-5 py-2.5 text-sm", lg: "px-7 py-3.5 text-base" };
  const cls = cn(base, variants[variant], sizes[size], className);

  if (to) return <Link to={to} className={cls}>{children}</Link>;
  if (href)
    return (
      <a href={href} target="_blank" rel="noreferrer" className={cls}>
        {children}
      </a>
    );
  return (
    <button type="button" onClick={onClick} disabled={disabled} className={cls}>
      {children}
    </button>
  );
}

/* -------------------------------------------------------------------------- */
/* Card                                                                       */
/* -------------------------------------------------------------------------- */
export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900",
        className,
      )}
    >
      {children}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Badge                                                                      */
/* -------------------------------------------------------------------------- */
export function Badge({
  children,
  color = "brand",
}: {
  children: ReactNode;
  color?: "brand" | "green" | "amber" | "slate";
}) {
  const colors = {
    brand: "bg-brand-100 text-brand-700 dark:bg-brand-500/15 dark:text-brand-300",
    green: "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300",
    amber: "bg-amber-100 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300",
    slate: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300",
  };
  return (
    <span className={cn("inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold", colors[color])}>
      {children}
    </span>
  );
}

/* -------------------------------------------------------------------------- */
/* Callout                                                                    */
/* -------------------------------------------------------------------------- */
export function Callout({
  children,
  kind = "note",
  title,
}: {
  children: ReactNode;
  kind?: "note" | "warning" | "tip";
  title?: string;
}) {
  const styles = {
    note: { wrap: "border-brand-200 bg-brand-50 dark:border-brand-500/30 dark:bg-brand-500/10", icon: Info, ic: "text-brand-600 dark:text-brand-400" },
    warning: { wrap: "border-amber-200 bg-amber-50 dark:border-amber-500/30 dark:bg-amber-500/10", icon: AlertTriangle, ic: "text-amber-600 dark:text-amber-400" },
    tip: { wrap: "border-emerald-200 bg-emerald-50 dark:border-emerald-500/30 dark:bg-emerald-500/10", icon: Lightbulb, ic: "text-emerald-600 dark:text-emerald-400" },
  };
  const s = styles[kind];
  const Icon = s.icon;
  return (
    <div className={cn("my-5 flex gap-3 rounded-xl border p-4", s.wrap)}>
      <Icon aria-hidden="true" className={cn("mt-0.5 h-5 w-5 shrink-0", s.ic)} />
      <div className="text-sm text-slate-700 dark:text-slate-300">
        {title && <p className="mb-1 font-semibold text-slate-900 dark:text-white">{title}</p>}
        {children}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* CodeBlock (with copy). Lightweight - no external highlighter.              */
/* -------------------------------------------------------------------------- */
export function CodeBlock({
  code,
  lang = "bash",
  title,
}: {
  code: string;
  lang?: string;
  title?: string;
}) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <div className="my-5 overflow-hidden rounded-xl border border-slate-800 bg-slate-900 text-slate-100 shadow-md">
      <div className="flex items-center justify-between border-b border-slate-700/60 bg-slate-800/50 px-4 py-2">
        <span className="font-mono text-xs uppercase tracking-wide text-slate-300">{title ?? lang}</span>
        <button
          type="button"
          onClick={copy}
          aria-label={copied ? "Copied to clipboard" : "Copy code"}
          className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-slate-300 transition-colors hover:bg-slate-700/60 hover:text-white"
        >
          {copied ? <Check className="h-3.5 w-3.5" aria-hidden="true" /> : <Copy className="h-3.5 w-3.5" aria-hidden="true" />}
          <span aria-hidden="true">{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>
      <pre
        tabIndex={0}
        aria-label={`${title ?? lang} code example`}
        className="aa-scrollbar overflow-x-auto p-4 text-sm leading-relaxed"
      >
        <code className="font-mono">{code}</code>
      </pre>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* ProgressBar                                                                */
/* -------------------------------------------------------------------------- */
export function ProgressBar({ value, className }: { value: number; className?: string }) {
  return (
    <div className={cn("h-2 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800", className)}>
      <div
        className="h-full rounded-full bg-gradient-to-r from-brand-500 to-brand-400 transition-all duration-500"
        style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
      />
    </div>
  );
}
