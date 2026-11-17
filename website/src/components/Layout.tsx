import { useEffect, useState, type ReactNode } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { Menu, Moon, Sun, X } from "lucide-react";
import { cn, getTheme, setTheme } from "../lib/util";
import SearchPalette from "./SearchPalette";

// The top bar's links (the docs' own sections are in the docs sidebar). The link check reads these targets.
const NAV = [
  { label: "Docs", to: "/docs" },
  { label: "Models", to: "/docs/choose-a-model" },
  { label: "CLI reference", to: "/docs/cli" },
  { label: "Python API", to: "/docs/python-api" },
];
const DEVELOPERS = { label: "Developers", to: "/docs/architecture" };
const PAPER = "https://doi.org/10.1093/nar/gkaf651";
const GITHUB = "https://github.com/MuteJester/AlignAIR";

function navClass({ isActive }: { isActive: boolean }) {
  return cn(
    "rounded-lg px-3 py-1.5 text-sm font-medium transition-colors",
    isActive
      ? "bg-brand-50 text-brand-700 dark:bg-brand-500/10 dark:text-brand-300"
      : "text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-white",
  );
}

function ThemeToggle() {
  const [theme, setThemeState] = useState<"light" | "dark">(() => getTheme());
  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    setThemeState(next);
  };
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      className="rounded-lg p-2 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-white"
    >
      {theme === "dark" ? <Sun className="h-4 w-4" aria-hidden="true" /> : <Moon className="h-4 w-4" aria-hidden="true" />}
    </button>
  );
}

function Navbar() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => setOpen(false), [loc.pathname]);

  return (
    <header className="sticky top-0 z-50 border-b border-slate-200 bg-[#fbfbfd]/85 backdrop-blur-md dark:border-slate-800 dark:bg-slate-950/85">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6">
        <Link to="/" className="font-display text-lg font-bold tracking-tight text-slate-900 dark:text-white">
          AlignAIR
        </Link>
        <nav aria-label="Primary" className="hidden items-center gap-1 md:flex">
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.to === "/docs"} className={navClass}>
              {item.label}
            </NavLink>
          ))}
          <NavLink to={DEVELOPERS.to} className={({ isActive }) => cn(navClass({ isActive }), "text-slate-400")}>
            {DEVELOPERS.label}
          </NavLink>
        </nav>
        <div className="flex items-center gap-1">
          <SearchPalette />
          <a href={GITHUB} target="_blank" rel="noreferrer" className="hidden rounded-lg px-3 py-1.5 text-sm font-medium text-slate-600 hover:text-slate-900 sm:inline dark:text-slate-400 dark:hover:text-white">
            GitHub
          </a>
          <a href={PAPER} target="_blank" rel="noreferrer" className="hidden rounded-lg px-3 py-1.5 text-sm font-medium text-slate-600 hover:text-slate-900 sm:inline dark:text-slate-400 dark:hover:text-white">
            Paper
          </a>
          <ThemeToggle />
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            aria-controls="mobile-nav"
            className="rounded-lg p-2 text-slate-700 md:hidden dark:text-slate-200"
          >
            {open ? <X className="h-5 w-5" aria-hidden="true" /> : <Menu className="h-5 w-5" aria-hidden="true" />}
          </button>
        </div>
      </div>
      {open && (
        <nav id="mobile-nav" aria-label="Primary" className="flex flex-col gap-1 border-t border-slate-200 px-4 py-3 md:hidden dark:border-slate-800">
          {[...NAV, DEVELOPERS].map((item) => (
            <NavLink key={item.to} to={item.to} end={item.to === "/docs"} className={navClass}>
              {item.label}
            </NavLink>
          ))}
          <a href={GITHUB} target="_blank" rel="noreferrer" className="rounded-lg px-3 py-1.5 text-sm font-medium text-slate-600 dark:text-slate-400">GitHub</a>
          <a href={PAPER} target="_blank" rel="noreferrer" className="rounded-lg px-3 py-1.5 text-sm font-medium text-slate-600 dark:text-slate-400">Paper</a>
        </nav>
      )}
    </header>
  );
}

function Footer() {
  const link = "text-sm font-medium text-slate-600 transition-colors hover:text-brand-700 dark:text-slate-400 dark:hover:text-brand-300";
  return (
    <footer className="mt-24 border-t border-slate-200 bg-[#fbfbfd] px-4 py-11 sm:px-6 dark:border-slate-800 dark:bg-slate-950">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <div className="flex flex-wrap items-start justify-between gap-6">
          <div className="max-w-sm">
            <p className="font-display text-lg font-bold text-slate-900 dark:text-white">AlignAIR</p>
            <p className="mt-3 text-sm leading-relaxed text-slate-500 dark:text-slate-400">
              V(D)J allele assignment for IG and TCR repertoires, with pretrained human models for every locus. Open
              source, GPL-3.0.
            </p>
          </div>
          <div className="flex gap-12">
            <div className="flex flex-col gap-2.5">
              <Link to="/docs" className={link}>Docs</Link>
              <Link to="/docs/choose-a-model" className={link}>Models</Link>
              <Link to="/docs/cli" className={link}>CLI reference</Link>
            </div>
            <div className="flex flex-col gap-2.5">
              <a href={GITHUB} target="_blank" rel="noreferrer" className={link}>GitHub</a>
              <a href="https://huggingface.co/AlignAIR" target="_blank" rel="noreferrer" className={link}>Hugging Face</a>
              <Link to="/docs/cite" className={link}>Cite</Link>
            </div>
          </div>
        </div>
        <div className="flex flex-wrap justify-between gap-3 border-t border-dashed border-slate-200 pt-6 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
          <p className="max-w-3xl">
            Cite: Konstantinovsky et al., Enhancing sequence alignment of adaptive immune receptors through multi-task
            deep learning, Nucleic Acids Research 2025, gkaf651.
          </p>
          <p>&copy; {new Date().getFullYear()} AlignAIR</p>
        </div>
      </div>
    </footer>
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  const skipToMain = () => {
    document.getElementById("main-content")?.focus({ preventScroll: false });
  };

  return (
    <div className="flex min-h-screen flex-col bg-[#fbfbfd] text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      {/* HashRouter owns the URL fragment, so an href="#main-content" would destroy the current
          route. Move focus directly instead and leave the route untouched. */}
      <button type="button" onClick={skipToMain} className="skip-link">Skip to main content</button>
      <Navbar />
      <main id="main-content" tabIndex={-1} className="flex-1">{children}</main>
      <Footer />
    </div>
  );
}
