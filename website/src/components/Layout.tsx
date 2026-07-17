import { useEffect, useState, type ReactNode } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { Menu, X } from "lucide-react";
import { cn } from "../lib/util";

function Navbar() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => setOpen(false), [loc.pathname]);

  return (
    <header
      style={{
        position: "sticky",
        top: 0,
        zIndex: 50,
        borderBottom: "1px solid #eae9f1",
        background: "rgba(251,251,253,0.82)",
        backdropFilter: "saturate(1.4) blur(12px)",
        WebkitBackdropFilter: "saturate(1.4) blur(12px)",
      }}
    >
      <div style={{ maxWidth: "1200px", margin: "0 auto", height: "64px", padding: "0 28px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <Link to="/" style={{ display: "flex", alignItems: "center", color: "#16151f" }}>
          <span style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: "18px", letterSpacing: "-0.01em" }}>
            AlignAIR
          </span>
        </Link>

        {/* Desktop Nav */}
        <nav aria-label="Primary" style={{ display: "flex", alignItems: "center", gap: "6px" }} className="hidden md:flex">
          <NavLink
            to="/learn"
            className={({ isActive }) =>
              cn(
                "transition-colors",
                isActive ? "text-indigo-700 bg-[#f2f1fb]" : "text-[#56546a] hover:text-indigo-700 hover:bg-indigo-50/30"
              )
            }
            style={{ padding: "7px 13px", borderRadius: "8px", fontSize: "14px", fontWeight: 500 }}
          >
            Learn
          </NavLink>
          <NavLink
            to="/docs"
            className={({ isActive }) =>
              cn(
                "transition-colors",
                isActive ? "text-indigo-700 bg-[#f2f1fb]" : "text-[#56546a] hover:text-indigo-700 hover:bg-indigo-50/30"
              )
            }
            style={{ padding: "7px 13px", borderRadius: "8px", fontSize: "14px", fontWeight: 500 }}
          >
            Docs
          </NavLink>
          <a
            href="https://github.com/MuteJester/AlignAIR"
            target="_blank"
            rel="noreferrer"
            style={{ padding: "7px 13px", borderRadius: "8px", fontSize: "14px", fontWeight: 500, color: "#56546a" }}
            className="hover:text-indigo-700 hover:bg-indigo-50/30 transition-colors"
          >
            GitHub
          </a>
          <span style={{ width: "1px", height: "22px", background: "#eae9f1", margin: "0 6px" }}></span>
          <Link
            to="/docs/getting-started"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "7px",
              padding: "8px 16px",
              borderRadius: "9px",
              background: "#574fd6",
              color: "#fff",
              fontSize: "14px",
              fontWeight: 600,
              boxShadow: "0 2px 10px rgba(87,79,214,0.28)",
              transition: "opacity 0.2s",
            }}
            className="hover:opacity-90"
          >
            Get started
          </Link>
        </nav>

        {/* Mobile menu trigger */}
        <div className="flex items-center gap-1 md:hidden">
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            aria-controls="mobile-nav"
            style={{ border: "none", background: "none", cursor: "pointer", padding: "6px", color: "#16151f" }}
          >
            {open ? <X className="h-6 w-6" aria-hidden="true" /> : <Menu className="h-6 w-6" aria-hidden="true" />}
          </button>
        </div>
      </div>

      {/* Mobile nav dropdown */}
      {open && (
        <nav
          id="mobile-nav"
          aria-label="Primary"
          style={{
            borderTop: "1px solid #eae9f1",
            background: "#fbfbfd",
            padding: "12px 28px",
            display: "flex",
            flexDirection: "column",
            gap: "8px",
          }}
          className="md:hidden"
        >
          <NavLink
            to="/learn"
            className={({ isActive }) =>
              cn(
                "block rounded-lg px-3 py-2 text-sm font-medium",
                isActive ? "bg-[#f2f1fb] text-indigo-700" : "text-[#56546a]"
              )
            }
          >
            Learn
          </NavLink>
          <NavLink
            to="/docs"
            className={({ isActive }) =>
              cn(
                "block rounded-lg px-3 py-2 text-sm font-medium",
                isActive ? "bg-[#f2f1fb] text-indigo-700" : "text-[#56546a]"
              )
            }
          >
            Docs
          </NavLink>
          <a
            href="https://github.com/MuteJester/AlignAIR"
            target="_blank"
            rel="noreferrer"
            className="block rounded-lg px-3 py-2 text-sm font-medium text-[#56546a]"
          >
            GitHub
          </a>
          <Link
            to="/docs/getting-started"
            className="block text-center rounded-lg py-2 text-sm font-medium bg-[#574fd6] text-white"
            style={{ marginTop: "4px" }}
          >
            Get started
          </Link>
        </nav>
      )}
    </header>
  );
}

function Footer() {
  return (
    <footer style={{ marginTop: "92px", borderTop: "1px solid #eae9f1", background: "#fbfbfd", padding: "44px 28px" }}>
      <div style={{ maxWidth: "1200px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "24px" }}>
        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "flex-start", gap: "24px" }}>
          <div style={{ maxWidth: "340px" }}>
            <div style={{ display: "flex", alignItems: "center", color: "#16151f" }}>
              <span style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: "18px", letterSpacing: "-0.01em" }}>
                AlignAIR
              </span>
            </div>
            <p style={{ marginTop: "14px", fontSize: "14px", lineHeight: 1.6, color: "#6f6d85" }}>
              An end-to-end neural aligner for immunoglobulin and T-cell-receptor repertoires. Open source, GPL-3.0.
            </p>
          </div>
          <div style={{ display: "flex", gap: "48px" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              <Link to="/learn" style={{ fontSize: "14px", fontWeight: 500, color: "#56546a" }} className="hover:text-indigo-700 transition-colors">
                Lessons
              </Link>
              <Link to="/docs" style={{ fontSize: "14px", fontWeight: 500, color: "#56546a" }} className="hover:text-indigo-700 transition-colors">
                Reference
              </Link>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              <a href="https://github.com/MuteJester/AlignAIR" target="_blank" rel="noreferrer" style={{ fontSize: "14px", fontWeight: 500, color: "#56546a" }} className="hover:text-indigo-700 transition-colors">
                GitHub
              </a>
              <a href="https://huggingface.co/AlignAIR/AlignAIR-pretrained" target="_blank" rel="noreferrer" style={{ fontSize: "14px", fontWeight: 500, color: "#56546a" }} className="hover:text-indigo-700 transition-colors">
                Models
              </a>
              <Link to="/docs/citation-support" style={{ fontSize: "14px", fontWeight: 500, color: "#56546a" }} className="hover:text-indigo-700 transition-colors">
                Cite &amp; support
              </Link>
            </div>
          </div>
        </div>
        <div style={{ borderTop: "1px dashed #eae9f1", paddingTop: "24px", display: "flex", flexWrap: "wrap", justifyContent: "space-between", gap: "12px" }}>
          <p style={{ fontSize: "12.5px", lineHeight: 1.6, color: "#6f6d85", margin: 0, maxWidth: "48em" }}>
            Cite: Konstantinovsky et al., Enhancing sequence alignment of adaptive immune receptors through multi-task deep learning, Nucleic Acids Research 2025, gkaf651.
          </p>
          <p style={{ fontSize: "12.5px", color: "#6f6d85", margin: 0 }}>
            &copy; {new Date().getFullYear()} AlignAIR
          </p>
        </div>
      </div>
    </footer>
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh", background: "#fbfbfd" }}>
      <a href="#main-content" className="skip-link">Skip to main content</a>
      <Navbar />
      <main id="main-content" style={{ flex: 1 }}>{children}</main>
      <Footer />
    </div>
  );
}
