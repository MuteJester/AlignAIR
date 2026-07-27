"""Archive license-terms provenance for the AA-guard profile-source clearance (Stage-2 Task 2a).

For each authoritative license page: fetch the raw HTML, store it PRIVATELY (git-ignored) with its response
headers, compute a content SHA-256, record the retrieval timestamp + URL, extract the relevant license
clause, and resolve a public archival (Wayback) URL. The COMMITTED artifact is counts/hashes/provenance
only — never the page body.

Records the July 1 2026 IMGT relicensing (CC BY-NC-ND -> CC BY-4.0) and the explicit usage distinction:
AlignAIR downloads germline DATA and builds profiles locally with pyhmmer (vendored HMMER, not an IMGT
tool); it does NOT redistribute IMGT-hosted files nor invoke IMGT web tools/services.

Run:  PYTHONPATH=src .venv/bin/python scripts/aa/applicability/capture_license_evidence.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

PRIV = Path(".private/aa-applicability/license_evidence")
ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability/license_evidence.json")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}

SOURCES = {
    "imgt_terms": "https://www.imgt.org/about/termsofuse.php",
    "imgt_home": "https://imgt.org/home.php",
    "imgt_downloads": "https://www.imgt.org/about/downloads.php",
    "ogrdb_licensing": "https://ogrdb.airr-community.org/render_page/licensing_statement.html",
    "interpro_licensing": "https://www.ebi.ac.uk/interpro/about/license/",
}
# license keyword(s) to locate the relevant clause in each page's text
CLAUSE_PATTERNS = {
    "imgt_terms": r"[^.]*CC BY[- ]?4\.0[^.]*\.",
    "imgt_home": r"[^.]*(?:July 1,? 2026|CC BY[- ]?4\.0|terms of use)[^.]*\.",
    "imgt_downloads": r"[^.]*(?:terms of use|CC BY[- ]?4\.0|licen[cs]e)[^.]*\.",
    "ogrdb_licensing": r"[^.]*CC0[^.]*\.",
    "interpro_licensing": r"[^.]*CC0[^.]*\.",
}
KEY_HEADERS = ("Date", "Last-Modified", "Content-Type", "Content-Length", "Server", "ETag")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _fetch(url):
    req = urllib.request.Request(url, headers=UA)
    resp = urllib.request.urlopen(req, timeout=60)
    body = resp.read()
    headers = {k: v for k, v in resp.getheaders()}
    return body, headers, resp.status


def _text(body):
    t = body.decode("utf-8", "replace")
    t = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _clause(key, text):
    m = re.search(CLAUSE_PATTERNS[key], text, re.IGNORECASE)
    return m.group(0).strip() if m else None


def _wayback(url):
    """Existing closest snapshot (availability API), then a save attempt; return the public archive URL."""
    out = {"available": None, "saved": None}
    try:
        api = "http://archive.org/wayback/available?url=" + urllib.parse.quote(url, safe="")
        j = json.loads(urllib.request.urlopen(urllib.request.Request(api, headers=UA), timeout=45).read())
        snap = (j.get("archived_snapshots") or {}).get("closest") or {}
        out["available"] = snap.get("url")
    except Exception as e:  # noqa: BLE001
        out["available"] = f"lookup_failed:{type(e).__name__}"
    try:
        save = urllib.request.urlopen(urllib.request.Request("https://web.archive.org/save/" + url, headers=UA), timeout=90)
        cl = save.geturl()
        out["saved"] = cl if "web.archive.org" in cl else f"https://web.archive.org/web/*/{url}"
    except Exception as e:  # noqa: BLE001
        out["saved"] = f"save_failed:{type(e).__name__}"
    return out


def main():
    PRIV.mkdir(parents=True, exist_ok=True)
    ART.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    for key, url in SOURCES.items():
        ts = _now()
        try:
            body, headers, status = _fetch(url)
        except Exception as e:  # noqa: BLE001
            records[key] = {"url": url, "retrieved_utc": ts, "error": f"{type(e).__name__}: {str(e)[:120]}"}
            continue
        sha = hashlib.sha256(body).hexdigest()
        (PRIV / f"{key}.html").write_bytes(body)
        (PRIV / f"{key}.headers.json").write_text(json.dumps(headers, indent=2, sort_keys=True))
        text = _text(body)
        records[key] = {
            "url": url, "retrieved_utc": ts, "http_status": status, "bytes": len(body),
            "content_sha256": sha, "private_html": str(PRIV / f"{key}.html"),
            "response_headers": {h: headers.get(h) for h in KEY_HEADERS if headers.get(h)},
            "relevant_clause": _clause(key, text),
            "public_archive": _wayback(url),
        }
        print(f"{key:20s} sha256={sha[:16]} status={status} clause={'yes' if records[key]['relevant_clause'] else 'JS-rendered/none'}")

    evidence = {
        "artifact": "license_evidence",
        "purpose": "provenance archive for the AA-guard profile-source clearance (Stage-2 Task 2a)",
        "captured_utc": _now(),
        "sources": records,
        "imgt_relicensing": {
            "from": "CC BY-NC-ND 4.0 (historically cited, e.g. 2022 IMGT NAR paper)",
            "to": "CC BY-4.0",
            "change_date": "2026-07-01",
            "change_date_source": "IMGT homepage (imgt.org/home.php) dates the terms change to 2026-07-01",
            "clause": "The IMGT data are provided to both public and private users under the CC BY-4.0 license.",
        },
        "usage_distinction": (
            "AlignAIR DOWNLOADS germline DATA (sequences) and builds profile HMMs LOCALLY with pyhmmer "
            "(a vendored HMMER library, not an IMGT tool). It does NOT redistribute IMGT-hosted files and "
            "does NOT invoke IMGT web tools/services. IMGT tool-use terms (separate from the CC BY-4.0 data "
            "terms) therefore do not apply."),
        "attribution_requirements": {
            "imgt": "Data derived from IMGT/GENE-DB (imgt.org), CC BY-4.0.",
            "modification_notice": "Modified: profile HMMs were built from the germline sequences.",
            "placement": ["shipped bundle NOTICE", "model card", "profile manifest"],
            "no_endorsement": "Attribution must not imply IMGT endorsement; the IMGT name must not be used "
                              "for publicity/business promotion without written approval (trademark).",
        },
        "cc0_sources": {
            "ogrdb": {"license": "CC0-1.0", "url": SOURCES["ogrdb_licensing"], "attribution": "none required"},
            "pfam_interpro": {"license": "CC0", "url": SOURCES["interpro_licensing"], "attribution": "none required"},
        },
        "note": "Technical provenance review, not formal legal advice. Raw HTML + full headers archived "
                "privately (git-ignored); this committed artifact is counts/hashes/provenance only.",
    }
    ART.write_text(json.dumps(evidence, indent=2, sort_keys=True))
    print("\nwrote", ART)


if __name__ == "__main__":
    main()
