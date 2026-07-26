"""Source 1 hydration: SAbDab VHH (camelid single-domain) + murine VH.

Pipeline per structure: RCSB FASTA (retry/backoff, per-PDB cached for resumability) -> heavy chain ->
**ANARCII** independent VH-domain extraction + heavy-chain verification (a tool separate from the
production guard/profile bank) -> record. Records are denylist-filtered against the verified pre-split
Gate-3 corpus and written with a counts-only census. No guard/annotator model is run.

Run:  PYTHONPATH=src .venv/bin/python scripts/aa/applicability/hydrate_sabdab.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import urllib.request
from pathlib import Path

from alignair.aa.guard.disjoint import build_gate3_global_denylist
from alignair.aa.guard.hydration import (
    atomic_write_bytes, atomic_write_text, fetch_with_retry, filter_disjoint, sha256_bytes, sha256_text,
)

STORE = Path(".private/aa-applicability/fresh_hydration")
FASTA_CACHE = STORE / "fasta_cache"
ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability/fresh_hydration")
CLEANED = ".private/aa-gate3-hydration/cleaned_cohort.jsonl"
FP = "docs/superpowers/artifacts/2026-07-24-aa-gate3/split_fingerprint.json"
GATE3_VHH = ".private/aa-gate3-hydration/sabdab_emitted.json"

SINGLE = "https://sabdab.opig.stats.ox.ac.uk/api/download/all-single-domain-summary"
ALLSUM = "https://sabdab.opig.stats.ox.ac.uk/api/download/all-summary"
UA = {"User-Agent": "Mozilla/5.0 (AlignAIR fresh-cohort recon)"}

TARGET = {"lama": 45, "nonlama": 45, "murine": 45}   # >= floor 36/subset with margin for rejects/dedup
CAM = ("lama", "vicugna", "camelus", "glama")


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read()


def _summary(url, name):
    raw = fetch_with_retry(lambda: _get(url), retries=4, sleep=__import__("time").sleep)
    atomic_write_bytes(STORE / f"{name}.csv", raw)
    atomic_write_text(STORE / f"{name}.csv.sha256", sha256_bytes(raw))
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"))))


def _pdb4(ext):
    return ext.replace("pdb_0000", "").replace("pdb_000", "").upper()


def _fasta(pdb_ext):
    code = _pdb4(pdb_ext)
    cache = FASTA_CACHE / f"{code}.fasta"
    if cache.exists():
        return cache.read_text()
    txt = fetch_with_retry(
        lambda: _get(f"https://www.rcsb.org/fasta/entry/{code}").decode(),
        retries=4, sleep=__import__("time").sleep)
    atomic_write_text(cache, txt)
    return txt


def _heavy_seq(fasta, hchain):
    seqs, hdr = {}, None
    for line in fasta.splitlines():
        if line.startswith(">"):
            hdr = line
            seqs[hdr] = ""
        elif hdr:
            seqs[hdr] += line.strip()
    for h, s in seqs.items():
        parts = h.split("|")
        if len(parts) >= 2:
            chains = parts[1].replace("Chains ", "").replace("Chain ", "")
            if hchain in {c.strip() for c in chains.split(",")}:
                return s
    return None


def _select(rows, single=True):
    """Return fresh (not-used) distinct-HEAVY_ID camelid VHH split lama/nonlama, or murine VH."""
    emi = json.load(open(GATE3_VHH))
    used = {r.get("sabdab_id") for r in emi.get("vhh", [])} | {r.get("sabdab_id") for r in emi.get("murine_vh", [])}
    used |= {(r.get("pdb") or "").lower() for r in emi.get("vhh", [])} | {(r.get("pdb") or "").lower() for r in emi.get("murine_vh", [])}

    def genus(r):
        return (r.get("heavy_species") or "").lower()

    def usedrow(r):
        return r.get("SABDAB_ID") in used or (r.get("PDB") or "").lower() in used

    used_heavy = {r.get("HEAVY_ID") for r in rows if r.get("SABDAB_ID") in used}
    seen, out = set(), []
    for r in rows:
        if single:
            if (r.get("type") or "").strip().lower() != "sd-h" or not any(t in genus(r) for t in CAM):
                continue
        else:
            if "mus musculus" not in genus(r) or (r.get("type") or "").strip().lower() == "sd-h":
                continue
        if usedrow(r) or r.get("HEAVY_ID") in used_heavy:
            continue
        h = r.get("HEAVY_ID")
        if not h or h in seen or not r.get("Hchain"):
            continue
        seen.add(h)
        out.append(r)
    return out


def _anarcii_extract(named_seqs):
    """ANARCII independent VH extraction: return {name: (vh_seq, chain_type)} for numbered heavy domains."""
    from anarcii import Anarcii
    model = Anarcii(seq_type="antibody", mode="speed", batch_size=32, cpu=True, ncpu=1, verbose=False)
    res = model.number(named_seqs)
    out = {}
    items = res.items() if isinstance(res, dict) else enumerate(res)
    for name, rec in items:
        if not isinstance(rec, dict) or rec.get("error"):
            continue
        ct = rec.get("chain_type")
        qs, qe = rec.get("query_start"), rec.get("query_end")
        if ct != "H" or qs is None or qe is None:
            continue
        seq = dict(named_seqs)[name]
        vh = seq[qs:qe + 1]
        if 90 <= len(vh) <= 160 and set(vh) <= set("ACDEFGHIKLMNPQRSTVWYX"):
            out[name] = (vh, ct)
    return out


def _collect(rows, category, group, target, single):
    cands = _select(rows, single=single)
    if single and category == "vhh":
        lama = [r for r in cands if "lama glama" in (r.get("heavy_species") or "").lower() or "glama" in (r.get("heavy_species") or "").lower()]
        nonl = [r for r in cands if r not in lama]
        cands = lama[:TARGET["lama"] * 2] + nonl[:TARGET["nonlama"] * 2]   # margin for ANARCII rejects
    else:
        cands = cands[:target * 2]
    # fetch heavy chains
    named = []
    meta = {}
    for r in cands:
        try:
            hs = _heavy_seq(_fasta(r["PDB"]), r["Hchain"])
        except Exception:
            hs = None
        if hs:
            nm = r["SABDAB_ID"] or r["PDB"]
            named.append((nm, hs))
            meta[nm] = r
    extracted = _anarcii_extract(named)
    recs = []
    for nm, (vh, ct) in extracted.items():
        r = meta[nm]
        recs.append({
            "id": f"{category}_{nm}",
            "aa": vh,
            "group": group,
            "category": category,
            "lineage_key": r.get("HEAVY_ID") or nm,
            "heavy_species": (r.get("heavy_species") or "").lower(),
            "deposition_date": r.get("date") or "",
            "source": "sabdab",
            "extraction": "anarcii_vh",
        })
    return recs


def main():
    STORE.mkdir(parents=True, exist_ok=True)
    FASTA_CACHE.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    denylist = build_gate3_global_denylist(CLEANED, json.load(open(FP)))   # raises on digest mismatch

    single_rows = _summary(SINGLE, "sabdab.summary_single")
    all_rows = _summary(ALLSUM, "sabdab.summary_all")

    vhh = _collect(single_rows, "vhh", "vhh_nonhuman", TARGET["lama"] + TARGET["nonlama"], single=True)
    murine = _collect(all_rows, "murine_vh", "vhh_nonhuman", TARGET["murine"], single=False)

    records = vhh + murine
    # within-source sequence dedup + denylist filter
    seen, dd = set(), []
    for r in records:
        h = sha256_text(r["aa"])
        if h in seen:
            continue
        seen.add(h)
        dd.append(r)
    kept, dropped = filter_disjoint(dd, denylist=denylist, phase3=({}, {}))

    atomic_write_text(STORE / "sabdab.normalized.json", json.dumps(kept))

    def genus_counts(cat):
        from collections import Counter
        return dict(Counter(r["heavy_species"] for r in kept if r["category"] == cat))

    census = {
        "source": "sabdab",
        "provenance": {"single_domain": SINGLE, "all_summary": ALLSUM,
                       "single_sha256": (STORE / "sabdab.summary_single.csv.sha256").read_text(),
                       "all_sha256": (STORE / "sabdab.summary_all.csv.sha256").read_text(),
                       "vh_extraction": "ANARCII (independent of guard/profile bank); chain_type==H, query_start:query_end"},
        "fetched_after_extraction": len(records),
        "after_within_source_dedup": len(dd),
        "kept_after_denylist": len(kept),
        "denylist_dropped": dropped,
        "by_category": {c: sum(1 for r in kept if r["category"] == c) for c in {r["category"] for r in kept}},
        "vhh_genus": genus_counts("vhh"),
        "vhh_lama": sum(1 for r in kept if r["category"] == "vhh" and ("lama glama" in r["heavy_species"] or "glama" in r["heavy_species"])),
        "vhh_nonlama": sum(1 for r in kept if r["category"] == "vhh" and not ("lama glama" in r["heavy_species"] or "glama" in r["heavy_species"])),
    }
    atomic_write_text(ART / "sabdab.census.json", json.dumps(census, indent=2, sort_keys=True))
    print(json.dumps(census, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
