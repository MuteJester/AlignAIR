"""Sources 2+5 (shared): OAS fresh-study heavy IGH positives + light IGK/IGL near-OOD negatives.

Pulls FRESH OAS studies (donor/study-disjoint from Gate-3), downloading each unit once (atomic cache +
sha256, rate-limited, size-capped, resumable). Heavy -> pos_igh_repertoire (positive_real); light IGKV/IGLV
-> igk/igl (igk_igl). Every real sequence is independently verified by **ANARCII** (chain H/K/L + locus +
completeness via the numbered variable-domain span); the ANARCII-numbered domain is the stored sequence.
Records are tagged repository/study/donor for the study-level split allocator, denylist-filtered, and
written with a counts-only census. No guard/annotator inference runs.

Run (background):  PYTHONPATH=src .venv/bin/python scripts/aa/applicability/hydrate_oas.py
"""
from __future__ import annotations

import csv
import io
import json
import re
import time
import urllib.request
import zlib
from pathlib import Path

from alignair.aa.guard.disjoint import build_gate3_global_denylist
from alignair.aa.guard.hydration import atomic_write_bytes, atomic_write_text, fetch_with_retry, filter_disjoint, sha256_bytes, sha256_text

STORE = Path(".private/aa-applicability/fresh_hydration")
UNITS = STORE / "oas_units"
ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability/fresh_hydration")
CLEANED = ".private/aa-gate3-hydration/cleaned_cohort.jsonl"
FP = "docs/superpowers/artifacts/2026-07-24-aa-gate3/split_fingerprint.json"
BASE = "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/"
UA = {"User-Agent": "Mozilla/5.0 (AlignAIR fresh-cohort recon)"}

TARGET = {"igk": 120, "igl": 120, "pos_igh_repertoire": 120}   # oversample vs floors (74/74/72)
MIN_STUDIES = 6            # donor diversity: draw from several fresh studies
MAX_UNITS = 80            # bound HUMAN-eligible downloads (only these consume quota)
MAX_UNIT_MB = 60          # skip giant units
PER_UNIT_CAP = 40        # candidate seqs extracted per unit (bound ANARCII work)
RATE_S = 0.4             # polite delay between downloads
HEAVY_MIN_AA = 100        # complete-read eligibility (catalog median) for heavy VH
LIGHT_MIN_AA = 95         # complete-read eligibility for light VL
AA = set("ACDEFGHIKLMNPQRSTVWYX")
CATALOG = STORE / "oas_species_probe.json"


def _load_catalog():
    """Cached species/completeness catalog from probe_oas_species.py — REQUIRED (never infer species)."""
    if not CATALOG.exists():
        raise SystemExit("catalog missing: run scripts/aa/applicability/probe_oas_species.py first")
    raw = CATALOG.read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def _eligible(entry, chain):
    """Per-chain eligibility from the catalog: the probe already classified explicit-human + complete-read
    as status 'eligible' (nested under entry[chain])."""
    return (entry.get(chain, {}) or {}).get("status") == "eligible"


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read()


def _norm(s):
    s = (s or "").lower()
    sn = re.split(r"[ _,]", s.strip())[0]
    yr = re.search(r"(?:19|20)\d\d", s)
    return sn + "_" + (yr.group() if yr else "")


def _used_studies():
    used = set()
    for line in open(CLEANED):
        e = json.loads(line)
        if e.get("category") in ("igk", "igl", "pos_igh_repertoire"):
            used.add(_norm(e["lineage_key"].split("::")[0]))
    return used


def _used_donors():
    """Gate-3 donors as (normalized_study, subject) for the untouched-donor audit."""
    donors = set()
    for line in open(CLEANED):
        e = json.loads(line)
        if e.get("category") in ("igk", "igl", "pos_igh_repertoire"):
            parts = e["lineage_key"].split("::")
            if len(parts) >= 2:
                donors.add((_norm(parts[0]), parts[1]))
    return donors


_PARTIAL = 1 << 20   # 1 MB partial read (metadata + thousands of rows; enough for PER_UNIT_CAP candidates)


def _decompress_multi(raw):
    """Decompress possibly-multi-member, possibly-TRUNCATED gzip (OAS metadata member + data member)."""
    out = b""
    while raw:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            out += d.decompress(raw)
        except Exception:
            break
        raw = d.unused_data
        if not raw:
            break
    return out


def _download_unit(study, unit):
    """Bounded PARTIAL download (no full-unit fetch); cached + checksummed + resumable + rate-limited."""
    cache = UNITS / f"{study}__{unit}.part"
    if cache.exists():
        return cache.read_bytes()
    url = f"{BASE}{study}/csv/{unit}"

    def _open():
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90)
        buf = b""
        while len(buf) < _PARTIAL:
            chunk = resp.read(_PARTIAL - len(buf))
            if not chunk:
                break
            buf += chunk
        resp.close()
        return buf

    gz = fetch_with_retry(_open, retries=4, sleep=time.sleep)
    atomic_write_bytes(cache, gz)
    atomic_write_text(UNITS / f"{study}__{unit}.part.sha256", sha256_bytes(gz))
    time.sleep(RATE_S)
    return gz


def _parse_unit(gz):
    txt = _decompress_multi(gz).decode("utf-8", "replace")
    lines = txt.split("\n")
    try:
        meta = json.loads(lines[0].strip().strip('"').replace('""', '"'))
    except Exception:
        meta = {}
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[1:]))))
    return meta, rows


def _candidates(meta, rows, chain):
    """Yield candidate (lineage_key, aa, category, donor, study) for productive complete rows."""
    study = meta.get("Author") or "unknown"
    subject = meta.get("Subject") or "unknown"
    out = []
    for r in rows:
        if r.get("productive") != "T" or r.get("stop_codon") == "T" or r.get("vj_in_frame") != "T":
            continue
        vcall = r.get("v_call") or ""
        if chain == "Heavy":
            cat, grp = ("pos_igh_repertoire", "positive_real") if vcall.startswith("IGHV") else (None, None)
        else:
            cat, grp = (("igk", "igk_igl") if vcall.startswith("IGKV")
                        else ("igl", "igk_igl") if vcall.startswith("IGLV") else (None, None))
        if cat is None:
            continue
        aa = (r.get("sequence_alignment_aa") or "").replace(".", "").replace("-", "").replace("*", "").upper()
        if not (80 <= len(aa) <= 170) or set(aa) - AA:
            continue
        lk = f"{study}::{subject}::{vcall}::{r.get('cdr3_aa') or ''}"
        out.append({"lineage_key": lk, "aa": aa, "category": cat, "group": grp,
                    "repository": "oas", "study": study, "donor": subject})
        if len(out) >= PER_UNIT_CAP:
            break
    return out


def _anarcii_verify(cands):
    """ANARCII chain/locus/completeness. Keep records whose numbered domain matches the expected chain and
    is complete; store the numbered domain as the sequence."""
    if not cands:
        return []
    from anarcii import Anarcii
    model = Anarcii(seq_type="antibody", mode="speed", batch_size=64, cpu=True, ncpu=1, verbose=False)
    named = [(c["lineage_key"], c["aa"]) for c in cands]
    res = model.number(named)
    if res is None:
        return []
    got = res if isinstance(res, dict) else dict(enumerate(res))
    expect = {"igk": "K", "igl": "L", "pos_igh_repertoire": "H"}
    by_lk = {}
    for c in cands:
        by_lk.setdefault(c["lineage_key"], c)
    out = []
    for name, rec in (got.items() if isinstance(got, dict) else enumerate(got)):
        if not isinstance(rec, dict) or rec.get("error"):
            continue
        c = by_lk.get(name)
        if not c:
            continue
        ct, qs, qe = rec.get("chain_type"), rec.get("query_start"), rec.get("query_end")
        if qs is None or qe is None or ct != expect[c["category"]]:
            continue
        dom = c["aa"][qs:qe + 1]
        min_len = 100 if c["category"] == "pos_igh_repertoire" else 95
        if len(dom) < min_len or set(dom) - AA:
            continue
        out.append({**c, "aa": dom, "id": f"{c['category']}_{sha256_text(c['lineage_key'])[:12]}",
                    "chain_type": ct, "verification": "anarcii_chain_locus_completeness (independent)"})
    return out


def main():
    STORE.mkdir(parents=True, exist_ok=True)
    UNITS.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    denylist = build_gate3_global_denylist(CLEANED, json.load(open(FP)))
    used = _used_studies()
    used_donors = _used_donors()
    catalog, catalog_sha = _load_catalog()
    per = catalog.get("per_study", {})

    listing = _get(BASE).decode("utf-8", "replace")
    studies = [h for h in re.findall(r'href="([A-Z][^"/]+)/"', listing)]
    fresh = [s for s in studies if _norm(s) not in used]
    # Only catalog-eligible (explicit human + complete-read median) studies enter the loop.
    eligible = [s for s in fresh if s in per and (_eligible(per[s], "heavy") or _eligible(per[s], "light"))]

    rej = {"non_human_unit": 0, "unknown_species_unit": 0, "partial_ineligible_chain": 0,
           "untouched_donor_audit_drop": 0, "anarcii_rejected": 0}
    raw, downloaded, raw_studies = [], 0, set()
    raw_have = {k: 0 for k in TARGET}
    for st in sorted(eligible):
        enough = all(raw_have[k] >= 2 * TARGET[k] for k in TARGET) and len(raw_studies) >= MIN_STUDIES
        if downloaded >= MAX_UNITS or enough:
            break
        e = per[st]
        try:
            units = re.findall(r'href="([^"]+\.csv\.gz)"', _get(f"{BASE}{st}/csv/").decode("utf-8", "replace"))
        except Exception:
            continue
        for unit in units:
            if downloaded >= MAX_UNITS:
                break
            chain = "Heavy" if "Heavy" in unit else "Light" if "Light" in unit else None
            if chain is None:
                continue
            clabel = "heavy" if chain == "Heavy" else "light"
            if not _eligible(e, clabel):          # complete-read eligibility BEFORE download; no quota spent
                rej["partial_ineligible_chain"] += 1
                continue
            gz = _download_unit(st, unit)
            if gz is None:
                continue
            meta, rows = _parse_unit(gz)
            sp = (meta.get("Species") or "").lower()
            if sp != "human":                     # explicit human only; unknown excluded; NO quota consumed
                rej["non_human_unit" if sp and sp != "?" else "unknown_species_unit"] += 1
                continue
            downloaded += 1                       # only confirmed-human eligible units consume the quota
            cands = _candidates(meta, rows, chain)
            # untouched study/donor audit BEFORE counting
            fresh_cands = [c for c in cands if (_norm(c["study"]), c["donor"]) not in used_donors]
            rej["untouched_donor_audit_drop"] += len(cands) - len(fresh_cands)
            if fresh_cands:
                raw_studies.add(_norm(st))
                for c in fresh_cands:
                    raw_have[c["category"]] = raw_have.get(c["category"], 0) + 1
                raw.extend(fresh_cands)

    # Phase B: single ANARCII pass over all raw candidates (one model instantiation).
    collected = _anarcii_verify(raw)
    rej["anarcii_rejected"] = len(raw) - len(collected)
    studies_hit = {_norm(r["study"]) for r in collected}

    # within-source lineage dedup + denylist
    seen, dd = set(), []
    for r in collected:
        if r["lineage_key"] in seen:
            continue
        seen.add(r["lineage_key"])
        dd.append(r)
    kept, dropped = filter_disjoint(dd, denylist=denylist, phase3=({}, {}))
    atomic_write_text(STORE / "oas.normalized.json", json.dumps(kept))

    def cnt(cat):
        return sum(1 for r in kept if r["category"] == cat)

    census = {
        "source": "oas_shared",
        "provenance": {"base": BASE, "draw": "catalog-eligible human studies only (explicit species)",
                       "verification": "ANARCII chain/locus/completeness on the numbered variable domain",
                       "catalog_sha256": catalog_sha, "human_units_downloaded": downloaded,
                       "unit_cache": str(UNITS)},
        "fresh_studies_available": len(fresh),
        "catalog_eligible_human_studies": len(eligible),
        "rejection_reasons": rej,
        "studies_contributing": sorted(studies_hit),
        "by_category": {c: cnt(c) for c in ("igk", "igl", "pos_igh_repertoire")},
        "kept_after_denylist": len(kept),
        "denylist_dropped": dropped,
        "distinct_donors": len({(r["study"], r["donor"]) for r in kept}),
    }
    atomic_write_text(ART / "oas.census.json", json.dumps(census, indent=2, sort_keys=True))
    print(json.dumps(census, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
