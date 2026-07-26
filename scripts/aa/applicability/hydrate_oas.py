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
import gzip
import io
import json
import re
import time
import urllib.request
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
MAX_UNITS = 60            # bound total downloads
MAX_UNIT_MB = 60          # skip giant units
PER_UNIT_CAP = 40        # candidate seqs extracted per unit (bound ANARCII work)
RATE_S = 0.4             # polite delay between downloads
AA = set("ACDEFGHIKLMNPQRSTVWYX")


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


def _download_unit(study, unit):
    cache = UNITS / f"{study}__{unit}"
    if cache.exists():
        return cache.read_bytes()
    url = f"{BASE}{study}/csv/{unit}"
    try:
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=30)
        size = int(head.headers.get("content-length") or 0)
        if size > MAX_UNIT_MB * 1_000_000:
            return None
    except Exception:
        pass
    gz = fetch_with_retry(lambda: _get(url), retries=4, sleep=time.sleep)
    atomic_write_bytes(cache, gz)
    atomic_write_text(UNITS / f"{study}__{unit}.sha256", sha256_bytes(gz))
    time.sleep(RATE_S)
    return gz


def _parse_unit(gz):
    txt = gzip.decompress(gz).decode("utf-8", "replace")
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
    from anarcii import Anarcii
    model = Anarcii(seq_type="antibody", mode="speed", batch_size=64, cpu=True, ncpu=1, verbose=False)
    named = [(c["lineage_key"], c["aa"]) for c in cands]
    res = model.number(named)
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

    listing = _get(BASE).decode("utf-8", "replace")
    studies = [h for h in re.findall(r'href="([A-Z][^"/]+)/"', listing)]
    fresh = [s for s in studies if _norm(s) not in used]

    collected, downloaded, studies_hit = [], 0, set()
    have = {k: 0 for k in TARGET}
    for st in fresh:
        if downloaded >= MAX_UNITS or (all(have[k] >= TARGET[k] for k in TARGET) and len(studies_hit) >= MIN_STUDIES):
            break
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
            gz = _download_unit(st, unit)
            if gz is None:
                continue
            downloaded += 1
            meta, rows = _parse_unit(gz)
            if (meta.get("Species") or "").lower() != "human":
                continue
            verified = _anarcii_verify(_candidates(meta, rows, chain))
            if verified:
                studies_hit.add(_norm(st))
                for v in verified:
                    have[v["category"]] = have.get(v["category"], 0) + 1
                collected.extend(verified)

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
        "provenance": {"base": BASE, "draw": "fresh studies (Gate-3 studies excluded), human only",
                       "verification": "ANARCII chain/locus/completeness on the numbered variable domain",
                       "units_downloaded": downloaded, "unit_cache": str(UNITS)},
        "fresh_studies_available": len(fresh),
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
