"""Light-diversity enrichment: fresh human IGK+IGL from MULTIPLE OAS PAIRED studies + MULTIPLE donors.

WHY: the initial light fallback drew all of one role's light chains from ONE paired study/donor
(Eccles_2020, Healthy-1) -> the validation (LOCO) IGL *and* IGK slices were single-donor. Owner directive:
the LOCO IGL slice must contain >=3 independent donors, >=2 independent studies, >=37 post-dedup lineages,
with no donor/study/lineage/sequence overlap with selection or Gate-3, same independent verification. IGK
has the identical single-donor flaw and is fixed at the same time (paired data carries both chains).

This source pulls IGK+IGL from several fresh paired studies (Gate-3 + vander/zhou/eccles excluded),
capping per-donor and per-study so DONOR and STUDY diversity is spread across the pool. With many multi-
donor light studies available the study/donor split allocator then places whole studies into both roles,
giving each role multiple light donors/studies. Same rigor as the other sources: explicit human species
(unknown/transient never inferred), complete-read light domains, ANARCII K/L chain/locus verification on
the numbered V domain, denylist + untouched-donor audit, atomic per-unit cache + checksums, bounded
partial reads, counts-only census with rejection reasons. No guard/annotator inference.

Run:  PYTHONPATH=src .venv/bin/python scripts/aa/applicability/hydrate_oas_paired_diverse.py
"""
from __future__ import annotations

import collections
import csv
import io
import json
import re
import socket
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

from alignair.aa.guard.disjoint import build_gate3_global_denylist
from alignair.aa.guard.hydration import (atomic_write_bytes, atomic_write_text, fetch_with_retry,
                                         filter_disjoint, sha256_bytes, sha256_text)

STORE = Path(".private/aa-applicability/fresh_hydration")
UNITS = STORE / "oas_paired_diverse_units"
ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability/fresh_hydration")
CLEANED = ".private/aa-gate3-hydration/cleaned_cohort.jsonl"
FP = "docs/superpowers/artifacts/2026-07-24-aa-gate3/split_fingerprint.json"
PBASE = "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/"
UA = {"User-Agent": "Mozilla/5.0 (AlignAIR fresh-cohort recon)"}
AA = set("ACDEFGHIKLMNPQRSTVWYX")
_PARTIAL = 1 << 20
# Existing light studies already in the cohort -> must stay disjoint from this enrichment source.
EXISTING_LIGHT = {"vander_2017", "zhou_2013", "eccles_2020"}
PER_DONOR_PER_CHAIN = 45       # >> the 37 per-locus floor after ANARCII/dedup
MAX_DONORS_PER_STUDY = 2       # spread donors across studies rather than draining one
MAX_UNITS_PER_STUDY = 6        # bound downloads/study; also caps waste on junk-subject studies
MAX_STUDIES = 10
TARGET_DONORS = 14             # comfortably powers >=3 igl + >=3 igk donors in BOTH roles after allocation
RATE_S = 0.3
# A donor must be an explicit, single, independent subject. Reject placeholders and POOLED subjects
# (e.g. "donor 21; 22 and 23") — a pool is not one independent donor, so it can never back a diversity claim.
JUNK_SUBJECT = {"no", "unknown", "", "?", "na", "none", "nan", "n/a", "pool", "pooled"}


def _valid_subject(subj):
    s = (subj or "").strip()
    if s.lower() in JUNK_SUBJECT:
        return False
    return not re.search(r"[;&+/]| and |,", s)   # pooled / multi-donor markers


def _get_full(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def _norm(s):
    s = (s or "").lower()
    sn = re.split(r"[ _,]", s.strip())[0]
    yr = re.search(r"(?:19|20)\d\d", s)
    return sn + "_" + (yr.group() if yr else "")


def _used():
    """Gate-3 light/heavy studies + donors (never reused)."""
    st, dn = set(), set()
    for line in open(CLEANED):
        e = json.loads(line)
        if e.get("category") in ("igk", "igl", "pos_igh_repertoire"):
            parts = e["lineage_key"].split("::")
            st.add(_norm(parts[0]))
            if len(parts) >= 2:
                dn.add((_norm(parts[0]), parts[1]))
    return st, dn


def _decompress_multi(raw):
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


def _list_units(study):
    """Return (subdir, units). Older studies use csv/, 2024+ studies use csv_paired/."""
    for sub in ("csv", "csv_paired"):
        try:
            body = _get_full(f"{PBASE}{study}/{sub}/").decode("utf-8", "replace")
        except Exception:
            continue
        units = re.findall(r'href="([^"]+\.csv\.gz)"', body)
        if units:
            return sub, units
    return None, []


def _download(study, sub, unit):
    cache = UNITS / f"{study}__{unit}.part"
    if cache.exists():
        return cache.read_bytes()

    def _open():
        resp = urllib.request.urlopen(urllib.request.Request(f"{PBASE}{study}/{sub}/{unit}", headers=UA), timeout=90)
        buf = b""
        while len(buf) < _PARTIAL:
            c = resp.read(_PARTIAL - len(buf))
            if not c:
                break
            buf += c
        resp.close()
        return buf
    try:
        gz = fetch_with_retry(_open, retries=4, sleep=time.sleep)
    except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError, OSError):
        return None
    atomic_write_bytes(cache, gz)
    atomic_write_text(UNITS / f"{study}__{unit}.part.sha256", sha256_bytes(gz))
    time.sleep(RATE_S)
    return gz


def _parse(gz):
    txt = _decompress_multi(gz).decode("utf-8", "replace")
    lines = txt.split("\n")
    try:
        meta = json.loads(lines[0].strip().strip('"').replace('""', '"'))
    except Exception:
        meta = {}
    return meta, list(csv.DictReader(io.StringIO("\n".join(lines[1:]))))


def _light_candidates(meta, rows):
    study = meta.get("Author") or "unknown"
    subject = meta.get("Subject") or "unknown"
    if not _valid_subject(subject):
        return []                                    # no independent single-donor identity -> unusable
    out = []
    for r in rows:
        if r.get("productive_light") != "T" or r.get("stop_codon_light") == "T" or r.get("vj_in_frame_light") != "T":
            continue
        vl = r.get("v_call_light") or ""
        cat = "igk" if vl.startswith("IGKV") else "igl" if vl.startswith("IGLV") else None
        if cat is None:
            continue
        aa = (r.get("sequence_alignment_aa_light") or "").replace(".", "").replace("-", "").replace("*", "").upper()
        if not (80 <= len(aa) <= 170) or set(aa) - AA:
            continue
        lk = f"{study}::{subject}::{vl}::{r.get('cdr3_aa_light') or ''}"
        out.append({"lineage_key": lk, "aa": aa, "category": cat, "group": "igk_igl",
                    "repository": "oas_paired", "study": study, "donor": subject})
    return out


def _anarcii_light(cands):
    if not cands:
        return []
    from anarcii import Anarcii
    model = Anarcii(seq_type="antibody", mode="speed", batch_size=64, cpu=True, ncpu=1, verbose=False)
    res = model.number([(c["lineage_key"], c["aa"]) for c in cands])
    if res is None:
        return []
    got = res if isinstance(res, dict) else dict(enumerate(res))
    expect = {"igk": "K", "igl": "L"}
    by = {c["lineage_key"]: c for c in cands}
    out = []
    for name, rec in (got.items() if isinstance(got, dict) else enumerate(got)):
        if not isinstance(rec, dict) or rec.get("error"):
            continue
        c = by.get(name)
        ct, qs, qe = rec.get("chain_type"), rec.get("query_start"), rec.get("query_end")
        if not c or qs is None or qe is None or ct != expect[c["category"]]:
            continue
        dom = c["aa"][qs:qe + 1]
        if len(dom) < 95 or set(dom) - AA:
            continue
        out.append({**c, "aa": dom, "id": f"{c['category']}_{sha256_text(c['lineage_key'])[:12]}",
                    "chain_type": ct, "verification": "anarcii_light_chain_locus_completeness (independent)"})
    return out


def main():
    STORE.mkdir(parents=True, exist_ok=True)
    UNITS.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    denylist = build_gate3_global_denylist(CLEANED, json.load(open(FP)))
    used_studies, used_donors = _used()
    exclude = used_studies | EXISTING_LIGHT

    studies = re.findall(r'href="([A-Z][^"/]+)/"', _get_full(PBASE).decode("utf-8", "replace"))
    fresh = sorted(s for s in studies if _norm(s) not in exclude)

    rej = {"non_human_unit": 0, "unknown_species_unit": 0, "temporarily_inaccessible": 0,
           "untouched_donor_audit_drop": 0, "anarcii_rejected": 0}
    raw = []
    donors_seen = set()          # (norm_study, subject)
    studies_hit = set()
    per_donor_have = collections.Counter()   # (study,subject,cat) -> n
    for st in fresh:
        if len(studies_hit) >= MAX_STUDIES or len(donors_seen) >= TARGET_DONORS:
            break
        sub, units = _list_units(st)
        if not units:
            rej["temporarily_inaccessible"] += 1
            continue
        study_donors = set()
        for u_i, unit in enumerate(sorted(units)):
            if len(study_donors) >= MAX_DONORS_PER_STUDY or u_i >= MAX_UNITS_PER_STUDY:
                break                                    # enough donors, or study exhausted its unit budget
            gz = _download(st, sub, unit)
            if gz is None:
                rej["temporarily_inaccessible"] += 1
                continue
            meta, rows = _parse(gz)
            sp = (meta.get("Species") or "").lower()
            if sp != "human":
                rej["non_human_unit" if sp and sp != "?" else "unknown_species_unit"] += 1
                continue
            cands = _light_candidates(meta, rows)
            # untouched-donor audit: never reuse a Gate-3 donor
            cands = [c for c in cands if (_norm(c["study"]), c["donor"]) not in used_donors]
            for c in cands:
                key = (_norm(c["study"]), c["donor"])
                if key not in study_donors and len(study_donors) >= MAX_DONORS_PER_STUDY:
                    continue                                   # already have enough donors from this study
                cap_key = (c["study"], c["donor"], c["category"])
                if per_donor_have[cap_key] >= PER_DONOR_PER_CHAIN:
                    continue                                   # per-donor per-chain cap (spread, not drain)
                per_donor_have[cap_key] += 1
                donors_seen.add(key)
                study_donors.add(key)
                studies_hit.add(_norm(st))
                raw.append(c)

    # within-source dedup by lineage_key then verify
    seen, dd = set(), []
    for r in raw:
        if r["lineage_key"] in seen:
            continue
        seen.add(r["lineage_key"])
        dd.append(r)
    collected = _anarcii_light(dd)
    rej["anarcii_rejected"] = len(dd) - len(collected)
    kept, dropped = filter_disjoint(collected, denylist=denylist, phase3=({}, {}))
    atomic_write_text(STORE / "oas_paired_diverse.normalized.json", json.dumps(kept))

    per_study = collections.defaultdict(lambda: collections.Counter())
    donors_by_study = collections.defaultdict(set)
    for r in kept:
        per_study[_norm(r["study"])][r["category"]] += 1
        donors_by_study[_norm(r["study"])].add(r["donor"])
    census = {
        "source": "oas_paired_diverse",
        "provenance": {"base": PBASE, "draw": "fresh human paired studies (Gate-3 + vander/zhou/eccles excluded)",
                       "verification": "ANARCII light chain/locus/completeness (K/L) on numbered V domain",
                       "diversity_policy": f"<= {MAX_DONORS_PER_STUDY} donors/study, <= {PER_DONOR_PER_CHAIN}/chain/donor"},
        "studies_contributing": sorted(studies_hit),
        "distinct_donors": len({(r["study"], r["donor"]) for r in kept}),
        "donors_by_study": {s: sorted(d) for s, d in donors_by_study.items()},
        "by_category": {c: sum(1 for r in kept if r["category"] == c) for c in ("igk", "igl")},
        "per_study_locus": {s: dict(c) for s, c in per_study.items()},
        "kept_after_denylist": len(kept),
        "denylist_dropped": dropped,
        "rejection_reasons": rej,
    }
    atomic_write_text(ART / "oas_paired_diverse.census.json", json.dumps(census, indent=2, sort_keys=True))
    print(json.dumps(census, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
