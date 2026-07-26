"""Fallback human LIGHT source: OAS PAIRED repertoires (IGK/IGL) — added because OAS unpaired had only 2
fresh human complete-light studies (IGL-limited), insufficient for donor-disjoint 37/37 across roles.

Same rigor as the unpaired puller: explicit human species (unknown/transient never inferred), complete-read
light domains, ANARCII chain/locus verification (K/L on the numbered light V-domain), denylist +
untouched-donor audit, repository/study/donor tags for the whole-study split allocator. Bounded partial
reads; counts-only census with rejection reasons. No guard/annotator inference.

Run:  PYTHONPATH=src .venv/bin/python scripts/aa/applicability/hydrate_oas_paired.py
"""
from __future__ import annotations

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
from alignair.aa.guard.hydration import atomic_write_bytes, atomic_write_text, fetch_with_retry, filter_disjoint, sha256_bytes, sha256_text

STORE = Path(".private/aa-applicability/fresh_hydration")
UNITS = STORE / "oas_paired_units"
ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability/fresh_hydration")
CLEANED = ".private/aa-gate3-hydration/cleaned_cohort.jsonl"
FP = "docs/superpowers/artifacts/2026-07-24-aa-gate3/split_fingerprint.json"
PBASE = "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/"
UA = {"User-Agent": "Mozilla/5.0 (AlignAIR fresh-cohort recon)"}
AA = set("ACDEFGHIKLMNPQRSTVWYX")
_PARTIAL = 1 << 20
TARGET = {"igk": 120, "igl": 120}
MAX_UNITS = 60
MAX_UNITS_PER_STUDY = 8
RATE_S = 0.4


def _get_full(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def _norm(s):
    s = (s or "").lower()
    sn = re.split(r"[ _,]", s.strip())[0]
    yr = re.search(r"(?:19|20)\d\d", s)
    return sn + "_" + (yr.group() if yr else "")


def _used():
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


def _download(study, unit):
    cache = UNITS / f"{study}__{unit}.part"
    if cache.exists():
        return cache.read_bytes()

    def _open():
        resp = urllib.request.urlopen(urllib.request.Request(f"{PBASE}{study}/csv/{unit}", headers=UA), timeout=90)
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

    studies = re.findall(r'href="([A-Z][^"/]+)/"', _get_full(PBASE).decode("utf-8", "replace"))
    fresh = [s for s in studies if _norm(s) not in used_studies]

    rej = {"non_human_unit": 0, "unknown_species_unit": 0, "temporarily_inaccessible": 0,
           "untouched_donor_audit_drop": 0, "anarcii_rejected": 0}
    raw, downloaded, studies_hit = [], 0, set()
    have = {k: 0 for k in TARGET}
    for st in sorted(fresh):
        if downloaded >= MAX_UNITS or all(have[k] >= 2 * TARGET[k] for k in TARGET):
            break
        try:
            units = re.findall(r'href="([^"]+\.csv\.gz)"', _get_full(f"{PBASE}{st}/csv/").decode("utf-8", "replace"))
        except Exception:
            rej["temporarily_inaccessible"] += 1
            continue
        study_units = 0
        for unit in units:
            if downloaded >= MAX_UNITS or study_units >= MAX_UNITS_PER_STUDY:
                break
            gz = _download(st, unit)
            if gz is None:
                rej["temporarily_inaccessible"] += 1
                continue
            meta, rows = _parse(gz)
            sp = (meta.get("Species") or "").lower()
            if sp != "human":
                rej["non_human_unit" if sp and sp != "?" else "unknown_species_unit"] += 1
                continue
            downloaded += 1
            study_units += 1
            cands = _light_candidates(meta, rows)
            fresh_cands = [c for c in cands if (_norm(c["study"]), c["donor"]) not in used_donors]
            rej["untouched_donor_audit_drop"] += len(cands) - len(fresh_cands)
            if fresh_cands:
                studies_hit.add(_norm(st))
                for c in fresh_cands:
                    have[c["category"]] = have.get(c["category"], 0) + 1
                raw.extend(fresh_cands)

    seen, dd = set(), []
    for r in raw:
        if r["lineage_key"] in seen:
            continue
        seen.add(r["lineage_key"])
        dd.append(r)
    collected = _anarcii_light(dd)
    rej["anarcii_rejected"] = len(dd) - len(collected)
    kept, dropped = filter_disjoint(collected, denylist=denylist, phase3=({}, {}))
    atomic_write_text(STORE / "oas_paired.normalized.json", json.dumps(kept))

    import collections
    per_study = collections.defaultdict(lambda: collections.Counter())
    for r in kept:
        per_study[_norm(r["study"])][r["category"]] += 1
    census = {
        "source": "oas_paired",
        "provenance": {"base": PBASE, "draw": "fresh human paired studies (Gate-3 studies + unpaired-used excluded)",
                       "verification": "ANARCII light chain/locus/completeness (K/L) on numbered V domain",
                       "units_downloaded": downloaded},
        "fresh_studies_available": len(fresh),
        "studies_contributing": sorted(studies_hit),
        "by_category": {c: sum(1 for r in kept if r["category"] == c) for c in ("igk", "igl")},
        "per_study_locus": {s: dict(c) for s, c in per_study.items()},
        "kept_after_denylist": len(kept),
        "denylist_dropped": dropped,
        "rejection_reasons": rej,
        "distinct_donors": len({(r["study"], r["donor"]) for r in kept}),
    }
    atomic_write_text(ART / "oas_paired.census.json", json.dumps(census, indent=2, sort_keys=True))
    print(json.dumps(census, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
