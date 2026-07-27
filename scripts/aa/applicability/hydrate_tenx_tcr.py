"""TCR negative source (5%-gate near-OOD): full-length human single-cell TRA/TRB V domains from 10x
Genomics public V(D)J datasets.

WHY 10x (not VDJServer): VDJServer/ADC fresh TCR is confirmed infeasible for full-length domains — 0 usable
full-length TRA (median 57 aa partial) and abundant TRB is Adaptive CDR3-amplicon (~29 aa). See
docs/superpowers/artifacts/2026-07-26-aa-applicability/vdjserver_tcr_feasibility.json. 10x single-cell
V(D)J provides full-length, real-donor TRA+TRB contigs. Owner chose 10x to preserve repertoire character +
study/donor allocator routing.

MODEL: each 10x dataset = one study; donor = the biological individual. The four
`vdj_v1_hs_aggregated_donor{1..4}` datasets are four distinct consented individuals (Donor 1-4), each a
separate dataset/provenance dir -> four genuinely-disjoint study-donor units. The study-split allocator
places whole units in one role each, so TRA+TRB of one donor share a role and no unit straddles roles.

RIGOR (same as the other sources):
  * Only the accessible `_all_contig_annotations.{json,csv}` objects are used (10x blocks `filtered_*`/
    `.fasta` via S3 ACL); the JSON carries `aa_sequence` (full translated contig) + region annotations.
  * Completeness/chain/locus verified INDEPENDENTLY by ANARCII in TCR mode (chain A/B on the numbered
    V-domain span); the numbered domain is the stored sequence. No guard/annotator inference runs.
  * productive + high_confidence contigs only; distinct clonotype (chain, v_gene, cdr3) per donor so
    expanded clones do not dominate; global sequence dedup happens later at the Stage-1 gate.
  * Bounded streaming read (early stop once enough distinct clonotypes) + atomic per-unit cache + sha256 +
    bounded retry/backoff + resumability; denylist filtering; counts-only census with rejection reasons.

Run:  PYTHONPATH=src .venv/bin/python scripts/aa/applicability/hydrate_tenx_tcr.py
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from alignair.aa.guard.disjoint import build_gate3_global_denylist
from alignair.aa.guard.hydration import (atomic_write_bytes, atomic_write_text, fetch_with_retry,
                                         filter_disjoint, sha256_bytes, sha256_text)

STORE = Path(".private/aa-applicability/fresh_hydration")
UNITS = STORE / "tenx_units"
ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability/fresh_hydration")
CLEANED = ".private/aa-gate3-hydration/cleaned_cohort.jsonl"
FP = "docs/superpowers/artifacts/2026-07-24-aa-gate3/split_fingerprint.json"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
      "Referer": "https://www.10xgenomics.com/"}
AA = set("ACDEFGHIKLMNPQRSTVWYX")

# Four distinct consented individuals (Donor 1-4), each its own 10x dataset/provenance dir -> study-disjoint.
BASE = "https://cf.10xgenomics.com/samples/cell-vdj/3.0.2"
UNITS_SPEC = [
    {"study": "10x_vdj_v1_hs_donor1", "donor": "HealthyDonor1",
     "url": f"{BASE}/vdj_v1_hs_aggregated_donor1/vdj_v1_hs_aggregated_donor1_all_contig_annotations.json"},
    {"study": "10x_vdj_v1_hs_donor2", "donor": "HealthyDonor2",
     "url": f"{BASE}/vdj_v1_hs_aggregated_donor2/vdj_v1_hs_aggregated_donor2_all_contig_annotations.json"},
    {"study": "10x_vdj_v1_hs_donor3", "donor": "HealthyDonor3",
     "url": f"{BASE}/vdj_v1_hs_aggregated_donor3/vdj_v1_hs_aggregated_donor3_all_contig_annotations.json"},
    {"study": "10x_vdj_v1_hs_donor4", "donor": "HealthyDonor4",
     "url": f"{BASE}/vdj_v1_hs_aggregated_donor4/vdj_v1_hs_aggregated_donor4_all_contig_annotations.json"},
]
PER_DONOR_PER_CHAIN = 120     # distinct clonotypes to bank per chain per donor (>> 37 floor after ANARCII/dedup)
RANGE_BYTES = 15_000_000      # bounded partial read; ~15 MB holds ~700 distinct clonotypes/chain (>> 120)
RATE_S = 0.3


def _chain_vgene(d):
    for a in d.get("annotations", []):
        f = a.get("feature", {})
        if "V-REGION" in (f.get("region_type") or ""):
            return f.get("chain"), f.get("gene_name")
    return None, None


def _objects(txt):
    """Yield complete top-level JSON objects from a (possibly truncated) array, string/escape-aware.
    A truncated trailing object never closes at depth 0, so it is naturally ignored."""
    depth = 0; start = None; instr = False; esc = False
    for i, ch in enumerate(txt):
        if instr:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': instr = False
        else:
            if ch == '"': instr = True
            elif ch == '{':
                if depth == 0: start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    yield txt[start:i + 1]
                    start = None


def _fetch_candidates(url, cache_stub):
    """Bounded HTTP Range read of the first RANGE_BYTES, batch-parsed for distinct-clonotype productive
    high-confidence TRA/TRB candidates (cap PER_DONOR_PER_CHAIN each). Caches the raw partial + sha256."""
    cache = Path(str(cache_stub) + ".partial.json")
    if cache.exists():
        raw = cache.read_bytes()
    else:
        def _open():
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes=0-{RANGE_BYTES}"})
            return urllib.request.urlopen(req, timeout=180)
        resp = fetch_with_retry(_open, retries=4, sleep=time.sleep)
        raw = b""
        while len(raw) <= RANGE_BYTES:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            raw += chunk
        resp.close()
        atomic_write_bytes(cache, raw)
        atomic_write_text(Path(str(cache_stub) + ".partial.json.sha256"), sha256_bytes(raw))
    txt = raw.decode("utf-8", "replace")
    cands = []; seen = set(); have = {"TRA": 0, "TRB": 0}; scanned = 0
    for obj in _objects(txt):
        scanned += 1
        _consider(obj, cands, seen, have)
        if have["TRA"] >= PER_DONOR_PER_CHAIN and have["TRB"] >= PER_DONOR_PER_CHAIN:
            break
    return cands, scanned, len(raw)


def _consider(obj, cands, seen, have):
    try:
        d = json.loads(obj)
    except Exception:
        return
    if not (d.get("productive") is True and d.get("high_confidence") is True):
        return
    ch, vg = _chain_vgene(d)
    if ch not in ("TRA", "TRB") or have[ch] >= PER_DONOR_PER_CHAIN:
        return
    cdr3 = d.get("cdr3") or ""
    key = (ch, vg, cdr3)
    if key in seen:
        return
    aa = (d.get("aa_sequence") or "").upper()
    if len(aa) < 100:
        return
    seen.add(key)
    have[ch] += 1
    # The V domain lives in the first ~150 aa (leader ~20 + V ~115 + FR4). Trim to 190 (< ANARCII's
    # SHORT_SEQ_MAX_LENGTH=200) so full-contig translations with a long C-region tail do not trip
    # ANARCII's long-sequence windowing path; the numbered V span is extracted from this same window.
    cands.append({"chain": ch, "v_gene": vg, "cdr3": cdr3, "aa_full": aa[:190]})


def _anarcii_tcr(cands):
    if not cands:
        return []
    from anarcii import Anarcii
    model = Anarcii(seq_type="tcr", mode="speed", batch_size=64, cpu=True, ncpu=1, verbose=False)
    named = [(f"{i}", c["aa_full"]) for i, c in enumerate(cands)]
    res = model.number(named)
    if res is None:
        return []
    got = res if isinstance(res, dict) else dict(enumerate(res))
    expect = {"TRA": "A", "TRB": "B"}
    out = []
    items = got.items() if isinstance(got, dict) else enumerate(got)
    for name, rec in items:
        idx = int(name) if isinstance(name, str) else name
        c = cands[idx]
        if not isinstance(rec, dict) or rec.get("error"):
            continue
        ct, qs, qe = rec.get("chain_type"), rec.get("query_start"), rec.get("query_end")
        if qs is None or qe is None or ct != expect[c["chain"]]:
            continue
        dom = c["aa_full"][qs:qe + 1]
        if len(dom) < 95 or set(dom) - AA:
            continue
        cat = "tra" if c["chain"] == "TRA" else "trb"
        out.append({"category": cat, "group": "tcr", "aa": dom, "chain_type": ct,
                    "verification": "anarcii_tcr_chain_locus_completeness (independent)"})
    return out


def hydrate_unit(spec, denylist):
    UNITS.mkdir(parents=True, exist_ok=True)
    stub = UNITS / spec["study"]
    norm_path = Path(str(stub) + ".normalized.json")
    if norm_path.exists():                                    # resume: unit already done
        recs = json.loads(norm_path.read_text())
        return recs, {"resumed": True, "kept": len(recs)}
    cands, scanned, nbytes = _fetch_candidates(spec["url"], stub)
    verified = _anarcii_tcr(cands)
    # attach provenance + lineage keys
    recs = []
    for c in verified:
        lk = f"{spec['study']}::{spec['donor']}::{c['category']}::{sha256_text(c['aa'])[:16]}"
        recs.append({**c, "repository": "10x", "study": spec["study"], "donor": spec["donor"],
                     "lineage_key": lk, "id": f"{c['category']}_{sha256_text(lk)[:12]}"})
    # dedup within unit by aa
    seen, dd = set(), []
    for r in recs:
        h = sha256_text(r["aa"])
        if h in seen:
            continue
        seen.add(h); dd.append(r)
    kept, dropped = filter_disjoint(dd, denylist=denylist, phase3=({}, {}))
    atomic_write_text(norm_path, json.dumps(kept))
    stats = {"scanned": scanned, "bytes": nbytes, "candidates": len(cands), "anarcii_verified": len(verified),
             "after_unit_dedup": len(dd), "denylist_dropped": dropped, "kept": len(kept),
             "tra": sum(1 for r in kept if r["category"] == "tra"),
             "trb": sum(1 for r in kept if r["category"] == "trb")}
    return kept, stats


def main():
    STORE.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    denylist = build_gate3_global_denylist(CLEANED, json.load(open(FP)))

    all_recs = []
    per_unit = {}
    for spec in UNITS_SPEC:
        recs, stats = hydrate_unit(spec, denylist)
        per_unit[spec["study"]] = stats
        all_recs.extend(recs)
        print(f"[{spec['study']}] {json.dumps(stats)}", flush=True)
        time.sleep(RATE_S)

    atomic_write_text(STORE / "tenx_tcr.normalized.json", json.dumps(all_recs))
    census = {
        "source": "tenx_tcr",
        "provenance": {"base": BASE, "datasets": [s["study"] for s in UNITS_SPEC],
                       "draw": "10x public single-cell V(D)J; all_contig_annotations.json (accessible S3 objects)",
                       "verification": "ANARCII TCR-mode chain/locus/completeness on numbered V domain",
                       "model": "each dataset=study, one biological donor; study+donor disjoint by construction"},
        "by_category": {c: sum(1 for r in all_recs if r["category"] == c) for c in ("tra", "trb")},
        "per_unit": per_unit,
        "kept_total": len(all_recs),
        "distinct_donors": len({(r["study"], r["donor"]) for r in all_recs}),
        "distinct_studies": len({r["study"] for r in all_recs}),
    }
    atomic_write_text(ART / "tenx_tcr.census.json", json.dumps(census, indent=2, sort_keys=True))
    print(json.dumps(census, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
