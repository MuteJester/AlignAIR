"""Probe every FRESH OAS unpaired study for species + read completeness, classifying each into exactly
one of four states so only CONFIRMED outcomes drive feasibility:

  * eligible_human_complete  — explicit human species + median read length >= completeness floor
  * confirmed_ineligible     — download+metadata OK, but non-human OR partial reads
  * unknown_metadata         — download OK but species/metadata unparseable (never inferred)
  * temporarily_inaccessible — transient network/HTTP failure after retries (retryable; NOT ineligible)

Transient failures are never counted as non-human or incomplete. Resumable + incremental: confirmed
studies are cached and skipped; only temporarily_inaccessible / unprobed studies are (re)tried.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import re
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from alignair.aa.guard.hydration import atomic_write_text

BASE = "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/"
UA = {"User-Agent": "Mozilla/5.0 (AlignAIR fresh-cohort recon)"}
OUT = Path(".private/aa-applicability/fresh_hydration/oas_species_probe.json")
CLEANED = ".private/aa-gate3-hydration/cleaned_cohort.jsonl"
HEAVY_MIN_AA = 100
LIGHT_MIN_AA = 95
RETRIES = 4

CONFIRMED = {"eligible_human_complete", "confirmed_ineligible", "unknown_metadata"}


class _Transient(Exception):
    pass


def _is_transient(e):
    if isinstance(e, urllib.error.HTTPError):
        return e.code in (408, 425, 429, 500, 502, 503, 504)
    return isinstance(e, (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError))


def _get(u, method="GET"):
    """Bounded retry; raise _Transient on transient network/HTTP after retries, else re-raise."""
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(u, headers=UA, method=method)
            return urllib.request.urlopen(req, timeout=45)
        except Exception as e:  # noqa: BLE001
            last = e
            if _is_transient(e) and attempt < RETRIES - 1:
                time.sleep(1.5 ** attempt)
                continue
            break
    raise _Transient(str(last)) if _is_transient(last) else last


def _norm(s):
    s = (s or "").lower()
    sn = re.split(r"[ _,]", s.strip())[0]
    yr = re.search(r"(?:19|20)\d\d", s)
    return sn + "_" + (yr.group() if yr else "")


def _used():
    u = set()
    for line in open(CLEANED):
        e = json.loads(line)
        if e.get("category") in ("igk", "igl", "pos_igh_repertoire"):
            u.add(_norm(e["lineage_key"].split("::")[0]))
    return u


def _head_size(url):
    try:
        return int(_get(url, "HEAD").headers.get("content-length") or 1 << 62)
    except _Transient:
        raise
    except Exception:
        return 1 << 62


def _sample_chain(study, units, min_aa):
    """Return a per-chain status dict. Transient -> temporarily_inaccessible (retryable)."""
    if not units:
        return {"status": "no_units"}
    try:
        smallest = min(units, key=lambda u: _head_size(f"{BASE}{study}/csv/{u}"))
        gz = _get(f"{BASE}{study}/csv/{smallest}").read()
    except _Transient as e:
        return {"status": "temporarily_inaccessible", "detail": str(e)[:80]}
    try:
        txt = gzip.decompress(gz).decode("utf-8", "replace")
        lines = txt.split("\n")
        meta = json.loads(lines[0].strip().strip('"').replace('""', '"'))
        rows = list(csv.DictReader(io.StringIO("\n".join(lines[1:250]))))
        lens = sorted(len((r.get("sequence_alignment_aa") or "").replace(".", "").replace("-", ""))
                      for r in rows if r.get("productive") == "T")
        sp = meta.get("Species")
        med = lens[len(lens) // 2] if lens else 0
    except Exception:
        return {"status": "unknown_metadata"}
    if not sp or sp == "?":
        return {"status": "unknown_metadata", "median_aa": med}
    if sp.lower() != "human":
        return {"status": "confirmed_ineligible", "species": sp, "median_aa": med, "reason": "non_human"}
    if med < min_aa:
        return {"status": "confirmed_ineligible", "species": sp, "median_aa": med, "reason": "partial_reads"}
    return {"status": "eligible", "species": sp, "median_aa": med}


def _study_status(heavy, light):
    if heavy.get("status") == "eligible" or light.get("status") == "eligible":
        return "eligible_human_complete"
    statuses = {heavy.get("status"), light.get("status")}
    if "temporarily_inaccessible" in statuses:
        return "temporarily_inaccessible"       # could still flip to eligible -> not confirmed
    if "unknown_metadata" in statuses:
        return "unknown_metadata"
    return "confirmed_ineligible"


def _probe_study(study):
    try:
        units = re.findall(r'href="([^"]+\.csv\.gz)"', _get(f"{BASE}{study}/csv/").read().decode("utf-8", "replace"))
    except _Transient as e:
        return {"status": "temporarily_inaccessible", "detail": str(e)[:80]}
    heavy = _sample_chain(study, [u for u in units if "Heavy" in u], HEAVY_MIN_AA)
    light = _sample_chain(study, [u for u in units if "Light" in u], LIGHT_MIN_AA)
    time.sleep(0.3)
    return {"status": _study_status(heavy, light), "heavy": heavy, "light": light}


def main():
    used = _used()
    studies = re.findall(r'href="([A-Z][^"/]+)/"', _get(BASE).read().decode("utf-8", "replace"))
    fresh = [s for s in studies if _norm(s) not in used]

    cache = json.loads(OUT.read_text()).get("per_study", {}) if OUT.exists() else {}
    per = {}
    for st in fresh:
        prev = cache.get(st)
        if prev and prev.get("status") in CONFIRMED:      # resume: keep confirmed, retry the rest
            per[st] = prev
            continue
        per[st] = _probe_study(st)
        # incremental write (resumable)
        _write(fresh, per)
    _write(fresh, per)
    summary = _summary(fresh, per)
    print(json.dumps(summary, indent=2))


def _summary(fresh, per):
    from collections import Counter
    by = Counter(e.get("status") for e in per.values())
    eligible = [s for s, e in per.items() if e.get("status") == "eligible_human_complete"]
    confirmed = sum(by[k] for k in CONFIRMED)
    return {
        "total_studies_listed": None, "fresh_studies": len(fresh),
        "by_status": dict(by),
        "eligible_human_complete": len(eligible),
        "confirmed_ineligible": by["confirmed_ineligible"],
        "unknown_metadata": by["unknown_metadata"],
        "temporarily_inaccessible": by["temporarily_inaccessible"],
        "confirmed_total": confirmed,
        "feasibility_note": "OAS feasibility uses CONFIRMED outcomes only; temporarily_inaccessible are retried, never counted ineligible",
        "eligible_studies": sorted(eligible),
    }


def _write(fresh, per):
    atomic_write_text(OUT, json.dumps({"per_study": per, "summary": _summary(fresh, per)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
