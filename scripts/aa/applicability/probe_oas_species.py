"""Probe every FRESH OAS unpaired study for species + read completeness, to target the puller and assess
fresh-human feasibility. For each fresh study: pick the smallest heavy + smallest light unit (HEAD), pull
it, record species and median sequence_alignment_aa length. Counts-only report; no models run.
"""
from __future__ import annotations

import gzip
import io
import json
import re
import time
import urllib.request
from pathlib import Path

BASE = "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/"
UA = {"User-Agent": "Mozilla/5.0 (AlignAIR fresh-cohort recon)"}
OUT = Path(".private/aa-applicability/fresh_hydration/oas_species_probe.json")
CLEANED = ".private/aa-gate3-hydration/cleaned_cohort.jsonl"


def _get(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read()


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
        h = urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=20)
        return int(h.headers.get("content-length") or 1 << 62)
    except Exception:
        return 1 << 62


def _sample(study, unit):
    gz = _get(f"{BASE}{study}/csv/{unit}")
    txt = gzip.decompress(gz).decode("utf-8", "replace")
    lines = txt.split("\n")
    meta = json.loads(lines[0].strip().strip('"').replace('""', '"'))
    import csv
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[1:250]))))
    lens = [len((r.get("sequence_alignment_aa") or "").replace(".", "").replace("-", ""))
            for r in rows if r.get("productive") == "T"]
    lens.sort()
    return meta.get("Species", "?"), (lens[len(lens) // 2] if lens else 0), len(rows)


def main():
    used = _used()
    studies = re.findall(r'href="([A-Z][^"/]+)/"', _get(BASE).decode("utf-8", "replace"))
    fresh = [s for s in studies if _norm(s) not in used]
    report = {"total_studies": len(studies), "fresh_studies": len(fresh), "per_study": {}}
    human_complete = []
    for st in fresh:
        try:
            units = re.findall(r'href="([^"]+\.csv\.gz)"', _get(f"{BASE}{st}/csv/").decode("utf-8", "replace"))
        except Exception:
            continue
        heavy = [u for u in units if "Heavy" in u]
        light = [u for u in units if "Light" in u]
        entry = {"n_heavy_units": len(heavy), "n_light_units": len(light)}
        for label, us in (("heavy", heavy), ("light", light)):
            if not us:
                continue
            smallest = min(us, key=lambda u: _head_size(f"{BASE}{st}/csv/{u}"))
            try:
                sp, med, nrows = _sample(st, smallest)
                entry[f"{label}_species"] = sp
                entry[f"{label}_median_aa"] = med
            except Exception as e:
                entry[f"{label}_err"] = str(e)[:60]
            time.sleep(0.3)
        report["per_study"][st] = entry
        if entry.get("heavy_species", "").lower() == "human" and entry.get("heavy_median_aa", 0) >= 100:
            human_complete.append(st)
    report["fresh_human_complete_heavy_studies"] = human_complete
    report["n_fresh_human_complete"] = len(human_complete)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("total_studies", "fresh_studies",
                                             "n_fresh_human_complete", "fresh_human_complete_heavy_studies")}, indent=2))


if __name__ == "__main__":
    main()
