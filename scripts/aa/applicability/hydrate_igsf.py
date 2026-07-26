"""Source 4 hydration: UniProt IgSF V-set domains (IPR013106, reviewed, non-IG/TR) — 10%-gate near-OOD.

Fresh UniProt query, excluding Gate-3's used accessions. The V-set domain is excised using UniProt's own
"Ig-like V-type" Domain feature coordinates — an INDEPENDENT complete-domain annotation (InterPro/UniProt),
never the production guard. IgSF proteins are not antigen receptors, so chain/locus is N/A; type is
confirmed as a non-IG/TR V-set domain by the InterPro cross-reference + gene exclusion. Counts-only census.

Run:  PYTHONPATH=src .venv/bin/python scripts/aa/applicability/hydrate_igsf.py
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

from alignair.aa.guard.disjoint import build_gate3_global_denylist
from alignair.aa.guard.hydration import atomic_write_bytes, atomic_write_text, fetch_with_retry, filter_disjoint, sha256_bytes, sha256_text

STORE = Path(".private/aa-applicability/fresh_hydration")
ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability/fresh_hydration")
CLEANED = ".private/aa-gate3-hydration/cleaned_cohort.jsonl"
FP = "docs/superpowers/artifacts/2026-07-24-aa-gate3/split_fingerprint.json"
GATE3_IGSF = ".private/aa-gate3-hydration/igsf_emitted.json"

AA = set("ACDEFGHIKLMNPQRSTVWYX")
QUERY = ("(xref:interpro-IPR013106) AND (reviewed:true) "
         "NOT gene:IGHV* NOT gene:IGKV* NOT gene:IGLV* "
         "NOT gene:TRAV* NOT gene:TRBV* NOT gene:TRGV* NOT gene:TRDV*")
FIELDS = "accession,gene_primary,organism_name,sequence,ft_domain,length"
URL = ("https://rest.uniprot.org/uniprotkb/stream?query=" + urllib.parse.quote(QUERY)
       + "&format=json&fields=" + FIELDS)
INTERPRO = "https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/interpro/IPR013106/?page_size=200"
UA = {"User-Agent": "Mozilla/5.0 (AlignAIR fresh-cohort recon)"}


def _fetch_interpro_coords():
    """Authoritative IPR013106 (V-set) match coordinates per reviewed accession {acc: (start, end)}.
    Independent of the production guard. Cached raw pages for resumability."""
    cache = STORE / "igsf.interpro_coords.json"
    if cache.exists():
        return {k: tuple(v) for k, v in json.loads(cache.read_text()).items()}
    coords, url = {}, INTERPRO
    while url:
        page = fetch_with_retry(
            lambda u=url: urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=90).read(),
            retries=4, sleep=__import__("time").sleep)
        d = json.loads(page)
        for r in d.get("results", []):
            acc = r.get("metadata", {}).get("accession")
            for ent in r.get("entries", []):
                locs = ent.get("entry_protein_locations") or []
                if acc and locs and locs[0].get("fragments"):
                    fr = locs[0]["fragments"][0]
                    coords[acc] = (fr["start"], fr["end"])
                    break
        url = d.get("next")
    atomic_write_text(cache, json.dumps({k: list(v) for k, v in coords.items()}, sort_keys=True))
    return coords


def main():
    STORE.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    denylist = build_gate3_global_denylist(CLEANED, json.load(open(FP)))
    used = {r["accession"] for r in json.load(open(GATE3_IGSF)) if r.get("accession")}

    raw_path = STORE / "igsf.raw.json"
    if raw_path.exists():
        raw = raw_path.read_bytes()
    else:
        raw = fetch_with_retry(
            lambda: urllib.request.urlopen(urllib.request.Request(URL, headers=UA), timeout=120).read(),
            retries=4, sleep=__import__("time").sleep)
        atomic_write_bytes(raw_path, raw)
        atomic_write_text(STORE / "igsf.raw.json.sha256", sha256_bytes(raw))
    data = json.loads(raw)
    coords = _fetch_interpro_coords()   # authoritative independent V-set spans {acc: (start, end)}

    picked, excluded_used, no_coord = {}, 0, 0
    for e in data.get("results", []):
        acc = e.get("primaryAccession")
        if not acc:
            continue
        if acc in used:
            excluded_used += 1
            continue
        if acc not in coords:
            no_coord += 1
            continue
        gene = ((e.get("genes") or [{}])[0].get("geneName") or {}).get("value") or acc
        org = (e.get("organism") or {}).get("scientificName", "")
        seq = (e.get("sequence") or {}).get("value", "")
        s, en = coords[acc]
        dom = (seq[s - 1:en] or "").upper()
        if not (70 <= len(dom) <= 160 and set(dom) <= AA):
            continue
        lk = f"{gene}::{org}"
        if lk not in picked:
            picked[lk] = {"id": f"igsf_{acc}", "aa": dom, "group": "igsf", "category": "igsf_vset",
                          "lineage_key": lk, "accession": acc, "organism": org, "gene": gene,
                          "domain_span": [s, en], "source": "uniprot_igsf",
                          "verification": "interpro_IPR013106_match_coords (independent of guard)"}

    records = sorted(picked.values(), key=lambda x: sha256_text(x["lineage_key"]))
    kept, dropped = filter_disjoint(records, denylist=denylist, phase3=({}, {}))
    atomic_write_text(STORE / "igsf.normalized.json", json.dumps(kept))

    from collections import Counter
    census = {
        "source": "igsf",
        "provenance": {"uniprot_query": QUERY, "uniprot_raw_sha256": sha256_bytes(raw),
                       "interpro_entry": "IPR013106", "interpro_endpoint": INTERPRO,
                       "vset_extraction": "InterPro IPR013106 match coords applied to UniProt sequence (independent of guard)"},
        "uniprot_results": len(data.get("results", [])),
        "interpro_coords_available": len(coords),
        "excluded_gate3_used_accessions": excluded_used,
        "no_interpro_coord": no_coord,
        "distinct_lineages_extracted": len(records),
        "kept_after_denylist": len(kept),
        "denylist_dropped": dropped,
        "distinct_organisms": len({r["organism"] for r in kept}),
        "top_organisms": dict(Counter(r["organism"] for r in kept).most_common(6)),
        "aa_len": {"min": min((len(r["aa"]) for r in kept), default=0),
                   "max": max((len(r["aa"]) for r in kept), default=0)},
    }
    atomic_write_text(ART / "igsf.census.json", json.dumps(census, indent=2, sort_keys=True))
    print(json.dumps(census, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
