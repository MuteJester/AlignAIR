"""UniProt IgSF V-set domains selected using independent InterPro coordinates."""
from __future__ import annotations

import argparse
import json
import urllib.parse
from collections import Counter
from pathlib import Path

from alignair.aa.guard.hydration import (
    SourceResult, atomic_write_bytes, atomic_write_text, filter_disjoint, sha256_bytes, sha256_text,
)
from alignair.errors import (
    AlignAIRError, ArtifactIntegrityError, DataContractError, InternalError,
    OperationCancelledError, OperationIOError, report_error,
)
from .http import HttpClient
from .numbering import AMINO_ACIDS

QUERY = ("(xref:interpro-IPR013106) AND (reviewed:true) "
         "NOT gene:IGHV* NOT gene:IGKV* NOT gene:IGLV* "
         "NOT gene:TRAV* NOT gene:TRBV* NOT gene:TRGV* NOT gene:TRDV*")
FIELDS = "accession,gene_primary,organism_name,sequence,ft_domain,length"
URL = ("https://rest.uniprot.org/uniprotkb/stream?query=" + urllib.parse.quote(QUERY)
       + "&format=json&fields=" + FIELDS)
INTERPRO = "https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/interpro/IPR013106/?page_size=200"


def _response(raw, source):
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DataContractError("Invalid IgSF source JSON.", context={"source": source}) from exc
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise DataContractError("IgSF source response requires a results list.", context={"source": source})
    return data


def _coordinates(value):
    if not isinstance(value, dict):
        raise DataContractError("InterPro coordinates must be an accession mapping.")
    result = {}
    for accession, span in value.items():
        if (not isinstance(span, (list, tuple)) or len(span) != 2 or
                any(isinstance(x, bool) or not isinstance(x, int) for x in span) or
                not 1 <= span[0] <= span[1]):
            raise DataContractError("InterPro domain span must use positive inclusive coordinates.",
                                    context={"accession": accession})
        result[accession] = tuple(span)
    return result


def fetch_interpro_coords(*, store, client):
    """Retain the original first-location/first-fragment selection per entry."""
    cache = Path(store) / "igsf.interpro_coords.json"
    if cache.exists():
        try:
            return _coordinates(json.loads(cache.read_text()))
        except json.JSONDecodeError as exc:
            raise DataContractError("Invalid cached InterPro coordinates.", context={"path": cache}) from exc
    coordinates, visited, url = {}, set(), INTERPRO
    while url:
        if url in visited:
            raise DataContractError("InterPro pagination repeated a page.", context={"url": url})
        visited.add(url)
        page = _response(client.get(url), "InterPro")
        for record in page["results"]:
            if not isinstance(record, dict):
                raise DataContractError("InterPro result must be a mapping.")
            metadata, entries = record.get("metadata") or {}, record.get("entries") or []
            if not isinstance(metadata, dict) or not isinstance(entries, list):
                raise DataContractError("Invalid InterPro result metadata or entries.")
            accession = metadata.get("accession")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise DataContractError("InterPro entry must be a mapping.")
                locations = entry.get("entry_protein_locations") or []
                if not isinstance(locations, list) or (locations and not isinstance(locations[0], dict)):
                    raise DataContractError("InterPro locations must be a list of mappings.")
                if accession and locations and locations[0].get("fragments"):
                    fragments = locations[0]["fragments"]
                    if not isinstance(fragments, list) or not isinstance(fragments[0], dict):
                        raise DataContractError("InterPro fragments must be a list of mappings.")
                    fragment = fragments[0]
                    coordinates[accession] = (fragment.get("start"), fragment.get("end"))
                    break
        following = page.get("next")
        if following is not None and not isinstance(following, str):
            raise DataContractError("InterPro next-page URL must be text or null.")
        url = urllib.parse.urljoin(url, following) if following else None
    coordinates = _coordinates(coordinates)
    atomic_write_text(cache, json.dumps(coordinates, sort_keys=True))
    return coordinates


def extract_domains(entries, coordinates, used_accessions):
    """Keep the first valid accession per gene/organism lineage, ordered by hash."""
    picked, excluded_used, no_coord = {}, 0, 0
    coordinates = _coordinates(coordinates)
    for entry in entries:
        if not isinstance(entry, dict):
            raise DataContractError("UniProt result must be a mapping.")
        accession = entry.get("primaryAccession")
        if not accession:
            continue
        if accession in used_accessions:
            excluded_used += 1
            continue
        if accession not in coordinates:
            no_coord += 1
            continue
        genes, organism_data, sequence_data = entry.get("genes") or [{}], entry.get("organism") or {}, entry.get("sequence") or {}
        if (not isinstance(genes, list) or not isinstance(genes[0], dict) or
                not isinstance(organism_data, dict) or not isinstance(sequence_data, dict)):
            raise DataContractError("Invalid UniProt gene, organism, or sequence fields.", context={"accession": accession})
        gene_name = genes[0].get("geneName") or {}
        if not isinstance(gene_name, dict):
            raise DataContractError("UniProt geneName must be a mapping.", context={"accession": accession})
        gene = gene_name.get("value") or accession
        organism = organism_data.get("scientificName", "")
        sequence = sequence_data.get("value", "")
        if not isinstance(sequence, str) or not isinstance(gene, str) or not isinstance(organism, str):
            raise DataContractError("UniProt sequence, gene, and organism must be text.")
        start, end = coordinates[accession]
        # Slicing must not silently shorten an out-of-bounds annotation.
        if end > len(sequence):
            continue
        domain = sequence[start - 1:end].upper()
        if not (70 <= len(domain) <= 160 and set(domain) <= AMINO_ACIDS):
            continue
        lineage = f"{gene}::{organism}"
        if lineage not in picked:
            picked[lineage] = {
                "id": f"igsf_{accession}", "aa": domain, "group": "igsf", "category": "igsf_vset",
                "lineage_key": lineage, "accession": accession, "organism": organism, "gene": gene,
                "domain_span": [start, end], "source": "uniprot_igsf",
                "verification": "interpro_IPR013106_match_coords (independent of guard)",
            }
    return sorted(picked.values(), key=lambda record: sha256_text(record["lineage_key"])), excluded_used, no_coord


def collect_igsf(denylist, used_accessions, *, store, census_dir, client=None):
    """Collect domains using explicit exclusions, source cache, and report paths."""
    # The UniProt response contains complete protein sequences and annotations.
    client = client or HttpClient(timeout=120, max_listing_bytes=128 << 20)
    store = Path(store)
    raw_path = store / "igsf.raw.json"
    checksum = store / "igsf.raw.json.sha256"
    if raw_path.exists():
        raw = raw_path.read_bytes()
        if not checksum.is_file() or checksum.read_text().strip() != sha256_bytes(raw):
            raise ArtifactIntegrityError("UniProt raw cache has a missing or mismatched checksum.",
                                         context={"path": raw_path})
    else:
        raw = client.get(URL)
        _response(raw, "UniProt")
        atomic_write_bytes(raw_path, raw)
        atomic_write_text(checksum, sha256_bytes(raw))
    data = _response(raw, "UniProt")
    coordinates = fetch_interpro_coords(store=store, client=client)
    records, excluded_used, no_coord = extract_domains(data["results"], coordinates, set(used_accessions))
    kept, dropped = filter_disjoint(records, denylist=denylist, phase3=({}, {}))
    census = {
        "source": "igsf",
        "provenance": {"uniprot_query": QUERY, "uniprot_raw_sha256": sha256_bytes(raw),
                       "interpro_entry": "IPR013106", "interpro_endpoint": INTERPRO,
                       "vset_extraction": "InterPro IPR013106 match coords applied to UniProt sequence (independent of guard)"},
        "uniprot_results": len(data["results"]), "interpro_coords_available": len(coordinates),
        "excluded_gate3_used_accessions": excluded_used, "no_interpro_coord": no_coord,
        "distinct_lineages_extracted": len(records), "kept_after_denylist": len(kept), "denylist_dropped": dropped,
        "distinct_organisms": len({record["organism"] for record in kept}),
        "top_organisms": dict(Counter(record["organism"] for record in kept).most_common(6)),
        "aa_len": {"min": min((len(record["aa"]) for record in kept), default=0),
                   "max": max((len(record["aa"]) for record in kept), default=0)},
    }
    atomic_write_text(store / "igsf.normalized.json", json.dumps(kept))
    atomic_write_text(Path(census_dir) / "igsf.census.json", json.dumps(census, indent=2, sort_keys=True))
    return SourceResult("igsf", kept, census)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--error-format", choices=("text", "json"), default="text")
    for option in ("cleaned", "fingerprint", "phase3-gold", "previous-records", "store", "census-dir"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from alignair.aa.guard.disjoint import build_gate3_global_denylist

        if not args.phase3_gold.is_file():
            raise OperationIOError("Phase-3 digest replay labels are missing.", context={"path": args.phase3_gold})
        fingerprint = json.loads(args.fingerprint.read_text())
        if not isinstance(fingerprint, dict) or not isinstance(fingerprint.get("record_digest", {}), dict):
            raise DataContractError("Pre-split fingerprint requires a record_digest mapping.")
        previous = json.loads(args.previous_records.read_text())
        if not isinstance(previous, list) or any(not isinstance(record, dict) for record in previous):
            raise DataContractError("Previous IgSF records must be a list of mappings.")
        used = {record["accession"] for record in previous if record.get("accession")}
        denylist = build_gate3_global_denylist(args.cleaned, fingerprint, phase3_gold_path=args.phase3_gold)
        result = collect_igsf(denylist, used, store=args.store, census_dir=args.census_dir)
        print(json.dumps(result.census, indent=2, sort_keys=True))
        return 0
    except AlignAIRError as exc:
        error = exc
    except KeyboardInterrupt:
        error = OperationCancelledError("IgSF collection interrupted.")
    except OSError as exc:
        error = OperationIOError("Cannot access IgSF collection inputs or outputs.", context={"reason": str(exc)})
        error.__cause__ = exc
    except json.JSONDecodeError as exc:
        error = DataContractError("Invalid JSON in IgSF collection input.", context={"line": exc.lineno})
        error.__cause__ = exc
    except Exception as exc:
        error = InternalError("Unexpected IgSF collection failure.", context={"cause_type": type(exc).__name__})
        error.__cause__ = exc
    return report_error(error, debug=args.debug, json_output=args.error_format == "json")


if __name__ == "__main__":
    raise SystemExit(main())
