"""SAbDab camelid VHH and murine VH cohort collection."""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
from collections import Counter
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

from alignair.aa.guard.hydration import (
    SourceResult, atomic_write_bytes, atomic_write_text, filter_disjoint, sha256_bytes, sha256_text,
)
from alignair.errors import (
    AlignAIRError, DataContractError, InternalError, MissingDependencyError,
    OperationCancelledError, OperationIOError, log_error, report_error,
)
from .http import HttpClient
from .numbering import AMINO_ACIDS

SINGLE = "https://sabdab.opig.stats.ox.ac.uk/api/download/all-single-domain-summary"
ALLSUM = "https://sabdab.opig.stats.ox.ac.uk/api/download/all-summary"
RCSB = "https://www.rcsb.org/fasta/entry/"
TARGET = {"lama": 45, "nonlama": 45, "murine": 45}
CAM = ("lama", "vicugna", "camelus", "glama")
_LOG = logging.getLogger(__name__)


def _used_identifiers(previous):
    if not isinstance(previous, Mapping):
        raise DataContractError("Previous SAbDab records must be a category mapping.")
    used = set()
    for category in ("vhh", "murine_vh"):
        records = previous.get(category, [])
        if not isinstance(records, list) or any(not isinstance(record, Mapping) for record in records):
            raise DataContractError("Previous SAbDab category must contain a list of records.", context={"category": category})
        used.update(record.get("sabdab_id") for record in records)
        used.update((record.get("pdb") or "").lower() for record in records)
    return used


def select_structures(rows, previous_records, *, single=True):
    """Preserve the original species, structure-ID, and HEAVY_ID exclusions."""
    used = _used_identifiers(previous_records)
    used_heavy = {row.get("HEAVY_ID") for row in rows if row.get("SABDAB_ID") in used}
    seen, selected = set(), []
    for row in rows:
        species = (row.get("heavy_species") or "").lower()
        structure_type = (row.get("type") or "").strip().lower()
        if single:
            if structure_type != "sd-h" or not any(genus in species for genus in CAM):
                continue
        elif "mus musculus" not in species or structure_type == "sd-h":
            continue
        if (row.get("SABDAB_ID") in used or (row.get("PDB") or "").lower() in used or
                row.get("HEAVY_ID") in used_heavy):
            continue
        heavy_id = row.get("HEAVY_ID")
        if not heavy_id or heavy_id in seen or not row.get("Hchain"):
            continue
        seen.add(heavy_id)
        selected.append(row)
    return selected


def heavy_sequence(fasta, chain):
    """Select an RCSB FASTA entry using its listed chain identifiers."""
    sequences, header = {}, None
    for line in fasta.splitlines():
        if line.startswith(">"):
            header = line
            sequences[header] = ""
        elif header:
            sequences[header] += line.strip()
    for header, sequence in sequences.items():
        fields = header.split("|")
        if len(fields) >= 2:
            chains = fields[1].replace("Chains ", "").replace("Chain ", "")
            if chain in {value.strip() for value in chains.split(",")}:
                return sequence
    return None


def extract_heavy_domains(named_sequences, *, numberer=None):
    """Retain numbered H-chain domains of 90–160 residues, using inclusive spans."""
    if not named_sequences:
        return {}
    if numberer is None:
        try:
            from anarcii import Anarcii
        except ModuleNotFoundError as exc:
            if exc.name != "anarcii":
                raise
            raise MissingDependencyError("SAbDab VH extraction requires ANARCII.", context={"module": "anarcii"}) from exc
        numberer = Anarcii(seq_type="antibody", mode="speed", batch_size=32, cpu=True, ncpu=1, verbose=False).number
    results = numberer(named_sequences)
    if not isinstance(results, Mapping):
        raise DataContractError("SAbDab domain numbering must return results keyed by input name.")
    sequences, domains = dict(named_sequences), {}
    for name, result in results.items():
        if not isinstance(result, Mapping) or result.get("error"):
            continue
        if name not in sequences:
            raise DataContractError("SAbDab domain numbering returned an unknown input name.")
        start, end = result.get("query_start"), result.get("query_end")
        if (result.get("chain_type") != "H" or
                any(isinstance(value, bool) or not isinstance(value, Integral) for value in (start, end)) or
                not 0 <= start <= end < len(sequences[name])):
            continue
        domain = sequences[name][start:end + 1]
        if 90 <= len(domain) <= 160 and set(domain) <= AMINO_ACIDS:
            domains[name] = (domain, "H")
    return domains


def _summary(url, name, store, client):
    raw = client.get(url)
    try:
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8", "replace")), strict=True))
    except csv.Error as exc:
        raise DataContractError("Invalid SAbDab summary CSV.", context={"url": url}) from exc
    atomic_write_bytes(store / f"{name}.csv", raw)
    atomic_write_text(store / f"{name}.csv.sha256", sha256_bytes(raw))
    return rows, sha256_bytes(raw)


def _fasta(pdb, store, client):
    code = pdb.replace("pdb_0000", "").replace("pdb_000", "").upper()
    if not re.fullmatch(r"[A-Z0-9]{4}", code):
        raise DataContractError("Invalid SAbDab PDB identifier.")
    cache = store / "fasta_cache" / f"{code}.fasta"
    if cache.exists():
        return cache.read_text()
    try:
        text = client.get(RCSB + code).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataContractError("Invalid RCSB FASTA encoding.", context={"pdb": code}) from exc
    atomic_write_text(cache, text)
    return text


def _is_lama(row):
    return "glama" in (row.get("heavy_species") or "").lower()


def _collect(rows, previous, category, target, *, single, store, client, numberer):
    selected = select_structures(rows, previous, single=single)
    if single:
        lama = [row for row in selected if _is_lama(row)]
        nonlama = [row for row in selected if not _is_lama(row)]
        selected = lama[:TARGET["lama"] * 2] + nonlama[:TARGET["nonlama"] * 2]
    else:
        selected = selected[:target * 2]
    named, metadata = [], {}
    for row in selected:
        if not row.get("PDB"):
            continue
        try:
            sequence = heavy_sequence(_fasta(row["PDB"], store, client), row["Hchain"])
        except (OperationIOError, DataContractError) as exc:
            log_error(_LOG, exc, level=logging.WARNING)
            continue
        if sequence:
            name = row.get("SABDAB_ID") or row["PDB"]
            named.append((name, sequence))
            metadata[name] = row
    records = []
    for name, (domain, _) in extract_heavy_domains(named, numberer=numberer).items():
        row = metadata[name]
        records.append({
            "id": f"{category}_{name}", "aa": domain, "group": "vhh_nonhuman", "category": category,
            "lineage_key": row.get("HEAVY_ID") or name, "heavy_species": (row.get("heavy_species") or "").lower(),
            "deposition_date": row.get("date") or "", "source": "sabdab", "extraction": "anarcii_vh",
        })
    return records


def collect_sabdab(denylist, previous_records, *, store, census_dir, client=None, numberer=None):
    """Collect VHH/murine domains with explicit inputs and unchanged output filenames."""
    _used_identifiers(previous_records)
    client = client or HttpClient(timeout=90, max_listing_bytes=128 << 20)
    store = Path(store)
    single_rows, single_sha = _summary(SINGLE, "sabdab.summary_single", store, client)
    all_rows, all_sha = _summary(ALLSUM, "sabdab.summary_all", store, client)
    options = dict(store=store, client=client, numberer=numberer)
    records = (_collect(single_rows, previous_records, "vhh", TARGET["lama"] + TARGET["nonlama"], single=True, **options) +
               _collect(all_rows, previous_records, "murine_vh", TARGET["murine"], single=False, **options))
    seen, unique = set(), []
    for record in records:
        digest = sha256_text(record["aa"])
        if digest not in seen:
            seen.add(digest)
            unique.append(record)
    kept, dropped = filter_disjoint(unique, denylist=denylist, phase3=({}, {}))
    census = {
        "source": "sabdab",
        "provenance": {"single_domain": SINGLE, "all_summary": ALLSUM, "single_sha256": single_sha,
                       "all_sha256": all_sha,
                       "vh_extraction": "ANARCII (independent of guard/profile bank); chain_type==H, query_start:query_end"},
        "fetched_after_extraction": len(records), "after_within_source_dedup": len(unique),
        "kept_after_denylist": len(kept), "denylist_dropped": dropped,
        "by_category": {category: sum(record["category"] == category for record in kept)
                        for category in {record["category"] for record in kept}},
        "vhh_genus": dict(Counter(record["heavy_species"] for record in kept if record["category"] == "vhh")),
        "vhh_lama": sum(record["category"] == "vhh" and _is_lama(record) for record in kept),
        "vhh_nonlama": sum(record["category"] == "vhh" and not _is_lama(record) for record in kept),
    }
    atomic_write_text(store / "sabdab.normalized.json", json.dumps(kept))
    atomic_write_text(Path(census_dir) / "sabdab.census.json", json.dumps(census, indent=2, sort_keys=True))
    return SourceResult("sabdab", kept, census)


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
        _used_identifiers(previous)
        denylist = build_gate3_global_denylist(args.cleaned, fingerprint, phase3_gold_path=args.phase3_gold)
        result = collect_sabdab(denylist, previous, store=args.store, census_dir=args.census_dir)
        print(json.dumps(result.census, indent=2, sort_keys=True))
        return 0
    except AlignAIRError as exc:
        error = exc
    except KeyboardInterrupt:
        error = OperationCancelledError("SAbDab collection interrupted.")
    except OSError as exc:
        error = OperationIOError("Cannot access SAbDab inputs or outputs.", context={"reason": str(exc)})
        error.__cause__ = exc
    except json.JSONDecodeError as exc:
        error = DataContractError("Invalid JSON in SAbDab input.", context={"line": exc.lineno})
        error.__cause__ = exc
    except Exception as exc:
        error = InternalError("Unexpected SAbDab collection failure.", context={"cause_type": type(exc).__name__})
        error.__cause__ = exc
    return report_error(error, debug=args.debug, json_output=args.error_format == "json")


if __name__ == "__main__":
    raise SystemExit(main())
