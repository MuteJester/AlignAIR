"""10x contig collection for TRA/TRB validation domains."""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

from alignair.aa.guard.hydration import SourceResult, atomic_write_text, filter_disjoint, sha256_text
from alignair.errors import (
    AlignAIRError, ConfigurationError, DataContractError, InternalError, MissingDependencyError,
    OperationCancelledError, OperationIOError, report_error,
)
from .http import HttpClient
from .numbering import AMINO_ACIDS

BASE = "https://cf.10xgenomics.com/samples/cell-vdj/3.0.2"
UNITS_SPEC = tuple({
    "study": f"10x_vdj_v1_hs_donor{index}", "donor": f"HealthyDonor{index}",
    "url": f"{BASE}/vdj_v1_hs_aggregated_donor{index}/vdj_v1_hs_aggregated_donor{index}_all_contig_annotations.json",
} for index in range(1, 5))
PER_DONOR_PER_CHAIN = 120
RANGE_BYTES = 15_000_000
RATE_S = 0.3
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Referer": "https://www.10xgenomics.com/", "Range": f"bytes=0-{RANGE_BYTES - 1}",
}
_LOG = logging.getLogger(__name__)


def iter_contigs(raw):
    """Decode complete array objects; ignore an unfinished trailing object.

    Framing is string/escape aware, so nested annotations and braces inside text
    do not split records. Only complete objects are passed to the JSON decoder.
    """
    text = raw.decode("utf-8", "replace")
    if not text.lstrip().startswith("["):
        raise DataContractError("10x contig data must begin with a JSON array.")
    depth, start, in_string, escaped = 0, None, False, False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                raise DataContractError("Unbalanced 10x contig JSON.")
            if depth == 0 and start is not None:
                try:
                    yield json.loads(text[start:index + 1])
                except json.JSONDecodeError as exc:
                    raise DataContractError("Invalid complete 10x contig record.") from exc
                start = None


def select_candidates(raw):
    """Retain productive, high-confidence clonotypes under the original chain caps."""
    candidates, seen, have, scanned = [], set(), {"TRA": 0, "TRB": 0}, 0
    for record in iter_contigs(raw):
        scanned += 1
        if record.get("productive") is not True or record.get("high_confidence") is not True:
            continue
        chain, v_gene = None, None
        annotations = record.get("annotations") or []
        if not isinstance(annotations, list):
            raise DataContractError("10x annotations must be a list.")
        for annotation in annotations:
            if not isinstance(annotation, Mapping):
                raise DataContractError("10x region annotation must be a mapping.")
            feature = annotation.get("feature") or {}
            if not isinstance(feature, Mapping):
                raise DataContractError("10x annotation feature must be a mapping.")
            if "V-REGION" in (feature.get("region_type") or ""):
                chain, v_gene = feature.get("chain"), feature.get("gene_name")
                break
        if chain not in ("TRA", "TRB") or have[chain] >= PER_DONOR_PER_CHAIN:
            continue
        cdr3 = record.get("cdr3") or ""
        if (v_gene is not None and not isinstance(v_gene, str)) or not isinstance(cdr3, str):
            raise DataContractError("10x V-gene and CDR3 identifiers must be text.")
        key = (chain, v_gene, cdr3)
        if key in seen:
            continue
        sequence = record.get("aa_sequence") or ""
        if not isinstance(sequence, str):
            raise DataContractError("10x aa_sequence must be text.")
        sequence = sequence.upper()
        if len(sequence) < 100:
            continue
        seen.add(key)
        have[chain] += 1
        # Preserve the original numbering window, excluding long constant-region tails.
        candidates.append({"chain": chain, "v_gene": v_gene, "cdr3": cdr3, "aa_full": sequence[:190]})
        if all(count >= PER_DONOR_PER_CHAIN for count in have.values()):
            break
    return candidates, scanned


def verify_tcr_domains(candidates, *, numberer=None):
    """Verify the expected A/B chain and extract its inclusive numbered span."""
    if not candidates:
        return []
    if numberer is None:
        try:
            from anarcii import Anarcii
        except ModuleNotFoundError as exc:
            if exc.name != "anarcii":
                raise
            raise MissingDependencyError("TCR domain verification requires ANARCII.", context={"module": "anarcii"}) from exc
        numberer = Anarcii(seq_type="tcr", mode="speed", batch_size=64, cpu=True, ncpu=1, verbose=False).number
    results = numberer([(str(index), record["aa_full"]) for index, record in enumerate(candidates)])
    if results is None:
        return []
    if isinstance(results, Mapping):
        items = results.items()
    elif isinstance(results, list):
        items = enumerate(results)
    else:
        raise DataContractError("Unexpected TCR numbering result format.")
    verified = []
    for name, result in items:
        try:
            if isinstance(name, bool) or not isinstance(name, (str, Integral)):
                raise ValueError
            index = int(name)
            if not 0 <= index < len(candidates):
                raise ValueError
        except ValueError as exc:
            raise DataContractError("TCR numbering returned an unknown input index.") from exc
        if not isinstance(result, Mapping) or result.get("error"):
            continue
        candidate = candidates[index]
        chain = result.get("chain_type")
        start, end = result.get("query_start"), result.get("query_end")
        if (chain != {"TRA": "A", "TRB": "B"}[candidate["chain"]] or
                any(isinstance(value, bool) or not isinstance(value, Integral) for value in (start, end)) or
                not 0 <= start <= end < len(candidate["aa_full"])):
            continue
        domain = candidate["aa_full"][start:end + 1]
        if len(domain) < 95 or set(domain) - AMINO_ACIDS:
            continue
        verified.append({"category": "tra" if candidate["chain"] == "TRA" else "trb", "group": "tcr",
                         "aa": domain, "chain_type": chain,
                         "verification": "anarcii_tcr_chain_locus_completeness (independent)"})
    return verified


def collect_unit(spec, denylist, *, cache_dir, client, numberer=None):
    """Reuse verified raw bytes, then reapply numbering and the current exclusions."""
    raw = client.prefix(spec["url"], byte_limit=RANGE_BYTES, cache_dir=cache_dir)
    candidates, scanned = select_candidates(raw)
    verified = verify_tcr_domains(candidates, numberer=numberer)
    unique, seen = [], set()
    for record in verified:
        digest = sha256_text(record["aa"])
        if digest in seen:
            continue
        seen.add(digest)
        lineage = f"{spec['study']}::{spec['donor']}::{record['category']}::{digest[:16]}"
        unique.append({**record, "repository": "10x", "study": spec["study"], "donor": spec["donor"],
                       "lineage_key": lineage, "id": f"{record['category']}_{sha256_text(lineage)[:12]}"})
    kept, dropped = filter_disjoint(unique, denylist=denylist, phase3=({}, {}))
    stats = {"scanned": scanned, "bytes": len(raw), "candidates": len(candidates), "anarcii_verified": len(verified),
             "after_unit_dedup": len(unique), "denylist_dropped": dropped, "kept": len(kept),
             "tra": sum(record["category"] == "tra" for record in kept),
             "trb": sum(record["category"] == "trb" for record in kept)}
    return kept, stats


def collect_tenx(denylist, *, store, census_dir, units=UNITS_SPEC, client=None, numberer=None):
    """Collect the declared study/donor units, preserving combined output filenames."""
    if not isinstance(units, (list, tuple)) or any(
        not isinstance(spec, Mapping) or any(not isinstance(spec.get(field), str) or not spec[field]
                                             for field in ("study", "donor", "url")) for spec in units
    ):
        raise ConfigurationError("10x units require study, donor, and URL strings.")
    if len({spec["study"] for spec in units}) != len(units):
        raise ConfigurationError("10x study identifiers must be unique.")
    client = client or HttpClient(timeout=180, headers=HEADERS)
    records, per_unit = [], {}
    for spec in units:
        kept, stats = collect_unit(spec, denylist, cache_dir=Path(store) / "tenx_units", client=client, numberer=numberer)
        records.extend(kept)
        per_unit[spec["study"]] = stats
        _LOG.info("%s: %s", spec["study"], json.dumps(stats))
        client.sleep(RATE_S)
    census = {
        "source": "tenx_tcr",
        "provenance": {"base": BASE, "datasets": [spec["study"] for spec in units],
                       "draw": "10x public single-cell V(D)J; all_contig_annotations.json (accessible S3 objects)",
                       "verification": "ANARCII TCR-mode chain/locus/completeness on numbered V domain",
                       "model": "each dataset=study, one biological donor; study+donor disjoint by construction"},
        "by_category": {category: sum(record["category"] == category for record in records) for category in ("tra", "trb")},
        "per_unit": per_unit, "kept_total": len(records),
        "distinct_donors": len({(record["study"], record["donor"]) for record in records}),
        "distinct_studies": len({record["study"] for record in records}),
    }
    atomic_write_text(Path(store) / "tenx_tcr.normalized.json", json.dumps(records))
    atomic_write_text(Path(census_dir) / "tenx_tcr.census.json", json.dumps(census, indent=2, sort_keys=True))
    return SourceResult("tenx_tcr", records, census)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--error-format", choices=("text", "json"), default="text")
    parser.add_argument("--units", type=Path, help="optional study/donor/URL list; defaults to the original four datasets")
    for option in ("cleaned", "fingerprint", "phase3-gold", "store", "census-dir"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from alignair.aa.guard.disjoint import build_gate3_global_denylist

        if not args.phase3_gold.is_file():
            raise OperationIOError("Phase-3 digest replay labels are missing.", context={"path": args.phase3_gold})
        fingerprint = json.loads(args.fingerprint.read_text())
        if not isinstance(fingerprint, dict) or not isinstance(fingerprint.get("record_digest", {}), dict):
            raise DataContractError("Pre-split fingerprint requires a record_digest mapping.")
        units = json.loads(args.units.read_text()) if args.units else UNITS_SPEC
        denylist = build_gate3_global_denylist(args.cleaned, fingerprint, phase3_gold_path=args.phase3_gold)
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        result = collect_tenx(denylist, store=args.store, census_dir=args.census_dir, units=units)
        print(json.dumps(result.census, indent=2, sort_keys=True))
        return 0
    except AlignAIRError as exc:
        error = exc
    except KeyboardInterrupt:
        error = OperationCancelledError("10x collection interrupted.")
    except OSError as exc:
        error = OperationIOError("Cannot access 10x inputs or outputs.", context={"reason": str(exc)})
        error.__cause__ = exc
    except json.JSONDecodeError as exc:
        error = DataContractError("Invalid JSON in 10x input.", context={"line": exc.lineno})
        error.__cause__ = exc
    except Exception as exc:
        error = InternalError("Unexpected 10x collection failure.", context={"cause_type": type(exc).__name__})
        error.__cause__ = exc
    return report_error(error, debug=args.debug, json_output=args.error_format == "json")


if __name__ == "__main__":
    raise SystemExit(main())
