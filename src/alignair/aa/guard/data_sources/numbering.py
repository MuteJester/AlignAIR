"""Independent, lazily loaded antibody-domain verification."""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

from alignair.aa.guard.hydration import sha256_text
from alignair.errors import DataContractError, MissingDependencyError

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYX")


def verify_antibody_domains(candidates, *, numberer=None, light_only=False):
    """Verify H/K/L identity and retain the complete numbered domain.

    ``numberer`` takes named sequences and returns ANARCII-compatible results.
    Coordinates are inclusive. Invalid spans or wrong chains reject a record.
    The first occurrence of a lineage supplies its sequence.
    """
    if not candidates:
        return []
    by_name = {}
    for candidate in candidates:
        by_name.setdefault(candidate["lineage_key"], candidate)
    if numberer is None:
        try:
            from anarcii import Anarcii
        except ModuleNotFoundError as exc:
            if exc.name != "anarcii":
                raise
            raise MissingDependencyError("Independent domain verification requires ANARCII.",
                                         context={"module": "anarcii"}) from exc
        numberer = Anarcii(seq_type="antibody", mode="speed", batch_size=64, cpu=True, ncpu=1, verbose=False).number
    results = numberer([(name, record["aa"]) for name, record in by_name.items()])
    if results is None:
        return []
    if not isinstance(results, Mapping):
        raise DataContractError("Domain numbering must return results keyed by input name.")
    expected = {"igk": "K", "igl": "L"}
    if not light_only:
        expected["pos_igh_repertoire"] = "H"
    verification = ("anarcii_light_chain_locus_completeness (independent)" if light_only else
                    "anarcii_chain_locus_completeness (independent)")
    kept = []
    for name, result in results.items():
        if not isinstance(result, Mapping) or result.get("error"):
            continue
        candidate = by_name.get(name)
        if candidate is None:
            raise DataContractError("Domain numbering returned an unknown input name.")
        start, end = result.get("query_start"), result.get("query_end")
        chain = result.get("chain_type")
        if (chain != expected.get(candidate["category"]) or chain is None or
                any(isinstance(x, bool) or not isinstance(x, Integral) for x in (start, end)) or
                not 0 <= start <= end < len(candidate["aa"])):
            continue
        domain = candidate["aa"][start:end + 1]
        minimum = 100 if candidate["category"] == "pos_igh_repertoire" else 95
        if len(domain) < minimum or set(domain) - AMINO_ACIDS:
            continue
        kept.append({**candidate, "aa": domain, "chain_type": chain, "verification": verification,
                     "id": f"{candidate['category']}_{sha256_text(candidate['lineage_key'])[:12]}"})
    return kept
