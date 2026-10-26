"""OAS metadata, compressed CSV prefixes, and normalized candidate records."""
from __future__ import annotations

import csv
import io
import json
import re
import zlib
from dataclasses import dataclass
from itertools import islice

from alignair.errors import DataContractError
from ..numbering import AMINO_ACIDS

UNPAIRED_BASE = "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/"
PAIRED_BASE = "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/"
_CATEGORIES = frozenset({"igk", "igl", "pos_igh_repertoire"})


def normalize_study(value):
    """Preserve the existing surname/year matching rule for cohort exclusions."""
    if value is not None and not isinstance(value, str):
        raise DataContractError("OAS study identity must be text.")
    text = (value or "").lower()
    surname = re.split(r"[ _,]", text.strip())[0]
    year = re.search(r"(?:19|20)\d\d", text)
    return surname + "_" + (year.group() if year else "")


@dataclass(frozen=True)
class Exclusions:
    studies: frozenset
    donors: frozenset
    denylist: dict

    @classmethod
    def from_records(cls, records, denylist):
        studies, donors = set(), set()
        for record in records:
            if not isinstance(record, dict):
                raise DataContractError("Cohort exclusion records must be mappings.")
            if record.get("category") not in _CATEGORIES:
                continue
            lineage = record.get("lineage_key")
            if not isinstance(lineage, str):
                raise DataContractError("Cohort exclusion records require a text lineage_key.")
            parts = lineage.split("::")
            study = normalize_study(parts[0])
            studies.add(study)
            if len(parts) >= 2:
                donors.add((study, parts[1]))
        return cls(frozenset(studies), frozenset(donors), denylist)


def _decompress_prefix(raw, max_bytes):
    output = bytearray()
    complete = False
    try:
        while raw:
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
            output.extend(decoder.decompress(raw, max_bytes - len(output) + 1))
            if len(output) > max_bytes:
                raise DataContractError("OAS CSV prefix exceeds its decompressed byte limit.")
            complete = decoder.eof
            raw = decoder.unused_data
    except zlib.error as exc:
        raise DataContractError("Invalid compressed OAS data.") from exc
    return output.decode("utf-8", "replace"), complete


def parse_unit(raw, *, max_bytes=32 << 20, row_limit=None):
    """Parse concatenated gzip members; discard an incomplete trailing CSV row."""
    text, complete = _decompress_prefix(raw, max_bytes)
    if not complete and not text.endswith("\n"):
        text = text.rsplit("\n", 1)[0] if "\n" in text else ""
    first, separator, body = text.partition("\n")
    if not separator:
        raise DataContractError("OAS prefix has no complete metadata line.")
    try:
        try:
            metadata = json.loads(first.strip())
        except json.JSONDecodeError:
            quoted = next(csv.reader([first], strict=True))
            if len(quoted) != 1:
                raise DataContractError("Invalid OAS metadata CSV field.")
            metadata = json.loads(quoted[0])
        if not isinstance(metadata, dict):
            raise DataContractError("OAS metadata must be a JSON mapping.")
        reader = csv.DictReader(io.StringIO(body), strict=True)
        rows = []
        for row in (reader if row_limit is None else islice(reader, row_limit)):
            # A short final record in a bounded prefix must not become a candidate.
            if None in row or any(value is None for value in row.values()):
                continue
            rows.append(row)
    except (json.JSONDecodeError, csv.Error, StopIteration) as exc:
        raise DataContractError("Cannot parse OAS metadata or CSV rows.") from exc
    return metadata, rows


def valid_subject(subject):
    if not isinstance(subject, str):
        return False
    text = subject.strip()
    if text.lower() in {"no", "unknown", "", "?", "na", "none", "nan", "n/a", "pool", "pooled"}:
        return False
    return not re.search(r"[;&+/]| and |,", text, re.IGNORECASE)


def candidates(metadata, rows, *, chain="Light", paired=False, diverse=False, cap=None):
    """Apply existing productivity, locus, alphabet, and length eligibility rules."""
    study, subject = metadata.get("Author") or "unknown", metadata.get("Subject") or "unknown"
    if diverse and not valid_subject(subject):
        return []
    suffix = "_light" if paired else ""
    kept = []
    for row in rows:
        if (row.get("productive" + suffix) != "T" or row.get("stop_codon" + suffix) == "T" or
                row.get("vj_in_frame" + suffix) != "T"):
            continue
        v_call = row.get("v_call" + suffix) or ""
        if chain == "Heavy" and not paired:
            category, group = ("pos_igh_repertoire", "positive_real") if v_call.startswith("IGHV") else (None, None)
        else:
            category = "igk" if v_call.startswith("IGKV") else "igl" if v_call.startswith("IGLV") else None
            group = "igk_igl"
        if category is None:
            continue
        sequence = (row.get("sequence_alignment_aa" + suffix) or "").replace(".", "").replace("-", "").replace("*", "").upper()
        if not 80 <= len(sequence) <= 170 or set(sequence) - AMINO_ACIDS:
            continue
        lineage = f"{study}::{subject}::{v_call}::{row.get('cdr3_aa' + suffix) or ''}"
        kept.append({"lineage_key": lineage, "aa": sequence, "category": category, "group": group,
                     "repository": "oas_paired" if paired else "oas", "study": study, "donor": subject})
        if cap is not None and len(kept) >= cap:
            break
    return kept
