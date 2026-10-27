"""Protein-equivalence classes: alleles with identical in-frame translation collapse to one
class (they are indistinguishable in protein space). The model predicts the class; inference
expands it back to the nucleotide-allele ambiguity set. Stamped with a germline-DB hash."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable
from collections.abc import Mapping, Sequence

from ..errors import ModelError


def db_hash(name_seq_pairs: Iterable[tuple]) -> str:
    """Order-independent 16-hex digest of the germline DB (sorted (name, seq) pairs)."""
    h = hashlib.sha1()
    for name, seq in sorted(name_seq_pairs):
        h.update(str(name).encode()); h.update(b"\0")
        h.update(str(seq).upper().encode()); h.update(b"\n")
    return h.hexdigest()[:16]


@dataclass
class EquivalenceClassMap:
    gene: str
    db_hash: str
    allele_class: dict          # allele name -> class id (int)
    class_members: dict         # class id (int) -> sorted [allele names]

    def validate(self) -> None:
        """Class IDs must be contiguous and the allele/member maps must be exact inverses."""
        def invalid(field, reason):
            raise ModelError(f"invalid equivalence label map: {reason}",
                             context={"stage": "reference_labels", "gene": self.gene, "field": field,
                                      "reason": reason})

        if (not isinstance(self.gene, str) or not self.gene or not isinstance(self.db_hash, str)
                or not isinstance(self.allele_class, Mapping) or not isinstance(self.class_members, Mapping)):
            invalid("metadata", "gene/hash metadata and mapping fields must have the expected types")
        if (any(type(key) is not int for key in self.class_members)
                or set(self.class_members) != set(range(len(self.class_members)))):
            invalid("class_members", "class IDs must be contiguous integers starting at zero")
        inverse = {}
        for label, members in self.class_members.items():
            if not isinstance(members, Sequence) or isinstance(members, str) or not members:
                invalid("class_members", "each class must contain a nonempty sequence of allele names")
            for name in members:
                if not isinstance(name, str) or not name or name in inverse:
                    invalid("class_members", "allele members must be distinct nonempty names")
                inverse[name] = label
        if (any(type(label) is not int for label in self.allele_class.values())
                or self.allele_class != inverse):
            invalid("allele_class", "allele and member maps must be exact inverses with integer class IDs")

    def class_of(self, name: str) -> int:
        return self.allele_class[name]

    def ambiguity_set(self, class_id: int) -> list:
        return list(self.class_members[class_id])

    def n_classes(self) -> int:
        return len(self.class_members)

    def to_json(self) -> dict:
        self.validate()
        return {
            "schema": "alignair.aa.equivalence.v1",
            "gene": self.gene,
            "db_hash": self.db_hash,
            "allele_class": dict(self.allele_class),
            "class_members": {str(k): list(v) for k, v in self.class_members.items()},
        }

    @classmethod
    def from_json(cls, d: dict) -> "EquivalenceClassMap":
        if not isinstance(d, Mapping) or d.get("schema") not in (None, "alignair.aa.equivalence.v1"):
            raise ModelError("unsupported equivalence label-map schema", context={"stage": "reference_labels"})
        try:
            members = {}
            for key, value in d["class_members"].items():
                if (isinstance(key, bool) or not isinstance(key, (int, str))
                        or isinstance(key, str) and str(int(key)) != key):
                    raise ValueError("class IDs must be canonical integers")
                members[int(key)] = value
            if len(members) != len(d["class_members"]):
                raise ValueError("duplicate serialized class ID")
            result = cls(d["gene"], d["db_hash"], d["allele_class"], members)
            result.validate()
            return cls(result.gene, result.db_hash, dict(result.allele_class),
                       {key: list(value) for key, value in members.items()})
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            if isinstance(exc, ModelError):
                raise
            raise ModelError("malformed equivalence label map", context={"stage": "reference_labels"}) from exc


def build_equivalence_classes(gene: str, aa_by_name: dict, db_hash: str) -> EquivalenceClassMap:
    """Assign a class id per allele; alleles sharing an identical translation share a class.

    Class ids are assigned in order of first appearance over ``sorted(aa_by_name)`` for
    determinism.
    """
    by_seq: dict[str, int] = {}
    members: dict[int, list] = {}
    allele_class: dict[str, int] = {}
    for name in sorted(aa_by_name):
        aa = aa_by_name[name]
        cid = by_seq.setdefault(aa, len(by_seq))
        members.setdefault(cid, []).append(name)
        allele_class[name] = cid
    return EquivalenceClassMap(gene, db_hash, allele_class,
                               {cid: sorted(m) for cid, m in members.items()})
