"""Protein-equivalence classes: alleles with identical in-frame translation collapse to one
class (they are indistinguishable in protein space). The model predicts the class; inference
expands it back to the nucleotide-allele ambiguity set. Stamped with a germline-DB hash."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


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

    def class_of(self, name: str) -> int:
        return self.allele_class[name]

    def ambiguity_set(self, class_id: int) -> list:
        return list(self.class_members[class_id])

    def n_classes(self) -> int:
        return len(self.class_members)

    def to_json(self) -> dict:
        return {
            "schema": "alignair.aa.equivalence.v1",
            "gene": self.gene,
            "db_hash": self.db_hash,
            "allele_class": dict(self.allele_class),
            "class_members": {str(k): list(v) for k, v in self.class_members.items()},
        }

    @classmethod
    def from_json(cls, d: dict) -> "EquivalenceClassMap":
        return cls(
            gene=d["gene"],
            db_hash=d["db_hash"],
            allele_class=dict(d["allele_class"]),
            class_members={int(k): list(v) for k, v in d["class_members"].items()},
        )


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
