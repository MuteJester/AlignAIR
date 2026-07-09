"""Phase 1 / Task 1: safe (no-pickle) reference serialization + canonical integrity hashes."""
import hashlib
import json

import GenAIRR.data as gd

from alignair.model_file import serialize
from alignair.reference.reference_set import ReferenceSet


def test_serializable_roundtrip_preserves_order_gapped_anchors():
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    ref2 = ReferenceSet.from_serializable(ref.to_serializable())
    for G in ("V", "D", "J"):
        a, b = ref.gene(G), ref2.gene(G)
        assert a.names == b.names                      # order == model-head index; must be exact
        assert a.sequences == b.sequences
        assert (a.anchors or {}) == (b.anchors or {})
        assert (a.gapped or {}) == (b.gapped or {})
    assert ref2.has_d == ref.has_d
    # the pieces plain FASTA lacks — proof the safe section carries them
    assert ref2.gene("V").gapped and ref2.gene("V").anchors and ref2.gene("J").anchors


def test_reference_json_bytes_roundtrip_no_pickle():
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    b = serialize.reference_to_json(ref)
    json.loads(b.decode("utf-8"))                      # it is plain JSON, no pickle
    ref2 = serialize.reference_from_json(b)
    assert ref2.gene("V").names == ref.gene("V").names
    assert ref2.gene("V").gapped == ref.gene("V").gapped


def test_allele_order_hash_deterministic_and_order_sensitive():
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    h1 = serialize.allele_order_sha256(ref)
    assert len(h1) == 64
    assert serialize.allele_order_sha256(ReferenceSet.from_serializable(ref.to_serializable())) == h1
    d = ref.to_serializable()                          # reorder V -> different hash (index guard)
    d["genes"]["V"]["names"] = list(reversed(d["genes"]["V"]["names"]))
    d["genes"]["V"]["sequences"] = list(reversed(d["genes"]["V"]["sequences"]))
    assert serialize.allele_order_sha256(ReferenceSet.from_serializable(d)) != h1


def test_reference_fasta_hash_matches_canonical_fasta():
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    assert serialize.reference_fasta_sha256(ref) == \
        hashlib.sha256(serialize.reference_fasta(ref).encode("utf-8")).hexdigest()


def test_allele_order_hash_has_empty_D_for_light_chain():
    ref = ReferenceSet.from_genotype({"V": {"IGKV1-1*01": "ACGT"}, "J": {"IGKJ1*01": "ACGT"}})
    payload = json.dumps({"V": ["IGKV1-1*01"], "D": [], "J": ["IGKJ1*01"]},
                         separators=(",", ":"), ensure_ascii=False)
    assert serialize.allele_order_sha256(ref) == hashlib.sha256(payload.encode("utf-8")).hexdigest()
