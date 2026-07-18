"""A fixed-head model's classification indices are tied to the embedded reference, so
a caller-supplied reference must match it exactly — same alleles, same order, AND the same germline
sequences / gapped / anchors (equal names with altered sequences must be rejected)."""
from types import SimpleNamespace

import pytest

from alignair.api import _assert_reference_matches


def _gene(names, seqs=None):
    seqs = list(seqs) if seqs is not None else [f"ACGT{i}" for i in range(len(names))]
    return SimpleNamespace(names=list(names), sequences=seqs, gapped=None, anchors=None)


def _ref(**genes):
    """A minimal ReferenceSet-like stub. Each value is names, or a (names, sequences) tuple."""
    g = {G: (_gene(*spec) if isinstance(spec, tuple) else _gene(spec)) for G, spec in genes.items()}
    return SimpleNamespace(genes=g, gene=lambda G, _g=g: _g[G.upper()])


def test_matching_reference_passes():
    emb = _ref(V=["V1", "V2"], J=["J1"])
    _assert_reference_matches(_ref(V=["V1", "V2"], J=["J1"]), emb)   # identical names + sequences -> ok


def test_reordered_reference_rejected():
    emb = _ref(V=["V1", "V2"], J=["J1"])
    with pytest.raises(ValueError, match="set/order differs|does not match"):
        _assert_reference_matches(_ref(V=["V2", "V1"], J=["J1"]), emb)   # same set, different order


def test_extra_or_missing_allele_rejected():
    emb = _ref(V=["V1", "V2"], J=["J1"])
    with pytest.raises(ValueError, match="does not match|differs"):
        _assert_reference_matches(_ref(V=["V1", "V2", "V3"], J=["J1"]), emb)   # novel allele appended


def test_mismatched_gene_set_rejected():
    emb = _ref(V=["V1"], D=["D1"], J=["J1"])                        # heavy (has D)
    with pytest.raises(ValueError, match="does not match|differs"):
        _assert_reference_matches(_ref(V=["V1"], J=["J1"]), emb)    # light ref for a heavy head


def test_same_names_but_different_sequences_rejected():
    """The subtle case: identical allele names/order but ALTERED germline sequences must be rejected
    (they would silently align/junction/score against the wrong biology)."""
    emb = _ref(V=(["V1", "V2"], ["ACGTACGT", "TTTTGGGG"]), J=(["J1"], ["CCCC"]))
    caller = _ref(V=(["V1", "V2"], ["ACGTACGT", "AAAACCCC"]), J=(["J1"], ["CCCC"]))   # V2 seq altered
    with pytest.raises(ValueError, match="germline sequences|anchors differ"):
        _assert_reference_matches(caller, emb)
