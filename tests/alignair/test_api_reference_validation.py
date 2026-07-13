"""P0-1: a fixed-head model's classification indices are tied to the embedded allele order, so a
caller-supplied reference must match it exactly — otherwise output columns would be mislabeled."""
from types import SimpleNamespace

import pytest

from alignair.api import _assert_reference_matches


def _ref(**genes):
    """A minimal ReferenceSet-like stub: {GENE: names}."""
    g = {G: SimpleNamespace(names=list(names)) for G, names in genes.items()}
    return SimpleNamespace(genes=g, gene=lambda G, _g=g: _g[G.upper()])


def test_matching_reference_passes():
    emb = _ref(V=["V1", "V2"], J=["J1"])
    _assert_reference_matches(_ref(V=["V1", "V2"], J=["J1"]), emb)   # identical order -> ok


def test_reordered_reference_rejected():
    emb = _ref(V=["V1", "V2"], J=["J1"])
    with pytest.raises(ValueError, match="does not match|embedded|order"):
        _assert_reference_matches(_ref(V=["V2", "V1"], J=["J1"]), emb)   # same set, different order


def test_extra_or_missing_allele_rejected():
    emb = _ref(V=["V1", "V2"], J=["J1"])
    with pytest.raises(ValueError, match="does not match|embedded"):
        _assert_reference_matches(_ref(V=["V1", "V2", "V3"], J=["J1"]), emb)   # novel allele appended


def test_mismatched_gene_set_rejected():
    emb = _ref(V=["V1"], D=["D1"], J=["J1"])                        # heavy (has D)
    with pytest.raises(ValueError, match="does not match|embedded|gene"):
        _assert_reference_matches(_ref(V=["V1"], J=["J1"]), emb)    # light ref for a heavy head
