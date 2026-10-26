"""The frozen, versioned domain-prep procedures (motif_fr4_junction_v1 / motif_cys_wgxg_v1)."""
from __future__ import annotations
import pytest

from alignair.aa.product.applicability.domain_trim import (
    MAX_VH, trim_vh, rebuild_cdr3_j, TRIM_PROCEDURE, REBUILD_PROCEDURE)

pytestmark = pytest.mark.component

_VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDT"
       "AVYYCAKDLYYYGMDVWGQGTTVTVSS")
_CH1 = "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKV"


def test_procedure_versions_are_frozen_strings():
    assert TRIM_PROCEDURE == "motif_fr4_junction_v1"
    assert REBUILD_PROCEDURE == "motif_cys_wgxg_v1"


def test_trim_cuts_a_fab_heavy_chain_back_to_the_vh_domain():
    trimmed, end = trim_vh(_VH + _CH1)
    assert trimmed == _VH                       # constant domain removed exactly at the FR4 junction
    assert end == len(_VH)
    assert trimmed.endswith("WGQGTTVTVSS")


def test_trim_returns_none_when_no_fr4_anchor():
    assert trim_vh("MKT" * 40) is None          # a non-antibody protein, no WGxG...VTVSS


def test_trim_drops_out_of_range_results():
    # anchor present but a huge N-terminal makes the trimmed domain implausibly long -> dropped
    assert trim_vh("A" * (MAX_VH + 50) + _VH) is None


def test_rebuild_extracts_cdr3_and_j_from_a_vh():
    cdr3, j = rebuild_cdr3_j(_VH)
    assert cdr3 == "AKDLYYYGMDV"                 # between the FR3 Cys and the FR4 Trp
    assert j == "WGQGTTVTVSS"                    # FR4 onward (the J-only fragment)
    assert _VH.endswith(cdr3 + j)


def test_rebuild_returns_none_without_both_anchors():
    assert rebuild_cdr3_j("EVQLVESGGGLVQPGG") is None   # no Cys / no WGxG
