"""Stage-1 HIGH-SPECIFICITY structural prefilter: hard-rejects only decisive defects, defers everything
else (including divergent/mutated complete VHs and all complete-domain non-IGH inputs) to Stage-2."""
from __future__ import annotations
import pytest

from alignair.aa.product.applicability.completeness import structural_prefilter, RULESET_VERSION

pytestmark = pytest.mark.component

_VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDT"
       "AVYYCAKDLYYYGMDVWGQGTTVTVSS")
_LEADER = "MEFGLSWVFLVALLRGVL"                          # hydrophobic signal peptide
_CH1 = "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKV"


def test_complete_vh_defers_not_rejected():
    r = structural_prefilter(_VH)
    assert r.rejected is False and r.reason is None and r.ruleset_version == RULESET_VERSION


def test_divergent_fr4_still_defers():
    # WGQG -> GGQG is biological divergence, NOT insufficient evidence -> must defer, never hard-reject
    divergent = _VH.replace("WGQGTTVTVSS", "GGQGATLVTVSA")
    assert structural_prefilter(divergent).rejected is False


def test_reversed_vh_is_hard_rejected():
    assert structural_prefilter(_VH[::-1]).reason == "reversed_order"


def test_cdr3_only_and_j_only_are_too_short():
    assert structural_prefilter("AKDLYYYGMDV").reason == "too_short"
    assert structural_prefilter("WGQGTTVTVSS").reason == "too_short"


def test_v_only_fragment_DEFERS_to_stage2():
    # a >=90aa V-only fragment is NOT a decisive defect the prefilter should reject on a missing FR4 motif
    v_only = _VH[:_VH.index("WGQG")]
    assert len(v_only) >= 90
    assert structural_prefilter(v_only).rejected is False


def test_leader_plus_vh_is_hard_rejected():
    assert structural_prefilter(_LEADER + _VH).reason == "leader_or_constant_contamination"
    assert structural_prefilter(_LEADER + _VH).detail.get("kind") == "leader"


def test_vh_plus_constant_is_hard_rejected():
    r = structural_prefilter(_VH + _CH1)
    assert r.reason == "leader_or_constant_contamination"
    assert r.detail.get("kind") == "constant"


def test_light_chain_defers_as_a_complete_domain():
    kappa = ("DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYC"
             "QQSYSTPPTFGQGTKVEIK")
    assert structural_prefilter(kappa).rejected is False    # complete domain -> Stage-2 rejects the locus
