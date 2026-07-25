"""Stage-1 completeness gate — rules verified from the domain contract on synthetic constructs (not
tuned on cohort negatives)."""
from __future__ import annotations

from alignair.aa.product.applicability.completeness import check_completeness, RULESET_VERSION

_VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDT"
       "AVYYCAKDLYYYGMDVWGQGTTVTVSS")
_LEADER = "MEFGLSWVFLVALLRGVQC"                          # signal peptide (~19 aa)
_CH1 = "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKV"


def test_complete_vh_passes():
    r = check_completeness(_VH)
    assert r.passed and r.reason is None and r.ruleset_version == RULESET_VERSION


def test_reversed_vh_is_rejected_as_reversed_order():
    assert check_completeness(_VH[::-1]).reason == "reversed_order"


def test_cdr3_only_is_too_short():
    assert check_completeness("AKDLYYYGMDV").reason == "too_short"


def test_j_only_fragment_is_too_short():
    assert check_completeness("WGQGTTVTVSS").reason == "too_short"


def test_v_only_without_fr4_is_missing_j_anchor():
    v_only = _VH[:_VH.index("WGQG")]                     # drop FR4 -> no J anchor, still long enough
    assert len(v_only) >= 90
    assert check_completeness(v_only).reason == "missing_j_anchor"


def test_leader_plus_vh_is_contamination():
    r = check_completeness(_LEADER + _VH)
    assert r.reason == "leader_or_constant_contamination"


def test_vh_plus_constant_is_rejected():
    # a VH + CH1 constant domain must be REJECTED (its J-region is buried before the constant, so the
    # locus-agnostic gate rejects it as missing_j_anchor or as constant contamination — either is a reject)
    r = check_completeness(_VH + _CH1)
    assert not r.passed
    assert r.reason in ("leader_or_constant_contamination", "missing_j_anchor")


def test_a_light_chain_is_a_complete_domain_and_passes_stage1():
    # locus-agnostic: a kappa light-chain V-domain is a COMPLETE Ig fold -> passes Stage-1 (Stage-2, the
    # membership score, is what rejects non-IGH loci). It must NOT be rejected here as incomplete.
    kappa = ("DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYC"
             "QQSYSTPPTFGQGTKVEIK")
    assert check_completeness(kappa).passed


def test_passing_is_not_tuned_on_a_specific_family():
    # a different-family human VH (IGHV1-ish) also passes on the same domain-contract rules
    vh1 = ("QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDT"
           "AVYYCARDRGYYYGMDVWGQGTTVTVSS")
    assert check_completeness(vh1).passed
