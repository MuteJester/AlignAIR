"""Endpoint-specific Stage-2 roles: completeness (locus-agnostic, no IgSF) vs membership (human IGH only)."""
from __future__ import annotations
import pytest

from alignair.aa.product.applicability.roles import (
    completeness_role, membership_role, assign_roles)

pytestmark = pytest.mark.component


def _rec(cat, lk, rid="x"):
    return {"id": rid, "category": cat, "lineage_key": lk}


def test_completeness_is_locus_agnostic_but_excludes_igsf_and_uses_structural_as_validation():
    sp = set()
    assert completeness_role(_rec("igk", "k1"), sp) in ("completeness_fit", "completeness_validation")
    assert completeness_role(_rec("tra", "t1"), sp) in ("completeness_fit", "completeness_validation")
    assert completeness_role(_rec("vhh", "v1"), sp) in ("completeness_fit", "completeness_validation")
    assert completeness_role(_rec("igsf_vset", "s1"), sp) is None                 # excluded
    assert completeness_role(_rec("reversed_vh", "r1"), sp) == "completeness_validation"  # negative


def test_membership_fits_only_human_igh_and_treats_others_as_validation_or_challenge():
    sp = set()
    assert membership_role(_rec("igk", "k1"), sp) == "membership_validation"
    assert membership_role(_rec("igsf_vset", "s1"), sp) == "membership_validation"
    assert membership_role(_rec("humanized", "h1"), sp) == "membership_challenge"
    assert membership_role(_rec("design_human_framework", "d1"), sp) == "membership_challenge"
    assert membership_role(_rec("pos_igh_repertoire", "p1"), sp) in ("membership_fit", "membership_validation")
    assert membership_role(_rec("reversed_vh", "r1"), sp) is None                 # not a membership concern


def test_structural_parent_never_enters_a_fit_manifold():
    sp = {"parentX"}
    assert completeness_role(_rec("pos_igh_repertoire", "parentX"), sp) == "completeness_validation"
    assert membership_role(_rec("pos_igh_repertoire", "parentX"), sp) == "membership_validation"


def test_assign_roles_enforces_disjoint_fit_validation():
    # many lineages across endpoints; assign_roles asserts no fit/val overlap and no parent-in-fit
    recs = ([_rec("pos_igh_repertoire", f"igh{i}", f"a{i}") for i in range(40)]
            + [_rec("igk", f"k{i}", f"b{i}") for i in range(20)]
            + [_rec("reversed_vh", f"a{i}", f"c{i}") for i in range(5)]      # parents overlap IGH lineages
            + [_rec("igsf_vset", f"s{i}", f"d{i}") for i in range(10)])
    out = assign_roles(recs)                                                  # asserts inside
    assert out["counts"].get("completeness_fit", 0) > 0
    assert out["counts"].get("membership_fit", 0) > 0
