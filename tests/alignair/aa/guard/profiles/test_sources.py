import pytest
from pathlib import Path

from alignair.aa.guard.profiles import (IMGT_CLEARANCE, PROFILE_SOURCES, RECEPTOR_GROUPS,
                                        resolve_production_bank, source_matrix,
                                        synthetic_hydration_resolution)

pytestmark = pytest.mark.component

LOCK = Path("requirements/aa-guard-build.lock")
CC0_GROUPS = {"human_igh", "human_igk", "human_igl", "igsf_vset"}
IMGT_GROUPS = {"tra", "trb", "trg", "trd", "murine_vh", "camelid_igh"}


def test_all_required_groups_have_a_source():
    assert RECEPTOR_GROUPS | {"igsf_vset"} <= set(PROFILE_SOURCES)
    for group in RECEPTOR_GROUPS:
        assert {"v", "j"} <= set(PROFILE_SOURCES[group].regions), group


def test_cc0_sources_are_production_without_attribution():
    for group in CC0_GROUPS:
        s = PROFILE_SOURCES[group]
        assert s.role == "production" and s.redistributable, group
        assert s.license.upper().startswith("CC0") and s.attribution == "", group


def test_imgt_cleared_as_cc_by_production_with_attribution():
    # owner cleared IMGT to CC BY-4.0 -> production + redistributable, but attribution is REQUIRED
    for group in IMGT_GROUPS:
        s = PROFILE_SOURCES[group]
        assert "imgt" in s.name.lower()
        assert s.role == "production" and s.redistributable, group
        assert s.license == "CC-BY-4.0" and s.attribution, group
    # the clearance is recorded with auditable evidence + full obligations
    assert IMGT_CLEARANCE["license"] == "CC-BY-4.0"
    assert IMGT_CLEARANCE["evidence_url"] and IMGT_CLEARANCE["evidence_quote"]
    assert IMGT_CLEARANCE["relicensing_date"] == "2026-07-01"
    assert IMGT_CLEARANCE["evidence_archive"].endswith("license_evidence.json")
    assert "pyhmmer" in IMGT_CLEARANCE["usage_distinction"] and "not" in IMGT_CLEARANCE["usage_distinction"].lower()
    ob = IMGT_CLEARANCE["obligations"]
    assert ob["attribution"] and ob["modification_notice"] and ob["no_endorsement"]
    assert {"shipped bundle NOTICE", "model card", "profile manifest"} == set(ob["placement"])


def test_all_required_profiles_resolve_to_go():
    res = resolve_production_bank()
    assert res.mode == "redistributable_bundle"
    assert res.all_required_profiles_resolved
    assert set(res.resolutions) == RECEPTOR_GROUPS | {"igsf_vset"}
    # the IMGT CC BY-4.0 attribution must be surfaced for the shipped bundle to emit
    assert any("IMGT" in a for a in res.attributions)


def test_source_matrix_reports_go_decision():
    m = source_matrix()
    assert m["resolution"]["decision"] == "GO"
    assert m["resolution"]["all_required_profiles_resolved"] is True
    assert any("IMGT" in a for a in m["resolution"]["attributions"])


def test_user_hydration_binds_exact_inputs_and_output():
    r = synthetic_hydration_resolution()
    assert r.input_sha256 and r.bank_sha256 and r.hydration_script_sha256
    assert r.offline_after_hydration is True and r.failure_semantics


def test_build_tool_lock_is_version_and_hash_pinned():
    text = LOCK.read_text()
    assert "pyhmmer==0.12.1" in text and "anarcii==2.0.8" in text
    assert "--hash=sha256:" in text
