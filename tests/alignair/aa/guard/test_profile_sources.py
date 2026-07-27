from pathlib import Path

import pytest

from alignair.aa.guard.profiles import (NoPublishableProfileSource, PROFILE_SOURCES,
                                        RECEPTOR_GROUPS, resolve_production_bank, source_matrix,
                                        synthetic_hydration_resolution)

LOCK = Path("requirements/aa-guard-build.lock")


def test_all_required_groups_have_a_source():
    assert RECEPTOR_GROUPS | {"igsf_vset"} <= set(PROFILE_SOURCES)
    for group in RECEPTOR_GROUPS:
        assert {"v", "j"} <= set(PROFILE_SOURCES[group].regions), group


def test_imgt_is_comparator_only_until_cleared():
    for s in PROFILE_SOURCES.values():
        if "imgt" in s.name.lower():
            assert not s.redistributable and s.role == "comparator_only"


RESOLVED_TODAY = {"human_igh", "human_igk", "human_igl", "igsf_vset"}
UNRESOLVED_TODAY = {"tra", "trb", "trg", "trd", "murine_vh", "vhh"}


def test_only_released_cc0_sources_are_production():
    # human IG (released AIRR-C sets) + Pfam igsf are the only CC0 redistributable production sources today
    for group in RESOLVED_TODAY:
        s = PROFILE_SOURCES[group]
        assert s.role == "production" and s.redistributable, group
        assert s.license.upper().startswith("CC0"), (group, s.license)
    # TR/mouse/camelid resolve only to IMGT comparator_only (not redistributable)
    for group in UNRESOLVED_TODAY:
        s = PROFILE_SOURCES[group]
        assert s.role == "comparator_only" and not s.redistributable, group


def test_unresolved_required_profiles_trigger_stop():
    # owner directive: any unresolved required profile -> STOP_NO_PUBLISHABLE_PROFILE_SOURCE.
    with pytest.raises(NoPublishableProfileSource) as ei:
        resolve_production_bank()
    assert set(ei.value.unresolved) == UNRESOLVED_TODAY
    assert "STOP_NO_PUBLISHABLE_PROFILE_SOURCE" in str(ei.value)


def test_source_matrix_reports_stop_decision():
    m = source_matrix()
    assert m["resolution"]["decision"] == "STOP_NO_PUBLISHABLE_PROFILE_SOURCE"
    assert set(m["resolution"]["unresolved"]) == UNRESOLVED_TODAY


def test_user_hydration_binds_exact_inputs_and_output():
    r = synthetic_hydration_resolution()
    assert r.input_sha256 and r.bank_sha256 and r.hydration_script_sha256
    assert r.offline_after_hydration is True and r.failure_semantics


def test_build_tool_lock_is_version_and_hash_pinned():
    text = LOCK.read_text()
    assert "pyhmmer==0.12.1" in text and "anarcii==2.0.8" in text
    assert "--hash=sha256:" in text
