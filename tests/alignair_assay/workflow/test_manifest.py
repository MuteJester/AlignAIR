"""The run directory's manifest, shared by ``all`` and ``real`` (``workflow.manifest``)."""
from __future__ import annotations

import pytest

from alignair_assay.workflow.manifest import ABSENT, changed_fields, describe_changes, flat, manifest_record

pytestmark = pytest.mark.integration

IDENTITY = {"donors": ["W-97"], "cap": 500, "settings": {"igblast": {"version": "1.22.0", "threads": 8}},
            "images": {"samtools": "quay.io/biocontainers/samtools:1.23.1@sha256:23cd"}}


def test_flat_names_each_tool_setting_and_image():
    assert flat(IDENTITY) == {"donors": ["W-97"], "cap": 500, "igblast.version": "1.22.0", "igblast.threads": 8,
                              "images.samtools": "quay.io/biocontainers/samtools:1.23.1@sha256:23cd"}
    assert flat({"seed": 7}) == {"seed": 7}


def test_changed_fields_name_what_differs_and_what_one_side_lacks():
    current = {**IDENTITY, "cap": 50000, "settings": {"igblast": {"version": "1.22.0", "threads": 4}},
               "reference_gapped_digest": "9f"}
    changed = changed_fields(IDENTITY, current)
    assert changed == ["cap", "igblast.threads", "reference_gapped_digest"]
    assert describe_changes(IDENTITY, current, changed) == (
        "cap 500 -> 50000; igblast.threads 8 -> 4; reference_gapped_digest (absent) -> 9f")
    # top-level keys left out on both sides
    assert changed_fields(IDENTITY, current, ignore=("settings", "cap")) == ["reference_gapped_digest"]
    assert changed_fields(IDENTITY, dict(IDENTITY)) == []
    assert flat(IDENTITY).get("missing", ABSENT) is ABSENT


def test_manifest_record_holds_the_configuration_and_where_it_ran():
    record = manifest_record(kind="k", version="1", config={"cap": 500}, identity=IDENTITY, config_hash="h",
                             split="dev", replaced=None, warnings=["w"])
    assert list(record)[:6] == ["kind", "version", "config", "identity", "config_hash", "split"]
    assert {"host", "git", "genairr", "created_utc", "replaced", "warnings"} <= set(record)
    assert set(record["host"]) == {"hostname", "platform", "python", "logical_cpus"}
    assert set(record["git"]) == {"git_commit", "git_dirty"} and "genairr_version" in record["genairr"]
    assert record["warnings"] == ["w"] and record["replaced"] is None
