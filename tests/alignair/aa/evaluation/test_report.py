import pytest
import json
from alignair.aa.report import assemble_report, render_markdown, write_report

pytestmark = pytest.mark.component


def _toy_report():
    return assemble_report(
        meta={"dc": "HUMAN_IGH_OGRDB", "n": 200, "seed": 0, "db_hash": "deadbeefdeadbeef"},
        identifiability={"V": {"n_alleles": 198, "n_classes": 150,
                               "frac_alleles_uniquely_identifiable": 0.6},
                         "J": {"n_alleles": 7, "n_classes": 6,
                               "frac_alleles_uniquely_identifiable": 0.71}},
        length_stats={"aa_len_p50": 120, "aa_len_p95": 128, "n_records": 190},
        residue_stats={"frac_X": 0.0, "frac_stop": 0.0},
        boundary_stats={"V_end_cut_frac": 0.66, "J_start_cut_frac": 0.66},
        d_stats={"mean_owned_d_residues": 1.4, "frac_zero_owned_d": 0.35},
        coverage_curve={"V": [{"lo": 0, "hi": 60, "frac_uniquely_identifiable": 0.3}]},
    )


def test_assemble_report_has_schema_and_sections():
    r = _toy_report()
    assert r["schema"] == "alignair.aa.phase0.v1"
    for key in ("meta", "identifiability", "length_stats",
                "residue_stats", "boundary_stats", "d_stats", "coverage_curve"):
        assert key in r


def test_render_markdown_mentions_key_numbers():
    md = render_markdown(_toy_report())
    assert "Phase 0" in md
    assert "HUMAN_IGH_OGRDB" in md
    assert "identifiab" in md.lower()


def test_write_report_emits_json_and_md(tmp_path):
    out = write_report(_toy_report(), str(tmp_path / "phase0"))
    with open(f"{out}/report.json") as f:
        loaded = json.load(f)
    assert loaded["schema"] == "alignair.aa.phase0.v1"
    assert (tmp_path / "phase0" / "report.md").exists()
