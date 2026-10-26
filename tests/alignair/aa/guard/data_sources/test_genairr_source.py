import itertools
import json

import pytest

from alignair.aa.guard.data_sources.genairr import BUDGET, START_SEED, WANT, collect_genairr, main
from alignair.errors import DataContractError

pytestmark = pytest.mark.component


def _record(**overrides):
    return {"productive": True, "sequence_aa": "a" * 105, "junction_start": 0,
            "v_call": "IGHV1*01,IGHV2*01", "j_call": "IGHJ1*01", **overrides}


def test_filters_and_seed_ids_preserve_simulator_order(tmp_path):
    stream = [
        _record(productive=False), _record(sequence_aa=""), _record(sequence_aa="A" * 100 + "*"),
        _record(junction_start=None), _record(sequence_aa="A" * 99), _record(),
    ]
    result = collect_genairr({}, store=tmp_path / "data", census_dir=tmp_path / "reports", record_stream=iter(stream))
    assert result.kept[0]["seed"] == START_SEED + 5
    assert result.kept[0]["id"] == f"pos_sim_{START_SEED + 5}"
    assert result.kept[0]["aa"] == "A" * 105
    assert result.kept[0]["v_call"] == "IGHV1*01"
    assert result.census["rejected"] == dict.fromkeys(["not_productive", "no_aa", "stop_codon", "no_frame", "bad_len"], 1)
    assert result.census["provenance"]["seed_range"] == [START_SEED, START_SEED + 5]
    assert json.loads((tmp_path / "data" / "genairr_sim.normalized.json").read_text()) == result.kept
    assert json.loads((tmp_path / "reports" / "genairr_sim.census.json").read_text()) == result.census


def test_target_is_applied_before_denylist_without_replacement(tmp_path):
    consumed = []

    def stream():
        for index in range(WANT + 2):
            consumed.append(index)
            yield _record()

    result = collect_genairr({"lineages": {f"genairr_fresh_seed_{START_SEED}"}}, store=tmp_path,
                             census_dir=tmp_path / "reports", record_stream=stream())
    assert len(consumed) == WANT
    assert len(result.kept) == WANT - 1
    assert result.census["denylist_dropped"]["lineage"] == 1


def test_injected_stream_cannot_exceed_original_scan_budget(tmp_path):
    result = collect_genairr({}, store=tmp_path, census_dir=tmp_path / "reports",
                             record_stream=itertools.repeat(_record(productive=False)))
    assert result.census["generated_scanned"] == BUDGET
    assert result.kept == []


@pytest.mark.parametrize("record", [None, _record(sequence_aa=["A"] * 105)])
def test_malformed_simulator_records_raise_without_writing_outputs(tmp_path, record):
    with pytest.raises(DataContractError):
        collect_genairr({}, store=tmp_path, census_dir=tmp_path, record_stream=iter([record]))
    assert not list(tmp_path.iterdir())


def test_cli_rejects_missing_replay_input_before_simulation(tmp_path, capsys):
    result = main(["--error-format", "json", "--cleaned", str(tmp_path / "absent.jsonl"),
                   "--fingerprint", str(tmp_path / "absent.json"), "--phase3-gold", str(tmp_path / "absent.tsv"),
                   "--store", str(tmp_path / "data"), "--census-dir", str(tmp_path / "reports")])
    assert result == 1
    assert json.loads(capsys.readouterr().err)["code"] == "ALIGNAIR_IO_FAILED"
