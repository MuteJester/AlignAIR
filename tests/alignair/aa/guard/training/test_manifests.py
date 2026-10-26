import copy
import hashlib
import json

import pytest

from alignair.aa.guard.train_data import partition_by_parent_lineage
from alignair.aa.guard.loco import LOCO_FOLDS, masked_profiles_for
from alignair.aa.guard.training import cli, manifests
from alignair.errors import ArtifactIntegrityError, ConfigurationError, DataContractError, InputError

pytestmark = pytest.mark.component


@pytest.fixture
def inputs():
    records = [{"id": str(index), "aa": "A" * (index + 10), "lineage_key": str(index)} for index in range(16)]
    records.append({"id": "child", "aa": "ACDEFG", "lineage_key": "child", "parent_lineage": "0"})
    settings = {
        "candidates": ["C1", "C2"], "seeds": [0, 7],
        "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.01},
        "schedule": {"name": "cosine", "warmup_steps": 1},
        "training": {"max_steps": 4, "batch_size": 4, "eval_every": 2,
                     "early_stop_metric": "dev_total_loss", "early_stop_patience": 6,
                     "min_delta": 0.001, "grad_clip": 1.0, "amp": False},
        "loss_weights": {"complete": 1.0, "app": 1.0, "oe": 0.5, "contrast": 0.5, "source": 0.5},
        "energy_margins": {"m_in": 5.0, "m_out": 10.0}, "temperatures": {"supcon": 0.1},
        "grl_lambda": 1.0, "oe": {"policy": "train-only shuffle", "cap_fraction": 0.15},
        "mining": {"rounds": 0, "policy": "disabled"},
        "production_seed_rule": "min_dev_total_loss, tie -> lowest seed (dev-only; raises on any fresh field)",
    }
    return dict(records=records, settings=settings, bank_checksum="fixture-bank", checkpoint_dir="models",
                forbidden_sequence_hashes={hashlib.sha256(b"VVVV").hexdigest()})


def test_sweep_binds_existing_parent_partition_and_copies_only_settings(inputs):
    settings = copy.deepcopy(inputs["settings"])
    inputs["settings"]["train_partition_hash"] = "stale"
    manifest = manifests.build_sweep_manifest(**inputs)
    train, dev, partition_hash = partition_by_parent_lineage(inputs["records"], seed=42, dev_frac=0.2)
    assert manifest["train_partition_hash"] == partition_hash
    assert manifest["n_train"] == len(train) and manifest["n_dev"] == len(dev)
    assert manifest["expected_runs"] == 4
    assert all(manifest[key] == value for key, value in settings.items())
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    assert manifest["manifest_sha256"] == hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    assert manifest["forbidden_fresh_seqhashes_count"] == 1
    inputs["settings"]["optimizer"]["lr"] = 99
    assert manifest["optimizer"]["lr"] == 0.001


def test_fresh_overlap_is_detected_by_sequence_even_with_a_different_record_id(inputs, tmp_path):
    inputs["records"][0]["aa"] = "VVVV"
    out = tmp_path / "study" / "manifest.json"
    with pytest.raises(ArtifactIntegrityError, match="overlap"):
        manifests.freeze_sweep_manifest(**inputs, output_path=out)
    assert not out.parent.exists()


def test_existing_manifest_is_preserved_byte_for_byte(inputs, tmp_path):
    out = tmp_path / "manifest.json"
    manifest = manifests.freeze_sweep_manifest(**inputs, output_path=out)
    before = out.read_bytes()
    assert json.loads(before) == manifest
    with pytest.raises(ArtifactIntegrityError, match="already exists"):
        manifests.freeze_sweep_manifest(**inputs, output_path=out)
    assert out.read_bytes() == before


def test_duplicate_record_ids_cannot_produce_an_ambiguous_partition_identity(inputs):
    inputs["records"][1]["id"] = inputs["records"][0]["id"]
    with pytest.raises(DataContractError, match="unique"):
        manifests.build_sweep_manifest(**inputs)


@pytest.mark.parametrize("key,value", [("seeds", [0, 0]), ("candidates", ["C0"]),
                                       ("production_seed_rule", "fresh_accuracy"), ("optimizer", {"lr": float("nan")})])
def test_invalid_program_or_seed_rule_cannot_be_frozen(inputs, key, value):
    inputs["settings"][key] = value
    with pytest.raises(ConfigurationError):
        manifests.build_sweep_manifest(**inputs)


def _cli_inputs(tmp_path, inputs):
    settings, records, bank = [tmp_path / name for name in ("settings.json", "records.jsonl", "bank.json")]
    settings.write_text(json.dumps(inputs["settings"]))
    records.write_text("\n".join(json.dumps(record) for record in inputs["records"]))
    bank.write_text(json.dumps({"bank_checksum": inputs["bank_checksum"]}))
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "fixture.normalized.json").write_text('[{"aa":"VVVV"}]')
    out = tmp_path / "study" / "manifest.json"
    args = ["--error-format", "json", "freeze-sweep", "--settings", str(settings),
            "--records", str(records), "--profile-manifest", str(bank), "--fresh-sources", str(fresh),
            "--checkpoints", "models", "--out", str(out)]
    return args, out, fresh


def test_package_command_freezes_explicit_inputs_without_loading_profiles(inputs, tmp_path, monkeypatch, capsys):
    from alignair.aa.guard import profiles

    def refuse(*args, **kwargs):
        pytest.fail("Freezing training settings must not load profile models.")

    monkeypatch.setattr(profiles, "load_bank", refuse)
    args, out, _ = _cli_inputs(tmp_path, inputs)
    assert cli.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["expected_runs"] == 4
    assert json.loads(out.read_text()) == manifests.build_sweep_manifest(**inputs)


@pytest.mark.parametrize("content", [None, '{"broken"', '{}', '[{"id":"missing-sequence"}]'])
def test_missing_or_invalid_exclusion_sources_stop_before_writing(inputs, tmp_path, capsys, content):
    args, out, fresh = _cli_inputs(tmp_path, inputs)
    path = fresh / "fixture.normalized.json"
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    assert cli.main(args) != 0
    output = capsys.readouterr()
    assert not output.out and json.loads(output.err)["code"] == DataContractError.code
    assert not out.parent.exists()


@pytest.fixture
def loco_inputs(inputs):
    groups = [("positive_real", "real"), ("positive_sim", "sim"), ("igk_igl", "igk"),
              ("tcr", "tra"), ("vhh_nonhuman", "vhh"), ("igsf", "igsf")]
    records = [{"id": f"{group}-{index}", "aa": "QVQL" + "A" * (index + offset + 10),
                "group": group, "category": category, "lineage_key": f"{group}-{index}"}
               for offset, (group, category) in enumerate(groups) for index in range(12)]
    sweep = manifests.build_sweep_manifest(**{**inputs, "records": records})
    results = {f"{cid}.{seed}": {"cid": cid, "seed": seed, "status": "ok", "dev_total_loss": loss}
               for cid in sweep["candidates"] for seed, loss in ((7, 0.5), (0, 0.5))}
    return dict(records=records, training_manifest=sweep, run_results=results,
                bank_checksum=inputs["bank_checksum"], checkpoint_dir="fold-models")


def test_loco_manifest_restricts_the_existing_partition_and_is_consumable_by_runner(loco_inputs, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from alignair.aa.guard.training import studies

    manifest = manifests.build_loco_manifest(**loco_inputs)
    assert manifest["frozen_seeds"] == {"C1": 0, "C2": 0}  # deterministic tie rule, regardless of result order
    train, dev, _ = partition_by_parent_lineage(loco_inputs["records"], seed=42, dev_frac=0.2)
    for group, fold in manifest["folds"].items():
        assert fold["masked_profiles"] == sorted(masked_profiles_for(group))
        assert fold["n_train_after_exclusion"] == sum(record["group"] != group for record in train)
        assert fold["n_dev_after_exclusion"] == sum(record["group"] != group for record in dev)
    calls = []

    def train_one(cid, seed, train, dev, *args, **kwargs):
        calls.append((kwargs["extra_provenance"]["fold"], train, dev, seed))
        return dict(tag=kwargs["run_tag"], cid=cid, seed=seed, status="ok", dev_total_loss=0.5)

    monkeypatch.setattr(studies, "precompute_features", lambda *args, **kwargs: {})
    monkeypatch.setattr(studies, "train_one", train_one)
    results = studies.run_loco(loco_inputs["training_manifest"], manifest, loco_inputs["records"],
                               SimpleNamespace(checksum=lambda: loco_inputs["bank_checksum"]),
                               artifact_dir=tmp_path / "reports", cache_dir=tmp_path / "cache")
    assert len(results) == len(LOCO_FOLDS) * 2
    for group, fold_train, fold_dev, seed in calls:
        assert fold_train == [record for record in train if record["group"] != group]
        assert fold_dev == [record for record in dev if record["group"] != group]
        assert seed == 0


@pytest.mark.parametrize("change", ["hash", "partition", "bank", "classes", "features"])
def test_loco_input_drift_stops_before_writing(loco_inputs, tmp_path, change):
    sweep = loco_inputs["training_manifest"]
    if change == "hash":
        sweep["optimizer"]["lr"] = 99
    elif change == "partition":
        loco_inputs["records"].pop()
    elif change == "bank":
        loco_inputs["bank_checksum"] = "other-bank"
    else:
        sweep["class_map_hash" if change == "classes" else "feature_order"] = "drifted"
        sweep["manifest_sha256"] = manifests._sha({key: value for key, value in sweep.items() if key != "manifest_sha256"})
    out = tmp_path / "study" / "loco.json"
    with pytest.raises(ArtifactIntegrityError):
        manifests.freeze_loco_manifest(**loco_inputs, output_path=out)
    assert not out.parent.exists()


@pytest.mark.parametrize("change", ["missing_candidate", "nonfinite", "fresh", "outside_grid", "duplicate"])
def test_loco_refuses_invalid_or_contaminated_seed_results(loco_inputs, change):
    results = loco_inputs["run_results"]
    first = next(iter(results.values()))
    if change == "missing_candidate":
        loco_inputs["run_results"] = {key: value for key, value in results.items() if value["cid"] != "C2"}
    elif change == "nonfinite":
        first["dev_total_loss"] = float("nan")
    elif change == "fresh":
        first["fresh_accuracy"] = 1.0
    elif change == "outside_grid":
        first["seed"] = 999
    else:
        results["duplicate"] = first.copy()
    with pytest.raises((ArtifactIntegrityError, DataContractError, InputError)):
        manifests.build_loco_manifest(**loco_inputs)


def test_loco_command_records_actual_input_paths_and_refuses_overwrite(loco_inputs, tmp_path, capsys, monkeypatch):
    from alignair.aa.guard import profiles

    def refuse(*args, **kwargs):
        pytest.fail("Manifest preparation must not load profile models.")

    monkeypatch.setattr(profiles, "load_bank", refuse)
    paths = {key: tmp_path / f"{key}.json" for key in ("records", "training_manifest", "run_results", "bank")}
    paths["records"].write_text("\n".join(json.dumps(record) for record in loco_inputs["records"]))
    for key in ("training_manifest", "run_results"):
        paths[key].write_text(json.dumps(loco_inputs[key]))
    paths["bank"].write_text(json.dumps({"bank_checksum": loco_inputs["bank_checksum"]}))
    out = tmp_path / "loco.json"
    args = ["--error-format", "json", "freeze-loco", "--records", str(paths["records"]),
            "--training-manifest", str(paths["training_manifest"]), "--runs", str(paths["run_results"]),
            "--profile-manifest", str(paths["bank"]), "--checkpoints", "fold-models", "--out", str(out)]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["expected_models"] == 8
    content = out.read_bytes()
    assert json.loads(content)["files_opened_audit"]["opened"] == [str(path) for path in paths.values()]
    assert cli.main(args) != 0
    diagnostic = capsys.readouterr()
    assert not diagnostic.out and json.loads(diagnostic.err)["code"] == ArtifactIntegrityError.code
    assert out.read_bytes() == content
