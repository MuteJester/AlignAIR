import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from alignair.aa.guard.loco import LOCO_FOLDS, LOCO_MASK_HASH, masked_profiles_for
from alignair.aa.guard.train_data import partition_by_parent_lineage
from alignair.aa.guard.training import cli, studies
from alignair.aa.guard.training.data import source_setup
from alignair.errors import ArtifactIntegrityError

pytestmark = pytest.mark.component


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@pytest.fixture
def study(tmp_path, monkeypatch):
    groups = [("positive_real", "real"), ("positive_sim", "sim"), ("igk_igl", "igk"),
              ("tcr", "tra"), ("vhh_nonhuman", "vhh"), ("igsf", "igsf")]
    records = [{"id": f"{group}-{i}", "aa": f"{group}-{i}", "group": group, "category": category,
                "lineage_key": f"{group}-{i}"} for group, category in groups for i in range(12)]
    train, dev, partition = partition_by_parent_lineage(records, seed=42, dev_frac=0.2)
    manifest = {"train_partition_hash": partition, "bank_checksum": "bank", "manifest_sha256": "training",
                "checkpoint_dir": str(tmp_path / "models"), "candidates": ["C1", "C2"], "seeds": [0, 1],
                "training": {"batch_size": 4}, "oe": {"cap_fraction": 0.25}}
    folds = {}
    for group in LOCO_FOLDS:
        fold_train = [r for r in train if r["group"] != group]
        fold_dev = [r for r in dev if r["group"] != group]
        excluded = [r for r in records if r["group"] == group]
        vocab, eligible, count = source_setup(fold_train)
        folds[group] = {
            "source_vocab_ordered": sorted(vocab), "n_sources": count,
            "source_eligible_domain_types": sorted(eligible), "masked_profiles": sorted(masked_profiles_for(group)),
            "loco_train_ids_sha256": _hash(sorted(r["id"] for r in fold_train)),
            "loco_dev_ids_sha256": _hash(sorted(r["id"] for r in fold_dev)),
            "excluded_ids_sha256": _hash(sorted(r["id"] for r in excluded)),
            "excluded_seqhash_sha256": _hash(sorted(hashlib.sha256(r["aa"].encode()).hexdigest() for r in excluded)),
            "n_train_after_exclusion": len(fold_train), "n_excluded": len(excluded),
        }
    loco = {"base_partition_hash": partition, "bank_checksum": "bank", "manifest_sha256": "loco",
            "checkpoint_dir": str(tmp_path / "fold-models"), "candidates": ["C1", "C2"],
            "frozen_seeds": {"C1": 0, "C2": 1}, "loco_mask_hash": LOCO_MASK_HASH,
            "stage4_manifest_sha256": "training", "folds": folds,
            "files_opened_audit": {"opened": ["development"], "NOT_opened": ["challenge", "selection"]}}
    calls, features = [], []

    def train_one(cid, seed, train, dev, oe, cache, vocabulary, eligible, count, **kw):
        calls.append(dict(cid=cid, seed=seed, train=train, dev=dev, oe=oe, vocabulary=vocabulary,
                          eligible=eligible, n_sources=count, **kw))
        return {"tag": kw.get("run_tag") or f"{cid}.s{seed}", "cid": cid, "seed": seed, "status": "ok",
                "dev_total_loss": 1 / (seed + 1), "applicability_auroc": None}

    def precompute(records, bank, **kwargs):
        features.append(dict(records=records, **kwargs))
        return {}

    monkeypatch.setattr(studies, "train_one", train_one)
    monkeypatch.setattr(studies, "precompute_features", precompute)
    return SimpleNamespace(manifest=manifest, loco=loco, records=records, calls=calls, features=features,
                           train=train, dev=dev, bank=SimpleNamespace(checksum=lambda: "bank"))


def test_sweep_preserves_grid_and_chooses_development_seeds(study, tmp_path):
    result = studies.run_sweep(study.manifest, study.records, study.bank,
                               artifact_dir=tmp_path / "out", cache_path=tmp_path / "cache.json",
                               checkpoint_dir=tmp_path / "fresh checkpoints")
    assert [(r["cid"], r["seed"]) for r in study.calls] == [("C1", 0), ("C1", 1), ("C2", 0), ("C2", 1)]
    assert result["frozen_seeds"] == {"C1": 1, "C2": 1}
    assert (tmp_path / "out" / "STAGE4_RESULT.md").exists()
    assert all(call["train"] == study.train and call["dev"] == study.dev for call in study.calls)
    assert all(call["config"].checkpoint_dir == tmp_path / "fresh checkpoints" for call in study.calls)


def test_loco_restricts_original_partition_and_masks_each_fold(study, tmp_path):
    results = studies.run_loco(study.manifest, study.loco, study.records, study.bank,
                               artifact_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    assert len(results) == len(LOCO_FOLDS) * 2
    for call in study.calls:
        fold = call["extra_provenance"]["fold"]
        assert call["mask_groups"] == masked_profiles_for(fold)
        assert all(record["group"] != fold for record in call["train"] + call["dev"])
        assert call["train"] == [record for record in study.train if record["group"] != fold]
        assert call["dev"] == [record for record in study.dev if record["group"] != fold]
        assert call["seed"] == study.loco["frozen_seeds"][call["cid"]]
        assert call["config"].manifest is study.manifest
    assert {tuple(sorted(f["mask_groups"])) for f in study.features} == {
        tuple(sorted(masked_profiles_for(fold))) for fold in LOCO_FOLDS}
    assert "unavailable" in (tmp_path / "out" / "STAGE5_LOCO_TRAINING.md").read_text()


@pytest.mark.parametrize("field", ["source_vocab_ordered", "loco_train_ids_sha256", "loco_dev_ids_sha256",
                                   "excluded_seqhash_sha256", "masked_profiles"])
def test_loco_drift_is_rejected_before_any_training_or_output(study, tmp_path, field):
    manifest = copy.deepcopy(study.loco)
    manifest["folds"][LOCO_FOLDS[-1]][field] = "changed"
    with pytest.raises(ArtifactIntegrityError):
        studies.run_loco(study.manifest, manifest, study.records, study.bank,
                          artifact_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    assert not study.calls
    assert not (tmp_path / "out").exists()


def test_partition_drift_cannot_start_a_sweep(study, tmp_path):
    manifest = dict(study.manifest, train_partition_hash="changed")
    with pytest.raises(ArtifactIntegrityError):
        studies.run_sweep(manifest, study.records, study.bank,
                           artifact_dir=tmp_path / "out", cache_path=tmp_path / "cache.json")
    assert not study.calls


def test_package_cli_runs_with_explicit_paths(study, tmp_path, monkeypatch, capsys):
    from alignair.aa.guard import profiles

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(study.manifest))
    records = tmp_path / "dev.jsonl"
    records.write_text("\n".join(json.dumps(record) for record in study.records))
    monkeypatch.setattr(profiles, "load_bank", lambda path: study.bank)
    status = cli.main(["sweep", "--manifest", str(manifest), "--records", str(records),
                       "--profiles", str(tmp_path), "--out", str(tmp_path / "study output"),
                       "--feature-cache", str(tmp_path / "cache.json"), "--device", "cpu"])
    assert status == 0
    assert json.loads(capsys.readouterr().out)["runs"] == 4


def test_package_cli_reports_missing_input_as_structured_io_error(tmp_path, capsys):
    status = cli.main(["--error-format", "json", "sweep", "--manifest", str(tmp_path / "absent.json"),
                       "--records", "absent.jsonl", "--profiles", "absent", "--out", "absent",
                       "--feature-cache", "absent.json"])
    assert status == 1
    assert json.loads(capsys.readouterr().err)["code"] == "ALIGNAIR_IO_FAILED"
