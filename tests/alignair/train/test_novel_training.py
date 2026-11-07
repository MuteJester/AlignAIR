"""Training with novel alleles is opt-in (``TrainingConfig.novel_alleles``): the run trains on reads simulated from
novel alleles, with their rows in each batch's bank, records the spec in its train args, and saves the dataconfigs'
reference (nothing novel reaches a model file); workers reproduce it; it refuses a reference with no eligible V or of
two species before any record is drawn."""
import pytest

from alignair import TrainingConfig, run_training
from alignair.errors import ConfigurationError
from alignair.model_file import load_model, read_metadata
from alignair.reference.variants import NOVEL_FIELD
from alignair.train import NovelAlleles
from tests.support.reference_pin import PIN, v1_digest

pytestmark = pytest.mark.integration
SPEC = NovelAlleles(epoch_records=16)


def request(*dataconfigs, **options):
    return TrainingConfig.from_genairr(*(dataconfigs or ("HUMAN_IGH_OGRDB",)), **{
        "steps": 3, "batch_size": 4, "device": "cpu", "seed": 5,
        "model_options": {"embed_dim": 4, "trunk_channels": 8}, "novel_alleles": SPEC, **options})


def trained(config, path):
    records = []
    run = run_training(config, out_path=str(path), on_iteration=lambda it: records.append((it.step, list(it.records))))
    return run, records


def test_a_novel_allele_run_trains_on_their_reads_and_saves_the_dataconfigs_reference(tmp_path):
    run, records = trained(request(steps=6, batch_size=16), tmp_path / "novel.alignair")
    assert any(NOVEL_FIELD in record for _, batch in records for record in batch)
    assert v1_digest(load_model(run.model_path).reference) == PIN["dataconfigs"]["HUMAN_IGH_OGRDB"]
    header = read_metadata(run.model_path)
    assert len(header["model"]["label_schema"]["genes"]["v"]) == 198
    assert header["training"]["train_args"]["novel_alleles"] == {"generator": 1, "gene_fraction": 0.4,
                                                                 "snps": [1, 6], "epoch_records": 16}


def test_a_run_without_novel_alleles_records_none(tmp_path):
    run, _ = trained(request(novel_alleles=None), tmp_path / "plain.alignair")
    assert "novel_alleles" not in read_metadata(run.model_path)["training"]["train_args"]


def test_parallel_training_with_novel_alleles_reproduces_the_serial_records_and_weights(tmp_path):
    import torch
    serial, serial_records = trained(request(steps=4, batch_size=16), tmp_path / "serial.alignair")
    parallel, parallel_records = trained(request(steps=4, batch_size=16, data_workers=3), tmp_path / "parallel.alignair")
    assert parallel_records == serial_records
    left, right = load_model(serial.model_path).model.state_dict(), load_model(parallel.model_path).model.state_dict()
    assert left.keys() == right.keys() and all(torch.equal(value, right[key]) for key, value in left.items())


def test_a_novel_allele_run_validates_on_batches_with_their_rows(tmp_path, capsys):
    trained(request(steps=2, batch_size=16, val_every=1), tmp_path / "validated.alignair")
    assert "VAL score" in capsys.readouterr().out


def test_novel_alleles_refuse_a_dataconfig_without_an_eligible_v(tmp_path):
    def never(*args, **kwargs):
        raise AssertionError("no record is drawn")
    with pytest.raises(ConfigurationError, match="MOUSE_TCRA_IMGT"):
        run_training(request("MOUSE_TCRA_IMGT"), out_path=str(tmp_path / "tra.alignair"), record_source=never)


def test_novel_alleles_refuse_a_reference_of_two_species(tmp_path):
    with pytest.raises(ConfigurationError, match="species"):
        run_training(request("HUMAN_IGK_OGRDB", "MOUSE_IGK_IMGT"), out_path=str(tmp_path / "igk.alignair"))

