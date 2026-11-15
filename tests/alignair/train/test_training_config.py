"""``TrainingConfig`` refuses a request it cannot run before anything is built, naming what is wrong."""
import pytest

from alignair.errors import ConfigurationError
from alignair.train import CosineSchedule, TrainingConfig

pytestmark = pytest.mark.component


@pytest.mark.parametrize("options,message", [
    ({"init_from": "a.alignair", "resume_path": "b.alignair"}, "choose init_from or resume_path"),
    ({"flanks": "default"}, "flanks must be a FlankPolicy"),
    ({"schedule": {"total_steps": 10}}, "schedule must be a CosineSchedule"),
    ({"input_space": "protein"}, "nucleotide reads with allele labels"),
], ids=["init_and_resume", "flanks", "schedule", "spaces"])
def test_an_unusable_request_is_refused(options, message):
    with pytest.raises(ConfigurationError, match=message):
        TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", **options)


def test_a_reference_source_list_is_not_a_string():
    with pytest.raises(ConfigurationError, match="nonempty collection"):
        TrainingConfig(dataconfigs="HUMAN_IGH_OGRDB")


def test_a_schedule_of_its_type_is_accepted_and_kept():
    schedule = CosineSchedule(total_steps=10, min_lr=1e-6)
    assert TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", schedule=schedule).schedule is schedule


def test_the_training_config_takes_a_novel_allele_spec():
    from alignair.train import NovelAlleles
    spec = NovelAlleles(epoch_records=16)
    assert TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", novel_alleles=spec).novel_alleles == spec
    assert TrainingConfig.from_genairr("HUMAN_IGH_OGRDB").novel_alleles is None
    with pytest.raises(ConfigurationError) as caught:
        TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", novel_alleles={"gene_fraction": 0.4})
    assert caught.value.context["field"] == "novel_alleles"
