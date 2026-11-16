"""``TrainingConfig`` refuses a request it cannot run before anything is built, naming what is wrong."""
import pytest

from alignair.errors import ConfigurationError
from alignair.train import CosineSchedule, TrainingConfig

pytestmark = pytest.mark.component


@pytest.mark.parametrize("options,message", [
    ({"init_from": "a.alignair", "resume_path": "b.alignair"}, "choose init_from or resume_path"),
    ({"flanks": "default"}, "flanks must be a FlankPolicy"),
    ({"schedule": {"total_steps": 10}}, "schedule must be a CosineSchedule"),
    ({"model_options": {"has_d": False}}, "cannot set the window, the D gene or the locus count"),
    ({"model_options": {"no_such_option": 1}}, "invalid model configuration options"),
    ({"seed": -1}, "seed must be a uint32"),
    ({"seed": True}, "seed must be a uint32"),
    ({"init_seed": 2**32}, "init_seed a uint32 or None"),
    ({"val_every": -1}, "val_every must be a nonnegative integer"),
    ({"lr": 0}, "lr must be a positive finite number"),
    ({"grad_clip": 0}, "grad_clip must be a positive finite number"),
    ({"inverted_d": 2}, "inverted_d"),
], ids=["init_and_resume", "flanks", "schedule", "reference_fields", "unknown_model_option", "negative_seed",
        "bool_seed", "init_seed", "val_every", "lr", "grad_clip", "inverted_d"])
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


def test_the_run_draws_its_records_with_the_recipe_and_its_own_settings():
    from alignair.data.flanks import FlankPolicy
    from alignair.train import NovelAlleles
    from alignair.train.records.streams import CurriculumStreamPolicy
    config = TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", short_boost=2, flanks=None, novel_alleles=NovelAlleles(),
                                         inverted_d=0.1)
    assert config.record_policy() == CurriculumStreamPolicy(short_boost=2, novel_alleles=NovelAlleles(),
                                                            inverted_d=0.1)
    assert TrainingConfig.from_genairr("HUMAN_IGH_OGRDB").record_policy().flanks == FlankPolicy()
