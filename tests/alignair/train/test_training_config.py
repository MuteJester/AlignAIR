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
    assert TrainingConfig.from_genairr("HUMAN_IGH_OGRDB").novel_alleles == NovelAlleles()   # the recipe's
    assert TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", novel_alleles=None).novel_alleles is None
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


@pytest.mark.parametrize("options, field", [
    ({"device": "gpu"}, "device"),
    ({"max_seq_length": -5}, "max_seq_length"),
    ({"max_seq_length": "576"}, "max_seq_length"),
    ({"has_d": "no"}, "has_d"),
    ({"model_options": {"embed_dim": -4}}, "embed_dim"),
    ({"resume_path": 123}, "resume_path"),
    ({"init_from": 5}, "init_from"),
], ids=["device", "negative_window", "string_window", "has_d", "model_option_value", "resume_path", "init_from"])
def test_every_value_is_refused_at_construction_not_inside_the_run(options, field):
    """The values the run used to refuse late (after resolving the reference), or not as AlignAIR errors."""
    with pytest.raises(ConfigurationError) as caught:
        TrainingConfig(dataconfigs=("HUMAN_IGH_OGRDB",), **options)
    assert caught.value.context["field"] == field


def test_a_device_torch_can_name_is_accepted_whether_or_not_it_is_here():
    for device in ("auto", "cpu", "cuda", "cuda:1", "mps"):
        assert TrainingConfig(dataconfigs=("HUMAN_IGH_OGRDB",), device=device).device == device


def test_a_novel_allele_s_gene_may_be_given_in_either_case():
    from alignair.reference import NovelAllele
    allele = NovelAllele.from_dict({"name": "IGHV1-2*02_nv", "base": "IGHV1-2*02", "sequence": "acgt", "gene": "v"})
    assert allele.gene == "V"


def test_novel_alleles_on_a_custom_reference_say_it_names_no_species():
    from types import SimpleNamespace

    from alignair.train import NovelAlleles
    from alignair.train.trainer import _check_novel_alleles
    reference = SimpleNamespace(loci=(SimpleNamespace(species=None),))
    with pytest.raises(ConfigurationError, match="names none"):
        _check_novel_alleles([], reference, NovelAlleles())
