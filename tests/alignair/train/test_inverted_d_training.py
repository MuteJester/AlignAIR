"""Training with inverted Ds is opt-in (``TrainingConfig.inverted_d``, ``CurriculumStreamPolicy.inverted_d``): that
share of every stream's records has its D inverted (GenAIRR's ``invert_d``), the run records the share in its train
args, and without it the mix's streams and train args are as before."""
import dataclasses

import GenAIRR.data as gd
import pytest

from alignair import TrainingConfig, run_training
from alignair.errors import ConfigurationError
from alignair.model_file import read_metadata
from alignair.train.gym.streams import CurriculumStreamPolicy


def _streams(policy):
    return [stream for component in policy.substreams_for(gd.HUMAN_IGH_OGRDB) for stream in component]


@pytest.mark.component
def test_the_mix_without_inverted_d_keeps_its_parameters():
    from alignair_lab.recipe import POLICY
    for policy in (CurriculumStreamPolicy(), POLICY):
        assert not any("invert_d_prob" in parameters for _, parameters in _streams(policy))


@pytest.mark.component
def test_inverted_d_reaches_every_stream_of_the_recipes_mix():
    from alignair_lab.recipe import POLICY
    streams = _streams(dataclasses.replace(POLICY, inverted_d=0.05))
    assert len(streams) == 28
    assert all(parameters["invert_d_prob"] == 0.05 for _, parameters in streams)


@pytest.mark.component
@pytest.mark.parametrize("value", [-0.1, 1.5, True, "0.05", float("nan")])
def test_an_invalid_inverted_d_share_is_refused(value):
    with pytest.raises(ConfigurationError) as caught:
        CurriculumStreamPolicy(inverted_d=value)
    assert caught.value.context["field"] == "inverted_d"
    with pytest.raises(ConfigurationError) as caught:
        TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", inverted_d=value)
    assert caught.value.context["field"] == "inverted_d"


def request(**options):
    return TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", **{
        "steps": 3, "batch_size": 8, "device": "cpu", "seed": 5,
        "model_options": {"embed_dim": 4, "trunk_channels": 8}, **options})


def trained(config, path):
    records = []
    run = run_training(config, out_path=str(path), on_iteration=lambda it: records.extend(it.records))
    return run, records


@pytest.mark.integration
def test_an_inverted_d_run_trains_on_inverted_ds_and_records_the_share(tmp_path):
    run, records = trained(request(inverted_d=1.0), tmp_path / "inverted.alignair")
    with_d = [record for record in records if record.get("d_call")]
    assert with_d and all(record["d_inverted"] for record in with_d)
    assert read_metadata(run.model_path)["training"]["train_args"]["inverted_d"] == 1.0


@pytest.mark.integration
def test_a_run_without_inverted_d_trains_on_forward_ds_and_records_none(tmp_path):
    run, records = trained(request(), tmp_path / "forward.alignair")
    assert records and not any(record.get("d_inverted") for record in records)
    assert "inverted_d" not in read_metadata(run.model_path)["training"]["train_args"]
