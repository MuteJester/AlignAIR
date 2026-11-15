"""Prepared batch contracts: dtypes, positional labels, ownership, and malformed samples."""
import numpy as np
import pytest
import torch

from alignair.data import PositionLabels, TrainingSample, collate_samples
from alignair.data.batching import pad_vectors
from alignair.errors import ConfigurationError, DataContractError, TrainingDataError

pytestmark = pytest.mark.component


def test_common_batch_preserves_target_types_and_position_ignore_values():
    samples = [
        TrainingSample(np.array([1, 2, 3]), {"v_class": torch.tensor(1), "v_start": torch.tensor([0.]),
                       "v_valid": torch.tensor(True), "state_labels": PositionLabels(np.array([0, 1, 2]))}),
        TrainingSample(np.array([4, 5]), {"v_class": torch.tensor(0), "v_start": torch.tensor([1.]),
                       "v_valid": torch.tensor(False), "state_labels": PositionLabels(np.array([0, 0]))}),
    ]
    batch = collate_samples(samples, max_length=5)
    inputs, targets = batch
    assert inputs["tokenized_sequence"].tolist() == [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]
    assert batch.lengths.tolist() == [3, 2]
    assert targets["v_class"].dtype == torch.long
    assert targets["v_valid"].dtype == torch.bool
    assert targets["v_start"].shape == (2, 1) and targets["v_start"].dtype == torch.float32
    assert targets["state_labels"].tolist() == [[0, 1, 2, -100, -100], [0, 0, -100, -100, -100]]
    assert batch.to("cpu").targets.keys() == targets.keys()
    assert samples[0].tokens.tolist() == [1, 2, 3]


def test_already_center_padded_samples_keep_their_coordinates():
    sample = TrainingSample(np.array([0, 1, 2, 0]), {"v_start": np.array([1.], np.float32)})
    batch = collate_samples([sample])
    assert batch.lengths.tolist() == [2]
    assert batch.token_mask.tolist() == [[False, True, True, False]]
    assert batch.targets["v_start"].item() == 1


@pytest.mark.parametrize("failure", ["empty_batch", "float_tokens", "negative_token", "keys",
                                     "dtypes", "shapes", "positional_shape", "positional_kind", "padding"])
def test_invalid_sample_contracts_raise_located_errors(failure):
    targets = {"v_class": torch.tensor(0)}
    samples = [TrainingSample(np.array([1, 2]), dict(targets)), TrainingSample(np.array([3]), dict(targets))]
    if failure == "empty_batch":
        samples = []
    elif failure == "float_tokens":
        samples[0] = TrainingSample(np.array([1.2, 2.]), targets)
    elif failure == "negative_token":
        samples[0] = TrainingSample(np.array([-1, 2]), targets)
    elif failure == "keys":
        samples[1] = TrainingSample(np.array([3]), {})
    elif failure == "dtypes":
        samples[1].targets["v_class"] = torch.tensor(1.)
    elif failure == "shapes":
        samples[1].targets["v_class"] = torch.tensor([0, 1])
    else:
        samples[0].targets["labels"] = PositionLabels(np.array([0, 1]))
        samples[1].targets["labels"] = PositionLabels(np.array([0]))
        if failure == "positional_shape":
            samples[1].targets["labels"] = PositionLabels(np.array([0, 1]))
        elif failure == "positional_kind":
            samples[1].targets["labels"] = np.array([0])
        else:
            samples[1].targets["labels"] = PositionLabels(np.array([0]), padding=-1)
    with pytest.raises(TrainingDataError):
        collate_samples(samples, max_length=4)


def test_fixed_windows_never_silently_truncate_and_reserve_endpoints_explicitly():
    sample = TrainingSample(np.array([1, 2, 3]), {"v_class": torch.tensor(0)})
    with pytest.raises(TrainingDataError, match="available model slots"):
        collate_samples([sample], max_length=2)
    with pytest.raises(TrainingDataError, match="available model slots"):
        collate_samples([sample], max_length=3, reserve_terminal_slot=True)
    assert collate_samples([sample], max_length=3).lengths.tolist() == [3]
    dynamic = collate_samples([sample], reserve_terminal_slot=True)
    assert dynamic.inputs["tokenized_sequence"].tolist() == [[1, 2, 3, 0]]
    with pytest.raises(DataContractError, match="exceeds"):     # padding also serves prediction: not a training error
        pad_vectors([[1, 2, 3]], width=2)


@pytest.mark.parametrize("width", [0, -1, 1.5, True])
def test_invalid_window_configuration_is_rejected(width):
    with pytest.raises(ConfigurationError, match="positive integer"):
        collate_samples([TrainingSample([1], {})], max_length=width)


def test_padding_rejects_nonvector_values():
    with pytest.raises(DataContractError, match="one-dimensional"):
        pad_vectors([[[1, 2]]])
