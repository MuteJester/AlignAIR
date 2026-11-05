"""Tests for batch-merge / clean stage."""
import pytest
import numpy as np

from alignair.predict.clean import clean

pytestmark = pytest.mark.component


def _batch(vp, vs, ve):
    return {"v_allele": np.array([vp]), "v_start": np.array([[vs]]), "v_end": np.array([[ve]]),
            "j_allele": np.array([[0.5, 0.5]]), "j_start": np.array([[60.0]]), "j_end": np.array([[90.0]]),
            "mutation_rate": np.array([[0.1]]), "indel_count": np.array([[0.0]])}


def test_clean_merges_the_batches():
    preds = clean([_batch([0.1, 0.9], 5.0, 50.0), _batch([0.3, 0.6], 6.0, 52.0)], genes=("v", "j"))
    assert preds.allele["v"].shape == (2, 2)
    assert preds.start["v"].tolist() == [5.0, 6.0]
    assert preds.orientation.tolist() == [0, 0] and preds.anchors is None and preds.d_presence is None
