"""Tests for batch-merge / clean stage."""
import numpy as np

from alignair.predict.clean import clean


def _batch(vp, vs, ve, prod):
    return {"v_allele": np.array([vp]), "v_start": np.array([[vs]]), "v_end": np.array([[ve]]),
            "j_allele": np.array([[0.5, 0.5]]), "j_start": np.array([[60.0]]), "j_end": np.array([[90.0]]),
            "mutation_rate": np.array([[0.1]]), "indel_count": np.array([[0.0]]),
            "productive": np.array([[prod]])}


def test_clean_merges_and_thresholds_productive():
    preds = clean([_batch([0.1, 0.9], 5.0, 50.0, 0.7), _batch([0.3, 0.6], 6.0, 52.0, 0.3)],
                  genes=("v", "j"))
    assert preds.allele["v"].shape == (2, 2)
    assert preds.start["v"].tolist() == [5.0, 6.0]
    assert preds.productive.tolist() == [True, False]        # 0.7>0.5, 0.3<0.5
