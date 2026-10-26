"""V+J penultimate representation extraction — fixed width, deterministic, prediction-preserving."""
from __future__ import annotations

import numpy as np
import pytest

import alignair
from alignair.aa.product.applicability.embedding import branch_features, REPRESENTATION_VERSION

pytestmark = pytest.mark.component

_VH1 = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDT"
        "AVYYCAKDLYYYGMDVWGQGTTVTVSS")
_VH2 = ("QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDT"
        "AVYYCARDRGYYYGMDVWGQGTTVTVSS")


@pytest.fixture(scope="module")
def predictor():
    return alignair.load_aa_model("alignair-aa-igh-human@2.0.0")


def test_features_are_fixed_width_and_deterministic(predictor):
    f1 = branch_features(predictor, [_VH1, _VH2])
    f2 = branch_features(predictor, [_VH1, _VH2])
    assert f1.shape[0] == 2 and f1.ndim == 2 and f1.shape[1] > 0
    assert np.array_equal(f1, f2)                       # deterministic
    assert REPRESENTATION_VERSION == "vj_penultimate_v1"


def test_v_plus_j_is_wider_than_v_alone(predictor):
    v = branch_features(predictor, [_VH1], branches=("v",))
    vj = branch_features(predictor, [_VH1], branches=("v", "j"))
    assert vj.shape[1] > v.shape[1]


def test_extraction_does_not_change_predictions(predictor):
    from alignair.aa.infer import class_probs
    before = class_probs(predictor.model, [_VH1], predictor.cfg, predictor.device, "v_class_logits")
    branch_features(predictor, [_VH1])                  # hooks attach + detach
    after = class_probs(predictor.model, [_VH1], predictor.cfg, predictor.device, "v_class_logits")
    assert np.array_equal(before, after)
