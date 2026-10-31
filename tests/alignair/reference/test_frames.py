"""Gene coordinate frames: every allele of a gene laid out on one shared set of columns."""
import numpy as np
import pytest

from alignair.reference.frames import gene_frames
from alignair.reference.reference_set import GeneReference, ReferenceSet

pytestmark = pytest.mark.component


def _reference(v_gapped=None, j_anchors=None):
    v = GeneReference(["V*01"], ["ACGT"], {"V*01": 0}, gapped=v_gapped)
    j_sequences = ["A" * 20, "C" * 20]
    j = GeneReference(["J1*01", "J2*01"], j_sequences, {"J1*01": 0, "J2*01": 1}, anchors=j_anchors)
    d = GeneReference(["D*01", "D*02"], ["GGG", "GGGTT"], {"D*01": 0, "D*02": 1})
    return ReferenceSet({"V": v, "D": d, "J": j}, has_d=True)


def test_v_columns_follow_the_gapped_form():
    frames = gene_frames(_reference(v_gapped={"V*01": "AC..GT"}))
    assert frames["v"].columns["V*01"].tolist() == [0, 1, 4, 5]
    assert frames["v"].width == 6


def test_a_v_allele_without_a_gapped_form_uses_its_own_positions():
    frames = gene_frames(_reference())
    assert frames["v"].columns["V*01"].tolist() == [0, 1, 2, 3]
    assert frames["v"].width == 4


def test_j_anchors_share_one_column():
    frames = gene_frames(_reference(j_anchors={"J1*01": 10, "J2*01": 13}))
    j = frames["j"]
    assert j.columns["J1*01"][10] == j.columns["J2*01"][13] == 13
    assert j.width == 23                                   # J1 is shifted 3 columns right
    assert np.array_equal(j.columns["J2*01"], np.arange(20))


def test_d_columns_are_its_positions():
    frames = gene_frames(_reference())
    assert frames["d"].columns["D*02"].tolist() == [0, 1, 2, 3, 4]
    assert frames["d"].width == 5
