"""Tests for the §7 calibrated-set wiring in `AAPredictor.predict` (Task 7): conformal `calibrated_*`
fields + `set_size` + real `applied_calibration_group`, and PARITY with the research path
(`class_probs` + a freshly-built `FrozenConformal` read straight from the bundle's own calibration
section, independent of the loader's `AAPredictor.conformal` holder)."""
from __future__ import annotations

import json

import pytest
import torch

from alignair.aa.conformal import FrozenConformal
from alignair.aa.equivalence import EquivalenceClassMap
from alignair.aa.infer import class_probs
from alignair.aa.product.calibration_group import SUPPLIED_WARNING
from alignair.aa.product.loader import _V2_BUNDLE_PATH, AAPredictor, load_aa_model
from alignair.model_file import container

pytestmark = pytest.mark.component

_REAL_MODEL_ID = "alignair-aa-igh-human@2.0.0"

# Same two real, full-length VH amino-acid reads used by test_predictor.py.
_VH_1 = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAV"
        "YYCSRWGGDGFYAMDYWGQGTLVTVSS")
_VH_2 = ("EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAV"
        "YYCAKYPHYYGSSHWYFDVWGQGTLVTVSS")


@pytest.fixture(scope="module")
def predictor() -> AAPredictor:
    return load_aa_model(_REAL_MODEL_ID)


def _raw_calibration(gene: str) -> tuple:
    """Read the bundle's `calibration` section directly (bypassing `AAPredictor.conformal`
    entirely) so the parity tests reconstruct the research-path `FrozenConformal` from first
    principles, not from the same loaded object the product path itself uses."""
    cal_bytes = container.read_section(str(_V2_BUNDLE_PATH), "calibration")
    cal = json.loads(cal_bytes.decode("utf-8"))
    method = cal["method"]
    frozen = cal["genes"][gene]["calibration_frozen"]
    return frozen["q_pooled"], frozen["q_mondrian"], method["lambda"], method["k_reg"]


# --------------------------------------------------------------------------------- parity (pooled)


@pytest.mark.parametrize("gene", ["V", "D", "J"])
@pytest.mark.parametrize("seq", [
    _VH_1, _VH_2, _VH_1[15:-15], _VH_2[15:-15], _VH_1[35:85], _VH_2[35:85],
], ids=["vh1_full", "vh2_full", "vh1_trunc", "vh2_trunc", "vh1_short", "vh2_short"])
def test_calibrated_class_set_parity_with_research_path_pooled(predictor, gene, seq):
    [result] = predictor.predict([seq])
    product_set = result.segments[gene]["calibrated_class_set"]

    q_pooled, _q_mondrian, lam, k_reg = _raw_calibration(gene)
    prob_row = class_probs(predictor.model, [seq], predictor.cfg, predictor.device,
                           f"{gene.lower()}_class_logits")[0]
    research_set = sorted(FrozenConformal(q_pooled, lam=lam, k_reg=k_reg).predict_set(prob_row))

    assert product_set == research_set
    assert result.segments[gene]["top_call"]["class_id"] == int(prob_row.argmax())
    assert result.applied_calibration_group == {"group": "pooled", "source": "pooled_fallback"}


def test_calibrated_class_set_pooled_matches_the_predictors_own_conformal_object(predictor):
    """Sanity companion to the from-scratch parity test above: the loaded `predictor.conformal`
    holder's own pooled `qhat` for V must equal the one read straight from the raw bundle bytes."""
    q_pooled, _, _, _ = _raw_calibration("V")
    assert predictor.conformal["V"]["pooled"].qhat == pytest.approx(q_pooled)


# --------------------------------------------------------------------------------- supplied group


@pytest.mark.parametrize("group", ["full", "trunc", "short"])
def test_supplied_group_uses_its_mondrian_quantile(predictor, group):
    seq = {"full": _VH_1, "trunc": _VH_1[15:-15], "short": _VH_1[35:85]}[group]
    [result] = predictor.predict([seq], coverage_group=group)

    _q_pooled, q_mondrian, lam, k_reg = _raw_calibration("V")
    prob_row = class_probs(predictor.model, [seq], predictor.cfg, predictor.device,
                           "v_class_logits")[0]
    expected = sorted(FrozenConformal(q_mondrian[group], lam=lam, k_reg=k_reg).predict_set(prob_row))

    assert result.segments["V"]["calibrated_class_set"] == expected
    assert result.applied_calibration_group == {"group": group, "source": "supplied"}
    assert SUPPLIED_WARNING in result.warnings


def test_set_size_matches_the_returned_sets(predictor):
    [result] = predictor.predict([_VH_1])
    for gene in ("V", "D", "J"):
        seg = result.segments[gene]
        assert seg["set_size"] == {
            "classes": len(seg["calibrated_class_set"]),
            "alleles": len(seg["calibrated_allele_set"]),
            "genes": len(seg["calibrated_gene_set"]),
        }
        # APS always includes at least the top-ranked (argmax) class.
        assert seg["top_call"]["class_id"] in seg["calibrated_class_set"]


def test_d_calibrated_allele_set_is_the_d_allele_names_of_the_class_set(predictor):
    [result] = predictor.predict([_VH_1])
    d = result.segments["D"]
    expected_alleles = sorted({predictor.d_allele_order[c] for c in d["calibrated_class_set"]})
    assert d["calibrated_allele_set"] == expected_alleles


# --------------------------------------------------------------------------------- crafted one-to-many


class _StubModel:
    def __init__(self, out: dict):
        self._out = out

    def eval(self):
        return self

    def __call__(self, batch):
        return self._out


class _StubTokenizer:
    def encode(self, sequence: str):
        import numpy as np
        return np.ones(len(sequence), dtype=np.int64)


class _StubCfg:
    max_seq_length = 32
    has_d = False


def _one_class_map(gene: str, members: list) -> EquivalenceClassMap:
    return EquivalenceClassMap(gene=gene, db_hash="fakehash",
                               allele_class={m: 0 for m in members}, class_members={0: members})


def _stub_predictor_for_expansion(*, mondrian: dict | None = None) -> AAPredictor:
    out = {
        "v_class_logits": torch.tensor([[2.0]]),      # single class -> softmax==1.0 -> class 0
        "j_class_logits": torch.tensor([[2.0]]),
        "v_start": torch.tensor([[0.0]]), "v_end": torch.tensor([[3.0]]),
        "j_start": torch.tensor([[4.0]]), "j_end": torch.tensor([[8.0]]),
    }
    mondrian = {} if mondrian is None else mondrian
    return AAPredictor(
        model=_StubModel(out), cfg=_StubCfg(), tokenizer=_StubTokenizer(),
        class_maps={"V": _one_class_map("V", ["IGHV1-1*01", "IGHV1-2*01"]),
                   "J": _one_class_map("J", ["IGHJ1*01"])},
        d_allele_order=[],
        # IGHV1-1*01 is a multi-IMGT ASC member (maps to TWO IMGT alleles); IGHV1-2*01 and IGHJ1*01
        # are one-to-one.
        asc_multi={"IGHV1-1*01": ["IGHV1-1*01", "IGHV1-1*02"],
                  "IGHV1-2*01": ["IGHV1-2*01"],
                  "IGHJ1*01": ["IGHJ1*01"]},
        conformal={
            "V": {"pooled": FrozenConformal(0.99), "mondrian": dict(mondrian)},
            "J": {"pooled": FrozenConformal(0.99), "mondrian": dict(mondrian)},
        },
        card={"model_id": "fake-model", "model_version": "0.0.0-fake",
             "alignair_version": "3.0.0-test", "calibration_scope": "TEST SIMULATOR SCOPE STRING",
             "integrity": {"weights_sha256": "abc123", "calibration_sha256": "def456"},
             "reference_provenance": {"class_map_db_hash": {"V": "vhash", "J": "jhash"}}},
        device="cpu",
    )


def test_calibrated_allele_set_is_the_one_to_many_expansion_of_the_class_set():
    predictor = _stub_predictor_for_expansion()
    [result] = predictor.predict(["ACDEFGHIKL"])

    v = result.segments["V"]
    assert v["calibrated_class_set"] == [0]                       # a single class ...
    assert v["calibrated_allele_set"] == ["IGHV1-1*01", "IGHV1-1*02", "IGHV1-2*01"]  # ... 3 alleles
    assert v["calibrated_gene_set"] == ["IGHV1-1", "IGHV1-2"]      # 2 genes (plural)
    assert v["calibrated_family_set"] == ["IGHV1"]                 # 1 family
    assert v["set_size"] == {"classes": 1, "alleles": 3, "genes": 2}


def test_supplied_group_unavailable_falls_back_to_pooled_with_warning():
    predictor = _stub_predictor_for_expansion(mondrian={})        # "full" not fitted for either gene
    [result] = predictor.predict(["ACDEFGHIKL"], coverage_group="full")

    assert result.applied_calibration_group == {"group": "pooled", "source": "pooled_fallback"}
    # When falling back to pooled, SUPPLIED_WARNING should NOT be present (the supplied group was not used)
    assert SUPPLIED_WARNING not in result.warnings
    assert any("unavailable" in w.lower() and "full" in w for w in result.warnings)
    # both V and J actually used the pooled FrozenConformal (not a half-applied group)
    assert result.segments["V"]["calibrated_class_set"] == [0]
    assert result.segments["J"]["calibrated_class_set"] == [0]


def test_supplied_group_present_uses_group_not_pooled():
    fc_full = FrozenConformal(0.5)
    predictor = _stub_predictor_for_expansion(mondrian={"full": fc_full})
    [result] = predictor.predict(["ACDEFGHIKL"], coverage_group="full")

    assert result.applied_calibration_group == {"group": "full", "source": "supplied"}
    # the group-specific FrozenConformal object was actually the one consulted
    assert result.segments["V"]["calibrated_class_set"] == sorted(
        fc_full.predict_set(torch.softmax(torch.tensor([2.0]), dim=-1).numpy()))


# ----------------------------------------------------------------- original-index regression (skip)


def test_predict_skip_coverage_group_list_aligns_by_original_batch_index(predictor):
    """`coverage_group` given as a per-sequence list is aligned to the ORIGINAL input batch
    positions, not to the position among only the valid (post-skip) records — regression guard for
    the invalid-record-dropping `on_invalid="skip"` path."""
    results = predictor.predict([_VH_1, "EV*QL", _VH_2], on_invalid="skip",
                                coverage_group=["full", "trunc", "short"])
    v1, err, v2 = results
    assert v1.applied_calibration_group["group"] == "full"
    assert err.error == "invalid_character"
    assert v2.applied_calibration_group["group"] == "short"


# --------------------------------------------------- §6 calibrated_classes (ranked class-level scores)
#
# Contract: `calibrated_classes` is the SAME calibrated set as `calibrated_class_set`, re-presented as
# a list ranked by the model's own softmax CLASS posterior. Scores are protein-class posteriors copied
# verbatim — never renormalized over the set, never divided among a class's one-to-many alleles. (D is
# the exception: its head predicts allele indices directly, so a D entry's single allele carries a
# defensible individual-allele probability.)


@pytest.mark.parametrize("gene", ["V", "J"])
@pytest.mark.parametrize("seq", [_VH_1, _VH_2], ids=["vh1", "vh2"])
def test_calibrated_classes_probabilities_are_softmax_verbatim(predictor, gene, seq):
    """Every ranked class carries the model's softmax posterior for that exact class id, taken
    straight from the class head (parity with `class_probs`) — not a renormalized or invented value."""
    [result] = predictor.predict([seq])
    seg = result.segments[gene]
    prob_row = class_probs(predictor.model, [seq], predictor.cfg, predictor.device,
                           f"{gene.lower()}_class_logits")[0]
    for entry in seg["calibrated_classes"]:
        assert entry["softmax_score"] == pytest.approx(float(prob_row[entry["class_id"]]))
    # rank-1 IS the argmax `top_call`, reported at its raw (un-renormalized) softmax_score
    top = seg["calibrated_classes"][0]
    assert top["rank"] == 1
    assert top["class_id"] == seg["top_call"]["class_id"]
    assert top["softmax_score"] == pytest.approx(seg["top_call"]["softmax_score"])


@pytest.mark.parametrize("gene", ["V", "D", "J"])
def test_calibrated_classes_are_rank_sorted_and_dense(predictor, gene):
    """Ranks are dense 1..N, probabilities are non-increasing, and the class ids are exactly the
    (order-independent) `calibrated_class_set` — no class added or dropped by the ranked view."""
    [result] = predictor.predict([_VH_1])
    seg = result.segments[gene]
    classes = seg["calibrated_classes"]
    assert [c["rank"] for c in classes] == list(range(1, len(classes) + 1))
    probs = [c["softmax_score"] for c in classes]
    assert probs == sorted(probs, reverse=True)
    assert sorted(c["class_id"] for c in classes) == seg["calibrated_class_set"]


@pytest.mark.parametrize("gene", ["V", "D", "J"])
def test_calibrated_classes_map_exhaustively_and_without_duplication(predictor, gene):
    """The per-class allele/gene/family lists union EXACTLY to the flattened `*_set` convenience
    fields (nothing added, nothing dropped), and no allele is emitted by more than one class."""
    [result] = predictor.predict([_VH_1])
    seg = result.segments[gene]
    classes = seg["calibrated_classes"]
    alleles_flat = [a for c in classes for a in c["alleles"]]
    assert sorted(set(alleles_flat)) == seg["calibrated_allele_set"]   # exhaustive + no extras
    assert len(alleles_flat) == len(set(alleles_flat))                 # never duplicated across classes
    assert sorted({g for c in classes for g in c["genes"]}) == seg["calibrated_gene_set"]
    assert sorted({f for c in classes for f in c["families"]}) == seg["calibrated_family_set"]


def test_calibrated_classes_scores_are_not_renormalized(predictor):
    """Scores are raw class posteriors: their sum equals the softmax mass over the covered classes,
    and is NEVER rescaled to sum to 1."""
    [result] = predictor.predict([_VH_1])
    seg = result.segments["V"]
    covered = seg["calibrated_class_set"]
    prob_row = class_probs(predictor.model, [_VH_1], predictor.cfg, predictor.device,
                           "v_class_logits")[0]
    total = sum(c["softmax_score"] for c in seg["calibrated_classes"])
    # (1) exactly the raw softmax posteriors of the covered classes — a renormalized set would sum
    #     to 1.0 instead of to this true covered mass.
    assert total == pytest.approx(sum(float(prob_row[cid]) for cid in covered))
    # (2) softmax assigns every one of the model's classes strictly positive mass, so any set that
    #     OMITS at least one class necessarily sums to < 1 (a renormalized set would be forced to 1).
    if len(covered) < len(prob_row):
        assert total < 1.0 - 1e-9


def test_calibrated_classes_D_carry_defensible_per_allele_probability(predictor):
    """D's head predicts allele indices directly, so each D entry maps to exactly one allele and its
    probability is that allele's own softmax posterior — the one place a per-allele score is honest."""
    [result] = predictor.predict([_VH_1])
    d = result.segments["D"]
    prob_row = class_probs(predictor.model, [_VH_1], predictor.cfg, predictor.device,
                           "d_class_logits")[0]
    for entry in d["calibrated_classes"]:
        assert len(entry["alleles"]) == 1
        assert entry["alleles"][0] == predictor.d_allele_order[entry["class_id"]]
        assert entry["softmax_score"] == pytest.approx(float(prob_row[entry["class_id"]]))


def test_calibrated_classes_one_class_keeps_one_score_across_expanded_alleles():
    """A single protein class that expands to three IMGT alleles yields exactly ONE ranked entry with
    ONE class-level probability and all three alleles — the score is never split per allele, and the
    flattened `calibrated_allele_set` is exactly that entry's `alleles`."""
    predictor = _stub_predictor_for_expansion()
    [result] = predictor.predict(["ACDEFGHIKL"])
    seg = result.segments["V"]
    [entry] = seg["calibrated_classes"]                 # one class, not three
    assert entry["class_id"] == 0 and entry["rank"] == 1
    assert entry["alleles"] == ["IGHV1-1*01", "IGHV1-1*02", "IGHV1-2*01"]   # 3 alleles ...
    assert entry["softmax_score"] == pytest.approx(1.0)                       # ... ONE probability
    assert seg["calibrated_allele_set"] == entry["alleles"]
