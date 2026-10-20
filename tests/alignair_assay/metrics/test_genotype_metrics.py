"""Genotype metrics: top-1 calls outside the donor's genotype, calls of its deleted genes, and excess alleles per
donor gene."""
import math

import pytest

from alignair_assay.core.genotype import DonorGenotype
from alignair_assay.core.prediction import PredSegment, Prediction
from alignair_assay.core.reference import Reference, gene_name
from alignair_assay.metrics import is_informational, lower_is_better
from alignair_assay.metrics.genotype import excess_alleles, genotype_metrics
from alignair_assay.scoring.pipeline import score_predictions
from alignair_assay.truth.alleles import AlleleTruth
from alignair_assay.truth.boundaries import BoundaryTruth
from alignair_assay.truth.model import ObservableTruth, SegmentTruth
from alignair_assay.truth.segment import ObservedSegment

OUTSIDE_V, OUTSIDE_J, DELETED = "v.outside_genotype", "j.outside_genotype", "v.deleted_gene_call"
# donor00 is heterozygous for IGHV1-2, homozygous for IGHV3-23 and IGHJ4, and has IGHV1-69 and IGHV4-34 deleted.
DONOR = DonorGenotype("donor00", carried=frozenset({"IGHV1-2*02", "IGHV1-2*04", "IGHV3-23*01", "IGHJ4*02"}),
                      deleted=frozenset({"IGHV1-69", "IGHV4-34"}))
OTHER = DonorGenotype("donor01", carried=frozenset({"IGHV3-23*04", "IGHJ6*02"}), deleted=frozenset({"IGHV1-2"}))


def _segment(primary, scored=(), length=100):
    """A forward segment observed over ``length`` germline bases; the scored set is ``primary`` plus ``scored``."""
    observed = ObservedSegment(primary, 0, length, 0, length, f"{length}M", 0, length, "A" * length)
    edge = (0, 0)
    alleles = frozenset({primary, *scored})
    truth = AlleleTruth(primary=primary, alleles=alleles, genes=frozenset(gene_name(a) for a in alleles),
                        simulator_ties=frozenset({primary}), disagrees_with_simulator=False,
                        generative=frozenset({primary}))
    return SegmentTruth(observed, BoundaryTruth(edge, edge, edge, edge), truth, observed_length=length,
                        inverted=False)


def _truth(case_id="c0", *, v=("IGHV1-2*02",), j=("IGHJ4*02",), v_length=100, j_length=40, subject="donor00",
           unscorable=(), excluded=None):
    """A read of ``subject`` whose V and J truths are ``(primary, *scored)`` (None: not observed)."""
    segments = {}
    if v is not None:
        segments["v"] = _segment(v[0], v[1:], v_length)
    if j is not None:
        segments["j"] = _segment(j[0], j[1:], j_length)
    return ObservableTruth(
        case_id=case_id, stratum="cohort", sequence="A" * 400, orientation=0, segments=segments,
        unscorable=frozenset(unscorable), junction=None, vj_in_frame=None, stop_codon=None, productive=None,
        clone_key=None, v_mismatches=None, v_frameshift=None, j_frameshift=None, excluded=excluded, warnings=(),
        groups={} if subject is None else {"subject": subject})


def _pred(v=None, j=None, case_id="c0"):
    """A prediction with top-1 calls ``v`` and ``j`` (None: no call)."""
    return Prediction(sequence_id=case_id, segments={"v": PredSegment(call=v), "j": PredSegment(call=j)})


def test_outside_genotype_hand():
    # A V truth whose scored set also holds a non-carried allele (a simulator re-call or an as-near allele).
    truth = _truth(v=("IGHV1-2*02", "IGHV1-3*01"))
    assert genotype_metrics(truth, _pred("IGHV1-2*02", "IGHJ4*02"), DONOR) == {OUTSIDE_V: 0.0, DELETED: 0.0,
                                                                                OUTSIDE_J: 0.0}
    # Carried but not in the truth set: wrong, but inside the genotype.
    assert genotype_metrics(truth, _pred("IGHV1-2*04", "IGHJ4*02"), DONOR)[OUTSIDE_V] == 0.0
    assert genotype_metrics(truth, _pred("IGHV3-23*01", "IGHJ4*02"), DONOR)[OUTSIDE_V] == 0.0
    # In the truth set but not carried: defensible, so not outside.
    assert genotype_metrics(truth, _pred("IGHV1-3*01", "IGHJ4*02"), DONOR)[OUTSIDE_V] == 0.0
    # Neither carried nor in the truth set.
    scores = genotype_metrics(truth, _pred("IGHV1-2*01", "IGHJ6*02"), DONOR)
    assert scores == {OUTSIDE_V: 1.0, DELETED: 0.0, OUTSIDE_J: 1.0}
    # The same calls are inside another donor's genotype only where it carries them.
    other = genotype_metrics(_truth(subject="donor01"), _pred("IGHV3-23*04", "IGHJ6*02"), OTHER)
    assert other == {OUTSIDE_V: 0.0, DELETED: 0.0, OUTSIDE_J: 0.0}

    # No call, or no prediction at all: applicable but not covered.
    no_v = genotype_metrics(truth, _pred(None, "IGHJ4*02"), DONOR)
    assert math.isnan(no_v[OUTSIDE_V]) and math.isnan(no_v[DELETED]) and no_v[OUTSIDE_J] == 0.0
    missing = genotype_metrics(truth, None, DONOR)
    assert set(missing) == {OUTSIDE_V, DELETED, OUTSIDE_J} and all(math.isnan(v) for v in missing.values())
    assert math.isnan(genotype_metrics(truth, Prediction(sequence_id="c0"), DONOR)[OUTSIDE_J])

    # Not applicable: a segment below its minimum observed length, not observed, unscorable, or an excluded read.
    assert set(genotype_metrics(_truth(v_length=19, j_length=9), _pred("IGHV1-2*01", "IGHJ6*02"), DONOR)) == set()
    assert set(genotype_metrics(_truth(v_length=20, j_length=10), _pred("IGHV1-2*01", "IGHJ6*02"), DONOR)) == {
        OUTSIDE_V, DELETED, OUTSIDE_J}
    assert set(genotype_metrics(_truth(v=None), _pred("IGHV1-2*01", "IGHJ6*02"), DONOR)) == {OUTSIDE_J}
    assert set(genotype_metrics(_truth(v=None, unscorable=("v",)), _pred("IGHV1-2*01"), DONOR)) == {OUTSIDE_J}
    assert genotype_metrics(_truth(excluded="non_biological_orientation"), _pred("IGHV1-2*01"), DONOR) == {}


def test_deleted_gene_call_hand():
    truth = _truth(v=("IGHV1-2*02", "IGHV1-69*02"))
    # An allele of a deleted gene outside the truth set.
    assert genotype_metrics(truth, _pred("IGHV1-69*01", "IGHJ4*02"), DONOR) == {OUTSIDE_V: 1.0, DELETED: 1.0,
                                                                                 OUTSIDE_J: 0.0}
    assert genotype_metrics(truth, _pred("IGHV4-34*01", "IGHJ4*02"), DONOR)[DELETED] == 1.0
    # An allele of a deleted gene that the truth set holds is defensible.
    assert genotype_metrics(truth, _pred("IGHV1-69*02", "IGHJ4*02"), DONOR)[DELETED] == 0.0
    # Outside the genotype but not of a deleted gene.
    assert genotype_metrics(truth, _pred("IGHV3-23*04", "IGHJ4*02"), DONOR)[DELETED] == 0.0
    # Deleted for one donor is carried by another: the donor's own genotype decides.
    assert genotype_metrics(_truth(subject="donor01"), _pred("IGHV1-2*04"), OTHER)[DELETED] == 1.0
    assert genotype_metrics(truth, _pred("IGHV1-2*04"), DONOR)[DELETED] == 0.0
    # No call: not covered. J genes are never deleted, so J has no deleted-gene metric.
    assert math.isnan(genotype_metrics(truth, _pred(None, "IGHJ4*02"), DONOR)[DELETED])
    assert math.isnan(genotype_metrics(truth, None, DONOR)[DELETED])
    assert "j.deleted_gene_call" not in genotype_metrics(truth, _pred("IGHV1-69*01", "IGHJ6*02"), DONOR)


def test_lower_is_better_registry():
    for name in (OUTSIDE_V, OUTSIDE_J, DELETED):
        assert lower_is_better(name) and not is_informational(name), name
    assert not lower_is_better("v.top1_in_set")


def test_score_predictions_applies_genotypes():
    hand = Reference(alleles={"v": {"IGHV1-2*02": "A" * 300}, "j": {"IGHJ4*02": "A" * 50}}, anchors={})
    truths = [_truth("c0"), _truth("c1", v=("IGHV3-23*04",), j=("IGHJ6*02",), subject="donor01"),
              _truth("c2", subject=None), _truth("c3", excluded="non_biological_orientation")]
    genotypes = {"donor00": DONOR, "donor01": OTHER}
    preds = {"c0": _pred("IGHV1-69*01", "IGHJ4*02", "c0"), "c1": None, "c2": _pred("IGHV1-69*01", None, "c2")}
    scores = score_predictions(truths, preds, hand, genotypes=genotypes)
    assert list(scores) == ["c0", "c1", "c2"]
    assert (scores["c0"][OUTSIDE_V], scores["c0"][DELETED], scores["c0"][OUTSIDE_J]) == (1.0, 1.0, 0.0)
    assert all(math.isnan(scores["c1"][name]) for name in (OUTSIDE_V, DELETED, OUTSIDE_J))
    # A case without a subject has no donor genotype.
    assert not {OUTSIDE_V, DELETED, OUTSIDE_J} & set(scores["c2"])
    # Without genotypes, no genotype metric at all.
    plain = score_predictions(truths, preds, hand)
    assert not any(name.endswith(("outside_genotype", "deleted_gene_call")) for case in plain.values()
                   for name in case)
    # A subject without a genotype is an error, naming the case and the subject.
    with pytest.raises(ValueError, match="c1.*donor01"):
        score_predictions(truths, preds, hand, genotypes={"donor00": DONOR})
    # Each truth's primary allele is inside its donor's genotype.
    primaries = {t.case_id: _pred(t.segments["v"].alleles.primary, t.segments["j"].alleles.primary, t.case_id)
                 for t in truths}
    primary = score_predictions(truths, primaries, hand, genotypes=genotypes)
    assert [primary[cid][name] for cid in ("c0", "c1") for name in (OUTSIDE_V, DELETED, OUTSIDE_J)] == [0.0] * 6


def test_excess_alleles_hand():
    calls = [
        # donor00 IGHV1-2 (two carried alleles): two alleles called on 3 reads each, one more on 2 reads.
        *[("donor00", "IGHV1-2*02")] * 3, *[("donor00", "IGHV1-2*04")] * 3, *[("donor00", "IGHV1-2*01")] * 2,
        # donor00 IGHV3-23 (one carried allele): two alleles called on 3 reads each, an excess.
        *[("donor00", "IGHV3-23*01")] * 4, *[("donor00", "IGHV3-23*04")] * 3,
        # donor00 IGHV1-69 is deleted: one allele called on 3 reads is an excess over none.
        *[("donor00", "IGHV1-69*01")] * 3,
        # donor00 IGHV4-34: 2 called reads, too few to count the gene.
        *[("donor00", "IGHV4-34*01")] * 2,
        # donor01 IGHV3-23 (one carried allele): one allele called, as many as carried.
        *[("donor01", "IGHV3-23*04")] * 5,
    ]
    truths, preds = [], {}
    for index, (subject, v_call) in enumerate(calls):
        case_id = f"c{index}"
        if subject == "donor00":
            truths.append(_truth(case_id))
            preds[case_id] = _pred(v_call, "IGHJ4*02", case_id)
        else:
            truths.append(_truth(case_id, v=("IGHV3-23*04",), j=("IGHJ6*02",), subject=subject))
            preds[case_id] = _pred(v_call, "IGHJ6*02", case_id)
    # Reads that do not count: no V call, no row, a V too short to require a call, unscorable V, excluded, no
    # subject. Any of the last four counted would make donor00 IGHV4-34 a counted pair (3 or more reads).
    truths += [_truth("n0"), _truth("n1"), _truth("n2", v_length=10), _truth("n3", v=None, unscorable=("v",)),
               _truth("n4", excluded="non_biological_orientation"), _truth("n5", subject=None)]
    preds |= {"n0": _pred(None, "IGHJ1*01", "n0"), "n2": _pred("IGHV4-34*01", None, "n2"),
              "n3": _pred("IGHV4-34*01", None, "n3"), "n4": _pred("IGHV4-34*01", None, "n4"),
              "n5": _pred("IGHV4-34*01", None, "n5")}
    genotypes = {"donor00": DONOR, "donor01": OTHER}

    table = excess_alleles(truths, preds, genotypes)
    # V: donor genes with >= 3 called reads: donor00 IGHV1-2, IGHV3-23, IGHV1-69 and donor01 IGHV3-23; two exceed.
    assert table["min_reads"] == 3
    assert table["v"] == {"n": 4, "excess": 2, "fraction": 0.5}
    # J: donor00 IGHJ4 (carried, one allele) and donor01 IGHJ6 (one allele): none exceeds.
    assert table["j"] == {"n": 2, "excess": 0, "fraction": 0.0}
    assert set(table) == {"min_reads", "v", "j"}

    # min_reads sets both thresholds: at 2, IGHV1-2*01 counts too (3 > 2), and IGHV4-34 is a deleted-gene pair.
    assert excess_alleles(truths, preds, genotypes, min_reads=2)["v"] == {"n": 5, "excess": 4, "fraction": 0.8}
    # Nothing to count: no pair, no fraction.
    assert excess_alleles(truths, {}, genotypes)["v"] == {"n": 0, "excess": 0, "fraction": None}
    # The oracle calls each read's primary allele, which the donor carries: no excess.
    oracle = {t.case_id: _pred(t.segments["v"].alleles.primary, t.segments["j"].alleles.primary, t.case_id)
              for t in truths if "v" in t.segments}
    assert excess_alleles(truths, oracle, genotypes)["v"]["excess"] == 0
    with pytest.raises(ValueError, match="donor01"):
        excess_alleles(truths, preds, {"donor00": DONOR})
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="min_reads"):
            excess_alleles(truths, preds, genotypes, min_reads=bad)
