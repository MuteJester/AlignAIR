"""As-near alleles: every same-gene allele that covers a forward segment's observed span and mismatches the read at
no more aligned columns than the primary allele is a defensible call, so it is in the scored set."""
import pytest

from alignair_assay.core.cigar import parse_cigar, walk
from alignair_assay.core.reference import Reference, gene_name
from alignair_assay.testing import assert_oracle_perfect
from alignair_assay.truth.build import truth_from_genairr_record
from alignair_assay.truth.genairr import record_calls

ga = pytest.importorskip("GenAIRR")

from alignair_assay.suites.donors import sample_donors  # noqa: E402

pytestmark = pytest.mark.component

CONFIG = ga.data.HUMAN_IGH_OGRDB

V1 = "ACGTACGTACGT"  # the generative allele
HAND = Reference(
    alleles={"v": {"X*01": V1,
                   "X*02": V1[:5] + "G" + V1[6:],                    # differs at 5: ties with the read
                   "X*03": V1[:5] + "G" + V1[6:8] + "C" + V1[9:],    # differs at 5 and 8: strictly farther
                   "X*04": V1[:-1],                                  # shorter than the observed span
                   "Y*01": V1[:5] + "G" + V1[6:]},                   # another gene, as near as X*02
             "j": {"J*01": "TTTT"}},
    anchors={})


def _hand_record(read_v: str) -> dict:
    return {"sequence": read_v + "TTTT", "rev_comp": False,
            "v_call": "X*01", "truth_v_call": "X*01", "v_sequence_start": 0, "v_germline_start": 0,
            "v_cigar": f"{len(V1)}M", "d_call": "",
            "j_call": "J*01", "truth_j_call": "J*01", "j_sequence_start": len(V1), "j_germline_start": 0,
            "j_cigar": "4M"}


def test_tied_sibling_scored_strictly_farther_excluded():
    # SHM put a third base at position 5: the read mismatches X*01 and X*02 there alike.
    read_v = V1[:5] + "T" + V1[6:]
    truth = truth_from_genairr_record(_hand_record(read_v), HAND, case_id="tie", stratum="test")
    v = truth.segments["v"].alleles
    assert truth.segments["v"].observed.n_mismatch == 1
    assert v.primary == "X*01" and v.generative == {"X*01"}
    assert v.alleles == {"X*01", "X*02"}  # X*03 (2 mismatches), X*04 (short) and Y*01 (other gene) are out
    assert v.simulator_ties == {"X*01"} and not v.disagrees_with_simulator
    assert v.genes == {"X"}
    assert_oracle_perfect([truth], HAND)
    # The unmutated read: only X*01 is as near.
    clean = truth_from_genairr_record(_hand_record(V1), HAND, case_id="clean", stratum="test")
    assert clean.segments["v"].alleles.alleles == {"X*01"}


def test_n_bases_never_count():
    # Read N at 5 and at 8: X*01, X*02 and X*03 all mismatch the read nowhere.
    read_v = V1[:5] + "N" + V1[6:8] + "N" + V1[9:]
    truth = truth_from_genairr_record(_hand_record(read_v), HAND, case_id="n", stratum="test")
    v = truth.segments["v"].alleles
    assert truth.segments["v"].observed.n_mismatch == 0
    assert v.alleles == {"X*01", "X*02", "X*03"} and v.generative == {"X*01"}


def _mismatches(read, seg, sequence):
    return sum(read[c.read_pos] != "N" and read[c.read_pos] != sequence[c.germ_pos]
               for c in walk(parse_cigar(seg.cigar), seg.seq_start, seg.germ_start)
               if c.read_pos is not None and c.germ_pos is not None)


def test_cohort_scored_set_keeps_simulator_calls_and_primary(reference):
    donors = sample_donors(reference, 3, 31)
    records = ga.Experiment.on(CONFIG).recombine().mutate(rate=0.15).run_cohort(
        [d.genotype(CONFIG) for d in donors], n_per_subject=200, seed=4, expose_provenance=True).records
    grew = {"v": 0, "d": 0, "j": 0}
    for i, record in enumerate(records):
        truth = truth_from_genairr_record(record, reference, case_id=f"c{i}", stratum="test")
        for g, segment in truth.segments.items():
            a = segment.alleles
            assert a.primary in a.alleles and a.primary in a.generative
            calls = set(record_calls(record, g)[1])  # GenAIRR's post-SHM calls, in either encoding
            assert {c for c in calls if reference.has(c)} <= a.alleles
            extra = a.alleles - a.generative - calls
            if extra:
                assert not segment.inverted
                grew[g] += 1
            o = segment.observed
            for name in extra:
                # Same gene, covering the span, and no more mismatches than the primary.
                sequence = reference.seq(name)
                assert gene_name(name) == gene_name(a.primary) and len(sequence) >= o.germ_end
                assert _mismatches(truth.sequence, o, sequence) <= o.n_mismatch
    # The rule fires on V (heavily mutated V reads come near to a sibling); D and J siblings rarely cover the span.
    assert grew["v"] > 0, grew
