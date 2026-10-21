"""R1 for real reads (``real.metrics``): each tool's top-1 V call against the donor's genomic alleles.

The offline tests build a tiny shared reference (IMGT-gapped V alleles named like OGRDB's F/G labels), its IUIS map,
a donor genotype with one gene per class and the reads' truths in the test. The gated test scores an oracle (each
known answer's own shared allele) on the pilot donors W-97 and SC-23 when ``ALIGNAIR_ASSAY_DATA`` names the data
root, ``ALIGNAIR_ASSAY_DOCKER=1`` lets the contig BAMs be read, and Task 4's preprocessed outputs are in
``.private/assay/p3a_task4_run``.
"""
from __future__ import annotations

import ast
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import pytest

import alignair_assay
from alignair_assay import metrics as registry
from alignair_assay.core.prediction import PredSegment, Prediction
from alignair_assay.core.reference import Reference
from alignair_assay.metrics import is_informational, lower_is_better
from alignair_assay.real.genotype import IuisMap, RealAllele, RealGenotype, merge_identical
from alignair_assay.real.metrics import (
    EXCLUSIONS,
    MIN_CLONOTYPES,
    MIN_NAIVE_CLONOTYPES,
    NO_CALL,
    SUPPORTED,
    UNSUPPORTED_ALLELE,
    UNSUPPORTED_GENE,
    ReadCalls,
    SupportIndex,
    classify_v_call,
    decoy_calls,
    exclusion_counts,
    gene_support,
    oracle_prediction,
    oracle_self_check,
    passing_genes,
    real_case_scores,
    shared_exclusions,
    through_318,
)
from alignair_assay.real.truth import RealTruth

DONOR = "SC-23"
# IMGT gap columns (0-based, half-open) of the synthetic V alleles: codon 104 (columns 309-311) is TGT.
GAPS = ((27, 30), (72, 84), (165, 171), (195, 198))


def _random(n: int, seed: int) -> str:
    return "".join(random.Random(seed).choices("ACGT", k=n))


def _gapped(seed: int, *mutated: int) -> str:
    """A 320-column IMGT-gapped V-REGION with Cys104 at columns 309-311, a base changed at each ``mutated`` column
    (0-based; column 318 is IMGT position 319)."""
    bases = list(_random(320, seed))
    bases[309:312] = "TGT"
    for column in mutated:
        bases[column] = "A" if bases[column] != "A" else "C"
    for start, end in GAPS:
        bases[start:end] = "." * (end - start)
    return "".join(bases)


# Shared V allele -> (IMGT-gapped sequence, IUIS labels with its sequence).
SHARED_V = {
    "IGHVF1-G1*01": (_gapped(1), ("IGHV1-2*02",)),
    "IGHVF1-G1*02": (_gapped(1, 150), ("IGHV1-2*04",)),
    "IGHVF6-G21*06": (_gapped(2), ("IGHV1-69*01", "IGHV1-69D*01")),  # an identical-sequence paralog pair
    "IGHVF6-G21*07": (_gapped(2, 150), ("IGHV1-69*02", "IGHV1-69D*02")),
    "IGHVF6-G21*08": (_gapped(2, 160), ("IGHV1-69*04",)),
    "IGHVF3-G9*04": (_gapped(3), ("IGHV4-28*01",)),
    "IGHVF3-G9*04_A319G": (_gapped(3, 318), ("IGHV4-28*03",)),  # differs only beyond IMGT 318
    "IGHVF4-G1*01": (_gapped(4), ("IGHV3-30*01",)),
    "IGHVF4-G1*02": (_gapped(4, 150), ("IGHV3-30*02",)),
    "IGHVF5-G1*01": (_gapped(5), ("IGHV3-23*01",)),
    "IGHVF7-G1*01": (_gapped(7), ("IGHV3-11*01",)),
    "IGHVF8-G1*01": (_gapped(8), ("IGHV3-62*03",)),
    "IGHVF8-G1*02": (_gapped(8, 150), ("IGHV3-62*01",)),
    "IGHVF9-G1*01": (_gapped(9), ("IGHV3-15*01",)),
    "IGHVF9-G1*02": (_gapped(9, 150), ("IGHV3-15*02",)),
    "IGHVF10-G33*01": (_gapped(10), ()),  # no label has its sequence; IGHV3-13*06 is inferred (INFERRED)
    "IGHVF10-G99*01": (_gapped(17), ()),  # no label and none inferred: unmapped
    "IGHVF2-G1*01": (_gapped(11), ("IGHV1-46*01",)),
    "IGHVF2-G1*02": (_gapped(11, 150), ("IGHV1-46*03",)),
    "IGHVF11-G1*01": (_gapped(12), ("IGHV5-51*01",)),
    "IGHVF11-G1*02": (_gapped(12, 150), ("IGHV5-51*02",)),
    "IGHVF12-G1*01": (_gapped(13), ("IGHV4-4*02",)),
    "IGHVF12-G1*02": (_gapped(13, 318), ("IGHV4-4*07",)),  # IGHV4-4*02 through 318; a nonfunctional donor allele
    "IGHVF13-G1*01": (_gapped(14), ("IGHV4-59*01",)),
    "IGHVF13-G1*02": (_gapped(14, 150), ("IGHV4-59*02",)),  # a nonfunctional donor allele, before 318
    "IGHVF14-G1*01": (_gapped(15), ("IGHV2-70*01", "IGHV2-70D*01")),  # both genes captured, neither carried
    "IGHVF15-G1*01": (_gapped(16), ("IGHV3-64*01", "IGHV3-64D*01")),  # IGHV3-64D not captured
    "IGHVF16-G1*01": (_gapped(18), ("IGHV4-30-2*01",)),  # spanned and held by the contigs, not called: uncalled
    "IGHVF17-G1*01": (_gapped(19), ("IGHV3-30-5*01", "IGHV3-30*18")),  # IGHV3-30-5 uncalled, IGHV3-30 carried
}
# Shared V allele without a label -> the OGRDB labels it equals through IMGT 318 (``IuisMap.inferred``).
INFERRED = {"IGHVF10-G33*01": ("IGHV3-13*06",)}
D_SEQ = {"IGHD3-10*01": _random(31, 20), "IGHD3-3*01": _random(31, 21), "IGHD2-2*01": _random(31, 22),
         "IGHD4-4*01": _random(20, 23), "IGHD2-15*01": _random(31, 24), "IGHD1-26*01": _random(20, 26)}
J_SEQ = {"IGHJ4*02": _random(48, 30), "IGHJ6*02": _random(63, 31), "IGHJ3*02": _random(50, 32)}
REFERENCE = Reference(
    alleles={"v": {name: gapped.replace(".", "") for name, (gapped, _) in SHARED_V.items()},
             "d": {**D_SEQ, "IGHD4-11*01": D_SEQ["IGHD4-4*01"]}, "j": dict(J_SEQ)},
    anchors={}, gapped_v={name: gapped for name, (gapped, _) in SHARED_V.items()})
LABELS = {**{name: labels for name, (_, labels) in SHARED_V.items()}, **{name: (name,) for name in D_SEQ},
          "IGHD4-11*01": ("IGHD4-11*01",), **{name: (name,) for name in J_SEQ}}


def _iuis_map(reference: Reference, labels: dict[str, tuple[str, ...]],
              inferred: dict[str, tuple[str, ...]]) -> IuisMap:
    """The ``IuisMap`` that ``IuisMap.from_ogrdb`` gives for OGRDB labels ``labels`` (shared name -> labels) and the
    labels ``inferred`` for unlabelled shared alleles."""
    by_sequence, shared = defaultdict(set), defaultdict(list)
    for name, sequence in reference.sequences().items():
        by_sequence[sequence].update(labels.get(name, ()))
        shared[sequence].append(name)
    labels_by_sequence = {sequence: tuple(sorted(names)) for sequence, names in by_sequence.items() if names}
    return IuisMap(labels_by_shared={name: labels_by_sequence.get(sequence, ())
                                     for name, sequence in reference.sequences().items()},
                   shared_by_sequence={sequence: tuple(sorted(names)) for sequence, names in shared.items()},
                   labels_by_sequence=labels_by_sequence, reference_digest=reference.digest(),
                   inferred_by_shared=dict(inferred))


IUIS = _iuis_map(REFERENCE, LABELS, INFERRED)
MERGED = merge_identical(REFERENCE)


def _allele(allele_id: str, haplotype: int, *, shared_as: str | None = None, gapped: str | None = None,
            gene: str | None = None, segment: str = "v", functional: bool = True,
            keep_gapped: bool = True) -> RealAllele:
    """A donor allele: by default the sequence of the shared allele ``shared_as`` (or the gapped V ``gapped``),
    with its ``shared`` and ``iuis`` names by exact sequence, as the VDJbase loader maps them."""
    if segment == "v":
        gapped = gapped or SHARED_V[shared_as][0]
        sequence = gapped.replace(".", "")
    else:
        sequence = REFERENCE.seq(shared_as)
    return RealAllele(allele_id=allele_id, gene=gene or allele_id.split("*")[0], segment=segment, sequence=sequence,
                      gapped=gapped if segment == "v" and keep_gapped else None, haplotype=haplotype,
                      functional=functional, shared=IUIS.shared_for_sequence(sequence),
                      iuis=IUIS.labels_for_sequence(sequence))


ALLELES = (
    _allele("IGHV1-2*02", 1, shared_as="IGHVF1-G1*01"),
    _allele("IGHV1-2*02", 2, shared_as="IGHVF1-G1*01"),
    _allele("IGHV1-69D*01", 0, shared_as="IGHVF6-G21*06"),  # found at IGHV1-69D: the shared allele's second label
    _allele("IGHV4-28*01", 1, shared_as="IGHVF3-G9*04"),
    _allele("IGHV4-28*01", 2, shared_as="IGHVF3-G9*04"),
    _allele("IGHV3-30*01", 1, shared_as="IGHVF4-G1*01"),
    _allele("IGHV3-30*18_t189c", 2, gapped=_gapped(4, 188)),  # novel: absent from the shared reference
    _allele("IGHV3-62*03", 1, shared_as="IGHVF8-G1*01", functional=False),
    _allele("IGHV3-62*03", 2, shared_as="IGHVF8-G1*01", functional=False),
    _allele("IGHV3-15*01", 1, shared_as="IGHVF9-G1*01"),  # haplotype 2 not spanned: half resolved
    _allele("IGHV1-46*01", 1, shared_as="IGHVF2-G1*01"),
    _allele("IGHV1-46*01", 2, shared_as="IGHVF2-G1*01"),
    _allele("IGHV5-51*01", 0, shared_as="IGHVF11-G1*01", keep_gapped=False),  # no usable gapped form
    _allele("IGHV4-4*02", 1, shared_as="IGHVF12-G1*01"),
    _allele("IGHV4-4*07", 2, shared_as="IGHVF12-G1*02", functional=False),
    _allele("IGHV4-59*01", 1, shared_as="IGHVF13-G1*01"),
    _allele("IGHV4-59*02", 2, shared_as="IGHVF13-G1*02", functional=False),
    _allele("IGHD3-10*01", 1, shared_as="IGHD3-10*01", segment="d"),
    RealAllele(allele_id="IGHD2-15*01_a5c", gene="IGHD2-15", segment="d", sequence=_random(31, 25), gapped=None,
               haplotype=0, functional=True, shared=(), iuis=()),  # a D allele absent from the shared reference
    _allele("IGHJ4*02", 0, shared_as="IGHJ4*02", segment="j"),
)
V_ALLELES = tuple(allele for allele in ALLELES if allele.segment == "v")
DROPPED = (RealAllele(allele_id="IGHD3-3*01_g8-", gene="IGHD3-3", segment="d", sequence=_random(30, 40),
                      gapped=None, haplotype=1, functional=True, shared=(), iuis=()),)
CARRIED = {allele.gene for allele in ALLELES}
# IGHV3-11, IGHV1-69, IGHV2-70, IGHV2-70D, IGHV3-64, IGHV3-13, IGHD2-2 and IGHJ6 are spanned and not carried, and
# absent from every capturing contig; IGHV4-30-2, IGHV3-30-5, IGHD1-26 and IGHJ3 are spanned and not carried, but the
# contigs hold them (uncalled); IGHV3-23, IGHV3-64D and IGHD4-4 / IGHD4-11 are not spanned.
UNCALLED = frozenset({"IGHV4-30-2", "IGHV3-30-5", "IGHD1-26", "IGHJ3"})
CAPTURED = frozenset(CARRIED | UNCALLED | {"IGHV3-11", "IGHV1-69", "IGHV2-70", "IGHV2-70D", "IGHV3-64", "IGHV3-13",
                                           "IGHD2-2", "IGHJ6"})
GENOTYPE = RealGenotype(donor=DONOR, alleles=ALLELES, captured=CAPTURED,
                        resolved=CAPTURED - {"IGHV3-15"} - UNCALLED, dropped=DROPPED, uncalled=UNCALLED)
# Clonotypes and naive clonotypes per gene, from the known answers: IGHV1-46 has too few naive clonotypes.
SUPPORT = {**{gene: (MIN_CLONOTYPES, MIN_NAIVE_CLONOTYPES) for gene in CARRIED},
           "IGHV1-46": (MIN_CLONOTYPES + 5, MIN_NAIVE_CLONOTYPES - 1)}
# The R1 primary: no gene-support thresholds. The sensitivity variant applies Mikelov et al. 2024's thresholds.
INDEX = SupportIndex(GENOTYPE, IUIS, MERGED, REFERENCE, support=SUPPORT)
THRESHOLDED = SupportIndex(GENOTYPE, IUIS, MERGED, REFERENCE, support=SUPPORT, thresholds=True)


def _pred(read_id: str = "r", v: str | None = None, *listed: str, d: str | None = None,
          j: str | None = None) -> Prediction:
    """A tool's row with top-1 V ``v`` (then ``listed``, as a ``v_call`` list), D ``d`` and J ``j``."""
    calls = [name for name in (v, *listed) if name]
    return Prediction(sequence_id=read_id, segments={
        "v": PredSegment(call=v, call_set=tuple(calls)), "d": PredSegment(call=d, call_set=(d,) if d else ()),
        "j": PredSegment(call=j, call_set=(j,) if j else ())})


def _truth(read_id: str, *, known: tuple[str, ...] = (), shared: tuple[str, ...] = (), gene: str | None = None,
           naive: bool = False, clonotype: str | None = None, donor: str = DONOR) -> RealTruth:
    """A read's truth: a known answer when ``known`` names donor alleles (novel when ``shared`` is empty)."""
    return RealTruth(read_id=read_id, donor=donor, isotype="IGHM" if naive else "IGHG", protocol="race",
                     observed=(0, 400), known=frozenset(known) if known else None,
                     known_shared=frozenset(shared) if known else None, known_gene=gene if known else None,
                     novel_known=bool(known) and not shared, naive=naive and bool(known),
                     clonotype=clonotype or f"{donor}/single:{read_id}", anchor="donor")


# --- one class per call -------------------------------------------------------------------------------------------

def test_wrong_allele_of_a_carried_gene_is_unsupported_allele():
    assert classify_v_call(_pred(v="IGHVF1-G1*02"), GENOTYPE, IUIS, MERGED, reference=REFERENCE,
                           support=SUPPORT, truth=None) == UNSUPPORTED_ALLELE
    assert classify_v_call(_pred(v="IGHVF1-G1*01"), GENOTYPE, IUIS, MERGED, reference=REFERENCE,
                           support=SUPPORT, truth=None) == SUPPORTED


def test_call_to_a_captured_gene_the_donor_lacks_is_unsupported_gene():
    assert INDEX.classify("IGHVF7-G1*01") == UNSUPPORTED_GENE  # IGHV3-11: spanned on both haplotypes, no allele
    assert INDEX.classify("IGHVF6-G21*08") == UNSUPPORTED_GENE  # IGHV1-69*04: IGHV1-69 is spanned, not carried


def test_call_to_an_uncaptured_gene_is_excluded():
    assert INDEX.classify("IGHVF5-G1*01") == "excluded:not_captured"  # IGHV3-23: its region is not spanned


def test_call_to_a_gene_the_contigs_hold_and_the_genotype_does_not_call_is_excluded():
    # VDJbase's calls can miss a gene: IGHV4-30-2 is spanned, and a capturing contig holds sequence there, so it is
    # not a gene the donor lacks, and a call to it, right or wrong, is excluded
    assert "IGHV4-30-2" in CAPTURED and not GENOTYPE.carried("IGHV4-30-2")
    assert INDEX.classify("IGHVF16-G1*01") == INDEX.classify("IGHVF16-G1*01", documented=True) == "excluded:uncalled"
    # D and J alike (informational)
    assert INDEX.classify("IGHD1-26*01", "d") == "excluded:uncalled"
    assert INDEX.classify("IGHJ3*02", "j") == "excluded:uncalled"
    # a paralog allele whose other gene is carried is still excluded by its uncalled gene
    assert IUIS.genes("IGHVF17-G1*01") == ("IGHV3-30", "IGHV3-30-5") and GENOTYPE.carried("IGHV3-30")
    assert INDEX.classify("IGHVF17-G1*01") == "excluded:uncalled"
    # a gene deleted on every capturing contig is one the donor lacks: unsupported_gene
    assert INDEX.classify("IGHVF7-G1*01") == UNSUPPORTED_GENE
    assert INDEX.classify("IGHD2-2*01", "d") == INDEX.classify("IGHJ6*02", "j") == UNSUPPORTED_GENE
    # not captured comes first, uncalled next
    assert EXCLUSIONS[:3] == ("excluded:dropped_truth", "excluded:not_captured", "excluded:uncalled")
    # a genotype whose uncalled genes are unknown is refused
    with pytest.raises(ValueError, match="uncalled"):
        SupportIndex(RealGenotype(donor=DONOR, alleles=ALLELES, captured=CAPTURED, resolved=CAPTURED),
                     IUIS, MERGED, REFERENCE, support=SUPPORT)


def test_call_differing_from_the_donor_allele_only_beyond_318_is_supported():
    # Peres et al. 2023: alleles that differ only past IMGT 318 cannot be told apart from rearranged reads. The
    # donor carries IGHV4-28*01 (IGHVF3-G9*04); IGHVF3-G9*04_A319G differs from it at IMGT 319 alone.
    a319g = REFERENCE.gapped("IGHVF3-G9*04_A319G")
    assert a319g[:318] == REFERENCE.gapped("IGHVF3-G9*04")[:318] and a319g != REFERENCE.gapped("IGHVF3-G9*04")
    assert INDEX.classify("IGHVF3-G9*04_A319G") == SUPPORTED
    assert INDEX.classify("IGHVF3-G9*04") == SUPPORTED


def test_multi_iuis_shared_allele_is_supported_through_its_second_label():
    # IGHVF6-G21*06 is IGHV1-69*01 and IGHV1-69D*01; the donor carries the sequence at IGHV1-69D only
    assert IUIS.genes("IGHVF6-G21*06") == ("IGHV1-69", "IGHV1-69D") and "IGHV1-69" not in CARRIED
    assert INDEX.classify("IGHVF6-G21*06") == SUPPORTED
    # an unsupported paralog call is allele-level when either of its genes is carried
    assert INDEX.classify("IGHVF6-G21*07") == UNSUPPORTED_ALLELE


def test_paralog_calls_by_both_genes():
    # IGHVF14-G1*01 is IGHV2-70*01 and IGHV2-70D*01: both regions captured, neither gene carried
    assert {"IGHV2-70", "IGHV2-70D"} <= CAPTURED and not {"IGHV2-70", "IGHV2-70D"} & CARRIED
    assert INDEX.classify("IGHVF14-G1*01") == UNSUPPORTED_GENE
    # IGHVF15-G1*01 is IGHV3-64*01 and IGHV3-64D*01: IGHV3-64D's region is not captured, so the call may be an
    # allele of it the donor's long reads did not resolve
    assert "IGHV3-64" in CAPTURED and "IGHV3-64D" not in CAPTURED
    assert INDEX.classify("IGHVF15-G1*01") == "excluded:not_captured"


def test_a_match_through_318_to_a_functional_allele_is_supported_even_if_exactly_a_nonfunctional_one():
    # IGHVF12-G1*02 is exactly the donor's nonfunctional IGHV4-4*07 and equals its functional IGHV4-4*02 through 318
    assert INDEX.classify("IGHVF12-G1*02") == SUPPORTED and INDEX.classify("IGHVF12-G1*01") == SUPPORTED


def test_a_call_equal_to_a_nonfunctional_allele_of_a_gene_with_a_functional_one_is_supported():
    # the donor's genome carries IGHV4-59*02 (a V-REGION defect note) beside the functional IGHV4-59*01: a call of it
    # is supported, as R1 asks whether the genome carries the call
    assert INDEX.classify("IGHVF13-G1*02") == SUPPORTED and INDEX.classify("IGHVF13-G1*01") == SUPPORTED


def test_a_gene_carried_only_as_nonfunctional_alleles_excludes_every_call_to_it():
    # IGHV3-62's donor alleles are all nonfunctional: a call equal to one and a call equal to none are excluded alike
    assert all(not allele.functional for allele in GENOTYPE.carried("IGHV3-62"))
    assert IUIS.genes("IGHVF8-G1*01") == IUIS.genes("IGHVF8-G1*02") == ("IGHV3-62",)
    assert IUIS.shared_for_sequence(GENOTYPE.carried("IGHV3-62")[0].sequence) == ("IGHVF8-G1*01",)
    assert INDEX.classify("IGHVF8-G1*01") == "excluded:nonfunctional"  # matches the donor's IGHV3-62*03
    assert INDEX.classify("IGHVF8-G1*02") == "excluded:nonfunctional"  # matches no donor allele


def test_no_call_is_no_call():
    assert INDEX.classify_v(_pred(v=None), truth=None) == NO_CALL
    assert INDEX.classify_v(None, truth=None) == NO_CALL  # the tool emitted no row
    assert INDEX.classify("", "v") == NO_CALL
    with pytest.raises(ValueError, match="IGHV1-2\\*02"):
        INDEX.classify("IGHV1-2*02")  # an IUIS name: not an allele of the shared reference


def test_the_reads_truth_is_required_by_keyword():
    # every tool is classified with the read's truth, so no caller can classify one tool with it and the other
    # without: leaving it out, or passing it by position, is an error
    pred = _pred(v="IGHVF4-G1*02")
    for call in (lambda: INDEX.classify_v(pred), lambda: INDEX.read_calls(pred),
                 lambda: INDEX.classify_v(pred, _truth("r")), lambda: INDEX.read_calls(pred, _truth("r")),
                 lambda: classify_v_call(pred, GENOTYPE, IUIS, MERGED, reference=REFERENCE, support=SUPPORT)):
        with pytest.raises(TypeError, match="'truth'|positional argument"):
            call()


def test_the_other_exclusions():
    # each is a property of the call's genes, so a right and a wrong call to the gene are excluded alike
    # a gene with an allele absent from the shared reference
    assert INDEX.classify("IGHVF4-G1*01") == INDEX.classify("IGHVF4-G1*02") == "excluded:novel_gene"
    # a gene whose donor alleles all have a V-REGION defect
    assert INDEX.classify("IGHVF8-G1*01") == INDEX.classify("IGHVF8-G1*02") == "excluded:nonfunctional"
    # a gene known on one haplotype only: its unknown second allele could be any call
    assert INDEX.classify("IGHVF9-G1*01") == INDEX.classify("IGHVF9-G1*02") == "excluded:half_resolved"
    # a gene with too few naive clonotypes among the known answers: excluded only in the gene-support variant; the R1
    # primary scores it like any carried gene
    assert THRESHOLDED.classify("IGHVF2-G1*01") == THRESHOLDED.classify("IGHVF2-G1*02") == "excluded:low_support"
    assert INDEX.classify("IGHVF2-G1*01") == SUPPORTED and INDEX.classify("IGHVF2-G1*02") == UNSUPPORTED_ALLELE
    # a gene with a functional donor allele without a usable IMGT-gapped form: no call can be compared through 318
    assert INDEX.classify("IGHVF11-G1*01") == INDEX.classify("IGHVF11-G1*02") == "excluded:beyond_318"
    assert set(EXCLUSIONS) == {"excluded:not_captured", "excluded:dropped_truth", "excluded:uncalled",
                               "excluded:unmapped", "excluded:novel_gene", "excluded:nonfunctional",
                               "excluded:beyond_318", "excluded:half_resolved", "excluded:low_support"}


def test_the_gene_support_thresholds_are_a_sensitivity_variant():
    # the R1 primary never excludes a call for low support: a thin carried gene is scored like any other, so its
    # supported calls are not dropped while calls to genes the donor lacks (which have no known answers) stay
    primary = {INDEX.classify(name) for name in SHARED_V}
    assert "excluded:low_support" not in primary
    variant = {name: THRESHOLDED.classify(name) for name in SHARED_V}
    assert {name for name, cls in variant.items() if cls == "excluded:low_support"} == {"IGHVF2-G1*01",
                                                                                         "IGHVF2-G1*02"}
    # the variant changes only classes the primary scores, never another exclusion
    assert all(variant[name] in ("excluded:low_support", INDEX.classify(name)) for name in SHARED_V)
    assert classify_v_call(_pred(v="IGHVF2-G1*01"), GENOTYPE, IUIS, MERGED, reference=REFERENCE, support=SUPPORT,
                           thresholds=True, truth=None) == "excluded:low_support"
    assert classify_v_call(_pred(v="IGHVF2-G1*01"), GENOTYPE, IUIS, MERGED, reference=REFERENCE,
                           truth=None) == SUPPORTED
    # the thresholds need the gene support; the primary does not
    with pytest.raises(ValueError, match="support"):
        SupportIndex(GENOTYPE, IUIS, MERGED, REFERENCE, thresholds=True)
    assert SupportIndex(GENOTYPE, IUIS, MERGED, REFERENCE).classify("IGHVF2-G1*01") == SUPPORTED

def test_novel_gene_is_not_excluded_on_a_documented_known_answer():
    # the read's V is exactly a documented donor allele (in the shared reference), so no novel allele can hide
    # behind a call there: a call to the gene with a novel allele is judged like any other
    documented = _truth("x", known=("IGHV3-30*01",), shared=("IGHVF4-G1*01",), gene="IGHV3-30")
    assert INDEX.classify_v(_pred("x", "IGHVF4-G1*01"), truth=documented) == SUPPORTED
    assert INDEX.classify_v(_pred("x", "IGHVF4-G1*02"), truth=documented) == UNSUPPORTED_ALLELE
    assert INDEX.read_calls(_pred("x", "IGHVF4-G1*02"), truth=documented).v == UNSUPPORTED_ALLELE
    assert classify_v_call(_pred("x", "IGHVF4-G1*02"), GENOTYPE, IUIS, MERGED, reference=REFERENCE, support=SUPPORT,
                           truth=documented) == UNSUPPORTED_ALLELE
    assert INDEX.classify("IGHVF4-G1*02", documented=True) == UNSUPPORTED_ALLELE
    # the same call on a read without a known answer, or with a novel one, stays excluded, right or wrong
    for truth in (_truth("u"), _truth("n", known=("IGHV3-30*18_t189c",), gene="IGHV3-30"), None):
        read_id = truth.read_id if truth else "r"
        for call in ("IGHVF4-G1*01", "IGHVF4-G1*02"):
            assert INDEX.classify_v(_pred(read_id, call), truth=truth) == "excluded:novel_gene", (truth, call)
    # the flag is the read's, so both tools get it: the wrong call is scored, not excluded, for both
    classes = [INDEX.classify_v(_pred("x", call), truth=documented) for call in ("IGHVF4-G1*01", "IGHVF4-G1*02")]
    assert shared_exclusions({"x": classes[0]}, {"x": classes[1]}) == set()
    # it is V's only: a D gene with an allele absent from the shared reference stays excluded
    assert INDEX.classify("IGHD2-15*01", "d") == INDEX.classify("IGHD2-15*01", "d", documented=True) == (
        "excluded:novel_gene")
    calls = INDEX.read_calls(_pred("x", "IGHVF4-G1*02", d="IGHD2-15*01"), truth=documented)
    assert calls.v == UNSUPPORTED_ALLELE and calls.d == "excluded:novel_gene"


def test_an_unlabelled_shared_allele_is_classified_by_its_inferred_gene():
    # IGHVF10-G33*01 has no label of its own; IuisMap infers IGHV3-13*06 through IMGT 318, a gene the donor lacks
    assert IUIS.labels("IGHVF10-G33*01") == () and IUIS.genes("IGHVF10-G33*01") == ("IGHV3-13",)
    assert INDEX.classify("IGHVF10-G33*01") == UNSUPPORTED_GENE
    # unmapped is a guard for an allele with no gene at all; HUMAN_IGH_OGRDB has none (the gated test)
    assert IUIS.unmapped() == ("IGHVF10-G99*01",) and INDEX.classify("IGHVF10-G99*01") == "excluded:unmapped"


def test_exclusion_never_depends_on_whether_the_call_is_right():
    # every pair of calls to the same genes, one right and one wrong, is excluded alike: the class of an excluded
    # call is a function of its IUIS genes alone
    by_genes = defaultdict(set)
    for name in SHARED_V:
        by_genes[IUIS.genes(name)].add(INDEX.classify(name))
    for genes, classes in by_genes.items():
        excluded = {cls for cls in classes if cls.startswith("excluded:")}
        assert not excluded or classes == excluded and len(excluded) == 1, (genes, classes)
    assert sum(len(classes) > 1 for classes in by_genes.values()) >= 2  # right and wrong calls do occur
    # two tools on one read of a half-resolved gene: the right call and the wrong one take the read out for both
    right, wrong = _pred("h", "IGHVF9-G1*01"), _pred("h", "IGHVF9-G1*02")
    classes = {"right": {"h": INDEX.classify_v(right, truth=_truth("h"))},
               "wrong": {"h": INDEX.classify_v(wrong, truth=_truth("h"))}}
    assert classes["right"]["h"] == classes["wrong"]["h"] == "excluded:half_resolved"
    assert shared_exclusions(classes["right"], classes["wrong"]) == {"h"}
    for pred in (right, wrong):
        cls = INDEX.classify_v(pred, truth=_truth("h"))
        assert real_case_scores(pred, _truth("h"), cls, excluded=True, index=INDEX) == {}


def test_d_and_j_calls_where_the_donor_has_their_truth():
    assert INDEX.classify("IGHD3-10*01", "d") == SUPPORTED
    assert INDEX.classify("IGHD3-3*01", "d") == "excluded:dropped_truth"  # its only call was dropped
    assert INDEX.classify("IGHD2-2*01", "d") == UNSUPPORTED_GENE
    assert INDEX.classify("IGHD4-11*01", "d") == "excluded:not_captured"
    assert INDEX.classify("IGHJ4*02", "j") == SUPPORTED and INDEX.classify("IGHJ6*02", "j") == UNSUPPORTED_GENE
    calls = INDEX.read_calls(_pred(v="IGHVF1-G1*02", d="IGHD3-10*01", j="IGHJ6*02"), truth=None)
    assert calls == ReadCalls(v=UNSUPPORTED_ALLELE, d=SUPPORTED, j=UNSUPPORTED_GENE, v_any_listed_supported=False)
    scores = real_case_scores(_pred(v="IGHVF1-G1*02", d="IGHD3-10*01", j="IGHJ6*02"), _truth("r"), calls,
                              excluded=False, index=INDEX)
    assert scores["real.d.unsupported"] == 0.0 and scores["real.j.unsupported"] == 1.0
    # applicability is the read's: where the donor has D and J truth and the read is not excluded for them, a tool
    # without a D or J call has not covered the metric (nan), not made it inapplicable (an absent key)
    no_dj = INDEX.read_calls(_pred(v="IGHVF1-G1*01"), truth=None)
    assert no_dj.d == no_dj.j == NO_CALL
    scores = real_case_scores(_pred(v="IGHVF1-G1*01"), _truth("r"), no_dj, excluded=False, index=INDEX)
    assert math.isnan(scores["real.d.unsupported"]) and math.isnan(scores["real.j.unsupported"])
    # excluded for D (the other tool's D call is), the read has no D metric
    scores = real_case_scores(_pred(v="IGHVF1-G1*01"), _truth("r"), no_dj, excluded={"d"}, index=INDEX)
    assert "real.d.unsupported" not in scores and math.isnan(scores["real.j.unsupported"])
    # a donor without D and J truth gets neither metric
    v_only = SupportIndex(RealGenotype(donor=DONOR, alleles=V_ALLELES, captured=CAPTURED, resolved=CAPTURED,
                                       uncalled=UNCALLED), IUIS, MERGED, REFERENCE, support=SUPPORT)
    calls = v_only.read_calls(_pred(v="IGHVF1-G1*01", d="IGHD3-10*01", j="IGHJ4*02"), truth=None)
    assert calls.d is None and calls.j is None
    scores = real_case_scores(_pred(v="IGHVF1-G1*01", d="IGHD3-10*01", j="IGHJ4*02"), _truth("r"), calls,
                              excluded=False, index=v_only)
    assert not {"real.d.unsupported", "real.j.unsupported"} & set(scores)


def test_any_listed_v_allele_supported():
    # IgBLAST lists ties: the top-1 is the first, and the informational metric asks whether the list hides a
    # supported allele
    calls = INDEX.read_calls(_pred("r", "IGHVF1-G1*02", "IGHVF1-G1*01"), truth=None)
    assert calls.v == UNSUPPORTED_ALLELE and calls.v_any_listed_supported
    scores = real_case_scores(_pred("r", "IGHVF1-G1*02", "IGHVF1-G1*01"), _truth("r"), calls, excluded=False,
                              index=INDEX)
    assert scores["real.v.unsupported"] == 1.0 and scores["real.v.any_listed_supported"] == 1.0
    assert not INDEX.read_calls(_pred("r", "IGHVF1-G1*02", "IGHVF7-G1*01"), truth=None).v_any_listed_supported
    assert not INDEX.read_calls(None, truth=None).v_any_listed_supported


# --- per-read scores ----------------------------------------------------------------------------------------------

def test_real_case_scores_headline_and_components():
    known = _truth("r", known=("IGHV1-2*02",), shared=("IGHVF1-G1*01",), gene="IGHV1-2")
    expected = {SUPPORTED: (0.0, 0.0, 0.0, 0.0), UNSUPPORTED_ALLELE: (1.0, 1.0, 0.0, 0.0),
                UNSUPPORTED_GENE: (1.0, 0.0, 1.0, 0.0), NO_CALL: (1.0, 0.0, 0.0, 1.0)}
    for call, cls in (("IGHVF1-G1*01", SUPPORTED), ("IGHVF1-G1*02", UNSUPPORTED_ALLELE),
                      ("IGHVF7-G1*01", UNSUPPORTED_GENE), (None, NO_CALL)):
        pred = _pred(v=call)
        assert INDEX.classify_v(pred, truth=known) == cls
        scores = real_case_scores(pred, known, cls, excluded=False, index=INDEX)
        assert tuple(scores[f"real.v.{name}"] for name in ("unsupported", "unsupported_allele",
                                                           "unsupported_gene", "no_call")) == expected[cls]
        # the wrong allele of the known gene is still the known gene
        assert scores["real.v.known_top1"] == float(cls == SUPPORTED)
        assert scores["real.v.known_gene_top1"] == float(cls in (SUPPORTED, UNSUPPORTED_ALLELE))
    # a read without a known answer has no known-answer metric
    scores = real_case_scores(_pred("u", "IGHVF1-G1*01"), _truth("u"), SUPPORTED, excluded=False, index=INDEX)
    assert set(scores) == {"real.v.unsupported", "real.v.unsupported_allele", "real.v.unsupported_gene",
                           "real.v.no_call", "real.v.unsupported_observable"}


def test_known_answer_metrics_follow_the_318_rule_and_paralog_genes():
    known = _truth("r", known=("IGHV4-28*01",), shared=("IGHVF3-G9*04",), gene="IGHV4-28")
    scores = real_case_scores(_pred(v="IGHVF3-G9*04_A319G"), known, SUPPORTED, excluded=False, index=INDEX)
    assert scores["real.v.known_top1"] == scores["real.v.known_gene_top1"] == 1.0
    paralog = _truth("p", known=("IGHV1-69D*01",), shared=("IGHVF6-G21*06",), gene="IGHV1-69D")
    scores = real_case_scores(_pred("p", "IGHVF6-G21*07"), paralog, UNSUPPORTED_ALLELE, excluded=False, index=INDEX)
    assert scores["real.v.known_top1"] == 0.0 and scores["real.v.known_gene_top1"] == 1.0
    # a novel known answer scores only where its calls go: to its gene or not
    novel = _truth("n", known=("IGHV3-30*18_t189c",), gene="IGHV3-30")
    for call, value in (("IGHVF4-G1*02", 1.0), ("IGHVF7-G1*01", 0.0), (None, 0.0)):
        scores = real_case_scores(_pred("n", call), novel, INDEX.classify_v(_pred("n", call), truth=novel),
                                  excluded=True, index=INDEX)
        assert scores == {"real.v.novel_known_call": value}, call


def _window_truth(read_id: str, *, start: int, cys_end: int = 288, n: tuple[int, ...] = (),
                  known: tuple[str, ...] = (), shared: tuple[str, ...] = (), gene: str | None = None) -> RealTruth:
    """A read whose observed V runs from read index ``start`` to Cys104 at ``cys_end``, with Ns at offsets ``n``
    from its first observed base."""
    truth = _truth(read_id, known=known, shared=shared, gene=gene)
    return RealTruth(**{**truth.__dict__, "observed": (start, cys_end + 100), "cys_end": cys_end, "v_window_n": n})


def test_the_tie_aware_metrics_compare_germline_windows_over_the_reads_observed_v():
    """Informational: a call is observably supported when its window over the read (its first observed base to the
    end of codon 104) equals a donor allele's, the read's Ns left out; an unsupported call that is observably
    supported is a tie the read cannot resolve, not an error it shows."""
    # IGHVF1-G1*02 (IGHV1-2*04) differs from the donor's IGHV1-2*02 at ungapped position 135 only
    wrong = _pred("t", "IGHVF1-G1*02")
    assert INDEX.classify_v(wrong, truth=None) == UNSUPPORTED_ALLELE

    def observable(pred, truth, cls):
        return real_case_scores(pred, truth, cls, excluded=False, index=INDEX)["real.v.unsupported_observable"]

    short = _window_truth("t", start=200)  # sees positions 200-287 of the V: not the difference, a tie
    assert observable(wrong, short, UNSUPPORTED_ALLELE) == 0.0
    assert real_case_scores(wrong, short, UNSUPPORTED_ALLELE, excluded=False, index=INDEX)["real.v.unsupported"] == 1
    full = _window_truth("t", start=0)
    assert observable(wrong, full, UNSUPPORTED_ALLELE) == 1.0
    # a base the read holds as N is not compared
    assert observable(wrong, _window_truth("t", start=0, n=(135,)), UNSUPPORTED_ALLELE) == 0.0
    assert observable(wrong, _window_truth("t", start=0, n=(134,)), UNSUPPORTED_ALLELE) == 1.0
    # a window longer than the allele (a 5'RACE read's leader before the V) is compared over the allele's length,
    # and its N offsets count from the read's first observed base
    assert observable(wrong, _window_truth("t", start=0, cys_end=358), UNSUPPORTED_ALLELE) == 1.0
    assert observable(wrong, _window_truth("t", start=0, cys_end=358, n=(205,)), UNSUPPORTED_ALLELE) == 0.0
    # a supported call is 0, a missing call 1, and a read without a placed Cys104 is not covered
    assert observable(_pred("t", "IGHVF1-G1*01"), full, SUPPORTED) == 0.0
    assert observable(_pred("t", None), full, NO_CALL) == 1.0
    assert math.isnan(observable(wrong, _truth("t"), UNSUPPORTED_ALLELE))
    # the window may equal any donor allele's, of any gene or functionality
    assert INDEX.observably_supported("IGHVF12-G1*02", full)  # exactly the nonfunctional IGHV4-4*07
    assert INDEX.observably_supported("IGHVF1-G1*02", short) and not INDEX.observably_supported("IGHVF1-G1*02",
                                                                                                full)
    # known_top1, observably: on a documented known answer, whether the call's window is the known one
    known = {"known": ("IGHV1-2*02",), "shared": ("IGHVF1-G1*01",), "gene": "IGHV1-2"}
    for start, value in ((200, 1.0), (0, 0.0)):
        truth = _window_truth("k", start=start, **known)
        scores = real_case_scores(_pred("k", "IGHVF1-G1*02"), truth, UNSUPPORTED_ALLELE, excluded=False,
                                  index=INDEX)
        assert scores["real.v.known_top1"] == 0.0 and scores["real.v.known_top1_observable"] == value
    # scored on an excluded read too, as known_top1 is; a call of another gene's window is not the known one
    scores = real_case_scores(_pred("k", "IGHVF5-G1*01"), _window_truth("k", start=0, **known),
                              "excluded:not_captured", excluded=True, index=INDEX)
    assert scores == {"real.v.known_top1": 0.0, "real.v.known_gene_top1": 0.0, "real.v.known_top1_observable": 0.0}
    assert "real.v.known_top1_observable" not in real_case_scores(wrong, full, UNSUPPORTED_ALLELE, excluded=False,
                                                                  index=INDEX)


def test_shared_exclusion_removes_read_for_both_tools():
    a = {"r1": SUPPORTED, "r2": "excluded:not_captured", "r3": UNSUPPORTED_ALLELE, "r4": NO_CALL}
    b = {"r1": "excluded:low_support", "r2": SUPPORTED, "r3": NO_CALL, "r4": SUPPORTED}
    excluded = shared_exclusions(a, b)
    assert excluded == {"r1", "r2"} == shared_exclusions(b, a)
    known = _truth("r1", known=("IGHV1-2*02",), shared=("IGHVF1-G1*01",), gene="IGHV1-2")
    for classes in (a, b):
        scores = real_case_scores(_pred("r1", "IGHVF1-G1*01"), known, classes["r1"], excluded="r1" in excluded,
                                  index=INDEX)
        # out of R1 for both tools; the known answer is still scored, its truth being tool-neutral
        assert not any(name.startswith("real.v.unsupported") or name == "real.v.no_call" for name in scores)
        assert scores["real.v.known_top1"] == 1.0
    # a read excluded for a tool cannot be scored as not excluded, and both tools are classified on the same reads
    with pytest.raises(ValueError, match="r2"):
        real_case_scores(_pred("r2", "IGHVF5-G1*01"), _truth("r2"), a["r2"], excluded=False, index=INDEX)
    with pytest.raises(ValueError, match="r5"):
        shared_exclusions(a, {**b, "r5": SUPPORTED})
    with pytest.raises(ValueError, match="read"):
        real_case_scores(_pred("other", "IGHVF1-G1*01"), known, SUPPORTED, excluded=False, index=INDEX)
    with pytest.raises(ValueError, match="W-97"):
        real_case_scores(_pred("w", "IGHVF1-G1*01"), _truth("w", donor="W-97"), SUPPORTED, excluded=False,
                         index=INDEX)
    # D and J classes are excluded the same way, segment by segment; a donor without their truth has None
    assert shared_exclusions({"r1": None, "r2": "excluded:dropped_truth"}, {"r1": SUPPORTED, "r2": None}) == {"r2"}


def test_exclusion_counts_per_tool():
    classes = {"r1": SUPPORTED, "r2": "excluded:not_captured", "r3": "excluded:not_captured",
               "r4": "excluded:low_support", "r5": NO_CALL}
    counts = exclusion_counts(classes)
    assert list(counts) == list(EXCLUSIONS)
    assert counts["excluded:not_captured"] == 2 and counts["excluded:low_support"] == 1
    assert sum(counts.values()) == 3
    assert exclusion_counts(list(classes.values())) == counts


# --- gene thresholds ----------------------------------------------------------------------------------------------

def test_low_support_gene_from_known_answers_only():
    truths = []
    # IGHV1-2: 10 clonotypes, 3 of them naive (a naive clonotype of two reads counts once)
    for k in range(10):
        truths.append(_truth(f"a{k}", known=("IGHV1-2*02",), shared=("IGHVF1-G1*01",), gene="IGHV1-2",
                             naive=k < 3, clonotype=f"{DONOR}/A{k}"))
    truths.append(_truth("a-dup", known=("IGHV1-2*02",), shared=("IGHVF1-G1*01",), gene="IGHV1-2", naive=True,
                         clonotype=f"{DONOR}/A0"))
    # IGHV4-28: 12 known reads in 9 clonotypes, with a read that is a clonotype of its own; 5 naive
    for k in range(11):
        truths.append(_truth(f"b{k}", known=("IGHV4-28*01",), shared=("IGHVF3-G9*04",), gene="IGHV4-28",
                             naive=k < 5, clonotype=f"{DONOR}/B{k % 8}"))
    truths.append(_truth("b-single", known=("IGHV4-28*01",), shared=("IGHVF3-G9*04",), gene="IGHV4-28"))
    # a paralog set counts toward each gene of it
    for k in range(10):
        truths.append(_truth(f"c{k}", known=("IGHV1-69*01", "IGHV1-69D*01"), shared=("IGHVF6-G21*06",),
                             gene="IGHV1-69|IGHV1-69D", naive=True, clonotype=f"{DONOR}/C{k}"))
    # reads without a known answer never count, whatever a tool would call them
    for k in range(30):
        truths.append(_truth(f"u{k}", clonotype=f"{DONOR}/U{k}"))
    support = gene_support(truths)
    assert support == {"IGHV1-2": (10, 3), "IGHV4-28": (9, 5), "IGHV1-69": (10, 10), "IGHV1-69D": (10, 10)}
    assert passing_genes(support) == {"IGHV1-2", "IGHV1-69", "IGHV1-69D"}
    index = SupportIndex(GENOTYPE, IUIS, MERGED, REFERENCE, support=support, thresholds=True)
    assert index.classify("IGHVF1-G1*01") == SUPPORTED and index.classify("IGHVF6-G21*06") == SUPPORTED
    assert index.classify("IGHVF3-G9*04") == "excluded:low_support"  # 9 clonotypes
    assert index.classify("IGHVF2-G1*01") == "excluded:low_support"  # carried, no known answer at all
    # the R1 primary applies no threshold
    primary = SupportIndex(GENOTYPE, IUIS, MERGED, REFERENCE, support=support)
    assert primary.classify("IGHVF3-G9*04") == primary.classify("IGHVF2-G1*01") == SUPPORTED
    with pytest.raises(ValueError, match="donor"):
        gene_support([*truths, _truth("w", known=("IGHV1-2*02",), shared=("IGHVF1-G1*01",), gene="IGHV1-2",
                                      donor="W-97")])


# --- the oracle -----------------------------------------------------------------------------------------------------

def test_oracle_scores_perfect():
    truths = [
        _truth("a", known=("IGHV1-2*02",), shared=("IGHVF1-G1*01",), gene="IGHV1-2", naive=True),
        _truth("p", known=("IGHV1-69D*01",), shared=("IGHVF6-G21*06",), gene="IGHV1-69D"),
        _truth("t", known=("IGHV4-28*01",), shared=("IGHVF3-G9*04",), gene="IGHV4-28"),
        # a gene with a novel allele: a documented known answer is not excluded for it
        _truth("x", known=("IGHV3-30*01",), shared=("IGHVF4-G1*01",), gene="IGHV3-30"),
        # excluded by a rule of the gene, whatever the call: half resolved, an allele without a usable gapped
        # form, too few clonotypes
        _truth("h", known=("IGHV3-15*01",), shared=("IGHVF9-G1*01",), gene="IGHV3-15"),
        _truth("m", known=("IGHV5-51*01",), shared=("IGHVF11-G1*01",), gene="IGHV5-51"),
        _truth("l", known=("IGHV1-46*01",), shared=("IGHVF2-G1*01",), gene="IGHV1-46"),
        # not scored by the oracle: a novel known answer, a read without one
        _truth("n", known=("IGHV3-30*18_t189c",), gene="IGHV3-30"),
        _truth("u"),
    ]
    assert oracle_prediction(truths[-1]) is None and oracle_prediction(truths[-2]) is None
    oracle = oracle_prediction(truths[0])
    assert oracle.sequence_id == "a" and oracle.segments["v"].call == "IGHVF1-G1*01"
    check = oracle_self_check(truths, INDEX)
    assert check["reads"] == 7
    assert check["classes"] == {SUPPORTED: 5, "excluded:half_resolved": 1, "excluded:beyond_318": 1}
    assert check["excluded_genes"] == {"excluded:beyond_318": ["IGHV5-51"], "excluded:half_resolved": ["IGHV3-15"]}
    assert check["scored"] == 5 and check["supported"] == 1.0 and check["unsupported"] == 0.0
    assert check["known_top1"] == check["known_gene_top1"] == 1.0
    assert check["passed"]
    # with the gene-support thresholds, IGHV1-46's read is excluded as well
    thresholded = oracle_self_check(truths, THRESHOLDED)
    assert thresholded["classes"] == {SUPPORTED: 4, "excluded:half_resolved": 1, "excluded:beyond_318": 1,
                                      "excluded:low_support": 1}
    assert thresholded["excluded_genes"]["excluded:low_support"] == ["IGHV1-46"] and thresholded["passed"]
    for truth in truths[:7]:
        pred = oracle_prediction(truth)
        cls = INDEX.classify_v(pred, truth=truth)
        scores = real_case_scores(pred, truth, cls, excluded=cls.startswith("excluded:"), index=INDEX)
        assert scores.get("real.v.unsupported", 0.0) == 0.0 and scores["real.v.known_top1"] == 1.0


def test_two_decoys_check_what_the_oracle_never_calls():
    # the oracle calls donor alleles only; the decoys are chosen by the truth alone and their classes are fixed by
    # the rules: an allele of a clean carried gene that matches no donor allele, and an allele of uncalled genes only
    decoys = decoy_calls(INDEX)
    assert decoys["unsupported_allele"] == {"call": "IGHVF1-G1*02", "genes": ["IGHV1-2"], "class": UNSUPPORTED_ALLELE,
                                            "expected": UNSUPPORTED_ALLELE, "passed": True}
    assert decoys["uncalled"] == {"call": "IGHVF16-G1*01", "genes": ["IGHV4-30-2"], "class": "excluded:uncalled",
                                  "expected": "excluded:uncalled", "passed": True}
    assert decoys["passed"]
    # a donor without an uncalled gene has no such decoy; a decoy classified otherwise fails
    plain = SupportIndex(RealGenotype(donor=DONOR, alleles=ALLELES, captured=CAPTURED - UNCALLED,
                                      resolved=GENOTYPE.resolved, uncalled=frozenset(), dropped=DROPPED),
                         IUIS, MERGED, REFERENCE)
    assert decoy_calls(plain)["uncalled"] is None and decoy_calls(plain)["passed"]
    wrong = SupportIndex(GENOTYPE, IUIS, MERGED, REFERENCE)
    wrong._classes[("IGHVF1-G1*02", "v", False)] = SUPPORTED  # as a broken rule would classify it
    assert not decoy_calls(wrong)["passed"] and not decoy_calls(wrong)["unsupported_allele"]["passed"]


def test_known_gene_accepts_the_iuis_genes_of_the_known_alleles_own_sequence():
    # VDJbase names an allele by the locus it was found at, whose sequence may be a paralog's documented allele
    # (SC-11: IGHV3-30-3*03_g150a is IGHV3-30*01's sequence, IGHV3-23D*01_t13g is IGHV3-23*04's): a call of that
    # sequence is of the known gene, and the oracle's own call passes the self-check
    paralog = _allele("IGHV3-30-3*03_g150a", 1, shared_as="IGHVF4-G1*01", gene="IGHV3-30-3")
    genotype = RealGenotype(donor=DONOR, alleles=(*ALLELES, paralog), captured=CAPTURED | {"IGHV3-30-3"},
                            resolved=(GENOTYPE.resolved | {"IGHV3-30-3"}), dropped=DROPPED, uncalled=UNCALLED)
    index = SupportIndex(genotype, IUIS, MERGED, REFERENCE, support=SUPPORT)
    truth = _truth("s", known=("IGHV3-30-3*03_g150a",), shared=("IGHVF4-G1*01",), gene="IGHV3-30-3")
    oracle = oracle_prediction(truth)
    cls = index.classify_v(oracle, truth=truth)
    scores = real_case_scores(oracle, truth, cls, excluded=cls.startswith("excluded:"), index=index)
    assert scores["real.v.known_top1"] == scores["real.v.known_gene_top1"] == 1.0
    check = oracle_self_check([truth], index)
    assert check["known_gene_top1"] == 1.0 and check["passed"]
    # another allele of the sequence's gene is of the known gene too; a call of another gene is not
    for call, value in (("IGHVF4-G1*02", 1.0), ("IGHVF7-G1*01", 0.0)):
        pred = _pred("s", call)
        scores = real_case_scores(pred, truth, index.classify_v(pred, truth=truth), excluded=False, index=index)
        assert scores["real.v.known_gene_top1"] == value, call


def test_support_index_refuses_a_genotype_it_cannot_score():
    with pytest.raises(ValueError, match="capture"):
        SupportIndex(RealGenotype(donor=DONOR, alleles=ALLELES), IUIS, MERGED, REFERENCE, support=SUPPORT)
    other = Reference(alleles={"v": dict(REFERENCE.alleles["v"])}, anchors={})
    with pytest.raises(ValueError, match="reference"):
        SupportIndex(GENOTYPE, IUIS, MERGED, other, support=SUPPORT)
    mapped = RealGenotype(donor=DONOR, alleles=ALLELES, captured=CAPTURED, resolved=CAPTURED, uncalled=UNCALLED,
                          source={"reference_digest": other.digest()})
    with pytest.raises(ValueError, match="reference"):
        SupportIndex(mapped, IUIS, MERGED, REFERENCE, support=SUPPORT)
    # the same alleles (and digest) without one V allele's gapped form: a call of it could not be compared through
    # 318, so the reference is refused rather than a call misclassified
    missing = {"IGHVF1-G1*02", "IGHVF7-G1*01"}
    partly = Reference(alleles=REFERENCE.alleles, anchors={},
                       gapped_v={name: gapped for name, gapped in REFERENCE.gapped_v.items() if name not in missing})
    assert partly.digest() == REFERENCE.digest()
    with pytest.raises(ValueError, match="IGHVF1-G1\\*02, IGHVF7-G1\\*01"):  # every offending allele, named
        SupportIndex(GENOTYPE, IUIS, MERGED, partly, support=SUPPORT)


# --- registration -------------------------------------------------------------------------------------------------

REAL_METRICS = {"real.v.unsupported": (False, True), "real.v.unsupported_allele": (False, True),
                "real.v.unsupported_gene": (False, True), "real.v.no_call": (False, True),
                "real.v.known_top1": (False, False), "real.v.known_gene_top1": (False, False),
                "real.v.novel_known_call": (True, False), "real.v.any_listed_supported": (True, False),
                "real.d.unsupported": (True, True), "real.j.unsupported": (True, True),
                "real.v.unsupported_observable": (True, True), "real.v.known_top1_observable": (True, False)}
NEW_INFORMATIONAL = {"real.v.novel_known_call", "real.v.any_listed_supported", "real.d.unsupported",
                     "real.j.unsupported", "real.v.unsupported_observable", "real.v.known_top1_observable"}
NEW_LOWER = (".unsupported", ".unsupported_allele", ".unsupported_gene", "real.v.unsupported_observable")


def _package_strings() -> set[str]:
    """Every string constant of the package outside ``real/metrics.py`` and the registry itself (f-string pieces
    included): the metric names and name suffixes the other layers write."""
    package = Path(alignair_assay.__file__).resolve().parent
    found = set()
    for path in package.rglob("*.py"):
        if path.relative_to(package) in (Path("real", "metrics.py"), Path("metrics", "__init__.py")):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 80:
                found.add(node.value)
    return found


def test_informational_names_registered():
    for name, (informational, lower) in REAL_METRICS.items():
        assert is_informational(name) is informational, name
        assert lower_is_better(name) is lower, name
    assert NEW_INFORMATIONAL <= registry.INFORMATIONAL and set(NEW_LOWER) <= set(registry._LOWER_IS_BETTER)
    # the registry as it was: the new names and suffixes change no other metric's classification
    old_informational = registry.INFORMATIONAL - NEW_INFORMATIONAL
    old_lower = tuple(suffix for suffix in registry._LOWER_IS_BETTER if suffix not in NEW_LOWER)

    def was_informational(name: str) -> bool:
        return name in old_informational or any(s.startswith(".") and name.endswith(s) for s in old_informational)

    names = {name for name in _package_strings() if not name.startswith("real.")}
    assert len(names) > 500
    changed = [name for name in names if is_informational(name) != was_informational(name)
               or lower_is_better(name) != name.endswith(old_lower)]
    assert changed == []
    # and a name ending with a new suffix outside real.* would have been caught
    assert lower_is_better("v.unsupported") and not "v.unsupported".endswith(old_lower)


# --- the pilot donors' oracle (gated) -------------------------------------------------------------------------------

DATA = os.environ.get("ALIGNAIR_ASSAY_DATA")
DOCKER = os.environ.get("ALIGNAIR_ASSAY_DOCKER") == "1"
RUN = Path(__file__).resolve().parents[3] / ".private" / "assay" / "p3a_task4_run"
PILOT_SAMPLES = {"W-97": "P25_I151_S1", "SC-23": "P25_I58_S1"}


def _read_fasta(path: Path) -> dict[str, str]:
    reads, name = {}, None
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
        elif name is not None:
            reads[name] = reads.get(name, "") + line.strip()
    return reads


def _read_tsv(path: Path) -> list[dict]:
    header, *rows = path.read_text(encoding="utf-8").splitlines()
    columns = header.split("\t")
    return [dict(zip(columns, row.split("\t"))) for row in rows]


@pytest.mark.skipif(not DATA or not DOCKER or not RUN.is_dir(),
                    reason="set ALIGNAIR_ASSAY_DATA to the PRJNA555323 data root and ALIGNAIR_ASSAY_DOCKER=1, with "
                           "Task 4's run in .private/assay/p3a_task4_run")
@pytest.mark.parametrize("donor", sorted(PILOT_SAMPLES))
def test_the_oracle_on_the_pilot_donors_real_truth(donor, reference):
    from alignair_assay.core.layout import RunLayout
    from alignair_assay.real.genotype import load_vdjbase_csv
    from alignair_assay.real.locus import donor_capture
    from alignair_assay.real.truth import real_truths

    root = Path(DATA)
    samples = root / "genotypes/vdjbase_genomic_samples"
    iuis = IuisMap.from_ogrdb(root / "genotypes/ogrdb_human_IGH_VDJ/Homo_sapiens_IGH_VDJ_rev_8.json", reference)
    genotype = load_vdjbase_csv(samples / f"{PILOT_SAMPLES[donor]}.csv", donor, reference, iuis)
    genotype = donor_capture(genotype, samples / f"{PILOT_SAMPLES[donor]}.bam",
                             root / "genotypes/igenotyper/gene_coords.bed", iuis)
    layout = RunLayout(RUN)
    truths = real_truths(_read_fasta(layout.real_reads(donor)), _read_tsv(layout.real_reads_meta(donor)), genotype,
                         reference)
    support = gene_support(truths)
    index = SupportIndex(genotype, iuis, merge_identical(reference), reference, support=support)
    thresholded = SupportIndex(genotype, iuis, merge_identical(reference), reference, support=support,
                               thresholds=True)
    check = oracle_self_check(truths, index)
    known = [t for t in truths if t.known is not None and not t.novel_known]
    assert check["reads"] == len(known) > 0
    assert check["passed"] and check["supported"] == 1.0 and check["unsupported"] == 0.0
    assert check["known_top1"] == check["known_gene_top1"] == 1.0
    # only the rules of the call's genes exclude an oracle call, and never novel_gene: every oracle read is a
    # documented known answer. The guard never fires on this reference.
    assert set(check["classes"]) <= {SUPPORTED, "excluded:not_captured", "excluded:nonfunctional",
                                     "excluded:beyond_318", "excluded:half_resolved", "excluded:uncalled"}
    for segment in ("v", "d", "j"):
        assert all(index.classify(name, segment) != "excluded:unmapped" for name in reference.alleles[segment])
    # each class is explained by the call's own genes, read off the genotype
    passing = passing_genes(support)
    v = [allele for allele in genotype.alleles if allele.segment == "v"]
    for truth in known:
        oracle = oracle_prediction(truth)
        name = oracle.segments["v"].call
        cls = index.classify_v(oracle, truth=truth)
        genes = set(index.genes(name))
        carried = {gene for gene in genes if genotype.carried(gene)}
        functional = {allele.gene for allele in v if allele.functional}
        reasons = {"excluded:not_captured": bool(genes - genotype.captured),
                   "excluded:uncalled": bool(genes & genotype.uncalled),
                   "excluded:nonfunctional": bool(carried - functional),
                   "excluded:beyond_318": any(allele.functional and through_318(allele.gapped) is None
                                              for allele in v if allele.gene in carried),
                   "excluded:half_resolved": bool(carried - genotype.resolved)}
        if cls == SUPPORTED:
            assert carried and not any(reasons.values()), truth.read_id
            assert any(allele.sequence == reference.seq(name) for allele in v), truth.read_id
            # the gene-support variant excludes it exactly when one of its carried genes misses the thresholds
            variant = thresholded.classify_v(oracle, truth=truth)
            assert variant == ("excluded:low_support" if carried - passing else SUPPORTED), truth.read_id
        else:
            assert reasons[cls], (truth.read_id, cls)
            assert thresholded.classify_v(oracle, truth=truth) == cls
    assert sum(check["classes"].values()) == len(known)
    json.dumps(check)  # plain data, for the report


DEV_RUN = Path(__file__).resolve().parents[3] / ".private" / "assay" / "p3a_dev"
DEV_SAMPLES = {"SC-23": "P25_I58_S1", "SC-11": "P25_I48_S1", "W-97": "P25_I151_S1", "W-133": "P25_I93_S1"}


@pytest.mark.skipif(not DATA or not DOCKER or not (DEV_RUN / "real").is_dir(),
                    reason="set ALIGNAIR_ASSAY_DATA to the PRJNA555323 data root and ALIGNAIR_ASSAY_DOCKER=1, with the "
                           "dev run in .private/assay/p3a_dev")
def test_no_dev_donors_call_is_an_unsupported_gene_once_uncalled_genes_are_excluded(reference):
    """Every unsupported_gene of the dev run was a VDJbase calling gap: the four dev donors' shared-arm predictions,
    re-scored against their new capture, have none. A gene the contigs hold and the genotype does not call is
    excluded, and no dev donor has a gene deleted on every capturing contig. The read's truth only lifts novel_gene,
    which applies to carried genes, so classifying without it is exact for unsupported_gene."""
    from alignair_assay.airr.reader import load_airr_tsv
    from alignair_assay.core.layout import RunLayout
    from alignair_assay.real.genotype import load_vdjbase_csv
    from alignair_assay.real.locus import donor_capture

    root = Path(DATA)
    samples = root / "genotypes/vdjbase_genomic_samples"
    iuis = IuisMap.from_ogrdb(root / "genotypes/ogrdb_human_IGH_VDJ/Homo_sapiens_IGH_VDJ_rev_8.json", reference)
    layout, merged = RunLayout(DEV_RUN), merge_identical(reference)
    uncalled_calls = {}
    for donor, sample in DEV_SAMPLES.items():
        genotype = donor_capture(load_vdjbase_csv(samples / f"{sample}.csv", donor, reference, iuis),
                                 samples / f"{sample}.bam", root / "genotypes/igenotyper/gene_coords.bed", iuis)
        assert genotype.source["capture"]["absent"] == []
        index = SupportIndex(genotype, iuis, merged, reference)
        for tool in ("igblast", "alignair", "alignair-api"):
            path = layout.real_predictions(tool, donor, "shared")
            if not path.is_file():
                continue
            predictions, _ = load_airr_tsv(path)
            classes = [index.classify_v(pred, truth=None) for pred in predictions.values()]
            assert UNSUPPORTED_GENE not in classes, (donor, tool)
            uncalled_calls[donor, tool] = classes.count("excluded:uncalled")
    assert uncalled_calls[("W-97", "igblast")] > 0 and uncalled_calls[("SC-11", "igblast")] == 0
