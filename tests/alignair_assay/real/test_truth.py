"""Tool-neutral truth for real reads (``real.truth``): observed span, known answer, naive flag and clonotype.

The offline tests build IMGT-gapped V alleles, reads and genotypes in the test; one OGRDB rev-8 gapped V
(IGHV3-23*01) is embedded as a string. The gated test runs on the pilot donors W-97 (multiplex) and SC-23 (5'RACE)
when ``ALIGNAIR_ASSAY_DATA`` names the data root (``.../PRJNA555323``) and Task 4's preprocessed outputs are in
``.private/assay/p3a_task4_run``.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

import pytest

from alignair_assay.core.layout import RunLayout
from alignair_assay.core.reference import Reference
from alignair_assay.real.genotype import IuisMap, RealAllele, RealGenotype, load_vdjbase_csv
from alignair_assay.real.truth import (
    ANCHOR_LENGTH,
    MAX_JUNCTION,
    MIN_JUNCTION,
    W118_MOTIF,
    RealTruth,
    clonotype_clusters,
    cys104_end,
    cys104_in_read,
    junction_by_motif,
    known_answer,
    known_match,
    mask_primers,
    observed_span,
    real_truths,
)

DATA = os.environ.get("ALIGNAIR_ASSAY_DATA")
RUN = Path(__file__).resolve().parents[3] / ".private" / "assay" / "p3a_task4_run"
DONOR = "SC-23"

# OGRDB rev 8 (Homo_sapiens_IGH_VDJ_rev_8.json), IGHV3-23*01: v_gene_delineations[0].aligned_sequence, the IMGT
# delineation, whose fwr3_end is 288 and cdr3_start 289 (1-based, ungapped).
OGRDB_IGHV3_23_01 = (
    "GAGGTGCAGCTGTTGGAGTCTGGGGGA...GGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCTTT............"
    "AGCAGCTATGCCATGAGCTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTCTCAGCTATTAGTGGTAGT......GGTGGTAGCACATACTACGCAG"
    "ACTCCGTGAAG...GGCCGGTTCACCATCTCCAGAGACAATTCCAAGAACACGCTGTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCCGTATATTAC"
    "TGTGCGAAAGA"
)
OGRDB_FWR3_END = 288

# IMGT gap columns of the synthetic alleles (0-based, half-open): 24 columns, as on most IGHV, so codon 104
# (columns 309-311) ends at ungapped index 288.
GAPS = ((27, 30), (72, 84), (165, 171), (195, 198))
CYS_END = 288
# IGHJ4*02: its W118 codon starts at 14 (TGG GGC CAG GGA is the W118 motif).
J4 = "ACTACTTTGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCAG"
J4_W118_END = 17
# IGHJ2*01: Trp118 Gly Arg Gly (TGG GGC CGT GGC), the IMGT W-G-X-G motif without IGHJ1/3/4/5/6's Gln120.
J2 = "CTACTGGTACTTCGATCTCTGGGGCCGTGGCACCCTGGTCACTGTCTCCTCAG"
J2_W118_END = 22
N_REGION = "GATAGTTATCGGAACT"  # 16 nt: an in-frame junction of 9 + 16 + 17 nt
C_START = "GGAGTGCATCCGCCCCAACCC"


def _random(n: int, seed: int) -> str:
    return "".join(random.Random(seed).choices("ACGT", k=n))


def _mutate(sequence: str, *positions: int) -> str:
    bases = list(sequence)
    for position in positions:
        bases[position] = "A" if bases[position] != "A" else "C"
    return "".join(bases)


def _gapped_v(seed: int, codon: str = "TGT") -> str:
    """A 320-column IMGT-gapped V-REGION: random bases, ``GAPS`` as ``.`` columns and ``codon`` at 309-311."""
    bases = list(_random(320, seed))
    bases[309:312] = codon
    for start, end in GAPS:
        bases[start:end] = "." * (end - start)
    return "".join(bases)


def _allele(allele_id: str, gapped: str | None, *, sequence: str | None = None, gene: str | None = None,
            shared: tuple[str, ...] = ("IGHVF1-G1*01",), haplotype: int = 1) -> RealAllele:
    return RealAllele(allele_id=allele_id, gene=gene or allele_id.split("*")[0], segment="v",
                      sequence=sequence or gapped.replace(".", ""), gapped=gapped, haplotype=haplotype,
                      functional=True, shared=shared, iuis=(allele_id,))


A_GAPPED = _gapped_v(1)
A = _allele("IGHV1-2*02", A_GAPPED)
B = _allele("IGHV3-23*01", _gapped_v(2), shared=("IGHVF3-G10*01",))
LEADER = _random(70, 3)  # 5'UTR and leader of a 5'RACE read


def _race(v: str, *, n_region: str = N_REGION, j: str = J4, cys_end: int = CYS_END) -> str:
    """A 5'RACE read: leader, the V-REGION through two codons past Cys104, N region, J and the start of C."""
    return LEADER + v[:cys_end + 6] + n_region + j + C_START


def _multiplex(v: str, *, primer: int = 22, start: int | None = None, cys_end: int = CYS_END,
               head: str | None = None) -> tuple[str, list[list[int]]]:
    """A multiplex read and its primer spans: a masked VH primer (``head``, or ``N``s) of ``primer`` nt, the V from
    allele position ``start`` (``primer`` by default) through two codons past Cys104, N region, J and a masked C
    primer of 20 nt."""
    start = primer if start is None else start
    body = (head if head is not None else "N" * primer) + v[start:cys_end + 6] + N_REGION + J4
    read = body + "N" * 20
    return read, [[0, primer], [len(body), len(read)]]


def _junction(v: str, *, n_region: str = N_REGION, cys_end: int = CYS_END) -> str:
    return v[cys_end - 3:cys_end + 6] + n_region + J4[:J4_W118_END]


def _meta(read_id: str, isotype: str = "IGHM", protocol: str = "race", spans=None) -> dict:
    return {"sequence_id": read_id, "isotype": isotype, "protocol": protocol, "consensus_count": "3",
            "dupcount": "1", "primer_masked": json.dumps(spans or [])}


def _genotype(*alleles: RealAllele, donor: str = DONOR) -> RealGenotype:
    return RealGenotype(donor=donor, alleles=tuple(alleles))


def _reference(*alleles: RealAllele) -> Reference:
    """A shared reference holding ``alleles``' V-REGIONs and their gapped forms."""
    return Reference(alleles={"v": {allele.allele_id: allele.sequence for allele in alleles}}, anchors={},
                     gapped_v={allele.allele_id: allele.gapped for allele in alleles if allele.gapped})


# A shared reference without a V allele: no fallback anchor.
NO_SHARED = Reference(alleles={}, anchors={})


# --- IMGT numbering ------------------------------------------------------------------------------------------------

def test_cys104_end_on_gapped_ogrdb_allele():
    ungapped = OGRDB_IGHV3_23_01.replace(".", "")
    assert len(OGRDB_IGHV3_23_01) == 320 and len(ungapped) == 296
    assert OGRDB_IGHV3_23_01[309:312] == "TGT"  # IMGT nt 310-312, codon 104
    end = cys104_end(OGRDB_IGHV3_23_01)
    assert end == OGRDB_FWR3_END == 288
    assert ungapped[end - 3:end] == "TGT"
    assert cys104_end(OGRDB_IGHV3_23_01.lower()) == 288


def test_cys104_end_refuses_a_shifted_or_short_codon_104():
    assert cys104_end(A_GAPPED) == CYS_END
    assert cys104_end(_gapped_v(1, codon="TGC")) == CYS_END
    # VDJbase gaps an insertion with an extra column, so codon 104 is no longer at 309-311
    inserted = A_GAPPED[:100] + "A" + A_GAPPED[100:]
    assert inserted[309:312] != "TGT" and cys104_end(inserted) is None
    assert cys104_end(A_GAPPED[:309] + "-GT" + A_GAPPED[312:]) is None  # a deleted base in the codon
    assert cys104_end(_gapped_v(1, codon="TAT")) is None  # second cysteine not found
    assert cys104_end(A_GAPPED[:311]) is None


# --- observed span -------------------------------------------------------------------------------------------------

def test_observed_span_leaves_out_primers_and_end_ns():
    read, spans = _multiplex(A.sequence)
    assert observed_span(read, spans) == (22, len(read) - 20)
    assert mask_primers("acgtACGTAC", [[0, 3], [8, 10]]) == "NNNTACGTNN"
    assert mask_primers("acgt", '[[1,2]]') == "ANGT" and mask_primers("ACGT", "[]") == "ACGT"
    assert observed_span("NNACGTNNACGNN", []) == (2, 11)
    # a primer span whose bases the assembly filled from the other mate is still not observed
    assert observed_span("ACGTACGTAC", [[0, 3], [8, 10]]) == (3, 8)
    assert observed_span("NNNN", []) == (0, 0) and observed_span("", []) == (0, 0)
    with pytest.raises(ValueError):
        observed_span("ACGT", [[2, 9]])
    with pytest.raises(ValueError):
        observed_span("ACGT", [[3, 1]])


# --- known answer --------------------------------------------------------------------------------------------------

def test_known_answer_exact():
    read = _race(A.sequence)
    assert known_answer(read, observed_span(read, []), [A, B]) == frozenset({"IGHV1-2*02"})
    match = known_match(read, observed_span(read, []), [A, B])
    assert match.window == A.sequence[:CYS_END]  # 5'RACE: the whole V-REGION through Cys104
    assert match.cys_end == len(LEADER) + CYS_END
    multiplex, spans = _multiplex(A.sequence)
    match = known_match(multiplex, observed_span(multiplex, spans), [A, B])
    assert match.window == A.sequence[22:CYS_END]  # multiplex: from the first base after the VH primer
    assert known_answer(_race(B.sequence), (0, len(_race(B.sequence))), [A, B]) == frozenset({"IGHV3-23*01"})


def test_one_mismatch_is_not_known():
    for position in (0, 150, CYS_END - 10, CYS_END - 1):  # first base, middle, in the anchor, in the codon
        read = _race(_mutate(A.sequence, position))
        assert known_answer(read, observed_span(read, []), [A, B]) is None, position
    # a mismatch past Cys104 is outside the window
    read = _race(_mutate(A.sequence, CYS_END + 2))
    assert known_answer(read, observed_span(read, []), [A, B]) == frozenset({"IGHV1-2*02"})


def test_an_n_inside_the_window_voids_the_known_answer():
    # N is no wildcard: an uncalled base in the middle of the observed V matches no allele, not even the read's own
    read = _race(A.sequence)
    middle = len(LEADER) + 150
    uncalled = read[:middle] + "N" + read[middle + 1:]
    assert observed_span(uncalled, []) == (0, len(read))
    assert known_answer(uncalled, observed_span(uncalled, []), [A, B]) is None
    truth, = real_truths({"r": uncalled}, [_meta("r")], _genotype(A, B), NO_SHARED)
    assert truth.known is None and not truth.naive and truth.anchor == "donor"


def test_identical_windows_form_one_set():
    # A2 differs from A only at allele position 5, inside a multiplex read's masked VH primer
    a2_gapped = _mutate(A_GAPPED, 5)
    a2 = _allele("IGHV1-2*04", a2_gapped)
    on_other_haplotype = _allele("IGHV1-2*02", A_GAPPED, haplotype=2)
    read, spans = _multiplex(A.sequence)
    assert known_answer(read, observed_span(read, spans), [A, a2, on_other_haplotype, B]) == frozenset(
        {"IGHV1-2*02", "IGHV1-2*04"})
    # a 5'RACE read observes position 5, so it tells them apart
    race = _race(A.sequence)
    assert known_answer(race, observed_span(race, []), [A, a2, B]) == frozenset({"IGHV1-2*02"})


def test_two_distinct_matches_is_none():
    # T is A without its first three bases: on a 5'RACE read both windows (A[0:288], A[3:288]) match exactly
    truncated = _allele("IGHV1-2*05", "..." + A_GAPPED[3:])
    assert cys104_end(truncated.gapped) == CYS_END - 3
    race = _race(A.sequence)
    assert known_answer(race, observed_span(race, []), [A, truncated, B]) is None
    # a multiplex read starts after both 5' ends, so the two windows are the same sequence: one answer
    read, spans = _multiplex(A.sequence)
    assert known_answer(read, observed_span(read, spans), [A, truncated, B]) == frozenset(
        {"IGHV1-2*02", "IGHV1-2*05"})


def test_an_allele_without_a_cys104_anchor_still_blocks_a_unique_answer():
    # the loader leaves out a gapped V-REGION that is not its sequence; such an allele cannot be windowed, but a
    # read matching it as well as A is no known answer
    unanchored = _allele("IGHV1-2*06", None, sequence=A.sequence + "GT")
    read = _race(A.sequence)
    assert known_answer(read, observed_span(read, []), [A, unanchored, B]) is None
    elsewhere = _allele("IGHV1-2*07", None, sequence=_random(296, 8))
    assert known_answer(read, observed_span(read, []), [A, elsewhere, B]) == frozenset({"IGHV1-2*02"})


def test_an_allele_without_a_cys104_takes_that_of_its_identical_sequence():
    # as VDJbase's IGHV4-38-2*02_..._i93agt_... call, which is IGHV4-39*01's sequence gapped with a spurious
    # insertion: its codon 104 is off 309-311, but a donor allele with its exact sequence places it
    misgapped = A_GAPPED[:93] + "..." + A_GAPPED[93:]
    assert cys104_end(misgapped) is None
    twin = _allele("IGHV4-38-2*02_i93agt", misgapped, gene="IGHV4-38-2")
    assert twin.sequence == A.sequence
    read = _race(A.sequence)
    assert known_answer(read, observed_span(read, []), [twin, A, B]) == frozenset({"IGHV1-2*02",
                                                                                   "IGHV4-38-2*02_i93agt"})
    truth, = real_truths({"r": read}, [_meta("r")], _genotype(twin, A, B), NO_SHARED)
    assert truth.known_gene == "IGHV1-2|IGHV4-38-2"
    # without its twin it cannot be windowed: no answer, and no anchor of its own
    assert known_answer(read, observed_span(read, []), [twin, B]) is None
    assert cys104_in_read(read, observed_span(read, []), [twin, B]) is None


def test_masked_primer_bases_are_not_observed():
    # the assembly filled the VH primer's bases from the other mate, with a mismatch to the allele
    head = _mutate(A.sequence[:22], 10)
    read, spans = _multiplex(A.sequence, head=head)
    assert known_answer(read, observed_span(read, spans), [A, B]) == frozenset({"IGHV1-2*02"})
    # read as observed from base 0, the same mismatch blocks the match
    assert known_answer(read, (0, spans[1][0]), [A, B]) is None


def test_short_window_below_200_is_not_known():
    read, spans = _multiplex(A.sequence, start=100)  # the observed V starts at allele position 100: 188 nt
    observed = observed_span(read, spans)
    assert known_answer(read, observed, [A, B]) is None
    assert known_answer(read, observed, [A, B], min_window=188) == frozenset({"IGHV1-2*02"})
    read, spans = _multiplex(A.sequence, start=CYS_END - 200)
    assert known_answer(read, observed_span(read, spans), [A, B]) == frozenset({"IGHV1-2*02"})
    # an allele whose region is too short to be an answer still matches exactly, so it blocks A's answer
    short = _allele("IGHV1-2*08", "." * 120 + A_GAPPED[120:])  # A from ungapped position 105
    assert cys104_end(short.gapped) == CYS_END - 105 < 200
    race = _race(A.sequence)
    assert known_answer(race, observed_span(race, []), [A, short]) is None


def test_d_and_j_alleles_are_not_v_windows():
    d = RealAllele(allele_id="IGHD3-10*01", gene="IGHD3-10", segment="d", sequence="GTATTACTATGGTTCGGGGAGTTATTATAAC",
                   gapped=None, haplotype=1, functional=True, shared=(), iuis=())
    read = _race(A.sequence)
    assert known_answer(read, observed_span(read, []), [A, d]) == frozenset({"IGHV1-2*02"})


# --- Cys104 by Hamming anchor, W118 by motif -----------------------------------------------------------------------

def test_cys104_anchor_tolerates_three_mismatches():
    expected = len(LEADER) + CYS_END
    for mismatches in range(4):
        positions = [CYS_END - 4 - 7 * k for k in range(mismatches)]
        read = _race(_mutate(A.sequence, *positions))
        assert cys104_in_read(read, observed_span(read, []), [A, B]) == expected, mismatches
    read = _race(_mutate(A.sequence, *[CYS_END - 4 - 7 * k for k in range(4)]))
    assert cys104_in_read(read, observed_span(read, []), [A, B]) is None
    # masked bases are N: never a match, so a masked anchor is not found
    read, spans = _multiplex(A.sequence)
    masked = read[:CYS_END - 20] + "N" * 4 + read[CYS_END - 16:]
    assert cys104_in_read(masked, observed_span(masked, spans), [A, B]) is None


def test_cys104_anchor_takes_the_fewest_mismatches_then_the_leftmost():
    anchor = A.sequence[CYS_END - ANCHOR_LENGTH:CYS_END]
    one, two = _mutate(anchor, 3), _mutate(anchor, 3, 17)
    filler = _random(40, 11)
    read = filler + two + filler + one + filler + one + filler
    first_one = len(filler) * 2 + len(two) + ANCHOR_LENGTH
    assert cys104_in_read(read, (0, len(read)), [A]) == first_one
    tied = filler + one + filler + one + filler
    assert cys104_in_read(tied, (0, len(tied)), [A]) == len(filler) + ANCHOR_LENGTH
    # only the observed span is scanned
    assert cys104_in_read(tied, (len(filler) + 1, len(tied)), [A]) == 2 * len(filler) + 2 * ANCHOR_LENGTH


def test_known_answer_cys104_agrees_with_the_hamming_anchor():
    for read, spans in ((_race(A.sequence), []), _multiplex(A.sequence), (_race(B.sequence), [])):
        observed = observed_span(read, spans)
        match = known_match(read, observed, [A, B])
        assert match is not None and match.cys_end == cys104_in_read(read, observed, [A, B])


def test_junction_by_motif():
    read = _race(A.sequence)
    cys = len(LEADER) + CYS_END
    junction = junction_by_motif(read, cys)
    assert junction == _junction(A.sequence)
    assert junction.startswith("TGT") and junction.endswith("TGG") and len(junction) % 3 == 0
    # the first W118 motif downstream of the Cys codon
    two = LEADER + A.sequence[:CYS_END + 6] + N_REGION + J4 + J4
    assert junction_by_motif(two, cys) == _junction(A.sequence)
    # N (masked or uncalled) bases between the anchors stay in the junction
    masked = read[:cys + 4] + "NN" + read[cys + 6:]
    assert junction_by_motif(masked, cys) == masked[cys - 3:cys + 6 + len(N_REGION) + J4_W118_END]
    no_w118 = LEADER + A.sequence[:CYS_END + 6] + N_REGION + J4.replace("TGGGG", "TAGGG") + C_START
    assert junction_by_motif(no_w118, cys) is None
    with pytest.raises(ValueError):
        junction_by_motif(read, 2)
    with pytest.raises(ValueError):
        junction_by_motif(read, len(read) + 1)


def test_w118_motif_is_the_imgt_w_g_x_g_so_ighj2_has_a_junction():
    # Trp118 Gly119 X Gly121: IGHJ2 has Arg at 120, the other IGHJ genes Gln
    for j, end in ((J4, J4_W118_END), (J2, J2_W118_END)):
        assert W118_MOTIF.search(j).start() + 3 == end
    read = _race(A.sequence, n_region=N_REGION + "A", j=J2)
    junction = junction_by_motif(read, len(LEADER) + CYS_END)
    assert junction == A.sequence[CYS_END - 3:CYS_END + 6] + N_REGION + "A" + J2[:J2_W118_END]
    assert junction.endswith("TGG") and len(junction) % 3 == 0
    truth, = real_truths({"j2": read}, [_meta("j2", "IGHG")], _genotype(A), NO_SHARED)
    assert truth.clonotype == f"{DONOR}/{junction}"


def test_the_junction_is_the_first_in_frame_w118_motif_of_15_to_150_nt():
    cys = len(LEADER) + CYS_END
    head = LEADER + A.sequence[:CYS_END]
    # the first motif is out of frame (43 nt): the next one, in frame (93 nt), ends the junction
    v = A.sequence[:CYS_END + 6]
    shifted = LEADER + v + N_REGION + "A" + J4 + "CC" + J4 + C_START
    assert junction_by_motif(shifted, cys) == v[CYS_END - 3:] + N_REGION + "A" + J4 + "CC" + J4[:J4_W118_END]
    # an out-of-frame motif and no other: no junction, and the read is a clonotype of its own
    out_of_frame = LEADER + v + N_REGION + "A" + J4 + C_START
    assert junction_by_motif(out_of_frame, cys) is None
    truth, = real_truths({"r": out_of_frame}, [_meta("r")], _genotype(A), NO_SHARED)
    assert truth.anchor == "donor" and truth.clonotype == f"{DONOR}/single:r"
    # a motif right after the Cys codon (a 6-nt junction) is too short to be W118: the next in-frame one is
    w118 = J4[J4_W118_END - 3:J4_W118_END + 9]  # TGG GGC CAG GGA
    early = head + w118 + "A" + J4 + C_START
    assert junction_by_motif(early, cys) == A.sequence[CYS_END - 3:CYS_END] + w118 + "A" + J4[:J4_W118_END]
    # the bounds: 15 and 150 nt are junctions, 12 and 153 are not
    for filler, expected in (("GATAGTTAT", 15), ("GATAGT", None), ("GAT" * 48, 150), ("GAT" * 49, None)):
        read = head + filler + J4[J4_W118_END - 3:] + C_START
        junction = junction_by_motif(read, cys)
        assert (junction if expected is None else len(junction)) == expected, len(filler)
    assert junction_by_motif(head + "GATAGT" + J4[J4_W118_END - 3:] + C_START, cys, min_length=12) is not None


def test_a_known_answer_takes_its_junction_from_its_exact_window():
    # B's anchor, exactly, in the 5' UTR: the leftmost exact Hamming anchor is there, but A's exact window ends at
    # A's Cys104, and that is where a known answer's junction starts
    b_anchor = B.sequence[CYS_END - ANCHOR_LENGTH:CYS_END]
    read = LEADER + b_anchor + _random(40, 14) + A.sequence[:CYS_END + 6] + N_REGION + J4 + C_START
    observed = observed_span(read, [])
    assert cys104_in_read(read, observed, [A, B]) == len(LEADER) + ANCHOR_LENGTH
    match = known_match(read, observed, [A, B])
    assert match.alleles == (A,) and match.cys_end == len(LEADER) + ANCHOR_LENGTH + 40 + CYS_END
    truth, = real_truths({"r": read}, [_meta("r")], _genotype(A, B), NO_SHARED)
    assert truth.known == frozenset({"IGHV1-2*02"}) and truth.anchor == "donor"
    assert truth.clonotype == f"{DONOR}/{_junction(A.sequence)}"


# --- real_truths ---------------------------------------------------------------------------------------------------

def test_naive_only_igm():
    genotype = _genotype(A, B)
    reads = {"m": _race(A.sequence), "g": _race(A.sequence + "C"), "mutated": _race(_mutate(A.sequence, 40))}
    meta = [_meta("m", "IGHM"), _meta("g", "IGHG"), _meta("mutated", "IGHM")]
    truths = {truth.read_id: truth for truth in real_truths(reads, meta, genotype, NO_SHARED)}
    assert truths["m"].naive and truths["m"].known == frozenset({"IGHV1-2*02"})
    assert not truths["g"].naive and truths["g"].known == frozenset({"IGHV1-2*02"})
    assert not truths["mutated"].naive and truths["mutated"].known is None


def test_clonotype_singleton_without_anchors():
    genotype = _genotype(A, B)
    no_w118 = LEADER + A.sequence[:CYS_END + 6] + N_REGION + J4.replace("TGGGG", "TAGGG") + C_START
    no_cys = _race(_mutate(A.sequence, *[CYS_END - 4 - 7 * k for k in range(4)]))
    reads = {"both": _race(A.sequence), "no_w118": no_w118, "no_cys": no_cys}
    truths = {t.read_id: t for t in real_truths(reads, [_meta(name) for name in reads], genotype, NO_SHARED)}
    assert truths["both"].clonotype == f"{DONOR}/{_junction(A.sequence)}"
    assert truths["no_w118"].clonotype == f"{DONOR}/single:no_w118"
    assert truths["no_cys"].clonotype == f"{DONOR}/single:no_cys"
    # the read with no W118 still has an exact V window: a known answer without a clonotype junction
    assert truths["no_w118"].known == frozenset({"IGHV1-2*02"})
    # ``anchor`` says where Cys104 was found, W118 or not
    assert [truths[name].anchor for name in reads] == ["donor", "donor", None]


def test_the_shared_reference_places_cys104_only_without_a_donor_anchor():
    genotype, reference = _genotype(A), _reference(A, B)
    three = [CYS_END - 4 - 7 * k for k in range(3)]
    four = [CYS_END - 4 - 7 * k for k in range(4)]
    reads = {"a": _race(A.sequence), "b": _race(B.sequence), "b3": _race(_mutate(B.sequence, *three)),
             "b4": _race(_mutate(B.sequence, *four)), "none": _race(_random(296, 12))}
    truths = {t.read_id: t for t in real_truths(reads, [_meta(name) for name in reads], genotype, reference)}
    assert truths["a"].anchor == "donor" and truths["a"].known == frozenset({"IGHV1-2*02"})
    # B is not the donor's: a shared anchor places its junction, under the same Hamming rule, and gives no answer
    for name in ("b", "b3"):
        truth = truths[name]
        assert truth.anchor == "shared" and truth.clonotype == f"{DONOR}/{_junction(B.sequence)}", name
        assert truth.known is None and truth.known_shared is None and truth.known_gene is None
        assert not truth.naive and not truth.novel_known
    for name in ("b4", "none"):
        assert truths[name].anchor is None and truths[name].clonotype == f"{DONOR}/single:{name}"
    alone, = real_truths({"b": reads["b"]}, [_meta("b")], genotype, NO_SHARED)
    assert alone.anchor is None and alone.clonotype == f"{DONOR}/single:b"


def test_a_shared_reference_whose_v_alleles_give_no_anchor_is_refused():
    # a reference read from a FASTA file has V alleles but no gapped form: it would silently disable the fallback
    read = _race(A.sequence)
    from_fasta = Reference(alleles={"v": {B.allele_id: B.sequence}}, anchors={})
    with pytest.raises(ValueError, match="gapped"):
        real_truths({"r": read}, [_meta("r")], _genotype(A), from_fasta)
    no_cys = _allele("IGHV3-23*01", _gapped_v(2, codon="TAT"))
    with pytest.raises(ValueError, match="gapped"):
        real_truths({"r": read}, [_meta("r")], _genotype(A), _reference(no_cys))
    # one allele with an anchor is enough, and a reference without V alleles has none to give
    assert real_truths({"r": read}, [_meta("r")], _genotype(A), _reference(no_cys, B))[0].anchor == "donor"
    assert real_truths({"r": read}, [_meta("r")], _genotype(A), NO_SHARED)[0].anchor == "donor"


def test_a_donor_anchor_wins_over_a_better_shared_one():
    # the donor's anchor with two mismatches, and a shared allele's exact anchor further left
    b_anchor = B.sequence[CYS_END - ANCHOR_LENGTH:CYS_END]
    v = _mutate(A.sequence, 100, CYS_END - 5, CYS_END - 20)
    read = LEADER + b_anchor + _random(40, 13) + v[:CYS_END + 6] + N_REGION + J4 + C_START
    truth, = real_truths({"r": read}, [_meta("r")], _genotype(A), _reference(A, B))
    assert truth.anchor == "donor" and truth.clonotype == f"{DONOR}/{_junction(A.sequence)}"
    assert truth.known is None


def test_mutated_clone_members_share_a_clonotype():
    genotype = _genotype(A, B)
    reads = {"c1": _race(_mutate(A.sequence, 50, 120)), "c2": _race(_mutate(A.sequence, 80, 200, CYS_END - 12)),
             "c3": _race(A.sequence), "other": _race(_mutate(A.sequence, 50), n_region="GGGACTACAG"),
             "b": _race(B.sequence)}
    meta = [_meta(name, "IGHG") for name in reads]
    truths = real_truths(reads, meta, genotype, NO_SHARED)
    clusters = clonotype_clusters(truths)
    assert clusters == {t.read_id: t.clonotype for t in truths}
    assert clusters["c1"] == clusters["c2"] == clusters["c3"] == f"{DONOR}/{_junction(A.sequence)}"
    assert clusters["other"] == f"{DONOR}/{_junction(A.sequence, n_region='GGGACTACAG')}"
    assert clusters["b"] == f"{DONOR}/{_junction(B.sequence)}"
    assert [t.known for t in truths] == [None, None, frozenset({"IGHV1-2*02"}), None, frozenset({"IGHV3-23*01"})]


def test_real_truths_fields():
    paralog = _gapped_v(4)
    p1 = _allele("IGHV1-69*01", paralog, shared=("IGHVF6-G21*06",))
    p2 = _allele("IGHV1-69D*01", paralog, gene="IGHV1-69D", shared=("IGHVF6-G21*06",))
    novel = _allele("IGHV3-30*18_t189c", _gapped_v(5), shared=())
    j = RealAllele(allele_id="IGHJ4*02", gene="IGHJ4", segment="j", sequence=J4, gapped=None, haplotype=0,
                   functional=True, shared=("IGHJ4*02",), iuis=("IGHJ4*02",))
    genotype = _genotype(A, p1, p2, novel, j)
    multiplex, spans = _multiplex(A.sequence)
    reads = {"r1": _race(p1.sequence), "r2": _race(novel.sequence), "r3": multiplex, "r4": _race(B.sequence)}
    meta = [_meta("r1"), _meta("r2", "IGHG"), {**_meta("r3", protocol="multiplex"), "primer_masked": spans},
            _meta("r4")]
    truths = real_truths(reads, meta, genotype, NO_SHARED)
    assert [t.read_id for t in truths] == ["r1", "r2", "r3", "r4"]
    r1, r2, r3, r4 = truths
    assert isinstance(r1, RealTruth) and r1.donor == DONOR and r1.protocol == "race" and r1.isotype == "IGHM"
    assert r1.known == frozenset({"IGHV1-69*01", "IGHV1-69D*01"})
    assert r1.known_shared == frozenset({"IGHVF6-G21*06"}) and r1.known_gene == "IGHV1-69|IGHV1-69D"
    assert not r1.novel_known and r1.naive
    assert r2.known == frozenset({"IGHV3-30*18_t189c"}) and r2.known_shared == frozenset()
    assert r2.novel_known and r2.known_gene == "IGHV3-30" and not r2.naive
    assert r3.protocol == "multiplex" and r3.observed == (22, len(multiplex) - 20)
    assert r3.known == frozenset({"IGHV1-2*02"}) and r3.known_shared == frozenset({"IGHVF1-G1*01"})
    assert r3.known_gene == "IGHV1-2" and r3.naive
    # B is not this donor's: its Cys104 anchor is not one of the donor's, so no anchor and no answer
    assert r4.known is None and r4.known_shared is None and r4.known_gene is None
    assert not r4.novel_known and not r4.naive and r4.clonotype == f"{DONOR}/single:r4"
    assert r4.observed == (0, len(reads["r4"]))
    assert [t.anchor for t in truths] == ["donor", "donor", "donor", None]


def test_real_truths_refuses_inconsistent_reads_and_metadata():
    genotype = _genotype(A)
    read = _race(A.sequence)
    with pytest.raises(ValueError, match="no read"):
        real_truths({}, [_meta("x")], genotype, NO_SHARED)
    with pytest.raises(ValueError, match="no metadata"):
        real_truths({"x": read, "y": read}, [_meta("x")], genotype, NO_SHARED)
    with pytest.raises(ValueError, match="twice"):
        real_truths({"x": read}, [_meta("x"), _meta("x")], genotype, NO_SHARED)
    with pytest.raises(ValueError):
        real_truths({"x": read}, [_meta("x", spans=[[0, len(read) + 5]])], genotype, NO_SHARED)
    # a genotype mapped to one shared reference is not read against another
    mapped = RealGenotype(donor=DONOR, alleles=(A,), source={"reference_digest": _reference(A).digest()})
    assert real_truths({"x": read}, [_meta("x")], mapped, _reference(A))[0].known == frozenset({"IGHV1-2*02"})
    with pytest.raises(ValueError, match="reference"):
        real_truths({"x": read}, [_meta("x")], mapped, _reference(A, B))


# --- the pilot donors' real reads ----------------------------------------------------------------------------------

PILOT_SAMPLES = {"W-97": "P25_I151_S1", "SC-23": "P25_I58_S1"}
# The least share of reads a donor Cys104 anchor places (measured: W-97 66.5%, SC-23 90.7%). W-97's genomic
# genotype has no IGHV4 gene, while about a quarter of its reads are IGHV4, whose anchors are therefore missing;
# the shared reference's anchors place them (donor or shared, measured: W-97 92.8%, SC-23 91.5%).
MIN_DONOR_ANCHORED = {"W-97": 0.6, "SC-23": 0.85}
MIN_ANCHORED = 0.9
# The least share of reads with an in-frame junction of 15 to 150 nt (measured: W-97 88.0%, SC-23 88.9%).
MIN_JUNCTIONS = 0.85
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


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


@pytest.mark.skipif(not DATA or not RUN.is_dir(),
                    reason="set ALIGNAIR_ASSAY_DATA to the PRJNA555323 data root, with Task 4's run in "
                           ".private/assay/p3a_task4_run")
@pytest.mark.parametrize("donor", sorted(PILOT_SAMPLES))
def test_the_pilot_donors_real_truth(donor, reference):
    root = Path(DATA)
    iuis = IuisMap.from_ogrdb(root / "genotypes/ogrdb_human_IGH_VDJ/Homo_sapiens_IGH_VDJ_rev_8.json", reference)
    genotype = load_vdjbase_csv(root / "genotypes/vdjbase_genomic_samples" / f"{PILOT_SAMPLES[donor]}.csv", donor,
                                reference, iuis)
    layout = RunLayout(RUN)
    reads, meta = _read_fasta(layout.real_reads(donor)), _read_tsv(layout.real_reads_meta(donor))
    truths = real_truths(reads, meta, genotype, reference)
    n = len(truths)
    assert n == len(reads) == len(meta)
    known = [t for t in truths if t.known is not None]
    assert len(known) / n > 0.05  # the design's flag line (measured: W-97 5.7%, SC-23 25.9%)
    assert all(t.naive == (t.known is not None and t.isotype == "IGHM") for t in truths)
    assert all(t.novel_known == (t.known is not None and not t.known_shared) for t in truths)
    spans = {row["sequence_id"]: json.loads(row["primer_masked"]) for row in meta}
    # every known answer's Cys104 (from its exact window) is where the donor's Hamming anchor puts it
    assert all(truth.anchor == "donor" for truth in known)
    for truth in known:
        masked = mask_primers(reads[truth.read_id], spans[truth.read_id])
        match = known_match(masked, truth.observed, genotype.alleles)
        assert match.cys_end == cys104_in_read(masked, truth.observed, genotype.alleles), truth.read_id
    sources = Counter(truth.anchor for truth in truths)
    assert set(sources) <= {"donor", "shared", None}
    assert sources["donor"] / n > MIN_DONOR_ANCHORED[donor]
    assert 1 - sources[None] / n > MIN_ANCHORED
    # pRESTO orients every read V to C: the anchors are found forward, and almost never on the reverse complement
    # of a read without one
    unanchored = [truth for truth in truths if truth.anchor is None]
    flipped = {t.read_id: mask_primers(reads[t.read_id], spans[t.read_id]).translate(COMPLEMENT)[::-1]
               for t in unanchored}
    reverse = real_truths(flipped, [{**_meta(t.read_id, t.isotype, t.protocol), "primer_masked": []}
                                    for t in unanchored], genotype, reference)
    assert sum(truth.anchor is not None for truth in reverse) / n < 0.001
    junctions = [t.clonotype.split("/", 1)[1] for t in truths if t.clonotype != f"{donor}/single:{t.read_id}"]
    assert all(len(junction) % 3 == 0 and MIN_JUNCTION <= len(junction) <= MAX_JUNCTION for junction in junctions)
    assert len(junctions) / n > MIN_JUNCTIONS
    sizes = Counter(clonotype_clusters(truths).values())
    assert max(sizes.values()) > 1
