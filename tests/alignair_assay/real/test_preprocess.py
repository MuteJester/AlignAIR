"""Annotation-free preprocessing of real AIRR-seq runs (``real.preprocess``).

The offline tests build read pairs with the synthetic generator below and never start Docker. The gated tests:

- ``ALIGNAIR_ASSAY_DOCKER=1``: both recipes end to end in the pinned Immcantation container, on synthetic pairs;
- ``ALIGNAIR_ASSAY_DATA`` (the PRJNA555323 data root, with the pilot reads fetched): the eight pilot runs classified.

**The generator.** Each molecule is a V(D)J built from one of two shared-reference V alleles (of different
families), a D, a J and random N1/N2 nucleotides, so every molecule's V(D)J is unique.

- A 5'RACE pair carries a 12-nt UMI + ``GTAC`` + ``GGG`` and a 10-nt 5'UTR before the V(D)J on R2, and a 7-nt
  random stagger, a 20-nt stand-in for the kit's primer and the start of CH1 (the reverse complement of the
  isotype's ``CREGION_RC`` line) on R1, which reads the reverse complement.
- A multiplex pair starts with the VH4 primer on R1 and the IgM primer on R2.
- Reads are 2 x 250 with Phred 40 qualities; read names are SRA's (``<run>.<spot> <spot>/<mate>``).
"""
from __future__ import annotations

import gzip
import json
import os
import random
import time
from pathlib import Path

import pytest

from alignair_assay.core.cases import evenly_spaced, read_fasta
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.seqops import revcomp
from alignair_assay.real import preprocess
from alignair_assay.real.preprocess import (
    C_PRIMERS_MULTIPLEX,
    CREGION_RC,
    IMMCANTATION_IMAGE,
    VH_PRIMERS,
    PreprocessError,
    cap_sequences,
    classify_protocol,
    preprocess_run,
    primer_mismatches,
    primer_spans,
    recipe,
)

pytestmark = pytest.mark.component

DATA = os.environ.get("ALIGNAIR_ASSAY_DATA")
DOCKER = os.environ.get("ALIGNAIR_ASSAY_DOCKER") == "1"
needs_data = pytest.mark.skipif(not DATA, reason="set ALIGNAIR_ASSAY_DATA to the PRJNA555323 data root")
needs_docker = pytest.mark.skipif(not DOCKER, reason="set ALIGNAIR_ASSAY_DOCKER=1 to run the Immcantation container")

V_ALLELES = ("IGHVF1-G1*01", "IGHVF3-G8*01")
D_ALLELE, J_ALLELE = "IGHD1-26*01", "IGHJ4*02"
READ_LENGTH = 250
UTR = "ACTGAGCACA"
KIT_PRIMER = {"IGHM": "CCGACGGGGAATTCTCACAG", "IGHG": "GAAGACCGATGGGCCCTTGG"}  # stand-ins, 20 nt
IGG_PRIMER = "CGATGGGCCCTTGGTGGAAGC"  # SD7's IgG primer with S=C and R=A, 21 nt
VH3_PRIMER = "AGGTGTCCAGTGTGAAGTGCAG"  # SD7's VH3 primer with R=A, 22 nt


def _random_dna(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _molecules(reference, n: int, rng: random.Random, *, isotypes=("IGHM", "IGHG")) -> list[dict]:
    """``n`` molecules: V alternating between ``V_ALLELES``, isotype cycling through ``isotypes``, each with its own
    N1/N2 (so its V(D)J, ``vdj``, is unique) and UMI."""
    molecules = []
    for index in range(n):
        vdj = (reference.seq(V_ALLELES[index % 2]) + _random_dna(rng, 6) + reference.seq(D_ALLELE)
               + _random_dna(rng, 6) + reference.seq(J_ALLELE))
        molecules.append({"vdj": vdj, "isotype": isotypes[index % len(isotypes)], "umi": _random_dna(rng, 12),
                          "v": V_ALLELES[index % 2]})
    return molecules


def _substitute(read: str, position: int) -> str:
    base = read[position]
    return read[:position] + ("A" if base != "A" else "C") + read[position + 1:]


def _race_pairs(molecules: list[dict], reads: int, rng: random.Random) -> tuple[list[str], list[str]]:
    """``reads`` pairs per molecule; the second pair of each molecule has one substitution in each mate, which the
    UMI consensus must correct."""
    r1, r2 = [], []
    for molecule in molecules:
        c_start = revcomp(CREGION_RC[molecule["isotype"]]) + revcomp(KIT_PRIMER[molecule["isotype"]])
        amplicon = UTR + molecule["vdj"] + c_start
        for copy in range(reads):
            mate1 = (_random_dna(rng, 7) + revcomp(amplicon))[:READ_LENGTH]
            mate2 = (molecule["umi"] + "GTAC" + "GGG" + amplicon)[:READ_LENGTH]
            if copy == 1:
                mate1, mate2 = _substitute(mate1, 150), _substitute(mate2, 150)
            r1.append(mate1)
            r2.append(mate2)
    return r1, r2


def _multiplex_amplicon(vdj: str, c_primer: str = C_PRIMERS_MULTIPLEX["IgM"],
                        v_primer: str = VH_PRIMERS["VH4"]) -> str:
    return v_primer + vdj + revcomp(c_primer)


def _multiplex_pairs(molecules: list[dict], copies: int) -> tuple[list[str], list[str]]:
    """``copies`` identical pairs per molecule (PCR duplicates)."""
    r1, r2 = [], []
    for molecule in molecules:
        amplicon = _multiplex_amplicon(molecule["vdj"])
        for _ in range(copies):
            r1.append(amplicon[:READ_LENGTH])
            r2.append(revcomp(amplicon)[:READ_LENGTH])
    return r1, r2


def _write_pairs(directory: Path, r1: list[str], r2: list[str], *, quality: str = "I") -> tuple[Path, Path]:
    """The mates as gzipped FASTQs with SRA's read names; ``quality`` is every base's quality symbol."""
    paths = []
    for mate, reads in ((1, r1), (2, r2)):
        path = directory / f"SYN_{mate}.fastq.gz"
        with gzip.open(path, "wt", encoding="ascii") as handle:
            for spot, read in enumerate(reads, start=1):
                handle.write(f"@SYN.{spot} {spot}/{mate}\n{read}\n+\n{quality * len(read)}\n")
        paths.append(path)
    return paths[0], paths[1]


@pytest.fixture(scope="module")
def rng():
    return random.Random(20261020)


# --- constants -----------------------------------------------------------------------------------------------------

def test_the_primers_are_the_papers_and_immcantations_verbatim():
    assert VH_PRIMERS == {"VH1": "ASKCCCAGGTBCAGCTKGTRC", "VH2": "CAGRTCACCTTGAAGGAGTCTG",
                          "VH3": "AGGTGTCCAGTGTGARGTGCAG", "VH4": "ATGGGTCCTGTCCCAGGTGC",
                          "VH5": "TCCACAGTGAGAGTCTGTGCC", "VH5-2": "TTCTCCAAGGAGTCTGTGCCG",
                          "VH6": "CAGGTACAGCTGCAGCAGTCA"}
    assert C_PRIMERS_MULTIPLEX == {"IgM": "GGTTGGGGCGGATGCACTCC", "IgG": "SGATGGGCCCTTGGTGGARGC",
                                   "IgA": "CGGGAAGACCTTGGGGCTGGT"}
    assert CREGION_RC == {"IGHA": "CGGGAAGACCTTGGGGCTGGTCGGGGATGC", "IGHD": "GGGGAACACATCCGGAGCCTTGGTGGGTGC",
                          "IGHE": "GGGGAAGACGGATGGGCTCTGTGTGGAGGC", "IGHG": "GGGGAAGACSGATGGGCCCTTGGTGGAAGC",
                          "IGHM": "GGGGAAAAGGGTTGGGGCGGATGCACTCCC"}


def test_the_image_is_the_pinned_immcantation_suite():
    assert IMMCANTATION_IMAGE == ("immcantation/suite:4.8.0@sha256:"
                                  "44e228a45b047bdb352363ab20a670336b003e98f3d1bac505f78af41a91ab67")


# --- the classifier ------------------------------------------------------------------------------------------------

def test_primer_mismatches_reads_iupac_codes():
    assert primer_mismatches("ATGGGTCCTGTCCCAGGTGCAGCTG", VH_PRIMERS["VH4"]) == 0
    # VH1 ASKCCCAGGTBCAGCTKGTRC: S = C/G, K = G/T, B = C/G/T, R = A/G
    assert primer_mismatches("ACGCCCAGGTGCAGCTGGTGC", VH_PRIMERS["VH1"]) == 0
    assert primer_mismatches("AAGCCCAGGTACAGCTGGTGC", VH_PRIMERS["VH1"]) == 2  # A for S, A for B
    assert primer_mismatches("ATGGGT", VH_PRIMERS["VH4"]) == 14  # a short read misses the primer's rest
    assert primer_mismatches("atgggtcctgtcccaggtgc", VH_PRIMERS["VH4"]) == 0


def test_classify_race(reference, rng):
    r1, r2 = _race_pairs(_molecules(reference, 100, rng), 3, rng)
    protocol, evidence = classify_protocol(r1, r2)
    assert protocol == "race"
    assert evidence["pairs"] == 300 and evidence["race_linker"] == 1.0
    assert evidence["vh_primer"] == 0.0 and evidence["threshold"] == 0.8 and evidence["max_mismatches"] == 2
    assert evidence["r2_lengths"] == {"min": 250, "median": 250, "max": 250}


def test_classify_multiplex(reference, rng):
    r1, r2 = _multiplex_pairs(_molecules(reference, 100, rng), 2)
    # a real run's reads carry errors: two mismatches in the VH primer still count, three do not
    r1[0] = _substitute(_substitute(r1[0], 3), 10)
    r1[1] = _substitute(_substitute(_substitute(r1[1], 3), 10), 15)
    protocol, evidence = classify_protocol(r1, r2)
    assert protocol == "multiplex"
    assert evidence["vh_primer"] == 0.995 and evidence["c_primer"] == 1.0 and evidence["race_linker"] == 0.0
    assert evidence["vh_primers"] == {"VH4": 199} and evidence["c_primers"] == {"IgM": 200}


def test_classify_mixed_is_unknown(reference, rng):
    """A 50/50 mix of the two layouts matches neither threshold: it is refused, with its fractions."""
    race1, race2 = _race_pairs(_molecules(reference, 50, rng), 2, rng)
    multi1, multi2 = _multiplex_pairs(_molecules(reference, 50, rng), 2)
    protocol, evidence = classify_protocol(race1 + multi1, race2 + multi2)
    assert protocol == "unknown"
    assert evidence["race_linker"] == 0.5 and evidence["vh_primer"] == 0.5 and evidence["c_primer"] == 0.5
    assert "neither" in evidence["reason"]
    # reads of neither layout are unknown too
    noise = [_random_dna(rng, 250) for _ in range(20)]
    assert classify_protocol(noise, noise)[0] == "unknown"
    with pytest.raises(ValueError, match="pairs"):
        classify_protocol(noise, noise[:-1])
    with pytest.raises(ValueError, match="pairs"):
        classify_protocol([], [])


def test_both_layouts_at_once_is_unknown(monkeypatch):
    """When both rules hold, neither recipe is trusted. No SD7 C primer can hold the linker at positions 12-15
    within two mismatches (each differs from GTAC there in three or four places), so the rule is exercised with a
    primer that does."""
    for primer in C_PRIMERS_MULTIPLEX.values():
        assert primer_mismatches(primer[:12] + "GTAC" + primer[16:], primer) >= 3
    monkeypatch.setitem(preprocess.C_PRIMERS_MULTIPLEX, "IgX", "CCCCCCCCCCCCGTACCCCC")
    r2 = "CCCCCCCCCCCCGTACCCCC" + "A" * 100
    protocol, evidence = classify_protocol([VH_PRIMERS["VH4"] + "A" * 100] * 10, [r2] * 10)
    assert protocol == "unknown" and "both" in evidence["reason"]
    assert evidence["race_linker"] == evidence["vh_primer"] == evidence["c_primer"] == 1.0


# --- the cap and the masked spans ------------------------------------------------------------------------------------

def _unique_rows(n: int, rng: random.Random) -> list[dict]:
    """``n`` rows of distinct content, as ``_rows`` makes them: sequence, isotype, primers and a pRESTO id."""
    return [{"sequence_id": f"SYN.{index}", "sequence": _random_dna(rng, 40), "isotype": ("IGHM", "IGHG")[index % 2],
             "primers": "VH4,IgM"} for index in range(n)]


def test_cap_is_evenly_spaced_over_the_content_order():
    """The cap orders the sequences by the SHA-256 of their content (sequence, isotype, primers), not by pRESTO's
    id, then takes them evenly spaced: which ids survive pRESTO's collapse depends on its workers' order."""
    rng = random.Random(3)
    rows = _unique_rows(100, rng)
    ordered = sorted(rows, key=preprocess.content_key)
    capped = cap_sequences(rows, 10)
    assert capped == evenly_spaced(ordered, 10)
    assert cap_sequences(rows, 1000) == ordered  # fewer than the cap: all of them, in content order
    assert cap_sequences([], 10) == []
    # the same contents in another order, and under other ids, give the same capped contents
    shuffled = rows[:]
    rng.shuffle(shuffled)
    assert cap_sequences(shuffled, 10) == capped
    ids = [row["sequence_id"] for row in rows]
    rng.shuffle(ids)
    renamed = [{**row, "sequence_id": f"OTHER.{identifier}"} for row, identifier in zip(rows, ids)]
    assert [(row["sequence"], row["isotype"]) for row in cap_sequences(renamed, 10)] == [
        (row["sequence"], row["isotype"]) for row in capped]
    # identical contents (Collapse's --keepmiss keeps them apart) fall back on the id
    twins = [{**rows[0], "sequence_id": "SYN.b"}, {**rows[0], "sequence_id": "SYN.a"}]
    assert [row["sequence_id"] for row in cap_sequences(twins, 2)] == ["SYN.a", "SYN.b"]
    with pytest.raises(ValueError, match="cap"):
        cap_sequences(rows, 0)
    with pytest.raises(ValueError, match="twice"):
        cap_sequences(rows + rows[:1], 10)


def test_sorting_a_fastq_by_read_id_keeps_every_record(tmp_path):
    """The inputs of the steps whose output depends on their read order (ClusterSets, AlignSets, CollapseSeq) are
    sorted by read id first, so a rerun does not depend on the workers' order of the step before."""
    records = [f"@SYN.{spot} {spot}/1|UMI=AAA\nACGT{spot}\n+\nIIII{spot}\n" for spot in (3, 1, 20, 2)]
    path = tmp_path / "reads.fastq"
    path.write_text("".join(records))
    preprocess.sort_fastq(path)
    assert path.read_text() == "".join(records[i] for i in (1, 3, 2, 0))  # SYN.1, SYN.2, SYN.20, SYN.3
    preprocess.sort_fastq(path)
    assert path.read_text() == "".join(records[i] for i in (1, 3, 2, 0))


def test_primer_spans_recorded():
    """pRESTO's mask mode turns a primer at the start of a mate into Ns. After assembly the R1 primer is at the
    start of the read and the reverse-complemented R2 primer at its end; the spans are half-open."""
    read = "N" * 20 + "ACGT" * 50 + "N" * 21
    assert primer_spans(len(read), 20, 21) == [[0, 20], [220, 241]]
    assert primer_spans(100, 0, 0) == []
    assert primer_spans(100, 20, 0) == [[0, 20]]
    assert primer_spans(100, 0, 20) == [[80, 100]]
    with pytest.raises(ValueError, match="overlap"):
        primer_spans(30, 20, 21)
    with pytest.raises(ValueError):
        primer_spans(30, -1, 0)


def test_assembly_evidence_splits_the_log_into_passed_and_failed_pairs():
    """``AssemblePairs --log`` writes one block per pair, with the best alignment it tried even for a pair it could
    not assemble; the pairs are told apart by whether their id reached the assembled output."""
    log = """      ID> 1AAAA
 FIELDS1> CONSCOUNT=3
    SEQ1> ACGTACGTAC
    SEQ2>         ACGTTT
ASSEMBLY> ACGTACGTACGTTT
  LENGTH> 14
 OVERLAP> 2
   ERROR> 0.0000
  PVALUE> 1.0000e-06

      ID> SYN.2 2/1|VPRIMER=VH4
    SEQ1> ACGTACGTAC
    SEQ2>            GGGGGGGG
ASSEMBLY> ACGTACGTACNGGGGGGGG
  LENGTH> 19
 OVERLAP> 8
   ERROR> 0.5000
  PVALUE> 4.0000e-01
"""
    evidence = preprocess.assembly_evidence(log.splitlines(keepends=True), {"1AAAA"})
    assert evidence["pass"]["n"] == 1 and evidence["fail"]["n"] == 1
    # why a pair failed, by AssemblePairs' own tests: an overlap over the error limit, one under it that is too
    # short to be significant, or no overlap tried at all
    assert evidence["fail"]["kinds"] == {"mismatches": 1, "not_significant": 0, "no_overlap": 0}
    more = log + """
      ID> SYN.3 3/1
    SEQ1> ACGTACGTAC
    SEQ2>           ACGT
ASSEMBLY> ACGTACGTACACGT
  LENGTH> 14
 OVERLAP> 8
   ERROR> 0.1000
  PVALUE> 2.0000e-03

      ID> SYN.4 4/1
    SEQ1> ACGTACGTAC
    SEQ2>           TTTT
ASSEMBLY> None
"""
    kinds = preprocess.assembly_evidence(more.splitlines(keepends=True), {"1AAAA"})["fail"]["kinds"]
    assert kinds == {"mismatches": 1, "not_significant": 1, "no_overlap": 1}
    assert evidence["pass"]["overlap"]["median"] == 2 and evidence["fail"]["overlap"]["median"] == 8
    assert evidence["fail"]["error"]["max"] == 0.5
    assert evidence["fail"]["head_length"]["min"] == 10 and evidence["fail"]["tail_length"]["min"] == 8
    # the insert-length estimates: the mates' combined length for both groups (a lower bound on a failed pair's
    # insert when its mates do not meet), and the assembled length for the pairs that passed; median and deciles
    assert evidence["pass"]["mates_length"] == {"n": 1, "min": 16, **{f"p{d}": 16 for d in range(10, 100, 10)},
                                                "median": 16, "max": 16}
    assert evidence["fail"]["mates_length"]["median"] == 18 and "insert_length" not in evidence["fail"]
    assert evidence["pass"]["insert_length"]["median"] == 14 and evidence["pass"]["insert_length"]["p90"] == 14
    empty = preprocess.assembly_evidence([], set())
    assert empty["pass"]["n"] == empty["fail"]["n"] == 0
    assert empty["pass"]["insert_length"] == empty["fail"]["mates_length"] == {"n": 0}
    assert empty["fail"]["kinds"] == {"mismatches": 0, "not_significant": 0, "no_overlap": 0}


def test_the_assembly_acceptance_line():
    """At most 35% of the pairs entering the assembly may fail; ``accept_assembly_loss`` lets a run over the line
    through deliberately, and says so."""
    within = preprocess.assembly_acceptance(100, 65, accept_assembly_loss=False)
    assert within == {"pairs": 100, "failed": 35, "failure_rate": 0.35, "max_failure_rate": 0.35,
                      "within_line": True, "override": False, "accepted": True}
    over = preprocess.assembly_acceptance(100, 64, accept_assembly_loss=False)
    assert over["within_line"] is False and over["accepted"] is False and over["failure_rate"] == 0.36
    let_through = preprocess.assembly_acceptance(100, 64, accept_assembly_loss=True)
    assert let_through["within_line"] is False and let_through["override"] is True and let_through["accepted"]
    assert preprocess.assembly_acceptance(0, 0, accept_assembly_loss=False)["accepted"] is False


def test_deciles_are_the_nearest_rank_quantiles():
    found = preprocess.deciles(list(range(1, 101)))
    assert found["n"] == 100 and found["min"] == 1 and found["max"] == 100
    assert [found[f"p{d}"] for d in range(10, 100, 10)] == [11, 21, 31, 41, 51, 61, 71, 81, 91]
    assert found["median"] == found["p50"] == 51
    assert preprocess.deciles([]) == {"n": 0}


# --- the recipes ---------------------------------------------------------------------------------------------------

def _names(steps) -> list[str]:
    return [step.name for step in steps]


def test_the_race_recipe_follows_the_paper():
    steps = recipe("race", quality="full", r1="/in1/a_1.fastq.gz", r2="/in2/a_2.fastq.gz", nproc=8)
    assert _names(steps) == ["quality-r1", "quality-r2", "umi", "linker", "c-primer", "pair", "cluster", "molecule",
                             "pair-molecule", "align-r1", "align-r2", "consensus-r1", "consensus-r2",
                             "pair-consensus", "assemble", "length", "conscount", "collapse", "conscount-2"]
    by_name = {step.name: step.argv for step in steps}
    assert "/in1/a_1.fastq.gz" in by_name["quality-r1"] and "/in2/a_2.fastq.gz" in by_name["quality-r2"]
    assert _flags(by_name["umi"], "-s") == ["R2-quality_quality-pass.fastq"]
    assert _flags(by_name["c-primer"], "-s") == ["R1-quality_quality-pass.fastq"]
    assert _flags(by_name["umi"], "--start", "--len", "--pf", "--mode") == ["0", "12", "UMI", "cut"]
    assert _flags(by_name["linker"], "--start", "--len", "--mode") == ["0", "7", "cut"]
    assert by_name["c-primer"][:2] == ("MaskPrimers.py", "align")
    assert _flags(by_name["c-primer"], "--maxlen", "--maxerror", "--mode", "--pf") == ["70", "0.2", "cut", "CPRIMER"]
    assert by_name["cluster"][:2] == ("ClusterSets.py", "set") and "vsearch" in by_name["cluster"]
    assert by_name["align-r1"][:2] == ("AlignSets.py", "muscle")
    assert by_name["assemble"][:2] == ("AssemblePairs.py", "align")
    # --scanrev as in presto-clontech-umi: a read-through pair, whose head overhangs its tail, can assemble
    assert "--scanrev" in by_name["assemble"]
    assert _flags(by_name["assemble"], "--maxerror", "--alpha") == ["0.3", "1e-05"]
    assert not any("sequential" in step.argv or "-r" in step.argv for step in steps)  # no reference-guided assembly
    # the same floor as multiplex: at least 250 nt once primers, UMI and linker are cut
    assert by_name["length"][:2] == ("FilterSeq.py", "length") and _flags(by_name["length"], "-n") == ["250"]
    assert "--inner" in by_name["length"] and _flags(by_name["conscount"], "-s") == ["filtered_length-pass.fastq"]
    assert _flags(by_name["conscount-2"], "-f", "--num") == ["CONSCOUNT", "2"]
    assert all(_flags(step.argv, "--nproc") == ["8"] for step in steps if "--nproc" in step.argv)
    # a consensus or a collapse counts groups, so it has no pass rate
    assert {step.name for step in steps if step.merges} == {"consensus-r1", "consensus-r2", "collapse"}
    # the steps whose output depends on their input's order read it sorted by read id
    assert {step.name for step in steps if step.sort_input} == {"cluster", "align-r1", "align-r2", "collapse"}


def test_the_multiplex_recipe_masks_both_primers_and_filters_the_length():
    steps = recipe("multiplex", quality="full", r1="/in1/a_1.fastq.gz", r2="/in2/a_2.fastq.gz", nproc=8)
    assert _names(steps) == ["quality-r1", "quality-r2", "v-primer", "c-primer", "both-primers", "assemble",
                             "length", "collapse"]
    by_name = {step.name: step.argv for step in steps}
    assert by_name["v-primer"][:2] == ("MaskPrimers.py", "score")
    assert _flags(by_name["v-primer"], "--start", "--mode", "--pf") == ["0", "mask", "VPRIMER"]
    assert _flags(by_name["c-primer"], "--start", "--mode", "--pf") == ["0", "mask", "CPRIMER"]
    assert _flags(by_name["length"], "-n") == ["250"] and "--inner" in by_name["length"]
    assert {step.name for step in steps if step.merges} == {"collapse"}
    assert {step.name for step in steps if step.sort_input} == {"collapse"}
    assert "--scanrev" not in by_name["assemble"]
    paper = {step.name: step.argv for step in recipe("multiplex", quality="full", r1="/in1/a", r2="/in2/b",
                                                     nproc=8, paper_length_filter=True)}
    assert _flags(paper["length"], "-n") == ["400"] and "--inner" not in paper["length"]


def test_the_default_quality_step_is_the_mean_quality_filter_and_the_option_trims():
    """Full-quality reads get Immcantation's mean-quality filter by default: whole reads below Q20 on average are
    dropped and none is shortened, which keeps the mates' overlap. ``paper_quality_trim`` switches to the paper's
    Q20 trimming and 125-nt minimum."""
    for protocol in ("race", "multiplex"):
        default = recipe(protocol, quality="full", r1="/in1/a", r2="/in2/b", nproc=2)
        assert _names(default)[:2] == ["quality-r1", "quality-r2"]
        for step, mate in zip(default[:2], ("/in1/a", "/in2/b")):
            assert step.argv[:2] == ("FilterSeq.py", "quality") and _flags(step.argv, "-s", "-q") == [mate, "20"]
        assert not any(part in ("trimqual", "length") for step in default[:2] for part in step.argv)
        trimmed = recipe(protocol, quality="full", r1="/in1/a", r2="/in2/b", nproc=2, paper_quality_trim=True)
        assert _names(trimmed)[:4] == ["trimqual-r1", "trimqual-r2", "length-r1", "length-r2"]
        assert _names(trimmed)[4:] == _names(default)[2:]
        by_name = {step.name: step.argv for step in trimmed}
        assert by_name["trimqual-r1"][:2] == ("FilterSeq.py", "trimqual")
        assert _flags(by_name["trimqual-r1"], "-s", "-q") == ["/in1/a", "20"]
        assert _flags(by_name["length-r2"], "-n") == ["125"]
        # the primer steps read the trimmed mates
        assert any("R1-quality_length-pass.fastq" in step.argv for step in trimmed[4:])
        assert any("R2-quality_length-pass.fastq" in step.argv for step in trimmed[4:])


def test_lite_reads_skip_the_quality_steps():
    for protocol in ("race", "multiplex"):
        full = recipe(protocol, quality="full", r1="/in1/a", r2="/in2/b", nproc=2)
        for paper_quality_trim in (False, True):
            lite = recipe(protocol, quality="lite", r1="/in1/a", r2="/in2/b", nproc=2,
                          paper_quality_trim=paper_quality_trim)
            assert _names(lite) == _names(full)[2:]
            # the raw mates go straight to the primer steps
            assert any("/in1/a" in step.argv for step in lite) and any("/in2/b" in step.argv for step in lite)
            assert not any("-quality_" in part for step in lite for part in step.argv)
        assert any("-quality_quality-pass" in part for step in full[2:] for part in step.argv)
    with pytest.raises(ValueError, match="quality"):
        recipe("race", quality="medium", r1="/in1/a", r2="/in2/b", nproc=2)
    with pytest.raises(ValueError, match="protocol"):
        recipe("unknown", quality="full", r1="/in1/a", r2="/in2/b", nproc=2)


def _flags(argv, *flags: str) -> list[str]:
    return [argv[argv.index(flag) + 1] for flag in flags]


# --- preprocess_run without Docker ------------------------------------------------------------------------------------

def _no_docker(*args, **kwargs):
    raise preprocess.ContainerError("docker is not available (test)")


def test_a_mixed_run_is_refused_with_its_evidence_recorded(reference, rng, tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "check_image", _no_docker)
    race1, race2 = _race_pairs(_molecules(reference, 50, rng), 2, rng)
    multi1, multi2 = _multiplex_pairs(_molecules(reference, 50, rng), 2)
    r1, r2 = _write_pairs(tmp_path, race1 + multi1, race2 + multi2)
    layout = RunLayout(tmp_path / "run")
    with pytest.raises(PreprocessError, match="neither") as refused:
        preprocess_run("W-117", r1, r2, layout, quality="full")
    assert refused.value.evidence["race_linker"] == 0.5
    record = json.loads(layout.real_preprocess_json("W-117").read_text())
    assert record["status"] == "refused" and record["protocol"] == "unknown"
    assert record["classifier"]["evidence"]["vh_primer"] == 0.5
    assert not layout.real_reads("W-117").exists() and not layout.real_preprocess_work("W-117").exists()


def test_lite_reads_cannot_pass_as_full_quality(reference, rng, tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "check_image", _no_docker)
    r1, r2 = _write_pairs(tmp_path, *_multiplex_pairs(_molecules(reference, 10, rng), 1), quality="?")
    with pytest.raises(PreprocessError, match="SRA-Lite"):
        preprocess_run("W-97", r1, r2, RunLayout(tmp_path / "run"), quality="full")
    with pytest.raises(ValueError, match="quality"):
        preprocess_run("W-97", r1, r2, RunLayout(tmp_path / "run"), quality="best")


def test_a_current_record_is_reused_without_docker(reference, rng, tmp_path, monkeypatch):
    """A record is current when its fingerprint (inputs, code, image, parameters) is the current one and its
    outputs are unchanged; otherwise the run is redone, which needs Docker."""
    monkeypatch.setattr(preprocess, "check_image", _no_docker)
    r1, r2 = _write_pairs(tmp_path, *_multiplex_pairs(_molecules(reference, 10, rng), 1))
    layout = RunLayout(tmp_path / "run")
    fingerprint = preprocess.run_fingerprint(r1, r2, quality="full", cap=50, paper_length_filter=False)
    preprocess._write_outputs(layout, "W-97", [{"sequence_id": "SYN.1", "sequence": "ACGT", "isotype": "IGHM",
                                                "protocol": "multiplex", "consensus_count": 1, "dupcount": 1,
                                                "primer_masked": []}])
    outputs = preprocess._output_digests(layout, "W-97")
    layout.real_preprocess_json("W-97").write_text(json.dumps(
        {"status": "done", "fingerprint": fingerprint["fingerprint"], "outputs": outputs}))
    record = preprocess_run("W-97", r1, r2, layout, quality="full", cap=50)
    assert record["reused"] is True and record["outputs"] == outputs
    # another cap, or an output changed since, is not current
    with pytest.raises(preprocess.ContainerError, match="not available"):
        preprocess_run("W-97", r1, r2, layout, quality="full", cap=51)
    layout.real_reads_meta("W-97").write_text("changed\n")
    with pytest.raises(preprocess.ContainerError, match="not available"):
        preprocess_run("W-97", r1, r2, layout, quality="full", cap=50)
    # a run rejected for its assembly loss is never reused as done, whatever its fingerprint and files
    preprocess._write_outputs(layout, "W-97", [{"sequence_id": "SYN.1", "sequence": "ACGT", "isotype": "IGHM",
                                                "protocol": "multiplex", "consensus_count": 1, "dupcount": 1,
                                                "primer_masked": []}])
    layout.real_preprocess_json("W-97").write_text(json.dumps(
        {"status": "rejected", "accepted": False, "fingerprint": fingerprint["fingerprint"],
         "outputs": preprocess._output_digests(layout, "W-97")}))
    with pytest.raises(preprocess.ContainerError, match="not available"):
        preprocess_run("W-97", r1, r2, layout, quality="full", cap=50)


def test_the_fingerprint_follows_inputs_code_image_and_parameters(tmp_path):
    r1, r2 = tmp_path / "a", tmp_path / "b"
    r1.write_bytes(b"one")
    r2.write_bytes(b"two")
    base = preprocess.run_fingerprint(r1, r2, quality="full", cap=10, paper_length_filter=False)
    assert base["inputs"]["r1"]["md5"] == "f97c5d29941bfb1b2fdab0874906ab82"
    assert base["image"] == IMMCANTATION_IMAGE and base["code_sha256"] == preprocess._CODE_SHA256
    assert base == preprocess.run_fingerprint(r1, r2, quality="full", cap=10, paper_length_filter=False)
    assert base["parameters"]["paper_quality_trim"] is False and base["parameters"]["nproc"] == preprocess.NPROC
    assert base["parameters"]["accept_assembly_loss"] is False
    for changed in ({"quality": "lite"}, {"cap": 11}, {"paper_length_filter": True}, {"paper_quality_trim": True},
                    {"nproc": 2}, {"accept_assembly_loss": True}):
        kwargs = {"quality": "full", "cap": 10, "paper_length_filter": False, **changed}
        assert preprocess.run_fingerprint(r1, r2, **kwargs)["fingerprint"] != base["fingerprint"]
    assert preprocess.run_fingerprint(r1, r2, quality="full", cap=10, paper_length_filter=False,
                                      paper_quality_trim=True)["parameters"]["paper_quality_trim"] is True
    r2.write_bytes(b"three")
    assert preprocess.run_fingerprint(r1, r2, quality="full", cap=10,
                                      paper_length_filter=False)["fingerprint"] != base["fingerprint"]


def test_the_container_runs_offline_on_read_only_inputs(tmp_path, monkeypatch):
    work, r1, r2 = tmp_path / "work", tmp_path / "in1" / "a_1.fastq.gz", tmp_path / "in2" / "a_2.fastq.gz"
    monkeypatch.setattr(preprocess.os, "cpu_count", lambda: 16)
    argv = preprocess.step_command(("FilterSeq.py", "length"), donor="SC-23", step="length", work=work, r1=r1,
                                   r2=r2, nproc=8)
    # never more CPUs than the host has (Docker refuses --cpus above it)
    for host, expected in ((4, "4"), (None, "8")):
        monkeypatch.setattr(preprocess.os, "cpu_count", lambda host=host: host)
        limited = preprocess.step_command(("true",), donor="SC-23", step="x", work=work, r1=r1, r2=r2, nproc=8)
        assert limited[limited.index("--cpus") + 1] == expected
    assert argv[argv.index("--network") + 1] == "none" and argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"
    assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert argv[argv.index("--cpus") + 1] == "8"
    assert argv[argv.index("--name") + 1] == f"alignair-preprocess-SC-23-length-{os.getpid()}"
    mounts = [argv[i + 1] for i, part in enumerate(argv) if part == "--mount"]
    assert mounts == [f"type=bind,source={r1.parent},target=/in1,readonly",
                      f"type=bind,source={r2.parent},target=/in2,readonly",
                      f"type=bind,source={work},target=/work"]
    assert argv[argv.index("-w") + 1] == "/work"
    assert argv[argv.index(IMMCANTATION_IMAGE) + 1:] == ["FilterSeq.py", "length"]


# --- gated: the pinned container -------------------------------------------------------------------------------------

@needs_docker
def test_both_recipes_end_to_end_in_the_container(reference, tmp_path):
    """The two recipes on synthetic pairs: every molecule's V(D)J comes back exactly, with its isotype; UMI reads
    collapse to consensus counts of at least 2; the multiplex primers are masked in the recorded spans."""
    started = time.monotonic()
    rng = random.Random(7)
    layout = RunLayout(tmp_path / "run")

    # 5'RACE: 24 molecules of 3 reads, 4 of a single read (dropped by the consensus count), and two molecules of
    # different V families sharing one UMI (ClusterSets keeps them apart)
    molecules = _molecules(reference, 24, rng)
    singles = _molecules(reference, 4, rng)
    collision = _molecules(reference, 2, rng)
    collision[1]["umi"] = collision[0]["umi"]
    r1, r2 = _race_pairs(molecules + collision, 3, rng)
    s1, s2 = _race_pairs(singles, 1, rng)
    race_dir = tmp_path / "race"
    race_dir.mkdir()
    record = preprocess_run("SC-T", *_write_pairs(race_dir, r1 + s1, r2 + s2), layout, quality="full", nproc=2)
    assert record["protocol"] == "race" and record["status"] == "done"
    assert record["quality_steps"] == preprocess.QUALITY_FILTER and record["parameters"]["paper_quality_trim"] is False
    assert [step["name"] for step in record["steps"]][:2] == ["quality-r1", "quality-r2"]
    rows = _rows(layout, "SC-T")
    expected = {UTR + molecule["vdj"]: molecule for molecule in molecules + collision}
    assert {row["sequence"] for row in rows} == set(expected)
    for row in rows:
        assert row["isotype"] == expected[row["sequence"]]["isotype"] and row["protocol"] == "race"
        assert int(row["consensus_count"]) == 3 and int(row["dupcount"]) == 1 and row["primer_masked"] == "[]"
    counts = record["counts"]
    assert counts["raw_pairs"] == 3 * 26 + 4 and counts["unique"] == 26 and counts["capped"] == 26
    assert counts["length"] == counts["assemble"] == 30  # every insert passes the 250-nt floor
    assert record["pass_rates"]["assembly"] == 1.0 and record["accepted"] is True
    # the 30 consensus pairs (the single-read molecules' too, dropped later) all assemble; their inserts are the
    # 10-nt UTR and the V(D)J, 386 or 391 nt
    assert record["assembly_acceptance"] == {"pairs": 30, "failed": 0, "failure_rate": 0.0, "max_failure_rate": 0.35,
                                             "within_line": True, "override": False, "accepted": True}
    assert record["assembly"]["fail"]["n"] == 0 and record["assembly"]["pass"]["n"] == 30
    assert {record["assembly"]["pass"]["insert_length"][key] for key in ("min", "max")} == {386, 391}
    assert record["isotypes"] == {"IGHG": 13, "IGHM": 13}

    # multiplex: 20 molecules (VH4, IgM) and one with primers of other lengths (VH3, 22 nt; IgG, 21 nt), so a
    # swapped head and tail span would show, each in 2 copies; one too short after primer removal; one pair without
    # a C primer
    molecules = _molecules(reference, 20, rng, isotypes=("IGHM",))
    r1, r2 = _multiplex_pairs(molecules, 2)
    igg = _molecules(reference, 1, rng, isotypes=("IGHG",))[0]
    for _ in range(2):
        r1.append(_multiplex_amplicon(igg["vdj"], IGG_PRIMER, VH3_PRIMER)[:READ_LENGTH])
        r2.append(revcomp(_multiplex_amplicon(igg["vdj"], IGG_PRIMER, VH3_PRIMER))[:READ_LENGTH])
    short = reference.seq(V_ALLELES[0])[150:] + reference.seq(J_ALLELE)
    r1.append(_multiplex_amplicon(short)[:READ_LENGTH])
    r2.append(revcomp(_multiplex_amplicon(short))[:READ_LENGTH])
    r1.append(_multiplex_amplicon(molecules[0]["vdj"])[:READ_LENGTH])
    r2.append(_random_dna(rng, READ_LENGTH))
    multiplex_dir = tmp_path / "multiplex"
    multiplex_dir.mkdir()
    record = preprocess_run("W-T", *_write_pairs(multiplex_dir, r1, r2), layout, quality="full", nproc=2)
    assert record["protocol"] == "multiplex" and record["status"] == "done" and record["accepted"] is True
    rows = _rows(layout, "W-T")
    expected = {"N" * 20 + molecule["vdj"] + "N" * 20: ("IGHM", 20, 20) for molecule in molecules}
    expected["N" * 22 + igg["vdj"] + "N" * 21] = ("IGHG", 22, 21)
    assert sorted(row["sequence"] for row in rows) == sorted(expected)
    for row in rows:
        isotype, v_mask, c_mask = expected[row["sequence"]]
        length = len(row["sequence"])
        assert json.loads(row["primer_masked"]) == [[0, v_mask], [length - c_mask, length]]
        assert row["isotype"] == isotype and int(row["dupcount"]) == 2 and int(row["consensus_count"]) == 2
    counts = record["counts"]
    assert counts["raw_pairs"] == 44 and counts["both-primers"] == 43 and counts["assemble"] == 43
    assert counts["length"] == 42 and counts["unique"] == 21
    assert record["assembly_acceptance"]["pairs"] == 43 and record["assembly_acceptance"]["accepted"] is True
    assert record["image"]["repo_digests"]
    assert record["classifier"]["evidence"]["vh_primers"] == {"VH3": 2, "VH4": 42}

    # a current record is reused; the logs are kept and the work directory is gone
    assert preprocess_run("W-T", multiplex_dir / "SYN_1.fastq.gz", multiplex_dir / "SYN_2.fastq.gz", layout,
                          quality="full", nproc=2)["reused"] is True
    assert (layout.real_preprocess_logs("W-T") / "assemble.log").is_file()
    assert not layout.real_preprocess_work("W-T").exists()

    # over the acceptance line: 6 of 10 inserts are longer than the mates span, so the run is rejected with its
    # evidence, and nothing is written but the record; accept_assembly_loss lets it through deliberately
    fits = _molecules(reference, 4, rng, isotypes=("IGHM",))
    too_long = [{**molecule, "vdj": molecule["vdj"] + _random_dna(rng, 250)}
                for molecule in _molecules(reference, 6, rng, isotypes=("IGHM",))]
    loss_dir = tmp_path / "loss"
    loss_dir.mkdir()
    pair = _write_pairs(loss_dir, *_multiplex_pairs(fits + too_long, 1))
    with pytest.raises(PreprocessError, match="assembl") as rejected:
        preprocess_run("W-L", *pair, layout, quality="full", nproc=2)
    assert rejected.value.evidence["failure_rate"] == 0.6
    record = json.loads(layout.real_preprocess_json("W-L").read_text())
    assert record["status"] == "rejected" and record["accepted"] is False
    assert record["assembly_acceptance"]["failed"] == 6 and record["assembly_acceptance"]["within_line"] is False
    assert record["assembly"]["fail"]["n"] == 6 and sum(record["assembly"]["fail"]["kinds"].values()) == 6
    assert record["assembly"]["fail"]["mates_length"]["median"] == 500
    assert not layout.real_reads("W-L").exists() and not layout.real_preprocess_work("W-L").exists()
    record = preprocess_run("W-L", *pair, layout, quality="full", nproc=2, accept_assembly_loss=True)
    assert record["status"] == "done" and record["accepted"] is True and record["counts"]["unique"] == 4
    assert record["assembly_acceptance"]["override"] is True and record["assembly_acceptance"]["within_line"] is False
    assert record["parameters"]["accept_assembly_loss"] is True
    assert time.monotonic() - started < 60


def _rows(layout: RunLayout, donor: str) -> list[dict]:
    """The TSV rows joined with their FASTA sequences, checking the two list the same ids in the same order."""
    lines = layout.real_reads_meta(donor).read_text().splitlines()
    header = lines[0].split("\t")
    assert header == ["sequence_id", "isotype", "protocol", "consensus_count", "dupcount", "primer_masked"]
    rows = [dict(zip(header, line.split("\t"))) for line in lines[1:]]
    fasta = read_fasta(layout.real_reads(donor))
    assert [row["sequence_id"] for row in rows] == [name for name, _ in fasta]
    for row, (_, sequence) in zip(rows, fasta):
        row["sequence"] = sequence
    return rows


# --- gated: the pilot runs ----------------------------------------------------------------------------------------

PILOT_PROTOCOLS = {"SC-23": "race", "SC-4": "race", "SC-11": "race", "SC-6": "race",
                   "W-97": "multiplex", "W-89": "multiplex", "W-133": "multiplex", "W-131": "multiplex"}


@needs_data
def test_the_pilot_runs_classify():
    from alignair_assay.real.fetch import PILOT_DONORS, fetch_reads

    found = {}
    for donor in PILOT_PROTOCOLS:  # the pilot's eight: the test donors' reads are read only by a claim run
        r1, r2, record = fetch_reads(PILOT_DONORS[donor], DATA)
        assert record.quality == "full"
        protocol, evidence = classify_protocol(*preprocess.read_head(r1, r2))
        found[donor] = protocol
        assert evidence["pairs"] == 2000
        if protocol == "race":
            assert evidence["race_linker"] >= 0.9 and evidence["vh_primer"] < 0.1
        else:
            assert evidence["vh_primer"] >= 0.9 and evidence["c_primer"] >= 0.8 and evidence["race_linker"] < 0.1
    assert found == PILOT_PROTOCOLS
