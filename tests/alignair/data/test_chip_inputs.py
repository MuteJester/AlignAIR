"""Chip inputs: the nucleotide adapter gives every base its germline column in
its gene's frame and the set of regions the read cannot tell apart at it, and every batch a bank mask per gene and
the teacher inputs (true regions and columns)."""
import numpy as np
import pytest
import torch

import GenAIRR.data as gd

from alignair.core import AlignAIRConfig
from alignair.data import NucleotideBatchAdapter
from alignair.data.alignment import RecordAlignment
from alignair.data.chip_inputs import chip_batch_inputs, chip_record_targets
from alignair.core.regions import REGION_INDEX, REGIONS as REGION_NAMES
from alignair.reference.frames import gene_frames
from alignair.reference.reference_set import GeneReference, ReferenceSet

pytestmark = pytest.mark.component

# read: V [0, 8) germline 0..8 of ACGTACGTAC (2 bases trimmed); N1 "AT" (A continues the V germline);
# D [10, 14) germline 1..5 of GGGCCC; N2 "CT" (T precedes the J germline); J [16, 21) germline 2..7 of TTTGGAA
SEQUENCE = "ACGTACGT" + "AT" + "GGCC" + "CT" + "TGGAA"
SPANS = {"v": (0, 8, 0, 8), "d": (10, 14, 1, 5), "j": (16, 21, 2, 7)}
REGIONS = ["V"] * 8 + ["N1"] * 2 + ["D"] * 4 + ["N2"] * 2 + ["J"] * 5


def _reference():
    v = GeneReference(["V*01"], ["ACGTACGTAC"], {"V*01": 0}, anchors={"V*01": 4},
                      gapped={"V*01": "ACGT..ACGTAC"})
    d = GeneReference(["D*01"], ["GGGCCC"], {"D*01": 0})
    j = GeneReference(["J*01"], ["TTTGGAA"], {"J*01": 0}, anchors={"J*01": 3})
    return ReferenceSet({"V": v, "D": d, "J": j}, has_d=True)


def _record(v_cigar="8M", v_germline_end=8, sequence=SEQUENCE):
    record = {"sequence": sequence, "sequence_id": "chip-read", "productive": None, "mutation_rate": 0.,
              "n_indels": 0, "n_quality_errors": 0, "v_call": "V*01", "d_call": "D*01", "j_call": "J*01"}
    for gene, (start, end, gs, ge) in SPANS.items():
        record.update({f"{gene}_sequence_start": start, f"{gene}_sequence_end": end, f"{gene}_germline_start": gs,
                       f"{gene}_germline_end": ge, f"{gene}_cigar": f"{end - start}M"})
    record.update({"v_cigar": v_cigar, "v_germline_end": v_germline_end})
    return record


def _region_labels():
    return np.array([REGION_INDEX[name] for name in REGIONS], dtype=np.int64)


def _bits(*names):
    return sum(1 << REGION_NAMES.index(name) for name in names)


def _targets(record=None):
    reference = _reference()
    return chip_record_targets(RecordAlignment.of(record or _record()), reference, gene_frames(reference),
                               _region_labels())


def test_coordinates_are_frame_columns_of_the_aligned_bases():
    targets = _targets()
    assert targets["coord_v"].values[:8].tolist() == [0, 1, 2, 3, 6, 7, 8, 9]     # gapped: 2 columns after base 3
    assert targets["coord_d"].values[10:14].tolist() == [1, 2, 3, 4]
    assert targets["coord_j"].values[16:21].tolist() == [47, 48, 49, 50, 51]    # the W/F codon at column 48
    assert (targets["coord_v"].values[8:] == -100).all() and targets["coord_v"].padding == -100


def test_coordinates_follow_the_cigar_through_an_insertion():
    targets = _targets(_record(v_cigar="4M1I3M", v_germline_end=7))
    assert targets["coord_v"].values[:8].tolist() == [0, 1, 2, 3, -100, 6, 7, 8]


def test_v_end_may_move_right_while_np_bases_match_the_germline():
    allowed = _targets()["region_allowed"].values
    assert allowed[8] == _bits("N1", "V")          # A == V germline base 8
    assert allowed[9] == _bits("N1")               # T != V germline base 9 (C)
    assert allowed[7] == _bits("V") and allowed[10] == _bits("D")


def test_j_start_may_move_left_while_np_bases_match_the_germline():
    allowed = _targets()["region_allowed"].values
    assert allowed[15] == _bits("N2", "J")         # T == J germline base 1


def _substituted(**bases):
    """SEQUENCE with the bases at the given positions (``p7="G"``) replaced."""
    read = list(SEQUENCE)
    for position, base in bases.items():
        read[int(position[1:])] = base
    return "".join(read)


def test_a_gene_may_shrink_over_its_mismatching_edge_bases():
    """A mutated edge base reads like an N base: the boundary may move inward over the run of edge bases that
    mismatch the germline (as the assay's boundary intervals allow), and no further than the first match."""
    allowed = _targets(_record(sequence=_substituted(p6="T", p7="A", p10="A", p13="A", p16="C")))["region_allowed"].values
    assert allowed[6] == allowed[7] == _bits("V", "N1")       # V's last two bases mismatch (germline G, T)
    assert allowed[5] == _bits("V")                            # base 5 matches: the run stops
    assert allowed[10] == _bits("D", "N1") and allowed[13] == _bits("D", "N2")
    assert allowed[11] == allowed[12] == _bits("D")
    assert allowed[16] == _bits("J", "N2") and allowed[17] == _bits("J")


def test_an_n_base_at_the_edge_counts_as_a_mismatch():
    allowed = _targets(_record(sequence=_substituted(p7="N")))["region_allowed"].values
    assert allowed[7] == _bits("V", "N1")


def test_a_gene_keeps_at_least_one_aligned_base():
    """D's 4 bases all mismatch: its end shrinks over 3 of them and its start not at all."""
    allowed = _targets(_record(sequence=_substituted(p10="A", p11="A", p12="T", p13="T")))["region_allowed"].values
    assert allowed[10] == _bits("D")
    assert allowed[11] == allowed[12] == allowed[13] == _bits("D", "N2")
    assert allowed[14] == _bits("N2", "D")         # C != J germline base 0 (T), but continues the D germline


def _outer_targets(record, regions):
    reference = _reference()
    labels = np.array([REGION_INDEX[name] for name in regions], dtype=np.int64)
    return chip_record_targets(RecordAlignment.of(record), reference, gene_frames(reference), labels)["region_allowed"].values


def test_v_start_and_j_end_shrink_over_their_mismatching_edge_bases_into_the_flanks():
    """The read's outer edges move like the inner ones (the assay's intervals): V's first base mismatching its
    germline may be read before V (pre), J's last base mismatching may be read after J (post)."""
    allowed = _targets(_record(sequence=_substituted(p0="C", p20="C")))["region_allowed"].values
    assert allowed[0] == _bits("V", "pre") and allowed[1] == _bits("V")
    assert allowed[20] == _bits("J", "post") and allowed[19] == _bits("J")


def test_v_start_and_j_end_extend_over_flank_bases_that_continue_their_germline():
    """A flank base equal to the germline base before V's first (after J's last) may be read as V (J)."""
    record = _record()
    record.update(v_sequence_start=1, v_germline_start=1, v_cigar="7M", j_sequence_end=20, j_germline_end=6,
                  j_cigar="4M")
    regions = ["pre"] + REGIONS[1:20] + ["post"]
    allowed = _outer_targets(record, regions)
    assert allowed[0] == _bits("pre", "V")         # A == V germline base 0
    assert allowed[20] == _bits("post", "J")       # A == J germline base 6
    mismatched = {**record, "sequence": "G" + SEQUENCE[1:20] + "C"}
    allowed = _outer_targets(mismatched, regions)
    assert allowed[0] == _bits("pre") and allowed[20] == _bits("post")


def test_region_is_the_read_region_index():
    region = _targets()["region"].values
    assert region.tolist() == [REGION_NAMES.index(name) for name in REGIONS]


def _allele_targets(reference, calls, gene="v"):
    n = len(reference.gene(gene.upper()))
    alleles = torch.zeros(len(calls), n)
    for row, call in enumerate(calls):
        alleles[row, call] = 1.
    return {f"{gene}_allele": alleles, f"{gene}_supervise": torch.ones(len(calls), dtype=torch.bool)}


def test_bank_masks_hold_the_batch_alleles_and_vary_in_size():
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    n = len(reference.gene("V"))
    generator = torch.Generator().manual_seed(0)
    sizes = []
    for draw in range(200):
        calls = [draw % n, (7 * draw + 3) % n]
        targets = {key: value for gene in ("v", "d", "j")
                   for key, value in _allele_targets(reference, [c % len(reference.gene(gene.upper()))
                                                                 for c in calls], gene).items()}
        inputs = chip_batch_inputs(targets, reference, generator=generator)
        mask = inputs["bank_mask_v"]
        assert mask.shape == (2, n) and mask.dtype == torch.bool and torch.equal(mask[0], mask[1])
        assert mask[0, calls].all()
        sizes.append(int(mask[0].sum()))
    assert min(sizes) <= 3 and max(sizes) >= n // 2


def test_unsupervised_rows_add_no_true_allele():
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    targets = {key: value for gene in ("v", "d", "j") for key, value in _allele_targets(reference, [0, 1], gene).items()}
    targets["v_supervise"] = torch.tensor([False, False])
    sizes = {int(chip_batch_inputs(targets, reference, generator=torch.Generator().manual_seed(seed))
                 ["bank_mask_v"][0].sum()) for seed in range(50)}
    assert min(sizes) == 1


def _config(**options):
    return AlignAIRConfig(has_d=True, max_seq_length=32, **options)


def test_the_adapter_adds_the_chip_inputs():
    batch = NucleotideBatchAdapter(_reference(), _config(), augment_orientation=False)([_record()])
    assert batch.targets["coord_v"].shape == (1, 32) and batch.targets["region_allowed"].shape == (1, 32)
    assert torch.equal(batch.inputs["teacher_coord_v"], batch.targets["coord_v"])
    assert torch.equal(batch.inputs["teacher_region"], batch.targets["region"])
    assert batch.inputs["bank_mask_v"].shape == (1, 1) and batch.inputs["bank_mask_v"].all()
