"""What each output column of the AIRR rearrangement table means (``io.airr.COLUMNS``), for the documentation's column
dictionary: its kind, its unit and one plain sentence. Standard AIRR fields follow the AIRR Rearrangement schema;
AlignAIR's own fields are described from what the writer puts in them."""
from __future__ import annotations

from typing import NamedTuple

from ..reference.genes import GENES
from .airr import COLUMNS


class ColumnDoc(NamedTuple):
    """One column: its ``kind`` (string, T/F, integer, number, nucleotides, amino acids, CIGAR), its ``unit`` ("" when
    none) and what it means."""
    kind: str
    unit: str
    meaning: str


_GENE = {"v": "V", "d": "D", "j": "J"}
_REGION = {"fwr1": "framework region 1", "cdr1": "CDR1", "fwr2": "framework region 2", "cdr2": "CDR2",
           "fwr3": "framework region 3", "cdr3": "CDR3", "fwr4": "framework region 4"}
_POSITION = "1-based position"

_FIXED = {
    "sequence_id": ColumnDoc("string", "", "The read's id: the input record's id (a repeated id gets a _dupN suffix), "
                                           "or the --id-column value of a table."),
    "sequence": ColumnDoc("nucleotides", "", "The read as aligned (cut to its first 575 nt when longer); for a "
                                             "reverse-complement read, the original read (see rev_comp)."),
    "rev_comp": ColumnDoc("T/F", "", "T when the read was reverse-complemented to align: coordinates and alignments "
                                     "then refer to the reverse complement of sequence."),
    "locus": ColumnDoc("string", "", "The locus of the V call (IGH, IGK, IGL, TRA, TRB, TRD or TRG); a model of one "
                                     "locus writes it for every read."),
    "c_call": ColumnDoc("string", "", "The constant-region gene. AlignAIR does not call it; a --metadata table may "
                                      "fill it (10x c_gene)."),
    "productive": ColumnDoc("T/F", "", "T when the rearrangement is productive: V and J in frame, no stop codon, "
                                       "conserved junction anchors."),
    "productive_prediction": ColumnDoc("T/F", "", "The same call as productive, kept for tools that read this "
                                                  "column."),
    "vj_in_frame": ColumnDoc("T/F", "", "T when V and J are in the same reading frame."),
    "stop_codon": ColumnDoc("T/F", "", "T when the aligned sequence holds a stop codon in the V reading frame."),
    "sequence_alignment": ColumnDoc("nucleotides", "", "The aligned read, IMGT-gapped (dots in the V gaps)."),
    "germline_alignment": ColumnDoc("nucleotides", "", "The germline the read aligns to, IMGT-gapped, position "
                                                       "for position with sequence_alignment (N in the junction's "
                                                       "untemplated bases)."),
    "sequence_alignment_aa": ColumnDoc("amino acids", "", "sequence_alignment translated."),
    "germline_alignment_aa": ColumnDoc("amino acids", "", "germline_alignment translated."),
    "junction": ColumnDoc("nucleotides", "", "The junction: CDR3 with its conserved anchors (V's Cys to J's Trp or "
                                             "Phe)."),
    "junction_aa": ColumnDoc("amino acids", "", "The junction translated (X for an incomplete codon)."),
    "junction_length": ColumnDoc("integer", "nt", "The junction's length in nucleotides."),
    "junction_aa_length": ColumnDoc("integer", "aa", "The junction's length in amino acids."),
    "np1": ColumnDoc("nucleotides", "", "The untemplated bases between V and D (between V and J without a D)."),
    "np1_length": ColumnDoc("integer", "nt", "The length of np1."),
    "np2": ColumnDoc("nucleotides", "", "The untemplated bases between D and J."),
    "np2_length": ColumnDoc("integer", "nt", "The length of np2."),
    "mutation_rate": ColumnDoc("number", "fraction 0-1", "AlignAIR's estimate of the read's somatic mutation rate "
                                                         "over its V, D and J."),
    "orientation": ColumnDoc("string", "", "How the read was turned to align: forward, reverse_complement, "
                                           "complement or reverse."),
    "input_sequence": ColumnDoc("nucleotides", "", "The read as given, before orientation."),
    "segmentation_low_quality": ColumnDoc("T/F", "", "T when the read's V or J segment came out empty or reversed: "
                                                     "treat its calls with caution."),
    "length_cropped": ColumnDoc("T/F", "", "T when the read was longer than 575 nt and only its first 575 nt were "
                                           "aligned."),
    "airr_assembly_status": ColumnDoc("string", "", "complete (every field derived), partial (valid calls, some "
                                                    "fields left blank: see airr_assembly_reason) or failed (see "
                                                    "airr_assembly_error)."),
    "airr_assembly_reason": ColumnDoc("string", "", "Why a partial row is partial (e.g. no_v: the read has no V "
                                                    "call)."),
    "airr_assembly_error": ColumnDoc("string", "", "The error that stopped a failed row's assembly."),
}


def _gene_columns(g: str) -> dict[str, ColumnDoc]:
    gene = _GENE[g]
    return {
        f"{g}_call": ColumnDoc("string", "", f"The {gene} allele call; alleles that match the read equally well "
                                             f"follow it, comma-separated (IgBLAST's convention)."),
        f"{g}_identity": ColumnDoc("number", "fraction 0-1", f"The fraction of the aligned {gene} bases that match "
                                                             f"the germline."),
        f"{g}_cigar": ColumnDoc("CIGAR", "", f"The {gene} alignment of the read against its germline, as a CIGAR "
                                             f"string."),
        f"{g}_sequence_start": ColumnDoc("integer", _POSITION, f"Where {gene} starts in sequence."),
        f"{g}_sequence_end": ColumnDoc("integer", _POSITION, f"Where {gene} ends in sequence."),
        f"{g}_germline_start": ColumnDoc("integer", _POSITION, f"Where the alignment starts in the {gene} germline."),
        f"{g}_germline_end": ColumnDoc("integer", _POSITION, f"Where the alignment ends in the {gene} germline."),
        f"{g}_sequence_alignment": ColumnDoc("nucleotides", "", f"The read's {gene} part of sequence_alignment."),
        f"{g}_germline_alignment": ColumnDoc("nucleotides", "", f"The {gene} germline's part of germline_alignment."),
        f"{g}_sequence_alignment_aa": ColumnDoc("amino acids", "", f"{g}_sequence_alignment translated."),
        f"{g}_germline_alignment_aa": ColumnDoc("amino acids", "", f"{g}_germline_alignment translated."),
        f"{g}_alignment_start": ColumnDoc("integer", "position", f"Where {gene} starts in sequence_alignment."),
        f"{g}_alignment_end": ColumnDoc("integer", "position", f"Where {gene} ends in sequence_alignment."),
        f"{g}_call_set": ColumnDoc("string", "", f"The candidate set: the {gene} alleles the model scored as "
                                                 f"possible before the germline refinement, comma-separated (the "
                                                 f"{g}_call can fall outside it)."),
    }


def _region_columns(region: str) -> dict[str, ColumnDoc]:
    name = _REGION[region]
    return {
        region: ColumnDoc("nucleotides", "", f"The read's {name}."),
        f"{region}_aa": ColumnDoc("amino acids", "", f"The {name} translated."),
        f"{region}_start": ColumnDoc("integer", _POSITION, f"Where the {name} starts in sequence."),
        f"{region}_end": ColumnDoc("integer", _POSITION, f"Where the {name} ends in sequence."),
    }


COLUMN_DOCS: dict[str, ColumnDoc] = {
    **_FIXED,
    **{key: doc for g in GENES for key, doc in _gene_columns(g).items()},
    **{key: doc for region in _REGION for key, doc in _region_columns(region).items()},
}
COLUMN_DOCS = {column: COLUMN_DOCS[column] for column in COLUMNS}
