"""Observable-truth records: what a perfect aligner can recover from one read."""
from __future__ import annotations

from dataclasses import dataclass, field

from alignair_assay.truth.alleles import AlleleTruth
from alignair_assay.truth.boundaries import BoundaryTruth
from alignair_assay.truth.segment import ObservedSegment


@dataclass(frozen=True)
class SegmentTruth:
    """Truth for one observed segment.

    For an inverted D (``inverted``), ``observed`` coordinates, CIGAR and germline, the boundaries and the
    allele set are all in the frame of the reverse-complemented allele, which is how the read shows it.
    """

    observed: ObservedSegment
    boundaries: BoundaryTruth
    alleles: AlleleTruth
    observed_length: int  # germline bases of the segment in the read
    inverted: bool


@dataclass(frozen=True)
class JunctionTruth:
    """The anchor-defined junction of a read: 0-based, half-open read coordinates in the canonical frame, its
    uppercase nucleotides and their translation when the length is a whole number of codons."""

    start: int  # read position of the V Cys codon
    end: int  # read position just after the J Trp/Phe codon
    nt: str
    aa: str | None  # None when length % 3 != 0
    length: int


@dataclass(frozen=True)
class ObservableTruth:
    """Truth for one read, projected onto what the read shows; coordinates are 0-based, half-open and in the
    canonical (forward) frame. Junction, productivity and clone key are None when the read does not show both
    anchors; an ``excluded`` case is never scored."""

    case_id: str
    stratum: str
    sequence: str  # canonical (forward) read, uppercase
    orientation: int  # 0 forward, 1 reverse complement (bench ids 2/3 are excluded)
    segments: dict[str, SegmentTruth]  # only observed segments
    # Genes the simulator called whose truth could not be built (reason in ``warnings``); never scored.
    unscorable: frozenset[str]
    junction: JunctionTruth | None
    vj_in_frame: bool | None
    stop_codon: bool | None
    productive: bool | None
    clone_key: tuple[frozenset[str], frozenset[str], int] | None
    v_mismatches: int | None
    # Net inserted minus deleted bases of the observed V / J alignment not a multiple of 3 (None: not observed).
    v_frameshift: bool | None
    j_frameshift: bool | None
    excluded: str | None
    warnings: tuple[str, ...]
    # Set by the suite engine: the suite id, the slice dimensions (``labels``) and the grouping keys
    # (``groups``: subject, clone, pair, ...); ``v_novel_snps`` are the read positions of a novel V allele's SNPs.
    suite: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)
    v_novel_snps: tuple[int, ...] | None = None

    @property
    def frame_unambiguous(self) -> bool:
        """True when the reading frame is known everywhere: an in-frame junction and no V/J frameshift.

        Translations and stop codons are scored only then; otherwise the frame depends on how a tool
        resolves the frameshift.
        """
        return bool(self.junction is not None and self.vj_in_frame
                    and not self.v_frameshift and not self.j_frameshift)
