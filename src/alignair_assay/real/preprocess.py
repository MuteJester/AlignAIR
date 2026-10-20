"""Annotation-free preprocessing of one real AIRR-seq run: its read pairs to the unique sequences every tool is given.

No V(D)J tool touches the reads here, and no germline reference is read: the steps are pRESTO's, run in the pinned
Immcantation container (``IMMCANTATION_IMAGE``), and the only sequences they are given are the library's primers.

**The protocol of each run is read from its reads** (``classify_protocol``, on the first ``CLASSIFIER_PAIRS``
pairs), since the paper's per-donor label is wrong for at least one run:

- ``"race"`` (Takara SMARTer 5'RACE + UMI) when R2[12:16] is the ``GTAC`` linker in at least 80% of the pairs:
  R2 is ``[12-nt UMI][GTAC][GGG][5'UTR, leader, V ...]`` and R1 ``[7 random nt][C-region primer ...]``;
- ``"multiplex"`` (the paper's SD7 VH primers) when at least 80% of R1 start with a VH primer and at least 80% of
  R2 with an IgM/IgG/IgA primer, each within ``PRIMER_MISMATCHES`` mismatches (IUPAC codes read as their bases);
- ``"unknown"`` otherwise, or when both rules hold. Such a run is refused with its evidence (the fractions), and its
  record says so; it is never processed under a guessed recipe.

**The recipes** (``recipe``; each step one pRESTO 0.7.9 command in its own container):

- Quality, full-quality reads only, on each mate before pairing and consensus:
  - by default the Immcantation and nf-core/airrflow standard, ``FilterSeq quality -q 20``: a read whose mean
    quality is below 20 is dropped whole, and no read is shortened (the image's ``presto-clontech-umi`` and
    ``presto-abseq``, and airrflow's ``filterseq_q`` of 20). Reference-free assembly needs the mates' overlap, which
    trimming destroys; ``BuildConsensus`` and ``AssemblePairs`` use the per-base qualities to resolve errors;
  - with ``paper_quality_trim``, the paper's trimming (Rodriguez et al. 2023, Methods: "All R1 and R2 reads were
    trimmed to Q = 20, and reads <125 bp were excluded"): ``FilterSeq trimqual -q 20`` then ``FilterSeq length -n
    125``, kept to reproduce it. On the pilot it costs the assembly a fifth to a half of its pairs, whose trimmed
    mates no longer overlap (the README has the numbers).

  SRA-Lite qualities (every base ``?`` or ``$``) carry no per-base information, so with ``quality="lite"`` neither
  runs, and the record says which ran (``quality_steps``). Reads said to be full quality are refused when their
  first records have only SRA-Lite symbols.
- **The assembly's acceptance line.** At most ``MAX_ASSEMBLY_FAILURE`` (35%) of the pairs entering the assembly
  may fail to assemble (``assembly_acceptance``). The pairs that fail are pairs whose mates do not overlap
  acceptably: inserts longer than the mates span, or overlaps with too many mismatches (the record counts them by
  AssemblePairs' reason, ``assembly_evidence``). They are lost reads, not wrong ones, and both tools see the same
  set; they are dropped and counted, never joined or assembled against a reference. A run over the line is
  rejected, as the classifier refuses a run: its record is written with ``status: "rejected"``, ``accepted``
  false and the evidence (the deciles and the failures by kind), nothing else is kept, and ``PreprocessError`` is
  raised. ``accept_assembly_loss`` lets such a run through deliberately; it is in the record and the fingerprint.
- 5'RACE:
  1. ``MaskPrimers extract`` on R2: the UMI (start 0, length 12) to the header as ``UMI``, cut;
  2. ``MaskPrimers extract`` on R2: the ``GTAC`` + ``GGG`` that follow (7 nt), cut;
  3. ``MaskPrimers align`` on R1 with Immcantation's ``Human_IG_CRegion_RC.fasta`` IGH lines (``CREGION_RC``),
     ``--maxlen 70 --maxerror 0.2 --skiprc``, cut: the isotype to the header as ``CPRIMER``;
  4. ``PairSeq``: the pairs whose mates both passed, each mate given the other's annotation;
  5. the paper's UMI consensus: ``ClusterSets set`` (vsearch, identity 0.9) splits each UMI's R1 reads into
     clusters, so two molecules that share a UMI are not merged; ``ParseHeaders copy --act cat`` joins the cluster
     to the UMI (``CLUSTER``), which ``PairSeq`` copies to R2; ``AlignSets muscle`` aligns each group's reads, on
     each mate; ``BuildConsensus`` (``--prcons 0.6 -n 1 -q 0 --maxerror 0.1 --maxgap 0.5``) builds each group's
     consensus, on each mate, with its isotype as ``PRCONS``;
  6. ``PairSeq`` of the consensus mates, then ``AssemblePairs align`` (R2 the head, R1 reverse-complemented):
     reference-free, so pairs that do not overlap acceptably are dropped and counted;
  7. ``ParseHeaders collapse``: the pair's consensus count is the smaller of its mates';
  8. ``CollapseSeq`` (exact, ``--inner``, within an isotype), summing the consensus counts;
  9. ``SplitSeq group``: the sequences with a summed consensus count of at least 2.
- Multiplex:
  1. ``MaskPrimers score`` (start 0, ``--maxerror 0.2``, mode mask): R1 with ``VH_PRIMERS`` (``VPRIMER``), R2 with
     ``C_PRIMERS_MULTIPLEX`` (``CPRIMER``);
  2. ``PairSeq``: the pairs with both primers found (pRESTO's ``AssemblePairs`` needs paired mates, so this comes
     before the assembly);
  3. ``AssemblePairs align`` (R1 the head, R2 reverse-complemented);
  4. ``FilterSeq length``: at least 250 nt after primer removal (``-n 250 --inner``: the masked primers are the
     ``N`` runs at the ends). Where the mates overlap a primer, the assembly can fill a masked base from the other
     mate, which ``--inner`` then counts, so the unique sequences are checked again here against the primers'
     lengths, and those shorter are dropped and counted (``checks``). The paper's ``FilterSeq length`` of 400 nt
     before primer removal removes about half of the reads, with a CDR3 and VH-family skew; it stays available as
     ``paper_length_filter``;
  5. ``CollapseSeq`` (exact, ``--inner``, within a VH and C primer pair).

**Differences from the image's ``presto-clontech-umi``** (``/usr/local/sbin/presto-clontech-umi``, 2021-09-11),
the Immcantation pipeline for this kit:

- quality: the same by default (``FilterSeq quality -q 20``); the paper's trimming is ``paper_quality_trim``;
- UMI: it extracts UMI and linker in one call (``extract --start 12 --len 7 --barcode``: the 12 nt before the 7-nt
  ``GTACGGG`` are the barcode); here two calls take the same 12 + 7 nt;
- UMI collisions: it has no ``ClusterSets``, and its ``AlignSets`` is off unless asked for (``-a``); here both run,
  as the paper describes;
- C primers: its file also holds IGKC, IGLC1 and IGLC3; only the IGH lines are given here, and the field is
  ``CPRIMER``, not ``C_CALL``; its read coordinates are Illumina's, these are SRA's;
- assembly: it uses ``AssemblePairs sequential``, which aligns unassembled pairs to IMGT's human V germlines with
  blastn: a reference-guided assembly, annotation before the tools under test, so it is not used; only
  ``AssemblePairs align`` runs, and the pairs it cannot assemble are dropped and counted;
- the consensus count of 2: as in the script, applied after the collapse, on the summed count (the paper drops
  consensus sequences of fewer than 2 reads before it);
- its ``ParseHeaders rename``, ``ParseLog``, report and archive steps are not run.

**Outputs** (``RunLayout``, per donor): the unique sequences as FASTA (``real_reads``), ordered by their content
(``content_key``: the SHA-256 of sequence, isotype and primer names, then the id) and capped at ``cap`` chosen
evenly spaced in that order (``core.cases.evenly_spaced``); a TSV with one row per sequence, in the same order
(``real_reads_meta``: ``TSV_COLUMNS``); each step's pRESTO output in ``real_preprocess_logs``; and the record
(``real_preprocess_json``, written last). The sequence id is pRESTO's: for a multiplex sequence the name of the
first of its reads the collapse saw, for 5'RACE the cluster and UMI of the first consensus. The isotype is the C
primer's gene (``IGHM``, ``IGHG``, ...).
``consensus_count`` is the reads behind the sequence: the summed UMI consensus count for 5'RACE, the duplicate
count for multiplex, where each read pair is one read. ``primer_masked`` is a JSON list of the ``[start, end)``
spans of primer bases in the read (``primer_spans``): the VH primer at the start of a multiplex read and the C
primer at its end, from the primers' lengths, so a primer base that the assembly took from the other mate is
still excluded; 5'RACE reads have none, their primers being cut. Those bases were never read from the template:
they carry no germline evidence.

**Reruns give the same outputs.** pRESTO's workers write their records in the order they finish, so with
``nproc`` above 1 a step's output order varies between runs. Three steps depend on their input's order: within a
UMI, ``ClusterSets`` (vsearch) numbers and builds its clusters, and ``AlignSets`` (muscle) aligns, in the order the
reads come; and ``CollapseSeq`` keeps, of identical sequences, the first it sees, whose id the output carries.
Their inputs are therefore sorted by read id first (``sort_fastq``), and the cap is keyed on content, not on
pRESTO's id. ``nproc`` (with ``cpus``, the container's limit: at most the host's CPUs) is in the record and the
fingerprint all the same.

**The record** holds the counts at every step and each step's pass rate, the quality and which quality steps ran,
the image found (``check_image``), the classifier's evidence, ``accepted`` and the assembly's acceptance and
evidence (``assembly_evidence``: overlap, error, mate lengths and the insert-length estimates of the pairs that
passed and failed, as deciles, and the failures by kind), the length distribution of the assembled reads, the
isotype split, the wall time and a ``fingerprint``: SHA-256 over the inputs' md5s, this module's SHA-256, the
pinned image and the parameters (``run_fingerprint``). The code digest covers this module alone, the assay's
convention for its steps' records: the ``core`` helpers it uses are not in it. A record is reused only while it is
current: it is ``done`` (never ``refused`` or ``rejected``), its fingerprint is the current one, and its FASTA and
TSV are the ones it recorded. Otherwise the run is redone from the reads: the
old record and outputs are removed first, the work directory (``real_preprocess_work``, the container's only
writable mount; the reads are mounted read-only) is removed once the outputs are written, and the record comes
last, so an interrupted run is never mistaken for a finished one. Docker or the pinned image missing stops the run
before anything is written; a refusal needs neither, being decided from the reads alone.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alignair_assay.core.cases import evenly_spaced
from alignair_assay.core.container import (
    ContainerError,
    Mount,
    check_image,
    container_name,
    hardened_run,
    run_container,
)
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import sha256_file

# immcantation/suite 4.8.0, released 2026-06-07: pRESTO 0.7.9, vsearch 2.30.0, muscle 3.8.425.
IMMCANTATION_IMAGE = ("immcantation/suite:4.8.0"
                      "@sha256:44e228a45b047bdb352363ab20a670336b003e98f3d1bac505f78af41a91ab67")

# The paper's multiplex primers (Rodriguez et al. 2023, Supplementary Data 7), verbatim; IUPAC codes included.
VH_PRIMERS: dict[str, str] = {
    "VH1": "ASKCCCAGGTBCAGCTKGTRC",
    "VH2": "CAGRTCACCTTGAAGGAGTCTG",
    "VH3": "AGGTGTCCAGTGTGARGTGCAG",
    "VH4": "ATGGGTCCTGTCCCAGGTGC",
    "VH5": "TCCACAGTGAGAGTCTGTGCC",
    "VH5-2": "TTCTCCAAGGAGTCTGTGCCG",
    "VH6": "CAGGTACAGCTGCAGCAGTCA",
}
C_PRIMERS_MULTIPLEX: dict[str, str] = {
    "IgM": "GGTTGGGGCGGATGCACTCC",
    "IgG": "SGATGGGCCCTTGGTGGARGC",
    "IgA": "CGGGAAGACCTTGGGGCTGGT",
}
MULTIPLEX_ISOTYPE = {"IgM": "IGHM", "IgG": "IGHG", "IgA": "IGHA"}
# The IGH lines of Immcantation's protocols/Universal/Human_IG_CRegion_RC.fasta, verbatim (as in the image).
CREGION_RC: dict[str, str] = {
    "IGHA": "CGGGAAGACCTTGGGGCTGGTCGGGGATGC",
    "IGHD": "GGGGAACACATCCGGAGCCTTGGTGGGTGC",
    "IGHE": "GGGGAAGACGGATGGGCTCTGTGTGGAGGC",
    "IGHG": "GGGGAAGACSGATGGGCCCTTGGTGGAAGC",
    "IGHM": "GGGGAAAAGGGTTGGGGCGGATGCACTCCC",
}

PROTOCOLS = ("race", "multiplex")
QUALITIES = ("full", "lite")
UMI_LENGTH = 12
RACE_LINKER = "GTAC"
LINKER_AND_TSO = 7  # GTAC + GGG, cut after the UMI
CLASSIFIER_PAIRS = 2000
CLASSIFIER_THRESHOLD = 0.8
PRIMER_MISMATCHES = 2
MIN_QUALITY = 20  # the mean-quality filter's threshold, and the paper's trimming threshold
MIN_READ_LENGTH = 125  # the paper's minimum after trimming
QUALITY_FILTER = "FilterSeq quality -q 20 on each mate: reads of mean quality below 20 dropped, none shortened"
QUALITY_TRIM = "FilterSeq trimqual -q 20 then length -n 125 on each mate: the paper's trimming"
QUALITY_SKIPPED = "skipped: SRA-Lite qualities carry no per-base quality"
MAX_ASSEMBLY_FAILURE = 0.35  # the share of pairs entering the assembly that may fail to assemble
ASSEMBLY_MAXERROR = 0.3  # AssemblePairs align: the most mismatches an overlap may have (pRESTO's default)
ASSEMBLY_ALPHA = 1e-5  # and the p-value it must reach (pRESTO's default)
C_PRIMER_MAXLEN = 70
PRIMER_MAXERROR = 0.2
CLUSTER_IDENTITY = 0.9
CONSENSUS_ARGS = ("--prcons", "0.6", "-n", "1", "-q", "0", "--maxerror", "0.1", "--maxgap", "0.5")
MIN_CONSCOUNT = 2
MIN_LENGTH = 250  # after primer removal
PAPER_MIN_LENGTH = 400  # before primer removal
CAP = 50_000
NPROC = 8
STEP_TIMEOUT = 6 * 3600.0  # seconds one step may take
LITE_QUALITIES = frozenset("?$")
TSV_COLUMNS = ("sequence_id", "isotype", "protocol", "consensus_count", "dupcount", "primer_masked")
RECORD_KIND = "alignair_assay.real.preprocess"
IN1, IN2, WORK = "/in1", "/in2", "/work"
PRIMER_FILES = {"cregion": "primers/cregion_rc.fasta", "vh": "primers/vh.fasta", "c": "primers/c_multiplex.fasta"}
ASSEMBLE_LOG = "assemble.records.log"

IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
         "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT"}
_CODE_SHA256 = sha256_file(__file__)


class PreprocessError(RuntimeError):
    """A run that cannot be preprocessed: refused by the classifier (``evidence`` holds its fractions), reads that
    are not what they are said to be, or a step that let no read through."""

    def __init__(self, message: str, evidence: dict | None = None):
        super().__init__(message)
        self.evidence = evidence


@dataclass(frozen=True)
class Step:
    """One pRESTO command: its name, its argv (paths relative to the work directory, or the read-only reads), the
    file whose records are its input and the files it writes (the first counted). A step that ``merges`` records
    (a consensus, a collapse) has no pass rate: its output counts groups, not records that passed. A step whose
    output depends on the order of its input (``sort_input``) reads it sorted by read id (``sort_fastq``). The step
    that ``assembles`` the pairs is the one the acceptance line is checked after."""

    name: str
    argv: tuple[str, ...]
    input: str
    outputs: tuple[str, ...]
    merges: bool = False
    sort_input: bool = False
    assembles: bool = False


# --- the classifier ----------------------------------------------------------------------------------------------

def primer_mismatches(read: str, primer: str) -> int:
    """Mismatches of ``primer`` against the start of ``read``, an IUPAC code matching any of its bases; each primer
    base past the end of a short read is a mismatch."""
    read = read.upper()
    missing = max(0, len(primer) - len(read))
    return missing + sum(base not in IUPAC[code] for base, code in zip(read, primer.upper()))


def _primer_at_start(read: str, primers: dict[str, str]) -> str | None:
    """The primer that starts ``read`` with the fewest mismatches, if within ``PRIMER_MISMATCHES``."""
    best, mismatches = None, PRIMER_MISMATCHES + 1
    for name, primer in primers.items():
        found = primer_mismatches(read, primer)
        if found < mismatches:
            best, mismatches = name, found
    return best


def classify_protocol(r1_head: Sequence[str], r2_head: Sequence[str]) -> tuple[str, dict]:
    """``"race"``, ``"multiplex"`` or ``"unknown"`` for read pairs ``(r1_head[i], r2_head[i])`` (the module
    docstring), and the evidence: the pair count, the fractions (``race_linker``, ``vh_primer``, ``c_primer``), the
    threshold, the primers found, the read lengths and the ``reason``. ValueError unless there are as many R1 as R2
    reads, and some."""
    if not r1_head or len(r1_head) != len(r2_head):
        raise ValueError(f"the classifier needs R1 and R2 reads of the same pairs, got {len(r1_head)} R1 and "
                         f"{len(r2_head)} R2 reads, not that many pairs")
    pairs = len(r1_head)
    linker = sum(read[UMI_LENGTH:UMI_LENGTH + len(RACE_LINKER)].upper() == RACE_LINKER for read in r2_head)
    vh = Counter(name for name in (_primer_at_start(read, VH_PRIMERS) for read in r1_head) if name)
    c = Counter(name for name in (_primer_at_start(read, C_PRIMERS_MULTIPLEX) for read in r2_head) if name)
    fractions = {"race_linker": round(linker / pairs, 4), "vh_primer": round(sum(vh.values()) / pairs, 4),
                 "c_primer": round(sum(c.values()) / pairs, 4)}
    race = fractions["race_linker"] >= CLASSIFIER_THRESHOLD
    multiplex = fractions["vh_primer"] >= CLASSIFIER_THRESHOLD and fractions["c_primer"] >= CLASSIFIER_THRESHOLD
    shares = ", ".join(f"{key} {value:.1%}" for key, value in fractions.items())
    if race and multiplex:
        protocol, reason = "unknown", f"the reads fit both layouts ({shares})"
    elif race:
        protocol, reason = "race", f"R2[12:16] is {RACE_LINKER} in {fractions['race_linker']:.1%} of the pairs"
    elif multiplex:
        protocol, reason = "multiplex", (f"R1 starts with a VH primer in {fractions['vh_primer']:.1%} and R2 with "
                                         f"a C primer in {fractions['c_primer']:.1%} of the pairs")
    else:
        protocol, reason = "unknown", f"the reads fit neither layout at {CLASSIFIER_THRESHOLD:.0%} ({shares})"
    evidence = {"pairs": pairs, **fractions, "threshold": CLASSIFIER_THRESHOLD, "max_mismatches": PRIMER_MISMATCHES,
                "vh_primers": dict(sorted(vh.items())), "c_primers": dict(sorted(c.items())),
                "r1_lengths": _lengths(r1_head), "r2_lengths": _lengths(r2_head), "reason": reason}
    return protocol, evidence


def _lengths(reads: Sequence[str]) -> dict:
    lengths = [len(read) for read in reads]
    return {"min": min(lengths), "median": statistics.median_low(lengths), "max": max(lengths)}


def read_head(r1, r2, n: int = CLASSIFIER_PAIRS) -> tuple[list[str], list[str]]:
    """The sequences of the first ``n`` pairs of gzipped FASTQs ``r1`` and ``r2`` (``_head``)."""
    first, second, _ = _head(Path(r1), Path(r2), n)
    return first, second


def _head(r1: Path, r2: Path, n: int) -> tuple[list[str], list[str], set[str]]:
    """The first ``n`` pairs' sequences and the quality symbols they use. PreprocessError for mates whose read
    names (first word) differ or that are not FASTQ."""
    mates: list[list[str]] = [[], []]
    symbols: set[str] = set()
    names: list[list[str]] = [[], []]
    for index, path in enumerate((r1, r2)):
        with gzip.open(path, "rt", encoding="ascii") as handle:
            for _ in range(n):
                header = handle.readline()
                if not header:
                    break
                sequence, separator, quality = (handle.readline().rstrip("\n") for _ in range(3))
                if not header.startswith("@") or not separator.startswith("+") or len(quality) != len(sequence):
                    raise PreprocessError(f"{path}: record {len(mates[index]) + 1} is not a FASTQ record")
                mates[index].append(sequence)
                names[index].append(header[1:].split()[0] if header[1:].split() else "")
                symbols.update(quality)
    if names[0] != names[1]:
        raise PreprocessError(f"{r1} and {r2}: the first {n} records are not the same pairs")
    return mates[0], mates[1], symbols


# --- primer spans and the cap --------------------------------------------------------------------------------------

def primer_spans(length: int, head: int, tail: int) -> list[list[int]]:
    """The ``[start, end)`` spans of primer bases in an assembled read of ``length`` nt: pRESTO's mask mode turns a
    primer at the start of a mate into ``N``s, so the head mate's primer (``head`` nt) is at the start of the read
    and the reverse-complemented tail mate's (``tail`` nt) at its end. A zero-length primer gives no span.
    ValueError for a negative length or primers that overlap."""
    if min(length, head, tail) < 0:
        raise ValueError(f"lengths must not be negative: read {length}, primers {head} and {tail}")
    if head + tail > length:
        raise ValueError(f"primers of {head} and {tail} nt overlap in a read of {length} nt")
    return [span for span in ([0, head], [length - tail, length]) if span[1] > span[0]]


def content_key(record: dict) -> tuple[str, str]:
    """The cap's sort key of a sequence: the SHA-256 of its content (``sequence``, ``isotype`` and ``primers``, the
    primer names it was found with), then its id, which only orders identical contents."""
    content = "\t".join((record["sequence"], record["isotype"], record.get("primers", "")))
    return hashlib.sha256(content.encode()).hexdigest(), record["sequence_id"]


def cap_sequences(records: Sequence[dict], cap: int) -> list[dict]:
    """``records`` ordered by their content (``content_key``), then at most ``cap`` of them evenly spaced
    (``core.cases.evenly_spaced``). The order is the content's, not pRESTO's id: which duplicate's id a collapse
    keeps can depend on its workers' order, while the contents do not. ValueError for a cap below 1 or a sequence
    id given twice."""
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(f"the cap must be a positive number of sequences, got {cap!r}")
    seen: set[str] = set()
    for record in records:
        if record["sequence_id"] in seen:
            raise ValueError(f"sequence id {record['sequence_id']} is given twice")
        seen.add(record["sequence_id"])
    ordered = sorted(records, key=content_key)
    return ordered if len(ordered) <= cap else evenly_spaced(ordered, cap)


def sort_fastq(path) -> None:
    """Rewrite FASTQ ``path`` with its records ordered by read id (pRESTO's, the header's text before the first
    ``|``; ties keep their order). Only the headers and their offsets are held in memory."""
    path = Path(path)
    index: list[tuple[str, int, int]] = []
    with open(path, "rb") as handle:
        while True:
            offset = handle.tell()
            header = handle.readline()
            if not header:
                break
            for _ in range(3):
                handle.readline()
            index.append((header[1:].split(b"|", 1)[0].rstrip(b"\r\n").decode("ascii"), offset,
                          handle.tell() - offset))
    index.sort(key=lambda entry: entry[0])
    part = path.with_name(path.name + ".sorted")
    with open(path, "rb") as source, open(part, "wb") as sorted_file:
        for _, offset, size in index:
            source.seek(offset)
            sorted_file.write(source.read(size))
    os.replace(part, path)


# --- the recipes ---------------------------------------------------------------------------------------------------

def recipe(protocol: str, *, quality: str, r1: str, r2: str, nproc: int, paper_length_filter: bool = False,
           paper_quality_trim: bool = False) -> list[Step]:
    """The pRESTO steps of ``protocol`` on raw mates ``r1`` and ``r2`` (their paths in the container), in order (the
    module docstring). Full-quality reads first get the mean-quality filter, or the paper's trimming with
    ``paper_quality_trim``; Lite reads get neither. ValueError for another protocol or quality."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"no recipe for protocol {protocol!r}: one of {', '.join(PROTOCOLS)}")
    if quality not in QUALITIES:
        raise ValueError(f"quality must be one of {', '.join(QUALITIES)}, got {quality!r}")
    # A step that may read the raw mates names its output with -o: from a gzipped input pRESTO would otherwise
    # gzip its output, and every later step's.
    n = ("--nproc", str(nproc))
    steps: list[Step] = []
    if quality == "full" and not paper_quality_trim:
        for mate, raw in (("1", r1), ("2", r2)):
            kept = f"R{mate}-quality_quality-pass.fastq"
            steps.append(Step(f"quality-r{mate}", ("FilterSeq.py", "quality", "-s", raw, "-q", str(MIN_QUALITY), *n,
                                                   "-o", kept), raw, (kept,)))
        r1, r2 = "R1-quality_quality-pass.fastq", "R2-quality_quality-pass.fastq"
    elif quality == "full":
        for mate, raw in (("1", r1), ("2", r2)):
            trimmed = f"R{mate}-quality_trimqual-pass.fastq"
            steps.append(Step(f"trimqual-r{mate}", ("FilterSeq.py", "trimqual", "-s", raw, "-q", str(MIN_QUALITY),
                                                    *n, "-o", trimmed), raw, (trimmed,)))
        for mate in ("1", "2"):
            trimmed = f"R{mate}-quality_trimqual-pass.fastq"
            steps.append(Step(f"length-r{mate}", ("FilterSeq.py", "length", "-s", trimmed, "-n", str(MIN_READ_LENGTH),
                                                  *n, "--outname", f"R{mate}-quality"),
                              trimmed, (f"R{mate}-quality_length-pass.fastq",)))
        r1, r2 = "R1-quality_length-pass.fastq", "R2-quality_length-pass.fastq"
    if protocol == "race":
        return steps + _race(r1, r2, n)
    return steps + _multiplex(r1, r2, n, paper_length_filter)


def _race(r1: str, r2: str, n: tuple[str, str]) -> list[Step]:
    pair = ("pair-1_pair-pass.fastq", "pair-2_pair-pass.fastq")
    molecule = ("molecule-1_pair-pass.fastq", "molecule-2_pair-pass.fastq")
    consensus = ("consensus-1_pair-pass.fastq", "consensus-2_pair-pass.fastq")  # R2 first: the assembly's head
    return [
        Step("umi", ("MaskPrimers.py", "extract", "-s", r2, "--start", "0", "--len", str(UMI_LENGTH), "--pf", "UMI",
                     "--mode", "cut", *n, "-o", "R2-umi_primers-pass.fastq"), r2, ("R2-umi_primers-pass.fastq",)),
        Step("linker", ("MaskPrimers.py", "extract", "-s", "R2-umi_primers-pass.fastq", "--start", "0", "--len",
                        str(LINKER_AND_TSO), "--pf", "LINKER", "--mode", "cut", *n, "--outname", "R2-linker"),
             "R2-umi_primers-pass.fastq", ("R2-linker_primers-pass.fastq",)),
        Step("c-primer", ("MaskPrimers.py", "align", "-s", r1, "-p", PRIMER_FILES["cregion"], "--maxlen",
                          str(C_PRIMER_MAXLEN), "--maxerror", str(PRIMER_MAXERROR), "--mode", "cut", "--skiprc",
                          "--pf", "CPRIMER", *n, "-o", "R1-cprimer_primers-pass.fastq"),
             r1, ("R1-cprimer_primers-pass.fastq",)),
        Step("pair", ("PairSeq.py", "-1", "R1-cprimer_primers-pass.fastq", "-2", "R2-linker_primers-pass.fastq",
                      "--1f", "CPRIMER", "--2f", "UMI", "--coord", "sra", "--outname", "pair"),
             "R1-cprimer_primers-pass.fastq", pair),
        Step("cluster", ("ClusterSets.py", "set", "-s", pair[0], "-f", "UMI", "-k", "CLUSTER", "--cluster", "vsearch",
                         "--ident", str(CLUSTER_IDENTITY), *n, "--outname", "R1-cluster"),
             pair[0], ("R1-cluster_cluster-pass.fastq",), sort_input=True),
        Step("molecule", ("ParseHeaders.py", "copy", "-s", "R1-cluster_cluster-pass.fastq", "-f", "UMI", "-k",
                          "CLUSTER", "--act", "cat", "--outname", "R1-molecule"),
             "R1-cluster_cluster-pass.fastq", ("R1-molecule_reheader.fastq",)),
        Step("pair-molecule", ("PairSeq.py", "-1", "R1-molecule_reheader.fastq", "-2", pair[1], "--1f", "CLUSTER",
                               "--coord", "sra", "--outname", "molecule"),
             "R1-molecule_reheader.fastq", molecule),
        *(Step(f"align-r{mate}", ("AlignSets.py", "muscle", "-s", source, "--bf", "CLUSTER", "--exec", "muscle", *n,
                                  "--outname", f"R{mate}-align"), source, (f"R{mate}-align_align-pass.fastq",),
               sort_input=True)
          for mate, source in (("1", molecule[0]), ("2", molecule[1]))),
        *(Step(f"consensus-r{mate}", ("BuildConsensus.py", "-s", f"R{mate}-align_align-pass.fastq", "--bf",
                                      "CLUSTER", "--pf", "CPRIMER", *CONSENSUS_ARGS, *n, "--outname",
                                      f"R{mate}-consensus"),
               f"R{mate}-align_align-pass.fastq", (f"R{mate}-consensus_consensus-pass.fastq",), merges=True)
          for mate in ("1", "2")),
        Step("pair-consensus", ("PairSeq.py", "-1", "R2-consensus_consensus-pass.fastq", "-2",
                                "R1-consensus_consensus-pass.fastq", "--coord", "presto", "--outname", "consensus"),
             "R2-consensus_consensus-pass.fastq", consensus),
        Step("assemble", ("AssemblePairs.py", "align", "-1", consensus[0], "-2", consensus[1], "--coord", "presto",
                          "--rc", "tail", "--1f", "CONSCOUNT", "--2f", "CONSCOUNT", "PRCONS", *n, "--outname",
                          "assembled", "--log", ASSEMBLE_LOG),
             consensus[0], ("assembled_assemble-pass.fastq",), assembles=True),
        Step("conscount", ("ParseHeaders.py", "collapse", "-s", "assembled_assemble-pass.fastq", "-f", "CONSCOUNT",
                           "--act", "min", "--outname", "conscount"),
             "assembled_assemble-pass.fastq", ("conscount_reheader.fastq",)),
        Step("collapse", ("CollapseSeq.py", "-s", "conscount_reheader.fastq", "-n", "0", "--uf", "PRCONS", "--cf",
                          "CONSCOUNT", "--act", "sum", "--inner", "--keepmiss", "--outname", "collapsed"),
             "conscount_reheader.fastq", ("collapsed_collapse-unique.fastq",), merges=True, sort_input=True),
        Step(f"conscount-{MIN_CONSCOUNT}", ("SplitSeq.py", "group", "-s", "collapsed_collapse-unique.fastq", "-f",
                                            "CONSCOUNT", "--num", str(MIN_CONSCOUNT), "--outname", "final"),
             "collapsed_collapse-unique.fastq", (f"final_atleast-{MIN_CONSCOUNT}.fastq",)),
    ]


def _multiplex(r1: str, r2: str, n: tuple[str, str], paper_length_filter: bool) -> list[Step]:
    primers = ("primers-1_pair-pass.fastq", "primers-2_pair-pass.fastq")
    length = ("-n", str(PAPER_MIN_LENGTH)) if paper_length_filter else ("-n", str(MIN_LENGTH), "--inner")
    return [
        Step("v-primer", ("MaskPrimers.py", "score", "-s", r1, "-p", PRIMER_FILES["vh"], "--start", "0",
                          "--maxerror", str(PRIMER_MAXERROR), "--mode", "mask", "--pf", "VPRIMER", *n, "-o",
                          "R1-vprimer_primers-pass.fastq"), r1, ("R1-vprimer_primers-pass.fastq",)),
        Step("c-primer", ("MaskPrimers.py", "score", "-s", r2, "-p", PRIMER_FILES["c"], "--start", "0",
                          "--maxerror", str(PRIMER_MAXERROR), "--mode", "mask", "--pf", "CPRIMER", *n, "-o",
                          "R2-cprimer_primers-pass.fastq"), r2, ("R2-cprimer_primers-pass.fastq",)),
        Step("both-primers", ("PairSeq.py", "-1", "R1-vprimer_primers-pass.fastq", "-2",
                              "R2-cprimer_primers-pass.fastq", "--coord", "sra", "--outname", "primers"),
             "R1-vprimer_primers-pass.fastq", primers),
        Step("assemble", ("AssemblePairs.py", "align", "-1", primers[0], "-2", primers[1], "--coord", "sra", "--rc",
                          "tail", "--1f", "VPRIMER", "--2f", "CPRIMER", *n, "--outname", "assembled", "--log",
                          ASSEMBLE_LOG), primers[0], ("assembled_assemble-pass.fastq",), assembles=True),
        Step("length", ("FilterSeq.py", "length", "-s", "assembled_assemble-pass.fastq", *length, *n, "--outname",
                        "filtered"), "assembled_assemble-pass.fastq", ("filtered_length-pass.fastq",)),
        Step("collapse", ("CollapseSeq.py", "-s", "filtered_length-pass.fastq", "-n", "0", "--uf", "VPRIMER",
                          "CPRIMER", "--inner", "--keepmiss", "--outname", "collapsed"),
             "filtered_length-pass.fastq", ("collapsed_collapse-unique.fastq",), merges=True, sort_input=True),
    ]


def step_command(args: Sequence[str], *, donor: str, step: str, work, r1, r2, nproc: int) -> list[str]:
    """The container run of pRESTO command ``args``: the pinned image on at most ``nproc`` CPUs, as the current user,
    with no network and no capabilities, the reads' directories mounted read-only at ``IN1`` and ``IN2`` and the
    work directory, its only writable mount, at ``WORK`` (``core.container``); named per donor, step and process."""
    return hardened_run(IMMCANTATION_IMAGE, list(args), name=container_name("preprocess", donor, step),
                        mounts=[Mount(Path(r1).absolute().parent, IN1), Mount(Path(r2).absolute().parent, IN2),
                                Mount(Path(work).absolute(), WORK, read_only=False)],
                        workdir=WORK, env={"HOME": "/tmp"}, cpus=container_cpus(nproc))


def container_cpus(nproc: int) -> int:
    """The container's CPU limit: ``nproc``, or the host's CPU count when it has fewer (Docker refuses more)."""
    return min(nproc, os.cpu_count() or nproc)


# --- the run -------------------------------------------------------------------------------------------------------

def run_fingerprint(r1, r2, *, quality: str, cap: int, paper_length_filter: bool, paper_quality_trim: bool = False,
                    accept_assembly_loss: bool = False, nproc: int = NPROC) -> dict:
    """What a record must match to be current: ``inputs`` (each mate's path, md5 and size), ``image`` (the pinned
    reference), ``code_sha256`` (this module's alone, as in the rest of the assay), ``parameters`` and
    ``fingerprint``, the SHA-256 over the inputs' md5s, the image, the code and the parameters."""
    inputs = {mate: {"path": str(Path(path)), "md5": _md5(Path(path)), "bytes": Path(path).stat().st_size}
              for mate, path in (("r1", r1), ("r2", r2))}
    parameters = {"quality": quality, "cap": cap, "paper_length_filter": paper_length_filter,
                  "min_length": PAPER_MIN_LENGTH if paper_length_filter else MIN_LENGTH,
                  "length_after_primer_removal": not paper_length_filter, "paper_quality_trim": paper_quality_trim,
                  "max_assembly_failure": MAX_ASSEMBLY_FAILURE, "accept_assembly_loss": accept_assembly_loss,
                  "nproc": nproc}
    digest = hashlib.sha256(json.dumps({"inputs": [inputs["r1"]["md5"], inputs["r2"]["md5"]],
                                        "image": IMMCANTATION_IMAGE, "code": _CODE_SHA256,
                                        "parameters": parameters}, sort_keys=True).encode()).hexdigest()
    return {"inputs": inputs, "image": IMMCANTATION_IMAGE, "code_sha256": _CODE_SHA256, "parameters": parameters,
            "fingerprint": digest}


def preprocess_run(donor: str, r1, r2, layout: RunLayout, *, quality: str, cap: int = CAP,
                   paper_length_filter: bool = False, paper_quality_trim: bool = False,
                   accept_assembly_loss: bool = False, nproc: int = NPROC, timeout: float = STEP_TIMEOUT) -> dict:
    """Preprocess donor ``donor``'s run from its gzipped mates ``r1`` and ``r2`` into ``layout`` (the module
    docstring) and return its record; a current record is returned as it is, with ``reused`` True.

    ``quality`` is the reads' quality alphabet (``"full"`` or ``"lite"``, as fetched); ``cap`` the most sequences
    kept; ``paper_length_filter`` the paper's multiplex length filter; ``paper_quality_trim`` the paper's Q20
    trimming in place of the mean-quality filter; ``accept_assembly_loss`` lets a run over the assembly's
    acceptance line through; ``nproc`` the CPUs given to the container (at most the host's) and to each pRESTO step;
    ``timeout`` the seconds one step may take. ValueError for another quality or cap; PreprocessError, its
    ``evidence`` set, for a run the classifier refuses or whose assembly loss is over the line (its record written
    either way, ``refused`` or ``rejected``), reads said to be full quality that have SRA-Lite qualities, mates
    that are not the same pairs, or a step that lets no read through; ContainerError when Docker or the pinned
    image is missing (before anything is written) or a step fails."""
    if quality not in QUALITIES:
        raise ValueError(f"quality must be one of {', '.join(QUALITIES)}, got {quality!r}")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(f"the cap must be a positive number of sequences, got {cap!r}")
    r1, r2 = Path(r1).absolute(), Path(r2).absolute()
    record_path = layout.real_preprocess_json(donor)
    pinned = run_fingerprint(r1, r2, quality=quality, cap=cap, paper_length_filter=paper_length_filter,
                             paper_quality_trim=paper_quality_trim, accept_assembly_loss=accept_assembly_loss,
                             nproc=nproc)
    current = _current(layout, donor, pinned["fingerprint"])
    if current is not None:
        return {**current, "reused": True}

    started = time.monotonic()
    first, second, symbols = _head(r1, r2, CLASSIFIER_PAIRS)
    if quality == "full" and symbols <= LITE_QUALITIES:
        raise PreprocessError(f"{donor}: {r1.name} and {r2.name} are said to be full quality, but their first "
                              f"{len(first)} records have only SRA-Lite qualities ({''.join(sorted(symbols))})")
    protocol, evidence = classify_protocol(first, second)
    record = {"kind": RECORD_KIND, "donor": donor, "protocol": protocol, "quality": quality,
              "classifier": {"protocol": protocol, "evidence": evidence, "quality_symbols": len(symbols)},
              **{key: pinned[key] for key in ("inputs", "parameters", "code_sha256", "fingerprint")}}
    if protocol == "unknown":
        _remove_outputs(layout, donor)
        _write_json(record_path, {**record, "status": "refused",
                                  "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        raise PreprocessError(f"{donor}: the run is refused: {evidence['reason']}", evidence)

    image = check_image(IMMCANTATION_IMAGE)  # before anything is written
    _remove_outputs(layout, donor)
    work, logs = layout.real_preprocess_work(donor), layout.real_preprocess_logs(donor)
    for directory in (work, logs):
        directory.mkdir(parents=True)
    _write_primers(work)
    steps = recipe(protocol, quality=quality, r1=f"{IN1}/{r1.name}", r2=f"{IN2}/{r2.name}", nproc=nproc,
                   paper_length_filter=paper_length_filter, paper_quality_trim=paper_quality_trim)
    raw = _count_gz(r1)
    if _count_gz(r2) != raw:
        raise PreprocessError(f"{donor}: {r1.name} and {r2.name} hold different numbers of reads")
    counts, file_counts = {"raw_pairs": raw}, {f"{IN1}/{r1.name}": raw, f"{IN2}/{r2.name}": raw}
    record.update(image=image, nproc=nproc, cpus=container_cpus(nproc),
                  quality_steps=(QUALITY_SKIPPED if quality == "lite" else
                                 QUALITY_TRIM if paper_quality_trim else QUALITY_FILTER),
                  container=step_command([], donor=donor, step="<step>", work=work, r1=r1, r2=r2, nproc=nproc))
    ran = []
    for step in steps:
        ran.append(_run_step(step, donor=donor, work=work, logs=logs, r1=r1, r2=r2, nproc=nproc, timeout=timeout,
                             file_counts=file_counts))
        counts[step.name] = ran[-1]["records"]
        if step.assembles:
            pairs_in, assembled_pairs = file_counts[step.input], ran[-1]["records"]
            assembled = _assembled_lengths(work / step.outputs[0])
            passed = ({identifier for identifier, _, _ in _read_presto_fastq(work / step.outputs[0])}
                      if (work / step.outputs[0]).is_file() else set())
            with open(work / ASSEMBLE_LOG, encoding="ascii") as handle:
                assembly = assembly_evidence(handle, passed)
            _archive_log(work / ASSEMBLE_LOG, logs / f"{ASSEMBLE_LOG}.gz")
            acceptance = assembly_acceptance(pairs_in, assembled_pairs, accept_assembly_loss=accept_assembly_loss)
            record.update(accepted=acceptance["accepted"], assembly_acceptance=acceptance, assembly=assembly)
            if not acceptance["accepted"]:
                shutil.rmtree(work)
                _write_json(record_path, {**record, "status": "rejected", "steps": ran, "counts": counts,
                                          "lengths": {"assembled": assembled},
                                          "wall_seconds": round(time.monotonic() - started, 1),
                                          "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                raise PreprocessError(
                    f"{donor}: {acceptance['failed']} of {pairs_in} pairs ({acceptance['failure_rate']:.1%}) failed "
                    f"to assemble, over the {MAX_ASSEMBLY_FAILURE:.0%} line; the run is rejected (see its record; "
                    "accept_assembly_loss lets it through)", {**acceptance, "kinds": assembly["fail"]["kinds"]})

    rows, checks = _rows(work / steps[-1].outputs[0], protocol, paper_length_filter)
    capped = cap_sequences(rows, cap)
    counts.update(unique=len(rows), capped=len(capped))
    _write_outputs(layout, donor, capped)
    shutil.rmtree(work)
    record.update(
        status="done", reused=False, steps=ran, counts=counts,
        pass_rates={"assembly": _rate(assembled_pairs, pairs_in), "yield": _rate(len(rows), raw),
                    **{entry["name"]: entry["pass_rate"] for entry in ran if entry["pass_rate"] is not None}},
        checks=checks, lengths={"assembled": assembled, "unique": _summary([row["observed"] for row in rows])},
        isotypes=dict(sorted(Counter(row["isotype"] for row in capped).items())),
        isotypes_unique=dict(sorted(Counter(row["isotype"] for row in rows).items())),
        outputs=_output_digests(layout, donor), wall_seconds=round(time.monotonic() - started, 1),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    _write_json(record_path, {key: value for key, value in record.items() if key != "reused"})
    return record


def assembly_acceptance(pairs: int, assembled: int, *, accept_assembly_loss: bool) -> dict:
    """Whether a run's assembly loss is acceptable: ``failure_rate``, the share of the ``pairs`` entering the
    assembly that did not assemble, is ``within_line`` at most ``MAX_ASSEMBLY_FAILURE``; a run is ``accepted``
    when it is, or when ``accept_assembly_loss`` (``override``) lets it through. No pair at all is not accepted."""
    failure = _rate(pairs - assembled, pairs)
    within = failure is not None and failure <= MAX_ASSEMBLY_FAILURE
    return {"pairs": pairs, "failed": pairs - assembled, "failure_rate": failure,
            "max_failure_rate": MAX_ASSEMBLY_FAILURE, "within_line": within, "override": accept_assembly_loss,
            "accepted": within or (accept_assembly_loss and pairs > 0)}


def _run_step(step: Step, *, donor: str, work: Path, logs: Path, r1: Path, r2: Path, nproc: int, timeout: float,
              file_counts: dict[str, int]) -> dict:
    """Run one step in the container (its input sorted by read id first when the step says so), keep its output as
    ``<step>.log`` and count its records. An assembly that lets no pair through is left to the acceptance line."""
    argv = step_command(step.argv, donor=donor, step=step.name, work=work, r1=r1, r2=r2, nproc=nproc)
    name = argv[argv.index("--name") + 1]
    started = time.monotonic()
    if step.sort_input:
        sort_fastq(work / step.input)
    try:
        completed = run_container(argv, name=name, timeout=timeout, what=f"{donor}'s pRESTO step {step.name}")
    except ContainerError as error:
        (logs / f"{step.name}.log").write_text(str(error) + "\n", encoding="utf-8")
        raise
    (logs / f"{step.name}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    for output in step.outputs:
        file_counts[output] = _count_fastq(work / output) if (work / output).is_file() else 0
    records = file_counts[step.outputs[0]]
    if records == 0 and not step.assembles:
        raise PreprocessError(f"{donor}: pRESTO step {step.name} let no read through (see {logs / step.name}.log)")
    return {"name": step.name, "argv": list(step.argv), "sorted_input": step.sort_input,
            "seconds": round(time.monotonic() - started, 1), "records": records,
            "pass_rate": None if step.merges else _rate(records, file_counts.get(step.input, 0))}


def _rows(path: Path, protocol: str, paper_length_filter: bool) -> tuple[list[dict], dict]:
    """The final unique sequences as TSV rows (plus ``sequence``, ``primers``, the primer names it was found with,
    and ``observed``, the bases outside the primers), and the checks: multiplex sequences shorter than the length
    filter once the primers' spans are removed are dropped and counted (pRESTO's ``--inner`` counts ``N`` runs,
    and the assembly may fill a masked primer base from the other mate), and primer spans holding a base other
    than ``N`` are counted."""
    rows, too_short, unmasked = [], 0, 0
    for identifier, annotations, sequence in _read_presto_fastq(path):
        if protocol == "race":
            isotype, spans, primers = annotations["PRCONS"], [], annotations["PRCONS"]
            consensus = int(annotations["CONSCOUNT"])
        else:
            isotype = MULTIPLEX_ISOTYPE[annotations["CPRIMER"]]
            spans = primer_spans(len(sequence), len(VH_PRIMERS[annotations["VPRIMER"]]),
                                 len(C_PRIMERS_MULTIPLEX[annotations["CPRIMER"]]))
            primers = f"{annotations['VPRIMER']},{annotations['CPRIMER']}"
            consensus = int(annotations["DUPCOUNT"])
        observed = len(sequence) - sum(end - start for start, end in spans)
        if protocol == "multiplex" and not paper_length_filter and observed < MIN_LENGTH:
            too_short += 1
            continue
        unmasked += any(set(sequence[start:end]) != {"N"} for start, end in spans)
        rows.append({"sequence_id": identifier, "sequence": sequence, "isotype": isotype, "primers": primers,
                     "protocol": protocol, "consensus_count": consensus, "dupcount": int(annotations["DUPCOUNT"]),
                     "primer_masked": spans, "observed": observed})
    return rows, {"shorter_than_filter_after_primers": too_short, "primer_spans_not_masked": unmasked}


def _read_presto_fastq(path: Path):
    """``(id, annotations, sequence)`` per record of a pRESTO FASTQ (``ID|FIELD=value|...``); the id is the first
    word of pRESTO's, which keeps the read name's mate suffix (``<run>.<spot> <spot>/<mate>``)."""
    with open(path, encoding="ascii") as handle:
        while True:
            header = handle.readline()
            if not header:
                return
            sequence = handle.readline().strip()
            handle.readline()
            handle.readline()
            identifier, *fields = header[1:].rstrip("\n").split("|")
            annotations = dict(field.split("=", 1) for field in fields if "=" in field)
            yield identifier.split()[0], annotations, sequence


def _write_outputs(layout: RunLayout, donor: str, rows: Sequence[dict]) -> None:
    """The FASTA and the TSV of ``rows``, each written to ``<name>.part`` and renamed."""
    fasta, tsv = layout.real_reads(donor), layout.real_reads_meta(donor)
    fasta.parent.mkdir(parents=True, exist_ok=True)
    _replace(fasta, "".join(f">{row['sequence_id']}\n{row['sequence']}\n" for row in rows))
    lines = ["\t".join(TSV_COLUMNS)]
    for row in rows:
        values = {**row, "primer_masked": json.dumps(row["primer_masked"], separators=(",", ":"))}
        lines.append("\t".join(str(values[column]) for column in TSV_COLUMNS))
    _replace(tsv, "\n".join(lines) + "\n")


def _output_digests(layout: RunLayout, donor: str) -> dict:
    return {"fasta": sha256_file(layout.real_reads(donor)), "tsv": sha256_file(layout.real_reads_meta(donor))}


def _current(layout: RunLayout, donor: str, fingerprint: str) -> dict | None:
    """The donor's record when it is current: done, of ``fingerprint``, with its FASTA and TSV unchanged."""
    path = layout.real_preprocess_json(donor)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("status") != "done" or record.get("fingerprint") != fingerprint:
        return None
    if not (layout.real_reads(donor).is_file() and layout.real_reads_meta(donor).is_file()):
        return None
    return record if _output_digests(layout, donor) == record.get("outputs") else None


def _remove_outputs(layout: RunLayout, donor: str) -> None:
    """The donor's record, outputs, logs and work directory, so none of an earlier run is left beside a new one."""
    for path in (layout.real_preprocess_json(donor), layout.real_reads(donor), layout.real_reads_meta(donor)):
        path.unlink(missing_ok=True)
    for directory in (layout.real_preprocess_logs(donor), layout.real_preprocess_work(donor)):
        shutil.rmtree(directory, ignore_errors=True)


def _write_primers(work: Path) -> None:
    for key, primers in (("cregion", CREGION_RC), ("vh", VH_PRIMERS), ("c", C_PRIMERS_MULTIPLEX)):
        path = work / PRIMER_FILES[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f">{name}\n{sequence}\n" for name, sequence in primers.items()), encoding="ascii")


def assembly_evidence(log_lines, passed: set[str]) -> dict:
    """What ``AssemblePairs align`` found per pair, from its ``--log`` lines: for the pairs whose id is in
    ``passed`` (``pass``) and the others (``fail``), the count and the distributions (``deciles``) of the overlap
    and error of the best alignment it tried, of each mate's length (``SEQ1`` the head, ``SEQ2`` the tail, as it
    logs them) and of the insert-length estimates: ``mates_length``, the mates' combined length (for a failed pair
    whose mates do not meet, a lower bound on its insert), and for the passed pairs ``insert_length``, the
    assembled length. The failed pairs are also counted by why they failed (``kinds``), by AssemblePairs' own
    tests: ``mismatches``, the best overlap over ``ASSEMBLY_MAXERROR`` (mates that do overlap, with too many
    mismatches, or mates that do not, whose best overlap is chance); ``not_significant``, an overlap within the
    error limit but too short for ``ASSEMBLY_ALPHA``; ``no_overlap``, none tried."""
    groups: dict[str, dict[str, list]] = {
        kind: {"overlap": [], "error": [], "head_length": [], "tail_length": [], "mates_length": []}
        for kind in ("pass", "fail")}
    groups["pass"]["insert_length"] = []
    kinds = {"mismatches": 0, "not_significant": 0, "no_overlap": 0}

    def close(entry: dict) -> None:
        if "ID" not in entry:
            return
        kind = "pass" if entry["ID"].split("|")[0].split()[0] in passed else "fail"
        group = groups[kind]
        if kind == "fail":
            if entry.get("OVERLAP") in (None, "", "None"):
                kinds["no_overlap"] += 1
            elif float(entry.get("ERROR") or 1) > ASSEMBLY_MAXERROR:
                kinds["mismatches"] += 1
            else:
                kinds["not_significant"] += 1
        for key, field in (("overlap", "OVERLAP"), ("error", "ERROR")):
            if entry.get(field) not in (None, "", "None"):
                group[key].append(float(entry[field]) if key == "error" else int(entry[field]))
        head, tail = len(entry.get("SEQ1", "").strip()), len(entry.get("SEQ2", "").strip())
        group["head_length"].append(head)
        group["tail_length"].append(tail)
        group["mates_length"].append(head + tail)
        if kind == "pass" and entry.get("LENGTH") not in (None, "", "None"):
            group["insert_length"].append(int(entry["LENGTH"]))

    entry: dict = {}
    for line in log_lines:
        key, separator, value = line.rstrip("\n").partition("> ")
        if not separator:
            close(entry)
            entry = {}
        else:
            entry[key.strip()] = value
    close(entry)
    evidence = {kind: {"n": len(values["head_length"]), **{key: deciles(found) for key, found in values.items()}}
                for kind, values in groups.items()}
    evidence["fail"]["kinds"] = kinds
    return evidence


def deciles(values: Sequence[float]) -> dict:
    """``n``, ``min``, the nine deciles ``p10``..``p90`` (nearest rank: the value at position ``floor(q * n)`` of
    the sorted values), ``median`` (``p50``) and ``max``; ``{"n": 0}`` for no values."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    found = {"n": len(ordered), "min": ordered[0]}
    found.update({f"p{decile}": ordered[min(len(ordered) - 1, decile * len(ordered) // 100)]
                  for decile in range(10, 100, 10)})
    return {**found, "median": found["p50"], "max": ordered[-1]}


def _assembled_lengths(path: Path) -> dict:
    """The length distribution of the assembled reads, primer ``N`` runs at the ends excluded."""
    lengths = [len(sequence.strip("N")) for _, _, sequence in _read_presto_fastq(path)] if path.is_file() else []
    return {**_summary(lengths), f"at_least_{MIN_LENGTH}": sum(length >= MIN_LENGTH for length in lengths),
            f"at_least_{PAPER_MIN_LENGTH}": sum(length >= PAPER_MIN_LENGTH for length in lengths)}


def _summary(values: Sequence[int]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def quantile(q: float) -> int:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {"n": len(ordered), "min": ordered[0], "p5": quantile(0.05), "p25": quantile(0.25),
            "median": quantile(0.5), "p75": quantile(0.75), "p95": quantile(0.95), "max": ordered[-1]}


def _archive_log(source: Path, destination: Path) -> None:
    if source.is_file():
        with open(source, "rb") as handle, gzip.open(destination, "wb") as archive:
            shutil.copyfileobj(handle, archive)


def _rate(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def _count_fastq(path: Path) -> int:
    with open(path, "rb") as handle:
        lines = sum(1 for _ in handle)
    if lines % 4:
        raise PreprocessError(f"{path} is not a FASTQ of four-line records ({lines} lines)")
    return lines // 4


def _count_gz(path: Path) -> int:
    with gzip.open(path, "rb") as handle:
        lines = sum(1 for _ in handle)
    if lines % 4:
        raise PreprocessError(f"{path} is not a FASTQ of four-line records ({lines} lines)")
    return lines // 4


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace(path: Path, text: str) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8")
    os.replace(part, path)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace(path, json.dumps(value, indent=2) + "\n")  # in insertion order: the steps' order
