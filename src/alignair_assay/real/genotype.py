"""Real donors' IGH genotypes from genomic long-read sequencing, mapped by sequence to the shared reference.

The truth for real AIRR-seq data is each donor's own alleles, read from long-read sequencing of the same person and
published by VDJbase (genomic section, study P25: Rodriguez et al. 2023, Nat Commun 14:4419; CC0). This module
reads them, and maps them to the shared reference the tools run on and to OGRDB's IUIS labels.

**Identity is the ungapped sequence, compared exactly**: uppercase, no mismatch, no length tolerance. Names are never
matched across sources. The shared reference names its V alleles by OGRDB F/G labels (``IGHVF6-G21*06``), which
carry no IUIS name, and VDJbase names a novel allele after its closest IMGT allele (``IGHV3-30*18_t189c``).

- ``IuisMap.from_ogrdb``: the OGRDB IGH_VDJ germline set (the AIRR-format JSON; rev 8 has 245 labels) against the
  shared reference. On ``HUMAN_IGH_OGRDB``, 236 of the 238 alleles have a label with their exact sequence;
  ``IGHVF10-G33*01`` and ``IGHVF10-G42*01`` have none, each being one base shorter than OGRDB's IGHV3-13*06 and
  IGHV3-43D*05. Eleven shared alleles carry two labels: identical-sequence paralogs (``IGHVF6-G21*06`` is
  IGHV1-69*01 and IGHV1-69D*01) and the identical D pairs.
- ``load_vdjbase_csv``: a VDJbase genomic per-sample CSV, one row per haplotype per gene, as a ``RealGenotype`` of
  ``RealAllele`` records. Haplotype 1 or 2 is phased; 0 is unphased, which is not homozygous, and is kept as 0.
  - The sequence is the row's V-REGION, D-REGION or J-REGION. A V allele's ``gapped`` is its V-REGION-GAPPED, whose
    gaps are ``.`` columns and ``-`` for a deleted base, when that is the V-REGION once the gaps are removed. On the
    pilot donors every one is; another is left out (None) and named in the genotype's ``source``.
  - ``functional`` is False when the row's ``notes`` flag a V-REGION defect (``DEFECT_NOTES``): a stop codon, a
    first or second cysteine or the conserved tryptophan not found, or a V-REGION length mismatch. Notes on the
    recombination signals do not count. On the eight pilot donors this rule reproduces VDJbase's genotype API
    allele for allele: the API lists exactly the alleles without such a note. So the donors' V genes with a
    functional allele are the API's V gene counts (47, 49, 50, 49, 39, 53, 40 and 48 for SC-23, SC-4, SC-11, SC-6,
    W-97, W-89, W-133 and W-131), while the CSVs carry 51, 53, 54, 52, 42, 57, 44 and 52 V genes in all.
    ``RealGenotype.genes`` counts every carried gene. OGRDB's own functional flag disagrees on two pilot alleles:
    IGHV3-20*02 is functional in OGRDB but noted "First cysteine not found", and IGHV8-51-1*02 is non-functional
    in OGRDB with no note. The note is the donor's own annotation, so it decides.
  - The D calls are filtered as they are loaded (``filter_d``); the dropped calls and their reasons are in
    ``source["d_dropped"]``. Identical rows (one allele on two contigs of a haplotype) are one record.
- ``filter_d``: VDJbase's D calls include assembly artefacts. A D call is dropped when its id carries a gap-indel
  token (a ``-`` for a deleted base, an ``i<pos><bases>`` insertion, or a block whose bases do not fill its span),
  or when it is under ``D_MIN_IDENTITY`` (90%) identical to its closest IMGT allele (VDJbase's
  ``imgt_allele_match``).
- ``merge_identical``: each shared allele and the shared alleles with its exact sequence (D4-4*01 / D4-11*01 and
  D5-5*01 / D5-18*01 on HUMAN_IGH_OGRDB; each V paralog pair is already one shared allele with two labels).
- ``upto``: an allele through IMGT position 318, ungapped. Two V alleles that differ only beyond it cannot be told
  apart from rearranged reads (Peres et al. 2023, PIgLET); on HUMAN_IGH_OGRDB that is ``IGHVF3-G9*04`` and
  ``IGHVF3-G9*04_A319G``.
- ``sd3_novel`` and ``sd3_only``: Supplementary Data 3 of Rodriguez et al. 2023 lists the paper's 137 novel alleles
  (132 IGHV, 5 IGHD), without donor ids. Each sequence is given reverse-complemented: all 125 whose closest IMGT
  allele is in OGRDB align better reversed. VDJbase lacks some of them, so R1 is also reported without the genes
  they touch (the sensitivity analysis). Exact equality finds 97 of them among the cohort's VDJbase undocumented
  sequences, as many as containment with 90% overlap does.
"""
from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.reference import Reference, gene_name

SEGMENTS = ("v", "d", "j")
# The VDJbase column holding each segment's ungapped sequence, and the V's IMGT-gapped one.
REGION = {"v": "V-REGION", "d": "D-REGION", "j": "J-REGION"}
GAPPED = "V-REGION-GAPPED"
REQUIRED_COLUMNS = ("subject", "haplotype", "gene", "vdjbase_allele", "imgt_allele_match", "notes",
                    *REGION.values(), GAPPED)
# A VDJbase note with one of these flags a V-REGION defect: the allele is not functional.
DEFECT_NOTES = ("Stop codon", "cysteine not found", "Trp not found", "length mismatch in V-REGION")
# A D call less identical than this (percent) to its closest IMGT allele is dropped.
D_MIN_IDENTITY = 90.0
# The last IMGT position rearranged reads can tell V alleles apart by (Peres et al. 2023).
POSITION_318 = 318
# The sheet of Rodriguez et al. 2023's supplementary workbook that lists the paper's novel alleles.
SD3_SHEET = "Supplementary Data 3"
SD3_COLUMNS = ("closest IMGT allele", "Gene", "Allele sequence")

_NO_GAPS = str.maketrans("", "", ".-")
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
_INSERTION = re.compile(r"i\d+[acgtn]+", re.IGNORECASE)
_BLOCK = re.compile(r"(\d+)([acgtn]+)(\d+)", re.IGNORECASE)
_LABEL = re.compile(r"\bIG[HKL][VDJ][^\s,*()]*\*\d+")
_XLSX_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_XLSX_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_XLSX_PACKAGE = "{http://schemas.openxmlformats.org/package/2006/relationships}"


@dataclass(frozen=True)
class RealAllele:
    """One allele a donor carries on one haplotype, as genomic sequencing found it."""

    # VDJbase's id: the IUIS label of a documented allele, else the closest IMGT allele with its differences
    allele_id: str
    # the IUIS gene at the locus where it was found (VDJbase's ``gene``)
    gene: str
    segment: str  # "v" | "d" | "j"
    # the uppercase ungapped V-, D- or J-REGION
    sequence: str
    # the uppercase IMGT-gapped V-REGION ("." gap columns, "-" for a deleted base); None for D and J
    gapped: str | None
    # 1 or 2 for a phased haplotype; 0 for unphased, which is not homozygous
    haplotype: int
    functional: bool
    # the shared-reference alleles with exactly this sequence, sorted; () for an allele absent from it
    shared: tuple[str, ...]
    # the OGRDB IUIS labels with exactly this sequence, sorted
    iuis: tuple[str, ...]
    # percent identity to the closest IMGT allele, as VDJbase reports it; None when not reported
    imgt_identity: float | None = None

    def __post_init__(self):
        if self.segment not in SEGMENTS:
            raise ValueError(f"{self.allele_id}: segment {self.segment!r} is not one of {', '.join(SEGMENTS)}")
        if self.haplotype not in (0, 1, 2):
            raise ValueError(f"{self.allele_id}: haplotype {self.haplotype!r} is not 0 (unphased), 1 or 2")
        if not self.sequence:
            raise ValueError(f"{self.allele_id}: no sequence")


@dataclass(frozen=True)
class RealGenotype:
    """A donor's alleles from genomic sequencing, with the genes its sequencing captured and where they came from."""

    donor: str
    alleles: tuple[RealAllele, ...]
    # the IUIS genes the donor's genomic contigs span (``real.locus``); None until that is known
    captured: frozenset[str] | None = None
    # provenance: the file, its SHA-256, the OGRDB release and what the loader left out
    source: dict = field(default_factory=dict)

    def carried(self, gene: str) -> tuple[RealAllele, ...]:
        """The donor's alleles of IUIS gene ``gene``, in file order; () for a gene it does not carry."""
        return tuple(allele for allele in self.alleles if allele.gene == gene)

    def genes(self, segment: str) -> set[str]:
        """The IUIS genes the donor carries an allele of, functional or not, for segment ``"v"``, ``"d"`` or ``"j"``
        (empty when its sequencing gave no allele of that segment)."""
        if segment not in SEGMENTS:
            raise ValueError(f"segment {segment!r} is not one of {', '.join(SEGMENTS)}")
        return {allele.gene for allele in self.alleles if allele.segment == segment}

    def absent_genes(self) -> set[str]:
        """The genes with an allele absent from the shared reference (``shared`` empty), functional or not."""
        return {allele.gene for allele in self.alleles if not allele.shared}


@dataclass(frozen=True)
class IuisMap:
    """The shared reference's alleles and OGRDB's IUIS labels, matched by exact ungapped sequence."""

    # shared allele -> the IUIS labels with its sequence, sorted
    labels_by_shared: dict[str, tuple[str, ...]]
    # uppercase sequence -> the shared alleles with it, sorted
    shared_by_sequence: dict[str, tuple[str, ...]]
    # uppercase sequence -> the IUIS labels with it, sorted
    labels_by_sequence: dict[str, tuple[str, ...]]
    # the digest (``Reference.digest``) of the shared reference the map was built against
    reference_digest: str
    # provenance: the file, its SHA-256 and the germline set's id, name, reference, release and label count
    source: dict = field(default_factory=dict)

    @classmethod
    def from_ogrdb(cls, json_path, reference: Reference) -> "IuisMap":
        """The map of an OGRDB germline set in AIRR format (one ``GermlineSet`` whose ``allele_descriptions`` hold
        each ``label`` and its ``coding_sequence``) against ``reference``. ValueError for another file, an allele
        without a label or sequence, or a label given twice."""
        path = Path(json_path)
        try:
            (germline_set,) = json.loads(path.read_text(encoding="utf-8"))["GermlineSet"]
            descriptions = list(germline_set["allele_descriptions"])
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path} is not an OGRDB germline set in AIRR format (one GermlineSet with its "
                             f"allele_descriptions): {error!r}") from error
        by_sequence: dict[str, list[str]] = defaultdict(list)
        labels: set[str] = set()
        for description in descriptions:
            label = description.get("label") if isinstance(description, dict) else None
            sequence = (description.get("coding_sequence") or "").strip().upper() if label else ""
            if not label or not sequence:
                raise ValueError(f"{path}: an allele description without a label or a coding_sequence ({label!r})")
            if label in labels:
                raise ValueError(f"{path}: label {label} is given twice")
            labels.add(label)
            by_sequence[sequence].append(label)
        labels_by_sequence = {sequence: tuple(sorted(names)) for sequence, names in by_sequence.items()}
        shared: dict[str, list[str]] = defaultdict(list)
        for name, sequence in reference.sequences().items():
            shared[sequence].append(name)
        source = {"path": str(path), "sha256": sha256_file(path),
                  **{key: germline_set.get(key) for key in ("germline_set_id", "germline_set_name",
                                                            "germline_set_ref", "release_version", "release_date")},
                  "labels": len(labels), "reference_digest": reference.digest()}
        return cls(labels_by_shared={name: labels_by_sequence.get(sequence, ())
                                     for name, sequence in reference.sequences().items()},
                   shared_by_sequence={sequence: tuple(sorted(names)) for sequence, names in shared.items()},
                   labels_by_sequence=labels_by_sequence, reference_digest=reference.digest(), source=source)

    def labels(self, shared_name: str) -> tuple[str, ...]:
        """The IUIS labels with the sequence of shared allele ``shared_name``, sorted; () for an unmapped one.
        KeyError for a name that is not in the shared reference."""
        try:
            return self.labels_by_shared[shared_name]
        except KeyError:
            raise KeyError(f"{shared_name} is not in the shared reference") from None

    def genes(self, shared_name: str) -> tuple[str, ...]:
        """The IUIS genes of ``labels(shared_name)``, each once, in label order (two for a paralog pair)."""
        return tuple(dict.fromkeys(gene_name(label) for label in self.labels(shared_name)))

    def shared_for_sequence(self, sequence: str) -> tuple[str, ...]:
        """The shared alleles whose sequence is exactly ``sequence`` (any case), sorted; () for none."""
        return self.shared_by_sequence.get(sequence.upper(), ())

    def labels_for_sequence(self, sequence: str) -> tuple[str, ...]:
        """The IUIS labels whose coding sequence is exactly ``sequence`` (any case), sorted; () for none."""
        return self.labels_by_sequence.get(sequence.upper(), ())

    def unmapped(self) -> tuple[str, ...]:
        """The shared alleles no IUIS label has the sequence of, sorted."""
        return tuple(sorted(name for name, labels in self.labels_by_shared.items() if not labels))


def load_vdjbase_csv(path, donor: str, reference: Reference, iuis_map: IuisMap) -> RealGenotype:
    """Donor ``donor``'s genotype from its VDJbase genomic per-sample CSV, as the module docstring describes, with
    each allele's ``shared`` and ``iuis`` names by exact sequence; D calls filtered by ``filter_d``.

    ``source`` holds ``kind``, ``path``, ``sha256``, ``sample`` (the file's stem, VDJbase's sample id), ``donor``,
    ``project``, ``rows``, ``duplicate_rows``, ``gapped_mismatch`` (V alleles whose gapped form is not their
    sequence), ``d_dropped`` (``[allele_id, reason]`` pairs), ``ogrdb`` (the IUIS map's release and SHA-256) and the
    ``reference_digest``. ValueError, naming the line, for a map built against another reference, a missing
    column, a row of another subject, a gene that is not V, D or J, an empty sequence, a haplotype other than 0, 1
    or 2, an identity that is not a number, or a file without rows."""
    path = Path(path)
    if iuis_map.reference_digest != reference.digest():
        raise ValueError(f"the IUIS map was built against another reference (digest {iuis_map.reference_digest[:12]}, "
                         f"not {reference.digest()[:12]})")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path} is not a VDJbase genomic sample CSV: no {', '.join(missing)} column")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} holds no allele")
    alleles: list[RealAllele] = []
    seen: set[RealAllele] = set()
    duplicates, mismatched = 0, []
    for line, row in enumerate(rows, start=2):
        if row["subject"] != donor:
            raise ValueError(f"{path} is the genotype of subject {row['subject']}, not {donor}")
        allele, gapped_mismatch = _row_allele(row, f"{path}: line {line} ({row['vdjbase_allele'] or 'no id'})",
                                              iuis_map)
        if allele in seen:
            duplicates += 1
            continue
        seen.add(allele)
        alleles.append(allele)
        if gapped_mismatch:
            mismatched.append(allele.allele_id)
    kept, dropped = filter_d(alleles)
    source = {"kind": "vdjbase_genomic_csv", "path": str(path), "sha256": sha256_file(path), "sample": path.stem,
              "donor": donor, "project": rows[0].get("project"), "rows": len(rows), "duplicate_rows": duplicates,
              "gapped_mismatch": mismatched, "d_dropped": [[allele_id, reason] for allele_id, reason in dropped],
              "ogrdb": {key: iuis_map.source.get(key)
                        for key in ("germline_set_id", "germline_set_ref", "release_version", "release_date",
                                    "sha256")},
              "reference_digest": reference.digest()}
    return RealGenotype(donor=donor, alleles=tuple(kept), source=source)


def _row_allele(row: dict, where: str, iuis_map: IuisMap) -> tuple[RealAllele, bool]:
    """A CSV row's allele, and whether its gapped V-REGION was left out for not being its sequence."""
    def cell(column: str) -> str:
        return (row.get(column) or "").strip()  # a short row leaves its last columns None

    gene = cell("gene")
    segment = gene[3:4].lower()
    if segment not in SEGMENTS:
        raise ValueError(f"{where}: gene {gene!r} is not a V, D or J gene")
    sequence = cell(REGION[segment]).upper()
    if not sequence:
        raise ValueError(f"{where}: no {REGION[segment]} sequence")
    try:
        haplotype = int(cell("haplotype"))
    except ValueError:
        haplotype = -1
    if haplotype not in (0, 1, 2):
        raise ValueError(f"{where}: haplotype {cell('haplotype')!r} is not 0 (unphased), 1 or 2")
    text = cell("imgt_allele_match")
    try:
        identity = float(text) if text else None
    except ValueError:
        raise ValueError(f"{where}: imgt_allele_match {text!r} is not a number") from None
    gapped = (cell(GAPPED).upper() or None) if segment == "v" else None
    mismatch = gapped is not None and gapped.translate(_NO_GAPS) != sequence
    allele = RealAllele(allele_id=cell("vdjbase_allele"), gene=gene, segment=segment, sequence=sequence,
                        gapped=None if mismatch else gapped, haplotype=haplotype,
                        functional=not any(note in cell("notes") for note in DEFECT_NOTES),
                        shared=iuis_map.shared_for_sequence(sequence), iuis=iuis_map.labels_for_sequence(sequence),
                        imgt_identity=identity)
    return allele, mismatch


def filter_d(alleles: Iterable[RealAllele]) -> tuple[list[RealAllele], list[tuple[str, str]]]:
    """``(kept, dropped)``: ``alleles`` without the D calls the module docstring's rule drops, in order, and each
    dropped call's ``(allele_id, reason)``. V and J alleles are always kept, and so is a D call without an identity."""
    kept, dropped = [], []
    for allele in alleles:
        reasons = _d_reasons(allele) if allele.segment == "d" else []
        if reasons:
            dropped.append((allele.allele_id, "; ".join(reasons)))
        else:
            kept.append(allele)
    return kept, dropped


def _d_reasons(allele: RealAllele) -> list[str]:
    reasons = []
    tokens = _indel_tokens(allele.allele_id)
    if tokens:
        reasons.append(f"gap-indel token{'s' if len(tokens) > 1 else ''} {', '.join(tokens)}")
    if allele.imgt_identity is not None and allele.imgt_identity < D_MIN_IDENTITY:
        reasons.append(f"{allele.imgt_identity:g}% identity to the closest IMGT allele, under {D_MIN_IDENTITY:g}%")
    return reasons


def _indel_tokens(allele_id: str) -> list[str]:
    """The tokens of a VDJbase allele id (``_``-separated, after the allele number) that change its length: one with
    a ``-`` (a deleted base: ``g8-``, ``8---a11``), an insertion (``i15ac``), or a block of replaced positions whose
    bases do not fill its span (``17accagc22`` replaces positions 17 to 22 with six bases: a substitution block)."""
    if "*" not in allele_id:
        return []
    found = []
    for token in allele_id.split("*", 1)[1].split("_")[1:]:
        block = _BLOCK.fullmatch(token)
        if ("-" in token or _INSERTION.fullmatch(token)
                or (block and len(block[2]) != int(block[3]) - int(block[1]) + 1)):
            found.append(token)
    return found


def merge_identical(reference: Reference) -> dict[str, frozenset[str]]:
    """Every allele of ``reference`` -> the alleles of its gene letter with exactly its sequence, itself included."""
    merged: dict[str, frozenset[str]] = {}
    for table in reference.alleles.values():
        groups: dict[str, set[str]] = defaultdict(set)
        for name, sequence in table.items():
            groups[sequence].add(name)
        merged.update({name: frozenset(groups[sequence]) for name, sequence in table.items()})
    return merged


def upto(gapped: str, position: int = POSITION_318) -> str:
    """The uppercase ungapped sequence of IMGT-gapped ``gapped`` through IMGT position ``position`` (its first
    ``position`` columns; ``.`` and ``-`` are gaps). ValueError for a position below 1."""
    if position < 1:
        raise ValueError(f"IMGT position {position} is below 1")
    return gapped[:position].translate(_NO_GAPS).upper()


class Sd3Allele(NamedTuple):
    """One of the paper's novel alleles (Supplementary Data 3)."""

    # the first IMGT allele the sheet names as closest, without species or functionality ("IGHV1-69*01"); "" for NA
    closest: str
    # the uppercase allele sequence, forward (the sheet gives it reverse-complemented)
    sequence: str
    # the IUIS gene where the paper found it (the sheet's ``Gene``; it may differ from the closest allele's gene)
    gene: str
    # every IMGT allele the sheet names as closest, in its order
    closest_all: tuple[str, ...] = ()


def sd3_novel(path) -> list[Sd3Allele]:
    """The novel alleles of Supplementary Data 3, from the supplementary workbook (``.xlsx``; sheet ``SD3_SHEET``,
    read with the standard library) or a tab-separated export of that sheet, each sequence reverse-complemented to
    its forward strand. Blank rows are skipped. ValueError for a file without the sheet or its ``SD3_COLUMNS``, or a
    row without a gene or with a sequence that is not nucleotides."""
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        rows = _xlsx_sheet(path, SD3_SHEET)
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
    header = [cell.strip() for cell in rows[0]] if rows else []
    if not all(column in header for column in SD3_COLUMNS):
        raise ValueError(f"{path}: {SD3_SHEET} has no {', '.join(repr(c) for c in SD3_COLUMNS)} columns")
    closest_at, gene_at, sequence_at = (header.index(column) for column in SD3_COLUMNS)
    novel = []
    for number, row in enumerate(rows[1:], start=2):
        cells = [cell.strip() for cell in row]
        if not any(cells):
            continue
        cells += [""] * (len(header) - len(cells))
        gene, sequence = cells[gene_at], cells[sequence_at].upper()
        if not gene or not sequence or set(sequence) - set("ACGTN"):
            raise ValueError(f"{path}: {SD3_SHEET} row {number} has no gene, or a sequence that is not nucleotides")
        closest = tuple(dict.fromkeys(_LABEL.findall(cells[closest_at])))
        novel.append(Sd3Allele(closest=closest[0] if closest else "", sequence=sequence.translate(_COMPLEMENT)[::-1],
                               gene=gene, closest_all=closest))
    return novel


def sd3_only(genotypes: Iterable[RealGenotype], sd3: Iterable[Sd3Allele]) -> set[str]:
    """The genes touched by a novel allele of ``sd3`` whose sequence is in none of ``genotypes``' alleles: its own
    ``gene`` and the gene of every closest allele the sheet names, so the sensitivity analysis leans to excluding
    more. Only the donors given count, so a subset of the cohort can only add genes; the D calls ``filter_d``
    dropped are not among a genotype's alleles."""
    carried = {allele.sequence for genotype in genotypes for allele in genotype.alleles}
    touched: set[str] = set()
    for allele in sd3:
        if allele.sequence not in carried:
            touched.add(allele.gene)
            # ``closest_all`` defaults to (): a record built with only ``closest`` still names that allele's gene
            closest = allele.closest_all or ((allele.closest,) if allele.closest else ())
            touched.update(gene_name(label) for label in closest)
    return touched


def _xlsx_sheet(path: Path, name: str) -> list[list[str]]:
    """The cells of sheet ``name`` of an .xlsx workbook as text, row by row (a missing cell is ``""``): a shared
    string, an inline string, or the value as written. ValueError for a file that is not a readable workbook or
    has no such sheet."""
    try:
        with zipfile.ZipFile(path) as archive:
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            sheet = next((s for s in workbook.iter(f"{_XLSX_MAIN}sheet") if s.get("name") == name), None)
            if sheet is None:
                raise ValueError(f"{path} has no sheet {name!r}")
            relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            targets = {r.get("Id"): r.get("Target") for r in relations.iter(f"{_XLSX_PACKAGE}Relationship")}
            target = targets[sheet.get(f"{_XLSX_REL}id")].lstrip("/")
            target = target if target.startswith("xl/") else f"xl/{target}"
            shared = []
            if "xl/sharedStrings.xml" in archive.namelist():
                for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).iter(f"{_XLSX_MAIN}si"):
                    shared.append("".join(text.text or "" for text in item.iter(f"{_XLSX_MAIN}t")))
            worksheet = ET.fromstring(archive.read(target))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as error:
        raise ValueError(f"{path} is not a readable .xlsx workbook: {error!r}") from error
    rows = []
    for row in worksheet.iter(f"{_XLSX_MAIN}row"):
        cells: dict[int, str] = {}
        for cell in row.iter(f"{_XLSX_MAIN}c"):
            kind, value = cell.get("t"), cell.find(f"{_XLSX_MAIN}v")
            if kind == "s" and value is not None:
                text = shared[int(value.text)]
            elif kind == "inlineStr":
                text = "".join(part.text or "" for part in cell.iter(f"{_XLSX_MAIN}t"))
            else:
                text = value.text or "" if value is not None else ""
            reference = cell.get("r")
            cells[_column(reference) if reference else max(cells, default=-1) + 1] = text
        rows.append([cells.get(index, "") for index in range(max(cells, default=-1) + 1)])
    return rows


def _column(reference: str) -> int:
    """The 0-based column of a cell reference (``"B7"`` -> 1, ``"AA1"`` -> 26)."""
    index = 0
    for letter in re.match(r"[A-Z]+", reference.upper()).group(0):
        index = index * 26 + ord(letter) - 64
    return index - 1
