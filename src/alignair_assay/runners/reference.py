"""The shared reference: one germline FASTA given to every tool, checked against the truth reference.

Every tool must annotate against the reference the truth was built from; otherwise its calls and germline
coordinates are scored against alleles it never saw. A tool's reference (written here, or exported by the tool)
is therefore verified allele by allele before anything runs.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.reference import Reference

_SEGMENTS = ("v", "d", "j")
# IgBLAST's chain type of a J allele, by the locus its name starts with (its auxiliary file's third column).
_J_CHAIN_TYPES = {"IGHJ": "JH", "IGKJ": "JK", "IGLJ": "JL", "TRAJ": "JA", "TRBJ": "JB", "TRDJ": "JD", "TRGJ": "JG"}


def igblast_auxiliary_lines(reference: Reference, alleles: Iterable[str] | None = None) -> list[str]:
    """IgBLAST's auxiliary-file lines (J reading frames and CDR3 ends) for the reference's J alleles, in name order
    and restricted to ``alleles`` when given, from the reference's own J anchors (the Trp/Phe codon start the truth
    uses): ``name``, the first coding frame start (``anchor % 3``), the chain type (``JH``, ``JK``, ``JL``, ``JA``,
    ``JB``, ``JD``, ``JG`` by the name's locus), the CDR3 stop (``anchor - 1``) and the bases past the last full
    codon, tab-separated and 0-based like the release's ``human_gl.aux``. A J allele without an anchor is left out
    (IgBLAST then reports no junction for it, as the truth defines none); one of no known locus raises ValueError."""
    lines = []
    for name in _selected(reference, alleles):
        if reference.gene_letter(name) != "j" or name not in reference.anchors:
            continue
        chain = _J_CHAIN_TYPES.get(name[:4])
        if chain is None:
            raise ValueError(f"J allele {name} is of no locus IgBLAST knows ({', '.join(_J_CHAIN_TYPES)})")
        anchor = reference.anchors[name]
        frame = anchor % 3
        lines.append(f"{name}\t{frame}\t{chain}\t{anchor - 1}\t{(len(reference.seq(name)) - frame) % 3}")
    return lines


def receptor_type(reference: Reference) -> str:
    """``"Ig"`` for a reference of immunoglobulin alleles, ``"TCR"`` for one of T-cell receptor alleles (IgBLAST's
    ``-ig_seqtype``), by the allele names; ValueError for a reference holding both."""
    kinds = {"TCR" if name.startswith("TR") else "Ig" for table in reference.alleles.values() for name in table}
    if len(kinds) > 1:
        raise ValueError("the reference holds both immunoglobulin and T-cell receptor alleles: IgBLAST searches one "
                         "receptor type per run")
    return kinds.pop() if kinds else "Ig"


def write_reference_fasta(reference: Reference, path, alleles: Iterable[str] | None = None) -> None:
    """Write the reference's alleles as FASTA: names in sorted order, each sequence on one line. ``alleles``
    restricts the file to those names; a name that is not in the reference raises ValueError."""
    with open(path, "w", encoding="utf-8") as fh:
        for name in _selected(reference, alleles):
            fh.write(f">{name}\n{reference.seq(name)}\n")


def write_segment_fastas(reference: Reference, directory, alleles: Iterable[str] | None = None) -> dict[str, Path]:
    """Write one FASTA per gene letter of the reference, ``germline_{v,d,j}.fasta`` in ``directory`` (created
    when missing), and return their paths by letter. ``alleles`` restricts every file to those names; a name that
    is not in the reference, or a letter left with no allele, raises ValueError."""
    directory = Path(directory)
    selected = _selected(reference, alleles)
    by_letter = {g: [name for name in selected if reference.gene_letter(name) == g]
                 for g in _SEGMENTS if g in reference.alleles}
    empty = [g for g, names in by_letter.items() if not names]
    if empty:
        raise ValueError(f"the selected alleles include no {empty[0]} allele")
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for g, names in by_letter.items():
        paths[g] = directory / f"germline_{g}.fasta"
        write_reference_fasta(reference, paths[g], alleles=names)
    return paths


def verify_reference(tool_fasta, truth: Reference) -> Reference:
    """The reference in ``tool_fasta``, when it holds exactly the truth reference's alleles and sequences (equal
    digests). Otherwise ValueError naming both references' allele counts and the first allele, in name order,
    that is missing from the tool's reference, extra in it, or has a different sequence there."""
    tool = Reference.from_fasta(tool_fasta)
    if tool.digest() == truth.digest():
        return tool
    raise ValueError(f"{tool_fasta} is not the truth reference: it has {tool.counts_text()}, the truth reference "
                     f"has {truth.counts_text()}; first difference: {tool.first_difference(truth, 'tool')}")


def ensure_shared_reference(layout: RunLayout, truth: Reference) -> Path:
    """Make the run's shared reference (``layout.reference_fasta``) the truth reference, and return its path.

    An existing file (for example a tool's exported reference) is verified with ``verify_reference``, which raises
    on any difference; otherwise the file is written from the truth reference. Either way ``reference.json`` is
    then written with the ``digest``, the allele ``counts`` per gene letter, the ``source`` (``"truth"`` when
    written here, ``"existing"`` when verified) and the file's ``sha256``. The old ``reference.json`` is removed
    first, so a refused reference leaves no record.
    """
    layout.reference_json.unlink(missing_ok=True)
    path = layout.reference_fasta
    if path.is_file():
        verify_reference(path, truth)
        source = "existing"
    else:
        layout.reference_dir.mkdir(parents=True, exist_ok=True)
        write_reference_fasta(truth, path)
        source = "truth"
    record = {"digest": truth.digest(), "counts": truth.counts(), "source": source, "sha256": sha256_file(path)}
    layout.reference_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def _selected(reference: Reference, alleles: Iterable[str] | None) -> list[str]:
    """Sorted allele names: every allele of the reference, or the given ones (ValueError for an unknown name)."""
    if alleles is None:
        return sorted(reference.sequences())
    names = sorted(set(alleles))
    unknown = [name for name in names if not reference.has(name)]
    if unknown:
        raise ValueError(f"allele {unknown[0]!r} is not in the reference ({len(unknown)} unknown in total)")
    return names
