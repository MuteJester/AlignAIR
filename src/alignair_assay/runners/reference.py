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

