"""The genotype arm: each donor's reads annotated by a tool restricted to that donor's genotype, then merged.

A suite with donors (``suites/<S>/donors.json``) is split by the ``subject`` group of its cases. Each donor with
reads gets its own FASTA and its own tool run, restricted (``Runner.run(..., alleles=...)``) to the reference
alleles the donor carries (``Donor.tool_genotype``, where a novel allele is given as its base). The donors' TSVs
are then merged into the arm's ``predictions.tsv``, which is scored like any other run's.
"""
from __future__ import annotations

import json
from pathlib import Path

from alignair_assay.core.cases import read_cases, read_fasta
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.reference import Reference
from alignair_assay.runners.base import Runner, RunResult, write_run_json
from alignair_assay.suites.donors import donors_from_json

ARM = "genotype"


def run_genotype_arm(runner: Runner, layout: RunLayout, suite: str, reference: Reference, *,
                     tool: str | None = None) -> RunResult:
    """Run a prepared ``runner`` on ``suite`` donor by donor, and record the arm in the layout. Donors run in
    ``donors.json`` order, and those without cases are skipped. Donor ``<subject>``'s reads go to
    ``layout.donor_reads(tool, suite, "genotype", subject)`` in suite order, its predictions to
    ``donor_predictions`` and its logs to ``donor_logs_dir``. Each run is restricted to the flattened
    ``donor.tool_genotype(reference)``. The runs stop at the first donor whose run fails, and the arm is then
    recorded with that return code, the ``failed_donor`` and the run's ``error`` line, and no merged
    predictions. Otherwise the donors' TSVs are merged into ``layout.predictions(tool, suite, "genotype")`` by
    ``merge_tsvs``, which raises ValueError naming the donor on a header that differs. The arm's ``RunResult``
    sums the donors' wall seconds and takes their largest peak RSS (None when none was measured), and its
    commands are the donors' in order. Its provenance holds, per donor, the ``reads`` count,
    wall seconds, peak RSS, return code, ``alleles`` per gene letter and the run's provenance. It also holds the
    ``skipped_donors``, ``donors_json`` (path and SHA-256) and ``reads`` (the suite's reads, as ``run_suite``
    records them). ``run.json`` and the old predictions are removed first and ``run.json`` is written last, so an
    arm that raises leaves no record. ``tool`` names the run directory (default ``runner.name``).
    FileNotFoundError when the suite has no ``donors.json``; ValueError when a read has no case, a case has no
    ``subject``, or a subject is not a donor.
    """
    tool = tool or runner.name
    run_json = layout.run_json(tool, suite, ARM)
    run_json.unlink(missing_ok=True)
    predictions = layout.predictions(tool, suite, ARM)
    predictions.unlink(missing_ok=True)
    if not layout.donors(suite).is_file():
        raise FileNotFoundError(f"suite {suite} has no donors ({layout.donors(suite)}); the genotype arm needs a "
                                "suite simulated from donors")
    donors = donors_from_json(json.loads(layout.donors(suite).read_text(encoding="utf-8")))
    reads = reads_by_donor(layout, suite, [donor.subject_id for donor in donors])

    results: list[tuple[str, RunResult]] = []
    record: dict[str, dict] = {}
    skipped, failed = [], None
    for donor in donors:
        subject = donor.subject_id
        if not reads[subject]:
            skipped.append(subject)
            continue
        fasta = layout.donor_reads(tool, suite, ARM, subject)
        fasta.parent.mkdir(parents=True, exist_ok=True)
        with open(fasta, "w", encoding="utf-8") as fh:
            fh.writelines(f">{read_id}\n{sequence}\n" for read_id, sequence in reads[subject])
        out_tsv = layout.donor_predictions(tool, suite, ARM, subject)
        out_tsv.unlink(missing_ok=True)
        genotype = donor.tool_genotype(reference)
        result = runner.run(fasta, out_tsv, suite=suite, arm=ARM,
                            log_dir=layout.donor_logs_dir(tool, suite, ARM, subject),
                            alleles=[name for names in genotype.values() for name in names])
        results.append((subject, result))
        record[subject] = {"reads": len(reads[subject]), "wall_seconds": result.wall_seconds,
                           "peak_rss_kb": result.peak_rss_kb, "returncode": result.returncode,
                           "alleles": {letter: len(names) for letter, names in genotype.items()},
                           "provenance": result.provenance}
        if result.returncode != 0:
            failed = (subject, result)
            break

    provenance = {"donors": record, "skipped_donors": skipped,
                  "donors_json": {"path": str(layout.donors(suite)), "sha256": sha256_file(layout.donors(suite))},
                  "reads": {"path": str(layout.reads(suite)), "sha256": sha256_file(layout.reads(suite))}}
    if failed is None:
        merge_tsvs([(subject, Path(result.predictions), [read_id for read_id, _ in reads[subject]])
                    for subject, result in results], predictions)
        returncode = 0
    else:
        provenance.update(failed_donor=failed[0], error=failed[1].provenance.get("error"))
        returncode = failed[1].returncode
    measured = [result.peak_rss_kb for _, result in results if result.peak_rss_kb is not None]
    arm = RunResult(tool=tool, suite=suite, arm=ARM, predictions=str(predictions),
                    n_reads=sum(result.n_reads for _, result in results),
                    wall_seconds=sum(result.wall_seconds for _, result in results),
                    peak_rss_kb=max(measured) if measured else None, returncode=returncode,
                    commands=[command for _, result in results for command in result.commands],
                    provenance=provenance)
    write_run_json(arm, run_json)
    return arm


def reads_by_donor(layout: RunLayout, suite: str, subjects: list[str]) -> dict[str, list[tuple[str, str]]]:
    """The suite's reads (``(id, sequence)``, in suite order) by donor subject id, every donor of ``subjects``
    included. A read's donor is its case's ``subject`` group. ValueError when a read is not a case of the suite,
    a case has no subject, or the subject is not among ``subjects``."""
    subject_of = {case["case_id"]: case["groups"].get("subject") for case in read_cases(layout.cases(suite))}
    by_donor: dict[str, list[tuple[str, str]]] = {subject: [] for subject in subjects}
    for read_id, sequence in read_fasta(layout.reads(suite)):
        if read_id not in subject_of:
            raise ValueError(f"{suite}: read {read_id} is not a case of {layout.cases(suite)}")
        subject = subject_of[read_id]
        if subject is None:
            raise ValueError(f"{suite}: case {read_id} has no subject, so it has no donor genotype")
        if subject not in by_donor:
            raise ValueError(f"{suite}: the subject of case {read_id}, {subject} is not a donor of "
                             f"{layout.donors(suite)}")
        by_donor[subject].append((read_id, sequence))
    return by_donor


def merge_tsvs(parts: list[tuple[str, Path, list[str]]], out_tsv: Path) -> int:
    """Concatenate the donors' AIRR TSVs into ``out_tsv`` under their shared header; return the rows written.
    ``parts`` are ``(subject, tsv, read ids)``. Rows are copied verbatim, in order. ValueError naming the donor
    when its TSV is empty, its header differs from the first donor's, or a row's ``sequence_id`` is not one of its
    reads. Since the donors' reads are disjoint, no case is then written by two donors. ``out_tsv`` is removed
    when the merge fails.
    """
    out_tsv = Path(out_tsv)
    header, column, rows = None, None, 0
    try:
        with open(out_tsv, "w", encoding="utf-8") as out:
            for subject, path, read_ids in parts:
                with open(path, encoding="utf-8") as fh:
                    first = fh.readline().rstrip("\r\n")
                    if not first:
                        raise ValueError(f"donor {subject}: {path} has no header")
                    if header is None:
                        header = first
                        names = header.split("\t")
                        if "sequence_id" not in names:
                            raise ValueError(f"donor {subject}: {path} has no sequence_id column")
                        column = names.index("sequence_id")
                        out.write(header + "\n")
                    elif first != header:
                        raise ValueError(f"donor {subject}: {path} has another header than the first donor's "
                                         f"({_header_difference(header, first)})")
                    allowed = set(read_ids)
                    for line in fh:
                        if not line.strip():
                            continue
                        fields = line.rstrip("\r\n").split("\t")
                        read_id = fields[column].strip() if column < len(fields) else ""
                        if read_id not in allowed:
                            raise ValueError(f"donor {subject}: {path} has a row for {read_id!r}, which is not one "
                                             "of its reads")
                        out.write(line if line.endswith("\n") else line + "\n")
                        rows += 1
    except BaseException:
        out_tsv.unlink(missing_ok=True)
        raise
    return rows


def _header_difference(expected: str, found: str) -> str:
    """The first column where two TSV headers differ."""
    a, b = expected.split("\t"), found.split("\t")
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"column {index + 1} is {y!r}, expected {x!r}"
    longer, extra = ("expected", a[len(b):]) if len(a) > len(b) else ("found", b[len(a):])
    return f"{len(extra)} more column(s) {longer}: {extra[:3]}"
