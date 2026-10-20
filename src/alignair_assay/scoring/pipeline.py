"""Score one tool against observable truth: load and check the truth, match predictions to cases, score every case.

Predictions are matched to truth by AIRR ``sequence_id`` == truth ``case_id``. Excluded truths are not scored and
their predictions are ignored; a truth case without a prediction is scored as missing (never dropped). The truth
reference of suites built in a run directory is ``truth_reference``.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from alignair_assay.airr.reader import load_airr_tsv
from alignair_assay.core.genotype import DonorGenotype
from alignair_assay.core.jsonl import read_jsonl, write_jsonl
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.prediction import Prediction
from alignair_assay.core.reference import Reference
from alignair_assay.metrics import CaseScores
from alignair_assay.metrics.case import score_case
from alignair_assay.metrics.genotype import donor_genotype, genotype_metrics
from alignair_assay.metrics.groups import clone_agreement
from alignair_assay.truth.io import read_truth_header, read_truth_jsonl
from alignair_assay.truth.model import ObservableTruth


def score_predictions(truths: list[ObservableTruth], preds: Mapping[str, Prediction | None],
                      reference: Reference, genotypes: Mapping[str, DonorGenotype] | None = None
                      ) -> dict[str, CaseScores]:
    """Per-case scores of one tool over the non-excluded truths, in truth order, keyed by case id.

    ``preds`` maps case ids to the tool's predictions; a case id that is absent, or mapped to None, is scored as a
    missing prediction. Entries for unknown or excluded case ids are ignored. Each case gets its own metrics
    (``metrics.case.score_case``) merged with its group metrics (``metrics.groups.clone_agreement``), which
    depend on the predictions for the rest of its clone. With ``genotypes`` (donor subject id -> genotype, e.g.
    ``Donor.view()`` of a suite's donors), every case with ``groups["subject"]`` also gets the genotype metrics of
    its donor (``metrics.genotype.genotype_metrics``); ValueError, naming the case, when its subject has no
    genotype. Every metric the assay reports for a single tool comes from here, so the oracle check holds it to the
    same rule as the pipeline.
    """
    scores = {t.case_id: score_case(t, preds.get(t.case_id), reference) for t in truths if t.excluded is None}
    for case_id, group_scores in clone_agreement(truths, preds).items():
        scores[case_id].update(group_scores)
    if genotypes is not None:
        for t in truths:
            if t.excluded is None and "subject" in t.groups:
                scores[t.case_id].update(genotype_metrics(t, preds.get(t.case_id), donor_genotype(genotypes, t)))
    return scores


def load_predictions(truths: list[ObservableTruth], predictions_path) -> tuple[dict[str, Prediction], dict]:
    """One tool's AIRR TSV as predictions by ``sequence_id``, plus its prediction/truth matching counts over
    ``truths``: ``truth_cases``, ``excluded``, ``scored`` (the non-excluded truths), the file's ``rows``,
    ``predictions``, ``duplicates`` and ``invalid_rows``, the scored cases without a prediction (``missing``),
    predictions for no truth case (``extra``) and predictions for excluded cases (``excluded_ignored``).

    A file whose ``sequence_id``s match no scored case raises ValueError (a renamed or wrong file, not a tool that
    missed every read).
    """
    preds, stats = load_airr_tsv(predictions_path)
    scored_ids = [t.case_id for t in truths if t.excluded is None]
    if not any(cid in preds for cid in scored_ids):
        example = f" (e.g. {next(iter(preds))!r} vs {scored_ids[0]!r})" if preds and scored_ids else ""
        raise ValueError(f"{predictions_path}: no prediction sequence_id matches a truth case_id{example}")
    truth_ids, scored = {t.case_id for t in truths}, set(scored_ids)
    matching = {
        "truth_cases": len(truths), "excluded": len(truths) - len(scored_ids), "scored": len(scored_ids),
        "rows": stats["rows"], "predictions": len(preds),
        "duplicates": stats["duplicates"], "invalid_rows": stats["invalid_rows"],
        "missing": sum(cid not in preds for cid in scored_ids),
        "extra": sum(pid not in truth_ids for pid in preds),
        "excluded_ignored": sum(pid in truth_ids and pid not in scored for pid in preds),
    }
    return preds, matching


def score_tool(truths: list[ObservableTruth], predictions_path, reference: Reference,
               genotypes: Mapping[str, DonorGenotype] | None = None) -> tuple[dict[str, CaseScores], dict]:
    """Per-case scores of one tool's AIRR TSV over the non-excluded truths (``score_predictions``, with the donor
    ``genotypes`` when given), plus the prediction/truth matching counts (``load_predictions``, which raises
    ValueError for a file matching no scored case)."""
    preds, matching = load_predictions(truths, predictions_path)
    return score_predictions(truths, preds, reference, genotypes), matching


def load_truth(path, reference_name: str, reference: Reference) -> tuple[dict, list[ObservableTruth]]:
    """Truth header and records; the header's reference digest must match ``reference`` and case ids must be
    unique (ValueError otherwise)."""
    header = read_truth_header(path)
    if header is None or "reference_digest" not in header:
        raise ValueError(f"{path} records no reference digest; rebuild it with the truth command")
    digest = reference.digest()
    if header["reference_digest"] != digest:
        raise ValueError(f"{path} was built against reference {header.get('reference')!r} (reference digest "
                         f"{header['reference_digest'][:12]}), but --reference {reference_name} has reference digest "
                         f"{digest[:12]}; score with the reference the truth was built from")
    truths = read_truth_jsonl(path)
    check_unique(truths, path)
    return header, truths


def truth_reference(layout: RunLayout, suites: list[str]) -> Reference:
    """The reference the named suites' truth was built against, from its truth headers. ValueError when a suite
    is not built (no ``build.json``), the suites name different references, or the reference's dataconfig no
    longer has the recorded digest."""
    recorded = {}
    for suite in suites:
        if not layout.build_json(suite).is_file():
            raise ValueError(f"suite {suite} is not built in {layout.root} (no {layout.build_json(suite).name}); "
                             f"build it first")
        header = read_truth_header(layout.truth(suite)) or {}
        recorded[suite] = (header.get("reference"), header.get("reference_digest"))
    if len(set(recorded.values())) > 1:
        listed = ", ".join(f"{suite}: {name} ({digest and digest[:12]})" for suite, (name, digest) in recorded.items())
        raise ValueError(f"the suites were built against different references: {listed}")
    name, digest = next(iter(recorded.values()))
    if name is None or digest is None:
        raise ValueError(f"{layout.truth(suites[0])} records no reference digest; rebuild the suite")
    reference = Reference.from_dataconfig(name)
    if reference.digest() != digest:
        raise ValueError(f"the truth was built against {name} with digest {digest[:12]}, but {name} now has digest "
                         f"{reference.digest()[:12]}; rebuild the suites")
    return reference


def check_unique(truths: list[ObservableTruth], source) -> None:
    """Raise ValueError naming ``source`` when two truths share a case id."""
    dup = [cid for cid, n in Counter(t.case_id for t in truths).items() if n > 1]
    if dup:
        raise ValueError(f"{source}: duplicate case ids, e.g. {dup[0]!r} ({len(dup)} in total)")


def write_scores(scores: dict[str, CaseScores], path: str | Path) -> None:
    """One ``{"case_id", "scores"}`` object per scored case; ``nan`` (applicable, not covered) becomes null."""
    write_jsonl(({"case_id": cid, "scores": {k: v if math.isfinite(v) else None for k, v in case.items()}}
                 for cid, case in scores.items()), path)


def read_scores(path: str | Path) -> dict[str, CaseScores]:
    """Inverse of ``write_scores``: per-case scores in file order, with null read back as ``nan``."""
    return {row["case_id"]: {k: math.nan if v is None else float(v) for k, v in row["scores"].items()}
            for row in read_jsonl(path)}
