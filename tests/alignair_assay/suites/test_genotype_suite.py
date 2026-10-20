"""S8 (genotype): cohort reads over donors with heterozygous and deleted genes, scored against each read's donor
genotype."""
import json
from dataclasses import replace

import pytest

pytest.importorskip("GenAIRR")

from alignair_assay.core.cases import read_cases  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.reference import gene_name  # noqa: E402
from alignair_assay.metrics.genotype import excess_alleles  # noqa: E402
from alignair_assay.scoring.pipeline import score_predictions  # noqa: E402
from alignair_assay.suites import get_suite  # noqa: E402
from alignair_assay.suites.donors import DonorParams, donors_from_json  # noqa: E402
from alignair_assay.testing import assert_oracle_perfect  # noqa: E402
from alignair_assay.truth.io import read_truth_jsonl  # noqa: E402
from alignair_assay.truth.oracle import oracle_prediction  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402
from alignair_assay.workflow.scoring import read_genotypes  # noqa: E402

REFERENCE = "HUMAN_IGH_OGRDB"
GENOTYPE_METRICS = {"v.outside_genotype", "v.deleted_gene_call", "j.outside_genotype"}


def test_s8_build_and_oracle_perfect(tmp_path, reference):
    suite = get_suite("S8")
    assert (suite.slug, suite.arms) == ("genotype", ("shared", "genotype"))
    assert suite.primary == ("v.outside_genotype", "v.deleted_gene_call", "j.outside_genotype", "v.top1_in_set",
                             "clone_key.exact")
    (stratum,) = suite.strata
    assert (stratum.name, stratum.n, stratum.cohort, stratum.rc_fraction) == ("cohort", 3000, True, 0.5)
    assert (suite.donors.n, suite.donors.tag, suite.donors.params) == (10, "S8", DonorParams())
    assert suite.purpose.endswith(".") and suite.purpose.count(". ") == 0

    layout = RunLayout(tmp_path)
    build = build_suite("S8", layout, reference_name=REFERENCE, split="dev", seed=17, scale=0.02, workers=1)
    cases, truths = read_cases(layout.cases("S8")), read_truth_jsonl(layout.truth("S8"))
    assert len(cases) == len(truths) == 60 and build["truth"]["excluded"] == {}
    assert all(case["record"]["end_loss_5_length"] == 0 for case in cases)  # the full design
    donors = donors_from_json(json.loads(layout.donors("S8").read_text()))
    genotypes = read_genotypes(layout.donors("S8"))
    assert genotypes == {donor.subject_id: donor.view() for donor in donors} and len(genotypes) == 10
    assert not any(donor.novel for donor in donors) and any(donor.deleted for donor in donors)

    scores = assert_oracle_perfect(truths, reference, genotypes=genotypes)
    emitted = {metric for case in scores.values() for metric in case}
    assert set(suite.primary) <= emitted, set(suite.primary) - emitted
    # Full-length reads: every read gets the genotype metrics against its own donor.
    assert all(GENOTYPE_METRICS <= set(case) for case in scores.values())
    # Without genotypes the pipeline emits none of them.
    plain = assert_oracle_perfect(truths, reference)
    assert not GENOTYPE_METRICS & {metric for case in plain.values() for metric in case}

    # The oracle calls each read's generative allele, which its donor carries: no excess alleles, even counting
    # every called allele (six reads per donor at this scale).
    oracle = {t.case_id: oracle_prediction(t) for t in truths}
    table = excess_alleles(truths, oracle, genotypes, min_reads=1)
    assert table["v"]["n"] >= 30 and table["j"]["n"] >= 10
    assert table["v"]["excess"] == table["j"]["excess"] == 0

    # A V call of one of the donor's deleted genes is caught.
    truth = next(t for t in truths if genotypes[t.groups["subject"]].deleted & {
        gene_name(name) for name in reference.alleles["v"]})
    genotype = genotypes[truth.groups["subject"]]
    deleted = next(name for name in sorted(reference.alleles["v"]) if gene_name(name) in genotype.deleted)
    pred = oracle[truth.case_id]
    wrong = replace(pred, segments={**pred.segments, "v": replace(pred.segments["v"], call=deleted,
                                                                   call_set=(deleted,))})
    case = score_predictions([truth], {truth.case_id: wrong}, reference, genotypes=genotypes)[truth.case_id]
    assert (case["v.outside_genotype"], case["v.deleted_gene_call"], case["v.top1_in_set"]) == (1.0, 1.0, 0.0)
