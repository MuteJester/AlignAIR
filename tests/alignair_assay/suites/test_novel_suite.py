"""S7 (novel alleles): SNP placement at invariant positions, novel donors, and truth re-based onto the base allele."""
import collections
import json
import math
import random
import re
from dataclasses import replace

import pytest

ga = pytest.importorskip("GenAIRR")

from alignair_assay.core.cases import make_case, read_cases  # noqa: E402
from alignair_assay.core.genotype import NovelAllele  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.prediction import PredSegment  # noqa: E402
from alignair_assay.core.reference import Reference, dataconfig, gene_name  # noqa: E402
from alignair_assay.metrics import is_informational, lower_is_better  # noqa: E402
from alignair_assay.metrics.case import score_case  # noqa: E402
from alignair_assay.metrics.novel import novel_metrics  # noqa: E402
from alignair_assay.suites import get_suite  # noqa: E402
from alignair_assay.suites.donors import (  # noqa: E402
    Donor,
    DonorParams,
    donors_from_json,
    donors_to_json,
    sample_donors,
)
from alignair_assay.suites.novel import MAX_ATTEMPTS, invariant_positions, make_novel_allele  # noqa: E402
from alignair_assay.testing import assert_oracle_perfect  # noqa: E402
from alignair_assay.truth.case import truth_from_case  # noqa: E402
from alignair_assay.truth.io import read_truth_jsonl  # noqa: E402
from alignair_assay.truth.oracle import oracle_prediction  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402

REFERENCE = "HUMAN_IGH_OGRDB"
CONFIG = dataconfig(REFERENCE)
NOVEL = DonorParams(novel_v_genes=25)
METRIC = "v_novel.snp_mutations"


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def _off_by_one_base(novel):
    """``novel`` with one more substitution than its ``snps`` list: intrinsically valid, inconsistent with its base."""
    extra = next(p for p in range(len(novel.sequence)) if p not in novel.snps)
    swapped = "C" if novel.sequence[extra] == "A" else "A"
    return replace(novel, sequence=novel.sequence[:extra] + swapped + novel.sequence[extra + 1:])


@pytest.fixture(scope="module")
def unmutated(reference):
    """Two novel donors and their full-length, unmutated cohort cases (300 per donor), with the truth built
    against each case's donor: ``(donors by subject, cases, truths)``."""
    donors = {d.subject_id: d for d in sample_donors(reference, 2, 5, NOVEL, config=CONFIG)}
    records = ga.Experiment.on(CONFIG).recombine().run_cohort(
        [d.genotype(CONFIG) for d in donors.values()], n_per_subject=300, seed=9, expose_provenance=True).records
    cases = [make_case(r, case_id=f"c{i}", suite="t", stratum="s", orientation=i % 2,
                       groups={"subject": r["subject_id"]}) for i, r in enumerate(records)]
    truths = [truth_from_case(c, reference, novel=donors[c["groups"]["subject"]].novel) for c in cases]
    return donors, cases, truths


def test_invariant_positions_hand_reference():
    base = "AAAAACCCGGGTTT"
    reference = Reference(
        alleles={"v": {"IGHV1-1*01": base,
                       "IGHV1-1*02": "AAAATCCCGGGTTA",       # differs at 4 and 13
                       "IGHV1-1*03": "ATAAAC",               # shorter: constrains positions < 6 only (1)
                       "IGHV1-1*04": base + "GGGG",          # longer, equal over the base
                       "IGHV2-1*01": "CCCCCCCCCCCCCC"},      # another gene: never constrains
                 "j": {"IGHJ1*01": "TTTT"}},
        anchors={"IGHV1-1*01": 6, "IGHV1-1*02": 6})
    # Every position but 1, 4 and 13 (another allele differs) and the anchor codon 6..8.
    assert invariant_positions(reference, "IGHV1-1*01") == [0, 2, 3, 5, 9, 10, 11, 12]
    # The longer allele: positions past the base length of the others are unconstrained there.
    assert invariant_positions(reference, "IGHV1-1*04") == [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]
    # A gene with one allele: everything outside the anchor codon (none for this J).
    assert invariant_positions(reference, "IGHJ1*01") == [0, 1, 2, 3]


def test_base_is_strictly_nearest(reference):
    rng = random.Random(11)
    validator = ga.Genotype.from_dataconfig(CONFIG)
    names = sorted(reference.alleles["v"])
    made = skipped = 0
    for draw in range(200):
        base, k = rng.choice(names), rng.randint(1, 6)
        name = f"{gene_name(base)}*{900 + draw}"
        try:
            novel = make_novel_allele(reference, base, k, rng, name, genotype=validator)
        except RuntimeError:
            skipped += 1  # a base whose coding frame has a stop codon that no draw repaired
            continue
        made += 1
        base_seq = reference.seq(base)
        assert (novel.name, novel.base, len(novel.sequence)) == (name, base, len(base_seq))
        assert len(novel.snps) == k and list(novel.snps) == sorted(set(novel.snps))
        assert set(novel.snps) <= set(invariant_positions(reference, base))
        anchor = reference.anchors[base]
        assert not any(anchor <= p < anchor + 3 for p in novel.snps)
        assert [p for p in range(len(base_seq)) if novel.sequence[p] != base_seq[p]] == list(novel.snps)
        assert name in validator.novel_allele_names()
        for other, seq in reference.alleles["v"].items():
            if other != base and gene_name(other) == gene_name(base) and len(seq) == len(base_seq):
                assert _hamming(novel.sequence, base_seq) == k < _hamming(novel.sequence, seq), (base, other)
    assert made >= 190 and made + skipped == 200

    class Rejecting:
        calls = 0

        def add_novel_allele(self, name, **kwargs):
            Rejecting.calls += 1
            raise ValueError("non-functional")

    base = names[0]
    with pytest.raises(RuntimeError, match=re.escape(base) + ".*non-functional"):
        make_novel_allele(reference, base, 2, rng, f"{gene_name(base)}*900", genotype=Rejecting())
    assert Rejecting.calls == MAX_ATTEMPTS == 50


def test_novel_reads_rebased(unmutated, reference):
    donors, cases, truths = unmutated
    novel_truths = 0
    for case, truth in zip(cases, truths):
        record, donor = case["record"], donors[case["groups"]["subject"]]
        assert not truth.warnings and not truth.unscorable, (truth.case_id, truth.warnings)
        v = truth.segments["v"]
        novel = donor.novel.get(record["truth_v_call"])
        if novel is None:
            assert truth.v_novel_snps is None and v.observed.n_mismatch == 0
            continue
        novel_truths += 1
        o, base_seq = v.observed, reference.seq(novel.base)
        # Re-based: the base allele, its germline, the same alignment, every observed SNP a mismatch.
        assert v.alleles.primary == o.allele == novel.base
        assert novel.base in v.alleles.generative and novel.name not in v.alleles.alleles
        assert novel.base in v.alleles.simulator_ties and novel.name not in v.alleles.simulator_ties
        assert not v.alleles.recalled
        # Unmutated, no other allele is as near to the read as the base unless it equals the base over the span.
        assert v.alleles.alleles == v.alleles.generative
        assert o.observed_germline == base_seq[o.germ_start:o.germ_end]
        assert o.cigar == f"{o.seq_end - o.seq_start}M"
        assert (record["v_sequence_start"], record["v_germline_start"]) == (o.seq_start, o.germ_start)
        shown = [p for p in novel.snps if o.germ_start <= p < o.germ_end]
        assert truth.v_novel_snps == tuple(o.seq_start + p - o.germ_start for p in shown)
        assert o.n_mismatch == len(truth.v_novel_snps) == truth.v_mismatches
        assert all(truth.sequence[r] == novel.sequence[p] != base_seq[p]
                   for r, p in zip(truth.v_novel_snps, shown))
        if truth.clone_key is not None:
            assert truth.clone_key[0] == v.alleles.genes and gene_name(novel.base) in v.alleles.genes
    assert novel_truths >= 150
    assert_oracle_perfect(truths, reference)

    # Without the donor's novel alleles a novel V cannot be built: unscorable, as for any allele not in the reference.
    case = next(c for c, t in zip(cases, truths) if t.v_novel_snps)
    plain = truth_from_case(case, reference)
    assert "v" in plain.unscorable and "v" not in plain.segments and plain.v_novel_snps is None
    # A novel allele that does not differ from its base at exactly its SNPs is refused, naming it.
    novel = donors[case["groups"]["subject"]].novel[case["record"]["truth_v_call"]]
    with pytest.raises(ValueError, match=re.escape(novel.name)):
        truth_from_case(case, reference, novel={novel.name: _off_by_one_base(novel)})


def test_sibling_as_near_as_the_base_is_scored(unmutated, reference):
    """SHM can bring the read nearer to another allele of the base's gene than to the base. GenAIRR's re-call cannot
    show it (it compares the read with the novel allele), so the truth adds every allele of the gene that covers
    the observed span and mismatches the read at no more aligned columns than the base."""
    donors, cases, truths = unmutated
    for case, truth in zip(cases, truths):
        if not truth.v_novel_snps:
            continue
        novel = donors[case["groups"]["subject"]].novel[case["record"]["truth_v_call"]]
        o = truth.segments["v"].observed
        base_seq = reference.seq(novel.base)
        siblings = [(name, seq) for name, seq in sorted(reference.alleles["v"].items())
                    if gene_name(name) == gene_name(novel.base) and name != novel.base and len(seq) >= o.germ_end
                    and 0 < _hamming(seq[o.germ_start:o.germ_end], base_seq[o.germ_start:o.germ_end]) <= 3]
        if siblings:
            break
    else:
        pytest.fail("no novel read with a near sibling")
    sibling, sibling_seq = siblings[0]
    differing = [p for p in range(o.germ_start, o.germ_end) if sibling_seq[p] != base_seq[p]]
    # Mutate the read to the sibling's bases wherever it differs from the base: the sibling is now strictly nearer.
    record = dict(case["record"])
    read = list(record["sequence"])
    for p in differing:
        read[o.seq_start + p - o.germ_start] = sibling_seq[p]
    record["sequence"] = "".join(read)
    novel_map = donors[case["groups"]["subject"]].novel
    mutated = truth_from_case({**case, "record": record, "orientation": 0, "sequence": record["sequence"].upper()},
                              reference, novel=novel_map)
    v = mutated.segments["v"]
    assert v.alleles.primary == novel.base and v.observed.n_mismatch == len(truth.v_novel_snps) + len(differing)
    assert sibling in v.alleles.alleles and sibling not in v.alleles.generative
    assert sibling in v.alleles.simulator_ties and novel.base in v.alleles.simulator_ties
    assert_oracle_perfect([mutated], reference)


def test_snp_mutations_metric(unmutated, reference):
    donors, cases, truths = unmutated
    case, truth = next((c, t) for c, t in zip(cases, truths) if t.v_novel_snps and len(t.v_novel_snps) >= 2)
    novel = donors[case["groups"]["subject"]].novel[case["record"]["truth_v_call"]]
    v = truth.segments["v"]
    oracle = oracle_prediction(truth).segments["v"]
    # A fixed-reference tool aligning the base reports every SNP as a mutation.
    assert novel_metrics(truth, oracle, reference) == {METRIC: float(len(truth.v_novel_snps))}
    # A tool that knows the novel allele reports none of them.
    knowing = Reference(alleles={**reference.alleles, "v": {**reference.alleles["v"], "IGHVX*01": novel.sequence}},
                        anchors=reference.anchors)
    assert novel_metrics(truth, replace(oracle, call="IGHVX*01"), knowing) == {METRIC: 0.0}
    # A prediction covering only the first SNP counts only that one.
    first = truth.v_novel_snps[0]
    span = first - v.observed.seq_start + 1
    short = replace(oracle, seq_end=first + 1, germ_end=v.observed.germ_start + span, cigar=f"{span}M")
    assert novel_metrics(truth, short, reference) == {METRIC: 1.0}
    # No usable V alignment: not covered.
    assert math.isnan(novel_metrics(truth, PredSegment(), reference)[METRIC])
    assert math.isnan(score_case(truth, None, reference)[METRIC])
    # Informational, not compared; reads without an observed novel SNP do not get it.
    assert is_informational(METRIC) and not lower_is_better(METRIC)
    assert all(METRIC not in score_case(t, oracle_prediction(t), reference) for t in truths if not t.v_novel_snps)


def test_tool_genotype_uses_base(reference):
    donors = sample_donors(reference, 3, 21, NOVEL, config=CONFIG)
    plain = sample_donors(reference, 3, 21)
    for donor, without in zip(donors, plain):
        assert len(donor.novel) == 25
        # Named {gene}*{900 + i}; the SNP count cycles 1..6; each gene is homozygous for its novel allele.
        by_index = sorted(donor.novel.values(), key=lambda n: int(n.name.split("*")[1]))
        assert [int(n.name.split("*")[1]) for n in by_index] == list(range(900, 925))
        assert [len(n.snps) for n in by_index] == [1, 2, 3, 4, 5, 6] * 4 + [1]
        for novel in by_index:
            gene = gene_name(novel.name)
            assert gene == gene_name(novel.base) and gene.startswith("IGHV") and gene not in donor.deleted
            assert donor.alleles[gene] == (novel.name,)
            # The novel alleles are drawn from their own stream: the rest of the genotype is the plain donor's.
            assert novel.base in without.alleles[gene]
        assert donor.deleted == without.deleted
        assert {g: a for g, a in donor.alleles.items() if not a[0] in donor.novel} == \
            {g: a for g, a in without.alleles.items() if g not in {gene_name(n) for n in donor.novel}}

        tool = donor.tool_genotype(reference)
        assert not set(tool["v"]) & set(donor.novel)
        assert {n.base for n in donor.novel.values()} <= set(tool["v"])
        assert set(tool["v"]) == {a for names in donor.alleles.values() for a in names
                                  if a not in donor.novel and reference.gene_letter(a) == "v"} | \
            {n.base for n in donor.novel.values()}
        genotype = donor.genotype(CONFIG)
        assert genotype.novel_allele_names() == set(donor.novel)
        assert all(genotype.carried_alleles("V", gene_name(n)) == {n} for n in donor.novel)

    assert donors_from_json(donors_to_json(donors)) == donors
    assert sample_donors(reference, 3, 21, NOVEL, config=CONFIG) == donors
    with pytest.raises(ValueError, match="config"):
        sample_donors(reference, 1, 21, NOVEL)

    # Malformed novel alleles and donors are refused, naming the allele.
    donor = donors[0]
    novel = next(iter(donor.novel.values()))
    gene = gene_name(novel.name)
    with pytest.raises(ValueError, match="not carried"):
        replace(donor, alleles={**donor.alleles, gene: (novel.base,)})
    for snps in ((5, 3), (3, 3), (len(novel.sequence),), ()):
        with pytest.raises(ValueError, match=re.escape(novel.name)):
            NovelAllele(novel.name, novel.base, novel.sequence, snps)
    with pytest.raises(ValueError, match=re.escape(novel.name)):
        replace(donor, novel={**donor.novel, novel.name: _off_by_one_base(novel)}).tool_genotype(reference)
    with pytest.raises(ValueError):
        DonorParams(novel_v_genes=-1)
    with pytest.raises(ValueError):
        DonorParams(novel_v_genes=3, novel_snp_cycle=())
    with pytest.raises(ValueError):
        DonorParams(novel_v_genes=3, novel_snp_cycle=(1, 0))


def test_oracle_perfect_on_s7(tmp_path, reference):
    suite = get_suite("S7")
    assert (suite.slug, suite.arms) == ("novel", ("shared", "genotype"))
    assert suite.primary == ("v.top1_in_set", "v.gene_top1", "v.mutation_count_err", "clone_key.exact",
                             "junction.nt_exact")
    (stratum,) = suite.strata
    assert (stratum.name, stratum.n, stratum.cohort, stratum.rc_fraction) == ("cohort", 3000, True, 0.5)
    assert suite.donors.n == 10 and suite.donors.tag == "S7" and suite.donors.params.novel_v_genes == 25
    assert suite.purpose.endswith(".") and suite.purpose.count(". ") == 0

    layout = RunLayout(tmp_path)
    build = build_suite("S7", layout, reference_name=REFERENCE, split="dev", seed=17, scale=0.02, workers=1)
    cases, truths = read_cases(layout.cases("S7")), read_truth_jsonl(layout.truth("S7"))
    donors = {d.subject_id: d for d in donors_from_json(json.loads(layout.donors("S7").read_text()))}
    assert len(cases) == len(truths) == 60 and build["truth"]["excluded"] == {}
    assert build["truth"]["unscorable"] == {"v": 0, "d": 0, "j": 0}
    assert all(len(d.novel) == 25 for d in donors.values()) and len(donors) == 10
    labels = collections.Counter()
    for case, truth in zip(cases, truths):
        novel = donors[case["groups"]["subject"]].novel.get(case["record"]["truth_v_call"])
        assert case["labels"] == {"novel_snps": "none" if novel is None else str(len(novel.snps))}
        labels[case["labels"]["novel_snps"]] += 1
        if novel is None:
            assert truth.v_novel_snps is None and truth.segments["v"].alleles.primary == case["record"]["truth_v_call"]
        else:
            assert truth.v_novel_snps is not None and truth.segments["v"].alleles.primary == novel.base
    assert labels["none"] < 60 and sum(labels.values()) == 60
    scores = assert_oracle_perfect(truths, reference)
    emitted = {metric for case in scores.values() for metric in case}
    assert set(suite.primary) <= emitted, set(suite.primary) - emitted
    assert METRIC in emitted
    for truth in truths:
        if truth.v_novel_snps:
            assert 0 <= scores[truth.case_id][METRIC] <= len(truth.v_novel_snps)
