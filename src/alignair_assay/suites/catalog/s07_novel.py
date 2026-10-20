"""S7 ``novel``: full-length IgG-level reads from donors carrying V alleles that are not in the tools' reference."""
from __future__ import annotations

from functools import partial

from alignair_assay.suites.base import RecipeContext, Stratum, Suite
from alignair_assay.suites.donors import DonorParams, DonorSpec
from alignair_assay.suites.recipes import SHM_CLASSES, repertoire

# Each donor carries 25 V genes as a homozygous novel allele with 1, 2, ..., 6, 1, ... SNPs.
_NOVEL_V_GENES = 25


def novel_snps_label(record: dict, ctx: RecipeContext) -> dict[str, str]:
    """``{"novel_snps": k}``: the SNP count of the read's generative V allele when it is one of its donor's novel
    alleles, ``"none"`` otherwise."""
    donor = next(d for d in ctx.donors if d.subject_id == record["subject_id"])
    novel = donor.novel.get(record["truth_v_call"])
    return {"novel_snps": "none" if novel is None else str(len(novel.snps))}


S7 = Suite(
    name="S7",
    slug="novel",
    title="Novel alleles",
    purpose="Measures allele and gene calls and mutation counts on reads from V alleles missing from the reference.",
    primary=("v.top1_in_set", "v.gene_top1", "v.mutation_count_err", "clone_key.exact", "junction.nt_exact"),
    strata=(Stratum("cohort", 3000, partial(repertoire, design="full", rate=SHM_CLASSES["igg"]), rc_fraction=0.5,
                    cohort=True, case_labels=novel_snps_label),),
    donors=DonorSpec(10, "S7", DonorParams(novel_v_genes=_NOVEL_V_GENES)),
    arms=("shared", "genotype"),
)
