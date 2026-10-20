"""S8 ``genotype``: full-length IgG-level reads from donors with heterozygous and deleted genes, scored against
each read's donor genotype."""
from __future__ import annotations

from functools import partial

from alignair_assay.suites.base import Stratum, Suite
from alignair_assay.suites.donors import DonorSpec
from alignair_assay.suites.recipes import SHM_CLASSES, repertoire

S8 = Suite(
    name="S8",
    slug="genotype",
    title="Donor genotypes",
    purpose="Measures how often a tool calls alleles its donor does not carry, including alleles of deleted genes.",
    primary=("v.outside_genotype", "v.deleted_gene_call", "j.outside_genotype", "v.top1_in_set", "clone_key.exact"),
    strata=(Stratum("cohort", 3000, partial(repertoire, design="full", rate=SHM_CLASSES["igg"]), rc_fraction=0.5,
                    cohort=True),),
    donors=DonorSpec(10, "S8"),
    arms=("shared", "genotype"),
)
