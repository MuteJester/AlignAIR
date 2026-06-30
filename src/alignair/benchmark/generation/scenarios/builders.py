"""Dynamic scenario builders for measurement-aligned GenAIRR benchmarks."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import re
from typing import Any

from ....reference.reference_set import ReferenceSet
from ...core.schema import BenchmarkSpec, StratumSpec
from ..genairr import dataconfig_by_name
from ..strata import isolated_params


@dataclass(frozen=True)
class AllelePanel:
    """One restricted allele panel for isolating allele/candidate-set behavior."""

    segment: str
    gene: str
    alleles: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def allele_gene_name(allele: str) -> str:
    """Return the gene-level name before the allele suffix."""

    return str(allele).split("*", 1)[0]


def select_sibling_allele_panels(
    reference_set: ReferenceSet,
    *,
    segment: str = "V",
    n_panels: int = 3,
    alleles_per_panel: int = 2,
) -> tuple[AllelePanel, ...]:
    """Select same-gene sibling allele panels from a reference set."""

    if n_panels < 1:
        raise ValueError("n_panels must be at least 1")
    if alleles_per_panel < 1:
        raise ValueError("alleles_per_panel must be at least 1")

    segment = segment.upper()
    ref = reference_set.gene(segment)
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for allele in ref.names:
        groups.setdefault(allele_gene_name(allele), []).append(allele)

    panels = [
        AllelePanel(segment=segment, gene=gene, alleles=tuple(alleles[:alleles_per_panel]))
        for gene, alleles in groups.items()
        if len(alleles) >= alleles_per_panel
    ]
    if not panels:
        raise ValueError(
            f"no {segment} genes have at least {alleles_per_panel} sibling alleles"
        )
    return tuple(panels[:n_panels])


def restricted_allele_panel_strata(
    reference_set: ReferenceSet,
    *,
    segment: str = "V",
    n_panels: int = 3,
    alleles_per_panel: int = 2,
    n_per_panel: int = 100,
    prefix: str = "allele_panel",
) -> tuple[StratumSpec, ...]:
    """Build strata that restrict GenAIRR sampling to sibling allele panels."""

    panels = select_sibling_allele_panels(
        reference_set,
        segment=segment,
        n_panels=n_panels,
        alleles_per_panel=alleles_per_panel,
    )
    strata: list[StratumSpec] = []
    for idx, panel in enumerate(panels):
        restrict_key = panel.segment.lower()
        panel_id = f"{prefix}_{restrict_key}_{idx:02d}_{_slug(panel.gene)}"
        strata.append(
            StratumSpec(
                name=panel_id,
                n=n_per_panel,
                progress=0.0,
                param_overrides=isolated_params(
                    restrict_alleles={restrict_key: panel.alleles},
                    metadata={
                        "benchmark_measurement": "allele_coverage_and_candidates",
                        "restricted_segment": panel.segment,
                        "restricted_gene": panel.gene,
                        "restricted_alleles": ",".join(panel.alleles),
                    },
                ),
                description=(
                    f"{panel.segment}-allele sibling panel for {panel.gene}; "
                    "all other background corruption is controlled."
                ),
                tags=("allele_panel", "sibling_allele", "genotype_size", "clean"),
            )
        )
    return tuple(strata)


def restricted_allele_panel_spec(
    *,
    dataconfig_name: str = "HUMAN_IGH_OGRDB",
    seed: int = 123,
    segment: str = "V",
    n_panels: int = 3,
    alleles_per_panel: int = 2,
    n_per_panel: int = 100,
) -> BenchmarkSpec:
    """Return a benchmark spec for allele/candidate-set isolation."""

    dataconfig = dataconfig_by_name(dataconfig_name)
    reference_set = ReferenceSet.from_dataconfigs(dataconfig)
    return BenchmarkSpec(
        name=f"{dataconfig_name.lower()}_{segment.lower()}_allele_panels",
        dataconfig_name=dataconfig_name,
        seed=seed,
        strata=restricted_allele_panel_strata(
            reference_set,
            segment=segment,
            n_panels=n_panels,
            alleles_per_panel=alleles_per_panel,
            n_per_panel=n_per_panel,
        ),
        description=(
            "GenAIRR restrict_alleles benchmark for sibling allele and "
            "candidate-set measurement."
        ),
    )


def genotype_subject_strata(
    *,
    n_subjects: int = 3,
    n_per_subject: int = 100,
    genotype_seed: int = 1000,
    validate_records: bool = False,
) -> tuple[StratumSpec, ...]:
    """Build genotype-subject strata using GenAIRR Genotype.sample at runtime."""

    if n_subjects < 1:
        raise ValueError("n_subjects must be at least 1")
    if n_per_subject < 1:
        raise ValueError("n_per_subject must be at least 1")

    strata = []
    for idx in range(n_subjects):
        subject_id = f"subject_{idx:03d}"
        strata.append(
            StratumSpec(
                name=f"genotype_subject_{idx:03d}",
                n=n_per_subject,
                progress=0.0,
                seed_offset=idx * 10_007,
                param_overrides=isolated_params(
                    genotype_seed=genotype_seed + idx,
                    genotype_subject_id=subject_id,
                    run_records=True,
                    validate_records=validate_records,
                    metadata={
                        "benchmark_measurement": "genotype_masked_inference",
                        "benchmark_subject": subject_id,
                    },
                ),
                description=(
                    "GenAIRR genotype-phased subject stratum with clean background "
                    "for genotype-mask and cohort-style measurement."
                ),
                tags=("genotype", "cohort", "genotype_size", "clean"),
            )
        )
    return tuple(strata)


def genotype_subject_spec(
    *,
    dataconfig_name: str = "HUMAN_IGH_OGRDB",
    seed: int = 123,
    n_subjects: int = 3,
    n_per_subject: int = 100,
    genotype_seed: int = 1000,
    validate_records: bool = False,
) -> BenchmarkSpec:
    """Return a benchmark spec for GenAIRR genotype-phased subject generation."""

    return BenchmarkSpec(
        name=f"{dataconfig_name.lower()}_genotype_subjects",
        dataconfig_name=dataconfig_name,
        seed=seed,
        strata=genotype_subject_strata(
            n_subjects=n_subjects,
            n_per_subject=n_per_subject,
            genotype_seed=genotype_seed,
            validate_records=validate_records,
        ),
        description="GenAIRR genotype-subject benchmark for genotype-aware AIRR alignment.",
    )


def multi_locus_specs(
    *,
    dataconfig_names: tuple[str, ...] = (
        "HUMAN_IGH_OGRDB",
        "HUMAN_IGK_OGRDB",
        "HUMAN_IGL_OGRDB",
    ),
    seed: int = 123,
    n_per_locus: int = 100,
) -> tuple[BenchmarkSpec, ...]:
    """Return one clean measurement spec per GenAIRR DataConfig/locus."""

    specs = []
    for idx, dataconfig_name in enumerate(dataconfig_names):
        dataconfig = dataconfig_by_name(dataconfig_name)
        reference_set = ReferenceSet.from_dataconfigs(dataconfig)
        locus = reference_set.infer_locus() or dataconfig_name
        specs.append(
            BenchmarkSpec(
                name=f"{dataconfig_name.lower()}_locus_probe",
                dataconfig_name=dataconfig_name,
                seed=seed + idx * 10_007,
                strata=(
                    StratumSpec(
                        name=f"{_slug(locus)}_clean_full",
                        n=n_per_locus,
                        progress=0.0,
                        param_overrides=isolated_params(
                            metadata={
                                "benchmark_measurement": "multi_locus_chain",
                                "benchmark_locus": locus,
                            },
                        ),
                        description=(
                            f"Clean GenAIRR {locus} locus probe for chain/locus routing measurement."
                        ),
                        tags=("locus", "chain", "clean"),
                    ),
                ),
                description=f"Clean GenAIRR {locus} benchmark for locus/chain support.",
            )
        )
    return tuple(specs)
