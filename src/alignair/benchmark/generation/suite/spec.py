"""Declarative benchmark suite specifications."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ...core.schema import BenchmarkSpec
from ..scenarios import genotype_subject_spec, multi_locus_specs, restricted_allele_panel_spec
from ..strata import default_igh_assay_spec


@dataclass(frozen=True)
class SuiteComponentSpec:
    """One named component inside a composed benchmark suite."""

    name: str
    role: str
    specs: tuple[BenchmarkSpec, ...]
    measurement_focus: tuple[str, ...] = ()
    readiness_profile: str = "assay"
    description: str = ""

    @property
    def n_cases(self) -> int:
        return sum(spec.n_cases for spec in self.specs)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["n_cases"] = self.n_cases
        return data


@dataclass(frozen=True)
class BenchmarkSuiteSpec:
    """A reproducible suite made from multiple benchmark generation components."""

    name: str
    seed: int
    components: tuple[SuiteComponentSpec, ...]
    version: str = "0.1"
    description: str = ""

    @property
    def n_cases(self) -> int:
        return sum(component.n_cases for component in self.components)

    @property
    def specs(self) -> tuple[BenchmarkSpec, ...]:
        return tuple(spec for component in self.components for spec in component.specs)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["components"] = [component.to_dict() for component in self.components]
        data["n_cases"] = self.n_cases
        return data


def default_measurement_suite_spec(
    *,
    dataconfig_name: str = "HUMAN_IGH_OGRDB",
    seed: int = 123,
    n_per_stratum: int = 200,
    n_per_focus: int = 200,
    n_panels: int = 3,
    alleles_per_panel: int = 2,
    n_per_panel: int = 100,
    n_subjects: int = 3,
    n_per_subject: int = 100,
    genotype_seed: int = 1000,
    validate_records: bool = False,
    locus_dataconfig_names: tuple[str, ...] = (
        "HUMAN_IGH_OGRDB",
        "HUMAN_IGK_OGRDB",
        "HUMAN_IGL_OGRDB",
    ),
    n_per_locus: int = 100,
    include_base_assay: bool = True,
    include_allele_panels: bool = True,
    include_genotype_subjects: bool = True,
    include_multi_locus: bool = True,
) -> BenchmarkSuiteSpec:
    """Return the standard measurement-aligned GenAIRR benchmark suite."""

    components: list[SuiteComponentSpec] = []
    if include_base_assay:
        components.append(
            SuiteComponentSpec(
                name="base_assay",
                role="base_assay",
                specs=(
                    default_igh_assay_spec(
                        n_per_stratum=n_per_stratum,
                        n_per_focus=n_per_focus,
                        seed=seed,
                    ),
                ),
                readiness_profile="assay",
                description="Broad, focused, and adaptive IGH assay strata.",
            )
        )
    if include_allele_panels:
        components.append(
            SuiteComponentSpec(
                name="allele_panels",
                role="measurement_slice",
                specs=(
                    restricted_allele_panel_spec(
                        dataconfig_name=dataconfig_name,
                        seed=seed + 10_000,
                        n_panels=n_panels,
                        alleles_per_panel=alleles_per_panel,
                        n_per_panel=n_per_panel,
                    ),
                ),
                measurement_focus=("allele_coverage_and_candidates",),
                readiness_profile="allele_stratified",
                description="Restricted sibling-allele panels for candidate-set measurement.",
            )
        )
    if include_genotype_subjects:
        components.append(
            SuiteComponentSpec(
                name="genotype_subjects",
                role="measurement_slice",
                specs=(
                    genotype_subject_spec(
                        dataconfig_name=dataconfig_name,
                        seed=seed + 20_000,
                        n_subjects=n_subjects,
                        n_per_subject=n_per_subject,
                        genotype_seed=genotype_seed,
                        validate_records=validate_records,
                    ),
                ),
                measurement_focus=("genotype_masked_inference",),
                readiness_profile="development",
                description="GenAIRR genotype-subject slices for genotype-mask measurement.",
            )
        )
    if include_multi_locus:
        components.append(
            SuiteComponentSpec(
                name="multi_locus",
                role="measurement_slice",
                specs=multi_locus_specs(
                    dataconfig_names=locus_dataconfig_names,
                    seed=seed + 30_000,
                    n_per_locus=n_per_locus,
                ),
                measurement_focus=("multi_locus_chain",),
                readiness_profile="development",
                description="IGH/IGK/IGL probes for locus and chain routing measurement.",
            )
        )
    return BenchmarkSuiteSpec(
        name=f"{dataconfig_name.lower()}_measurement_suite",
        seed=seed,
        components=tuple(components),
        description=(
            "Measurement-aligned GenAIRR suite composed of base assay, allele-panel, "
            "genotype-subject, and multi-locus components."
        ),
    )
