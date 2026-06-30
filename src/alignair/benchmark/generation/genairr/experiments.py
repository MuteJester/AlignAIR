"""GenAIRR experiment construction helpers for benchmark strata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from ....gym.curriculum import Curriculum
from ....gym.gym import build_experiment
from ...core.schema import StratumSpec


@dataclass(frozen=True)
class ResolvedGenAIRRExperiment:
    """Compiled GenAIRR experiment plus benchmark sampling metadata."""

    experiment: Any
    params: dict[str, Any]
    sampling_kwargs: dict[str, Any]
    run_records_kwargs: dict[str, Any]


def dataconfig_by_name(name: str):
    """Resolve a GenAIRR data config by name from ``GenAIRR.data``."""

    import GenAIRR.data as gdata

    return getattr(gdata, name)


def resolve_stratum_params(
    stratum: StratumSpec,
    curriculum: Curriculum | None = None,
) -> dict[str, Any]:
    """Resolve benchmark curriculum parameters plus stratum overrides."""

    curriculum = curriculum or Curriculum()
    params = dict(curriculum.params(stratum.progress))
    params.update(stratum.param_overrides)
    return params


def sampling_kwargs_from_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return GenAIRR stream/run options that are not compile-time experiment steps."""

    sampling_kwargs: dict[str, Any] = {}
    if "strict_sampling" in params:
        sampling_kwargs["strict"] = bool(params["strict_sampling"])
    return sampling_kwargs


def run_records_kwargs_from_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return GenAIRR run_records options needed for materialized generation."""

    kwargs = dict(sampling_kwargs_from_params(params))
    if "validate_records" in params:
        kwargs["validate_records"] = bool(params["validate_records"])
    if "expose_genairr_provenance" in params:
        kwargs["expose_provenance"] = bool(params["expose_genairr_provenance"])
    return kwargs


def apply_record_metadata(record: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Apply benchmark-managed metadata to a streamed GenAIRR record."""

    metadata = params.get("metadata")
    if not metadata:
        return record
    collisions = sorted(set(metadata) & set(record))
    if collisions:
        raise ValueError(f"metadata keys already present in GenAIRR record: {collisions}")
    out = dict(record)
    out.update(metadata)
    return out


def stream_stratum_records(
    resolved: ResolvedGenAIRRExperiment,
    *,
    n: int,
    seed: int,
) -> Iterator[dict[str, Any]]:
    """Stream records from a resolved GenAIRR experiment with benchmark postprocessing."""

    use_run_records = (
        bool(resolved.params.get("validate_records"))
        or bool(resolved.params.get("expose_genairr_provenance"))
        or resolved.params.get("genotype") is not None
        or resolved.params.get("genotype_seed") is not None
        or bool(resolved.params.get("run_records"))
    )
    if use_run_records:
        records = resolved.experiment.run_records(n=n, seed=seed, **resolved.run_records_kwargs).records
    else:
        records = resolved.experiment.stream_records(n=n, seed=seed, **resolved.sampling_kwargs)

    for record in records:
        yield apply_record_metadata(record, resolved.params)


def validate_stratum_records(
    dataconfig,
    stratum: StratumSpec,
    *,
    n: int = 1,
    seed: int = 0,
    curriculum: Curriculum | None = None,
) -> dict[str, Any]:
    """Run a small GenAIRR validate_records preflight for one stratum."""

    from dataclasses import replace

    params = dict(stratum.param_overrides)
    params["validate_records"] = True
    validation_stratum = replace(stratum, n=n, param_overrides=params)
    try:
        resolved = build_stratum_experiment(dataconfig, validation_stratum, curriculum)
        records = list(stream_stratum_records(resolved, n=n, seed=seed))
    except Exception as exc:  # pragma: no cover - exception type comes from GenAIRR
        return {
            "valid": False,
            "n_records": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "valid": True,
        "n_records": len(records),
        "error_type": None,
        "error": None,
    }


def build_stratum_experiment(
    dataconfig,
    stratum: StratumSpec,
    curriculum: Curriculum | None = None,
    *,
    allow_curatable: bool = False,
) -> ResolvedGenAIRRExperiment:
    """Build the GenAIRR experiment for one benchmark stratum."""

    params = resolve_stratum_params(stratum, curriculum)
    return ResolvedGenAIRRExperiment(
        experiment=build_experiment(dataconfig, params, allow_curatable=allow_curatable),
        params=params,
        sampling_kwargs=sampling_kwargs_from_params(params),
        run_records_kwargs=run_records_kwargs_from_params(params),
    )
