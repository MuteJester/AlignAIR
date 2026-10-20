"""The runner registry: every tool the assay can run, by name.

A runner is one module of this package implementing ``base.Runner``; it is registered here with a factory that
builds it from ``base.RunnerOptions``. Tests may register fakes in ``RUNNERS``.
"""
from __future__ import annotations

from collections.abc import Callable

from alignair_assay.runners.alignair import AlignAIRImage, AlignAIRLocal
from alignair_assay.runners.alignair_api import AlignAIRApi
from alignair_assay.runners.base import Runner, RunnerOptions
from alignair_assay.runners.igblast import IgBlastRunner
from alignair_assay.runners.oracle import OracleRunner

RUNNERS: dict[str, Callable[[RunnerOptions], Runner]] = {
    "igblast": IgBlastRunner.from_options,
    "alignair": AlignAIRImage.from_options,
    "alignair-local": AlignAIRLocal.from_options,
    "alignair-api": AlignAIRApi.from_options,
    "oracle": OracleRunner.from_options,
}


def make_runner(name: str, options: RunnerOptions) -> Runner:
    """The runner registered as ``name``, built from ``options``; ValueError listing the known tools for an
    unknown name, or when the options lack what the tool needs."""
    try:
        factory = RUNNERS[name]
    except KeyError:
        raise ValueError(f"unknown tool {name!r}; known tools: {', '.join(RUNNERS)}") from None
    return factory(options)
