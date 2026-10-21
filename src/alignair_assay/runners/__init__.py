"""The runner registry: every tool the assay can run, by name.

A runner is one module of this package implementing ``base.Runner``; it is registered here with a factory that
builds it from ``base.RunnerOptions``. A checkpoint tool may also run as a variant, ``<base>@<label>``
(``variants.parse_tool``), built by its ``VARIANTS`` factory from the label and the options, which hold the label's
checkpoint (``RunnerOptions.checkpoints``). Tests may register fakes in ``RUNNERS`` and ``VARIANTS``.
"""
from __future__ import annotations

from collections.abc import Callable

from alignair_assay.runners.alignair import AlignAIRImage, AlignAIRLocal
from alignair_assay.runners.alignair_api import AlignAIRApi
from alignair_assay.runners.base import Runner, RunnerOptions
from alignair_assay.runners.igblast import IgBlastRunner
from alignair_assay.runners.oracle import OracleRunner
from alignair_assay.runners.variants import (
    AlignAIRApiVariant,
    AlignAIRLocalVariant,
    parse_tool,
    variant_checkpoint,
)

RUNNERS: dict[str, Callable[[RunnerOptions], Runner]] = {
    "igblast": IgBlastRunner.from_options,
    "alignair": AlignAIRImage.from_options,
    "alignair-local": AlignAIRLocal.from_options,
    "alignair-api": AlignAIRApi.from_options,
    "oracle": OracleRunner.from_options,
}
# The checkpoint tools' variant factories, by base: ``factory(label, options)``.
VARIANTS: dict[str, Callable[[str, RunnerOptions], Runner]] = {
    "alignair-local": AlignAIRLocalVariant.from_label,
    "alignair-api": AlignAIRApiVariant.from_label,
}

__all__ = ["RUNNERS", "VARIANTS", "make_runner", "parse_tool"]


def make_runner(name: str, options: RunnerOptions) -> Runner:
    """The runner registered as ``name``, built from ``options``; ValueError listing the known tools for an
    unknown name, or when the options lack what the tool needs. A variant ``<base>@<label>`` is checked first
    (``variants.parse_tool``: the label's syntax and the base; ``variants.variant_checkpoint``: the label among
    ``options.checkpoints`` and its file present, FileNotFoundError otherwise), so a bad one is refused before
    anything is built."""
    base, label = parse_tool(name)
    if label is not None:
        variant_checkpoint(label, options.checkpoints, tool=name)
        return VARIANTS[base](label, options)
    try:
        factory = RUNNERS[name]
    except KeyError:
        raise ValueError(f"unknown tool {name!r}; known tools: {', '.join(RUNNERS)}") from None
    return factory(options)
