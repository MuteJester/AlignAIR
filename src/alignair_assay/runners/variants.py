"""Checkpoint variants: one AlignAIR checkpoint run as a tool of its own, ``<base>@<label>``.

A tool id is ``<base>`` or ``<base>@<label>`` (``parse_tool``). Only the checkpoint tools take a label
(``VARIANT_BASES``: ``alignair-local`` and ``alignair-api``), and a label is 1 to 32 of ``A-Z a-z 0-9 _ . -``
(``LABEL``), so a tool id is always one path component. A variant runs the checkpoint its label names in
``RunnerOptions.checkpoints`` (``variant_checkpoint``), with every other setting of its base, and is its base's runner
under the full id: its ``name`` is ``alignair-api@cand``, so its run records, run directory and work directory are
the variant's own, and its ``fingerprint()`` adds the ``label`` to the base's (whose ``checkpoint_sha256`` already
names the weights). Two variants of one checkpoint are therefore two tools, compared on the same reads.

The plain tools are untouched: their classes, fingerprints and paths are the base runners' as before.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from alignair_assay.runners.alignair import AlignAIRLocal
from alignair_assay.runners.alignair_api import AlignAIRApi
from alignair_assay.runners.base import RunnerOptions, absolute_path

# The tools that run a checkpoint file, so may run another one under a label.
VARIANT_BASES = ("alignair-local", "alignair-api")
# A label: one path component, no separator, no space, nothing a shell or a file name would read differently.
LABEL = re.compile(r"[A-Za-z0-9_.-]{1,32}")


def parse_tool(name: str) -> tuple[str, str | None]:
    """``(base, label)`` of tool id ``name``: ``(name, None)`` without an ``@``, else split at the ``@``. ValueError
    for a label that is not ``LABEL`` (an empty one, a second ``@``, a path separator, over 32 characters), or a
    label on a tool that runs no checkpoint (only ``VARIANT_BASES`` take one)."""
    base, at, label = name.partition("@")
    if not at:
        return name, None
    if base not in VARIANT_BASES:
        raise ValueError(f"tool {name!r}: only {' and '.join(VARIANT_BASES)} take a checkpoint label "
                         "(<tool>@<label>)")
    check_label(label, tool=name)
    return base, label


def check_label(label: str, *, tool: str | None = None) -> str:
    """``label`` when it is a valid checkpoint label (``LABEL``); ValueError naming it otherwise."""
    if not LABEL.fullmatch(label):
        where = f" of {tool!r}" if tool else ""
        raise ValueError(f"checkpoint label {label!r}{where} is not 1 to 32 of the letters, digits, '_', '.' and '-'")
    return label


def variant_checkpoint(label: str, checkpoints: Mapping[str, str], *, tool: str) -> str:
    """The absolute path of the checkpoint ``label`` names in ``checkpoints``, for the variant ``tool``. ValueError
    for a label not among them (naming the known labels), FileNotFoundError for a file that does not exist: a
    variant is refused before anything runs."""
    if label not in checkpoints:
        known = ", ".join(sorted(checkpoints)) or "none"
        raise ValueError(f"unknown checkpoint label {label!r} ({tool}); known labels: {known} (give it as "
                         f"--checkpoint {label}=PATH)")
    path = absolute_path(checkpoints[label])
    if not Path(path).is_file():
        raise FileNotFoundError(f"{tool}: no AlignAIR checkpoint {path} (--checkpoint {label}=...)")
    return path


class _Variant:
    """A checkpoint runner under a label: the base runner (the next class in the MRO) on the label's checkpoint,
    named ``<base>@<label>``, its fingerprint the base's plus the ``label``."""

    label: str

    def __init__(self, label: str, checkpoint, **settings):
        super().__init__(checkpoint, **settings)
        self.label = check_label(label)
        # an instance attribute: the class keeps its base's name, so a plain runner is never renamed
        self.name = f"{type(self).name}@{self.label}"

    @classmethod
    def from_label(cls, label: str, options: RunnerOptions):
        """The registry factory (``runners.VARIANTS``): the checkpoint ``options.checkpoints[label]``
        (``variant_checkpoint``), and the base's ``device``, ``batch_size``, ``threads`` and ``cpus``."""
        checkpoint = variant_checkpoint(label, options.checkpoints, tool=f"{cls.name}@{label}")
        return cls(label, checkpoint, device=options.device, batch_size=options.batch_size, threads=options.threads,
                   cpus=options.cpus)

    def fingerprint(self) -> dict:
        """The base runner's fingerprint (the checkpoint's path and SHA-256 among it) and the ``label``."""
        return {**super().fingerprint(), "label": self.label}


class AlignAIRLocalVariant(_Variant, AlignAIRLocal):
    """``alignair-local@<label>``: a checkpoint run with this checkout's AlignAIR CLI."""


class AlignAIRApiVariant(_Variant, AlignAIRApi):
    """``alignair-api@<label>``: a checkpoint run through AlignAIR's public Python API."""
