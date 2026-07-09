"""Focused CLI command handler modules."""
from __future__ import annotations

from .catalog import register_catalog_commands
from .evaluation import register_evaluation_commands
from .generation import register_generation_commands
from .io import register_io_commands
from .validation import register_validation_commands


def register_commands(subparsers) -> None:
    """Register every benchmark CLI command group."""

    register_generation_commands(subparsers)
    register_io_commands(subparsers)
    register_catalog_commands(subparsers)
    register_validation_commands(subparsers)
    register_evaluation_commands(subparsers)


__all__ = ["register_commands"]
