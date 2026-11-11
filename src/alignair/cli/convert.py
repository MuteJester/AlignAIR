"""`alignair convert` — a lab chip model file becomes a safe, pickle-free ``chip-v1`` `.alignair`.

A model file trained under a lab chip name (``lab-chip-v1``, ``lab-chipw64-v1``: ``model_file.convert``) is rewritten
as ``chip-v1`` without unpickling anything. An AlignAIR 3 checkpoint (a `.pt`, or a conv-v1 `.alignair`) does not
convert: AlignAIR 4 runs chip models only (``alignair<4`` runs it).
"""
from __future__ import annotations

from ..model_file import container
from ..errors import ConfigurationError, report_error


def register(sub) -> None:
    p = sub.add_parser("convert", help="convert a lab chip model file into a safe, pickle-free chip-v1 .alignair")
    p.add_argument("src")
    p.add_argument("dst")
    p.set_defaults(func=run)


def _lab_chip(path) -> bool:
    """Whether ``path`` is a model file whose configuration (JSON, read without unpickling) names a lab chip."""
    from ..model_file import serialize
    from ..model_file.convert import LAB_CHIPS
    if not container.is_alignair_file(path):
        return False
    return serialize.lab_config_from_bytes(container.read_section(path, "config")).architecture in LAB_CHIPS


def _v3_architecture(path) -> str | None:
    """The architecture an AlignAIR 3 model file's configuration names (``conv-v1``, or none at all), else None; the
    configuration is read as JSON, nothing unpickled."""
    import json

    from ..model_file.serialize import V3_ARCHITECTURE
    raw = json.loads(container.read_section(path, "config").decode("utf-8"))
    architecture = raw.get("architecture", V3_ARCHITECTURE)
    return architecture if architecture == V3_ARCHITECTURE else None


def run(args) -> int:
    if not container.is_alignair_file(args.src):
        from ..model_file import v3_checkpoint_error
        return report_error(v3_checkpoint_error(args.src))
    v3 = _v3_architecture(args.src)
    if v3 is not None:                       # an AlignAIR 3 model file: refused with the version that runs it
        from ..model_file.serialize import _v3_refusal
        return report_error(_v3_refusal(v3))
    if not _lab_chip(args.src):
        from ..model_file.convert import LAB_CHIPS
        return report_error(ConfigurationError(
            f"{args.src} is not a lab chip model file", context={"path": args.src},
            hint=f"Only {', '.join(sorted(LAB_CHIPS))} model files convert to chip-v1."))
    from ..model_file.convert import convert_lab_chip
    summary = convert_lab_chip(args.src, args.dst)
    print(f"converted {args.src} ({summary['from']}) -> {args.dst} ({summary['to']}, trunk_channels "
          f"{summary['trunk_channels']}; safe, pickle-free)")
    return 0
