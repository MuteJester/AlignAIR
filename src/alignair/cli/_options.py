"""The options several commands share, each defined once: the device, where a model comes from, and a report's
output."""
from __future__ import annotations

from ..errors import AlignAIRError, RegistryError
from ._errors import classify


def add_device(parser) -> None:
    """``--device``: the torch device, ``auto`` by default (as the API's ``device="auto"``)."""
    parser.add_argument("--device", default="auto",
                        help="cpu / cuda / mps / auto (default: auto: CUDA, then Apple MPS, then CPU)")


def add_registry(parser) -> None:
    """``--registry`` (repeatable) and ``--offline``: where catalog ids are looked up, and whether the network may
    be used."""
    parser.add_argument("--registry", action="append", help="registry url for a shipped model id (repeatable)")
    parser.add_argument("--offline", action="store_true", help="never touch the network")


def add_model_source(parser) -> None:
    """The options of a command that resolves a model: ``add_registry``'s, ``--revision`` and ``--hf-token``."""
    add_registry(parser)
    parser.add_argument("--revision", default=None,
                        help="pin a Hugging Face branch/tag/commit (for hf://org/repo) or a catalog version")
    parser.add_argument("--hf-token", default=None,
                        help="Hugging Face token for a private/gated repo (falls back to $HF_TOKEN, then "
                             "$HUGGING_FACE_HUB_TOKEN)")


def model_sources(args) -> list[str]:
    """The registries ``--registry`` names, else the configured ones (``registry.sources.resolve_sources``)."""
    from ..registry import sources
    return sources.resolve_sources(args.registry)


def resolve_model_path(args, sources: list[str] | None = None) -> str:
    """The local path of the model ``args.model`` names (a path, a registry id or a Hub repo), resolved with the
    ``add_model_source`` options against ``sources`` (default: ``model_sources``). A failure is raised classified: an
    AlignAIRError as it is, an OSError or a ValueError as a RegistryError."""
    from ..registry import resolve_model
    try:
        return str(resolve_model(args.model, sources=model_sources(args) if sources is None else sources,
                                 offline=args.offline, token=args.hf_token, revision=args.revision))
    except (AlignAIRError, OSError, ValueError) as error:
        raise classify(error, error_type=RegistryError, message=f"could not resolve model '{args.model}'",
                       context={"model_id": args.model})                # classify keeps or sets the cause


def add_report_output(parser, *, json_help: str) -> None:
    """``--out`` (the report's file, default stdout) and ``--json`` (``json_help`` says what it emits)."""
    parser.add_argument("--out", default=None, help="write the report here (default: stdout)")
    parser.add_argument("--json", action="store_true", help=json_help)


def emit(args, text: str) -> None:
    """Write ``text`` to ``--out`` and say so, or print it."""
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
