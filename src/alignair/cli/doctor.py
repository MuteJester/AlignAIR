"""``alignair doctor`` — print environment diagnostics (Python / Torch / CUDA / deps / versions).

Used as the Docker health check and the first step of the release smoke test, so it must exit 0 on a
usable environment and only surface WARNINGs (never a non-zero exit) for optional-but-missing pieces.
"""
from __future__ import annotations

import json
import platform
import sys

_CRITICAL = ("torch", "numpy")
_OPTIONAL = ("pandas", "GenAIRR", "safetensors", "zstandard", "huggingface_hub")


def register(sub) -> None:
    p = sub.add_parser("doctor", help="print environment diagnostics (python/torch/cuda/deps)")
    p.add_argument("--json", action="store_true", help="emit the diagnostics as JSON")
    p.set_defaults(func=run)


def _module_version(name: str) -> str:
    try:
        m = __import__(name)
    except Exception:                                  # noqa: BLE001 - report any import failure
        return "MISSING"
    return getattr(m, "__version__", "installed")


def diagnostics() -> dict:
    from .. import __version__
    info = {"alignair": __version__, "python": sys.version.split()[0],
            "platform": platform.platform(), "executable": sys.executable}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if getattr(torch.backends, "mps", None) is not None:
            info["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception as e:                             # noqa: BLE001
        info["torch"] = f"MISSING ({e})"
    for mod in _OPTIONAL + tuple(m for m in _CRITICAL if m != "torch"):
        info[mod] = _module_version(mod)
    return info


def run(args) -> int:
    info = diagnostics()
    if getattr(args, "json", False):
        print(json.dumps(info, indent=2))
    else:
        print("AlignAIR environment")
        for k, v in info.items():
            print(f"  {k:16s} {v}")
    missing = [m for m in _CRITICAL if str(info.get(m, "")).startswith("MISSING")]
    if missing:
        print(f"WARNING: missing critical dependency: {missing} — install with `pip install alignair[cli]`")
    return 0                                            # doctor is a diagnostic: healthy exit is always 0
