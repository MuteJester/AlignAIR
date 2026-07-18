"""Shared provenance helpers — used by both the bundle metadata (training-time) and the
prediction run.json (inference-time) so artifacts carry enough to reproduce/audit them:
package versions, the AlignAIR source commit, CUDA detail, and content hashes."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from typing import Optional

_DEFAULT_PACKAGES = ("torch", "GenAIRR", "numpy", "pandas", "airr", "parasail", "huggingface-hub")


def alignair_version() -> str:
    try:
        from importlib.metadata import version
        return version("AlignAIR")
    except Exception:
        return "0+unknown"


def package_versions(names=_DEFAULT_PACKAGES) -> dict:
    """{dist_name: version-or-None} for the packages that shape a result, plus python."""
    from importlib.metadata import version
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except Exception:
            out[n] = None
    out["python"] = sys.version.split()[0]
    return out


def git_commit_sha(short: bool = True) -> Optional[str]:
    """The commit of the AlignAIR source tree, or None when not a git checkout (e.g. a wheel
    install). Best-effort: never raises, short timeout."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        r = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=2)
        if r.returncode == 0 and r.stdout.strip():
            sha = r.stdout.strip()
            return sha[:12] if short else sha
    except Exception:
        pass
    return None


def cuda_detail() -> dict:
    """Device/CUDA detail for the active torch install (honest absence if torch is missing)."""
    try:
        import torch
        avail = bool(torch.cuda.is_available())
        return {"available": avail, "cuda_version": getattr(torch.version, "cuda", None),
                "device": torch.cuda.get_device_name(0) if avail else "cpu",
                "torch": torch.__version__}
    except Exception as e:
        return {"available": False, "error": str(e)}


def hash_json(obj) -> Optional[str]:
    """Stable content hash of any JSON-able object (None -> None)."""
    if obj is None:
        return None
    return "sha256:" + hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def reference_hash(reference_set) -> Optional[str]:
    """Content hash of a ReferenceSet: every gene's sorted (allele -> sequence) mapping."""
    if reference_set is None:
        return None
    try:
        payload = {}
        for g in reference_set.genes:
            ref = reference_set.gene(g)
            payload[g] = dict(sorted(zip(ref.names, ref.sequences)))
        return hash_json(payload)
    except Exception:
        return None
