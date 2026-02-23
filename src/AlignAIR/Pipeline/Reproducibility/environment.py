"""Environment fingerprint — capture runtime versions for reproducibility."""
from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Snapshot of the runtime environment for provenance tracking.

    The composite ``fingerprint`` is a SHA-256 of the version-critical fields
    (python, tensorflow, numpy, alignair, genairr). If two runs share the same
    fingerprint AND determinism is enabled, outputs should be bit-identical.
    """

    python_version: str = ""
    platform: str = ""
    tf_version: str = ""
    numpy_version: str = ""
    pandas_version: str = ""
    scipy_version: str = ""
    sklearn_version: str = ""
    alignair_version: str = ""
    genairr_version: str = ""
    cuda_version: str = ""
    gpu_info: str = ""
    git_commit: str = ""
    fingerprint: str = ""

    @classmethod
    def capture(cls) -> EnvironmentFingerprint:
        """Snapshot the current environment."""
        fields: Dict[str, str] = {}

        fields["python_version"] = sys.version.split()[0]
        fields["platform"] = platform.platform()

        fields["tf_version"] = _safe_version("tensorflow")
        fields["numpy_version"] = _safe_version("numpy")
        fields["pandas_version"] = _safe_version("pandas")
        fields["scipy_version"] = _safe_version("scipy")
        fields["sklearn_version"] = _safe_version("sklearn")
        fields["alignair_version"] = _safe_version("AlignAIR")
        fields["genairr_version"] = _safe_version("GenAIRR")

        # CUDA
        fields["cuda_version"] = _get_cuda_version()

        # GPU
        fields["gpu_info"] = _get_gpu_info()

        # Git
        fields["git_commit"] = _get_git_commit()

        # Composite fingerprint of version-critical fields
        critical = "|".join([
            fields["python_version"],
            fields["tf_version"],
            fields["numpy_version"],
            fields["alignair_version"],
            fields["genairr_version"],
        ])
        fields["fingerprint"] = hashlib.sha256(critical.encode()).hexdigest()[:16]

        return cls(**fields)

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _safe_version(package: str) -> str:
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return "unknown"


def _get_cuda_version() -> str:
    try:
        import tensorflow as tf
        build = tf.sysconfig.get_build_info()
        return build.get("cuda_version", "none")
    except Exception:
        return "none"


def _get_gpu_info() -> str:
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return "; ".join(g.name for g in gpus)
        return "none"
    except Exception:
        return "none"


def _get_git_commit() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"
