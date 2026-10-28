"""Own fresh convergence output directories and publish finite report artifacts."""
import json
from pathlib import Path

from ...errors import ConfigurationError, InputError, OperationIOError, error_context
from ...io.atomic import atomic_write_text


class StudyArtifacts:
    def __init__(self, art_dir, ckpt_dir):
        with error_context(stage="convergence_artifacts"):
            paths = []
            for field, value in (("art_dir", art_dir), ("ckpt_dir", ckpt_dir)):
                try:
                    if value == "":
                        raise ValueError
                    path = Path(value)
                    resolved = path.resolve()
                    exists = path.exists() or path.is_symlink()
                except (TypeError, ValueError) as exc:
                    raise ConfigurationError("study destinations must be directory paths", context={"field": field}) from exc
                except OSError as exc:
                    raise OperationIOError("cannot inspect study destination", context={"field": field}) from exc
                if exists:
                    raise ConfigurationError("study output directory already exists; choose a fresh destination",
                                             context={"field": field, "path": str(path)})
                paths.append((path, resolved))
            (self.art_dir, art_resolved), (self.ckpt_dir, ckpt_resolved) = paths
            if (art_resolved == ckpt_resolved or art_resolved in ckpt_resolved.parents
                    or ckpt_resolved in art_resolved.parents):
                raise ConfigurationError("study output directories must be distinct and nonoverlapping")
        self._created = False

    def create(self):
        created = []
        try:
            for path in (self.art_dir, self.ckpt_dir):
                path.mkdir(parents=True, exist_ok=False)
                created.append(path)
        except OSError as exc:
            # Only remove empty directories claimed by this call, never saved checkpoints or user files.
            for owned in reversed(created):
                try:
                    owned.rmdir()
                except OSError:
                    pass
            raise OperationIOError("cannot claim fresh study output directories",
                                   context={"stage": "convergence_artifacts", "path": str(path)}) from exc
        self._created = True

    def write_text(self, name, text):
        if not self._created or not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
            raise ConfigurationError("publish a named artifact only after claiming study outputs",
                                     context={"stage": "convergence_artifacts"})
        path = self.art_dir / name
        try:
            atomic_write_text(path, text, stage="convergence_artifacts")
        except OSError as exc:
            raise OperationIOError("cannot publish convergence artifact",
                                   context={"stage": "convergence_artifacts", "path": str(path)}) from exc

    def write_json(self, name, value):
        try:
            text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise InputError("convergence artifacts require finite JSON values",
                             context={"stage": "convergence_artifacts", "field": name}) from exc
        self.write_text(name, text)
