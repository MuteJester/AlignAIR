"""Public exception contract, safe context snapshots, and worker-process serialization."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from enum import IntEnum
import json

from .context import current_error_context, snapshot


class ExitCode(IntEnum):
    SUCCESS = 0
    ERROR = 1
    USAGE = 2
    INTERNAL = 70
    INTERRUPTED = 130


def _restore_error(cls, args, state):
    # Constructors such as AirrAssemblyError(ident, original) do not accept .args.
    error = cls.__new__(cls)
    Exception.__init__(error, *args)
    error.__dict__.update(state)
    return error


class AlignAIRError(Exception):
    """Base for classified AlignAIR failures; built-in compatibility is added by subclasses.

    ``str(error)`` remains the plain message for existing callers. Codes and the v1
    ``to_dict`` envelope are the machine interface; callers must not parse messages.
    Cause chains use Python's ``raise ... from ...``. Nothing logs automatically.
    """
    code = "ALIGNAIR_ERROR"
    category = "operation"
    exit_code = ExitCode.ERROR

    def __init__(self, message: str = "", *, context: Mapping | None = None,
                 hint: str | None = None, retryable: bool = False):
        self.message = str(message)
        self.hint = hint
        self.retryable = retryable
        self._context = snapshot({**current_error_context(), **(context or {})})
        super().__init__(self.message)

    @property
    def context(self) -> dict:
        """A copy: modifying the returned data cannot alter the captured diagnostic."""
        return deepcopy(self._context)

    def to_dict(self) -> dict:
        """Versioned JSON-compatible diagnostic; excludes raw causes and tracebacks."""
        return {"schema_version": 1, "type": type(self).__name__, "code": self.code,
                "category": self.category, "exit_code": int(self.exit_code),
                "message": self.message, "hint": self.hint,
                "context": self.context, "retryable": self.retryable}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True)

    def __reduce__(self):
        # Tracebacks/causes are process-local, matching built-in exception behavior.
        return _restore_error, (type(self), self.args, self.__dict__)
