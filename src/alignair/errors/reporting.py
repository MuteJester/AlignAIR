"""Opt-in text/JSON reporting and logging; never configures global logging."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import sys
import traceback

from .base import AlignAIRError

_OPTIONS: ContextVar[tuple[bool, bool]] = ContextVar("alignair_error_reporting", default=(False, False))


@contextmanager
def error_reporting(*, debug: bool = False, json_output: bool = False):
    """Task-local CLI reporting options, restored even when an operation fails."""
    token = _OPTIONS.set((debug, json_output))
    try:
        yield
    finally:
        _OPTIONS.reset(token)


def format_error(error: AlignAIRError, *, json_output: bool = False) -> str:
    """Format a classified failure without logging it or changing its traceback."""
    if json_output:
        return error.to_json()
    lines = [f"[{error.code}] {error.message}"]
    if error.context:
        lines.append("Context: " + json.dumps(error.context, ensure_ascii=False, allow_nan=False,
                                             sort_keys=True))
    if error.hint:
        lines.append("Hint: " + error.hint)
    return "\n".join(lines)


def report_error(error: AlignAIRError, *, stream=None, debug: bool | None = None,
                 json_output: bool | None = None) -> int:
    """Write to stderr and return the exit code. Debug tracebacks are opt-in.

    JSON remains a single diagnostic object, including in debug mode. Debug output
    can contain upstream messages; context redaction does not sanitize tracebacks.
    """
    default_debug, default_json = _OPTIONS.get()
    debug = default_debug if debug is None else debug
    json_output = default_json if json_output is None else json_output
    stream = sys.stderr if stream is None else stream
    if json_output and debug:
        payload = error.to_dict()
        payload["traceback"] = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
    else:
        text = format_error(error, json_output=json_output)
    print(text, file=stream)
    if debug and not json_output:
        traceback.print_exception(type(error), error, error.__traceback__, file=stream)
    return int(error.exit_code)


def log_error(logger: logging.Logger, error: AlignAIRError, *, level: int = logging.ERROR,
              debug: bool = False) -> None:
    """Attach the same diagnostic under LogRecord.alignair_error; log once at the boundary."""
    logger.log(level, "[%s] %s", error.code, error.message,
               extra={"alignair_error": error.to_dict()},
               exc_info=(type(error), error, error.__traceback__) if debug else None)
