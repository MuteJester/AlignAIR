"""Report a command's expected failure and return its exit code."""
from ..errors import AlignAIRError, InputError, report_error
from ..errors.context import snapshot


def fail(error: Exception, *, error_type=InputError, message=None, context=None) -> int:
    """Report a caught, expected failure. Unexpected exceptions must propagate to main.

    An already-classified error keeps its type, message and cause chain, and gains the ``context`` keys it lacks (its
    own keys win). A built-in error is reported as ``error_type`` with ``message`` (default: its own) and ``context``.
    """
    if isinstance(error, AlignAIRError):
        if context:
            error._context = snapshot({**context, **error._context})
        return report_error(error)
    classified = error_type(str(error) if message is None else message, context=context)
    classified.__cause__ = error
    return report_error(classified)
