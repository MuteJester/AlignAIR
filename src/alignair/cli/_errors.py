"""Report a command's expected failure and return its exit code."""
from ..errors import AlignAIRError, InputError, report_error
from ..errors.context import snapshot


def classify(error: Exception, *, error_type=InputError, message=None, context=None) -> AlignAIRError:
    """``error`` as an AlignAIRError. An already-classified error keeps its type, message and cause chain, and gains
    the ``context`` keys it lacks (its own keys win). A built-in error becomes ``error_type`` with ``message``
    (default: its own) and ``context``, caused by ``error``."""
    if isinstance(error, AlignAIRError):
        if context:
            error._context = snapshot({**context, **error._context})
        return error
    classified = error_type(str(error) if message is None else message, context=context)
    classified.__cause__ = error
    return classified


def fail(error: Exception, *, error_type=InputError, message=None, context=None) -> int:
    """Report a caught, expected failure (classified as ``classify`` does). Unexpected exceptions must propagate to
    main."""
    return report_error(classify(error, error_type=error_type, message=message, context=context))
