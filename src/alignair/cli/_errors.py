"""Compatibility bridge for command handlers that return exit codes directly."""
from ..errors import AlignAIRError, InputError, report_error


def fail(error: Exception, *, error_type=InputError, message=None, context=None) -> int:
    """Report a caught, expected failure. Unexpected exceptions must propagate to main.

    Already-classified errors keep their exact type, diagnostic, and cause chain.
    Legacy built-in errors are translated explicitly by the command's caller.
    """
    if isinstance(error, AlignAIRError):
        return report_error(error)
    classified = error_type(str(error) if message is None else message, context=context)
    classified.__cause__ = error
    return report_error(classified)
