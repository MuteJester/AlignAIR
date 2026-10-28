"""Calibration-group selection contract (§7 of the AA inference product)."""

__all__ = [
    "SUPPLIED_WARNING",
    "VALID_GROUPS",
    "resolve_group",
]

SUPPLIED_WARNING = (
    "Group supplied by caller; conditional coverage applies only if this label "
    "matches the documented calibration-group definition and the input is "
    "exchangeable with that simulator group."
)

VALID_GROUPS = ("full", "trunc", "short")


def resolve_group(coverage_group, sequence_id, index):
    """
    Resolve the calibration group for a sequence.

    Priority order (§7):
    1. Explicit — caller supplies a group string or per-sequence mapping.
    2. Pooled fallback (default) — no group supplied or not found in mapping.

    Args:
        coverage_group: One of:
            - None -> pooled fallback
            - str in VALID_GROUPS -> supplied group
            - str not in VALID_GROUPS -> raise ValueError
            - dict {sequence_id: group} -> look up sequence_id, resolve by str rules
            - list/tuple aligned by index -> take [index], resolve by rules
        sequence_id: The sequence identifier (used for dict lookup)
        index: The sequence index in the batch (used for list/tuple lookup)

    Returns:
        tuple[str, str, str|None]: (group, source, warning)
            - group: "pooled", "full", "trunc", or "short"
            - source: "pooled_fallback" or "supplied"
            - warning: None or SUPPLIED_WARNING text
    """
    # Case 1: None -> pooled fallback (the honest default)
    if coverage_group is None:
        return ("pooled", "pooled_fallback", None)

    # Case 2: dict -> per-sequence lookup
    if isinstance(coverage_group, dict):
        if sequence_id in coverage_group:
            # Recursively resolve the looked-up value (may be None or str)
            return resolve_group(
                coverage_group[sequence_id], sequence_id, index
            )
        else:
            # Absent sequence_id -> pooled fallback
            return ("pooled", "pooled_fallback", None)

    # Case 3: list or tuple -> per-sequence lookup by index
    if isinstance(coverage_group, (list, tuple)):
        if index < 0 or index >= len(coverage_group):
            raise ValueError(
                f"Index {index} out of range for coverage_group of length {len(coverage_group)}"
            )
        # Recursively resolve the looked-up value (may be None or str)
        return resolve_group(coverage_group[index], sequence_id, index)

    # Case 4: string -> validate and resolve
    if isinstance(coverage_group, str):
        if coverage_group not in VALID_GROUPS:
            raise ValueError(
                f"coverage_group must be one of {VALID_GROUPS} or None, got {coverage_group!r}"
            )
        # Valid group string -> supplied source + warning
        return (coverage_group, "supplied", SUPPLIED_WARNING)

    # Unknown type
    raise TypeError(
        f"coverage_group must be None, str, dict, or list/tuple, got {type(coverage_group).__name__}"
    )
