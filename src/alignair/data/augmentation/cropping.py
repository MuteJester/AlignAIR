"""Validated crop windows and explicit reference-relative compatibility helpers."""
from operator import index

from ...errors import ConfigurationError, TrainingDataError


def _integer(value):
    if isinstance(value, bool):
        raise TypeError
    return index(value)


def crop_window(n_res: int, kind: str, keep: int, start: int | None = None):
    try:
        n_res, keep = _integer(n_res), _integer(keep)
        if n_res <= 0:
            raise ValueError
        if start is not None:
            start = _integer(start)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("crop lengths must be integers and the source must be nonempty") from exc
    if kind not in ("nterm", "cterm", "internal"):
        raise ConfigurationError("crop kind must be nterm, cterm, or internal")
    keep = max(1, min(keep, n_res))
    if kind == "nterm":
        return 0, keep
    if kind == "cterm":
        return n_res - keep, n_res
    lo = start if start is not None else max(0, (n_res - keep) // 2)
    lo = max(0, min(lo, n_res - keep))
    return lo, lo + keep


def shift_clip(boundary, lo: int, hi: int):
    """Intersect a half-open span with the crop and express it in the crop's coordinates."""
    try:
        s, e = map(_integer, boundary)
        lo, hi = _integer(lo), _integer(hi)
        if not 0 <= s <= e or not 0 <= lo < hi:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise TrainingDataError("invalid half-open crop or gene span") from exc
    ns, ne = max(s, lo) - lo, min(e, hi) - lo
    return (ns, ne) if ne > ns else None


def compatible_classes(true_class: int, ga: int, gb: int, class_protos: dict) -> set:
    """Legacy helper for already aligned, reference-relative prototype windows."""
    if true_class not in class_protos:
        raise ConfigurationError("true class is absent from the prototype vocabulary")
    try:
        ga, gb = _integer(ga), _integer(gb)
        if not 0 <= ga < gb <= len(class_protos[true_class]):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise TrainingDataError("prototype window must contain observed reference residues") from exc
    key = class_protos[true_class][ga:gb]
    return {label for label, sequence in class_protos.items() if sequence[ga:gb] == key}


def fragment_compatible_classes(fragments, class_protos) -> set:
    """Classes containing each reliable fragment in order, with unknown positions/gaps permitted."""
    fragments = tuple(fragments)
    if not fragments or any(not fragment or "X" in fragment for fragment in fragments):
        raise TrainingDataError("fragment compatibility requires nonempty, known residue fragments")
    compatible = set()
    for label, sequence in class_protos.items():
        cursor = 0
        for fragment in fragments:
            match = sequence.find(fragment, cursor)
            if match < 0:
                break
            cursor = match + len(fragment)
        else:
            compatible.add(label)
    return compatible
