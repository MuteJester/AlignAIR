"""Validate observed simulator spans without inventing missing germline truth."""
from operator import index

from ...errors import TrainingDataError

GENES = ("v", "d", "j")


def _coordinate(value):
    # Native simulator integers and serialized integer strings are supported.
    # int(1.9) and int(True) would silently turn invalid annotations into labels.
    if isinstance(value, bool):
        raise ValueError("boolean coordinate")
    return int(value) if isinstance(value, str) else index(value)


def segment_spans(record):
    """Return query/germline maps for visible, nonempty V/D/J segments.

    GenAIRR keeps a zero-length query boundary for a fully removed gene, while
    omitting its alignment and germline coordinates. This is an absent segment.
    A nonempty query segment still requires actual germline annotations.
    """
    length = len(record["sequence"])
    query, germline = {}, {}
    for gene in GENES:
        fields = [f"{gene}_sequence_start", f"{gene}_sequence_end"]
        values = [record.get(field) for field in fields]
        if values == [None, None]:
            continue
        try:
            start, end = map(_coordinate, values)
            if not 0 <= start <= end <= length:
                raise ValueError("query coordinates outside the read")
        except (TypeError, ValueError, OverflowError) as exc:
            raise TrainingDataError("Invalid query-coordinate span.",
                                    context={"gene": gene.upper(), "fields": fields}) from exc
        if start == end:
            continue
        fields = [f"{gene}_germline_start", f"{gene}_germline_end"]
        values = [record.get(field) for field in fields]
        if any(value is None for value in values):
            raise TrainingDataError(
                f"{gene.upper()} query coordinates exist, but germline coordinates are unavailable.",
                context={"gene": gene.upper(), "fields": fields},
                hint="Inspect the simulator record and coordinate contract; no germline coordinates were inferred."
            ) from KeyError(gene)
        try:
            gs, ge = map(_coordinate, values)
            if not 0 <= gs <= ge:
                raise ValueError("invalid germline coordinates")
        except (TypeError, ValueError, OverflowError) as exc:
            raise TrainingDataError("Invalid germline-coordinate span.",
                                    context={"gene": gene.upper(), "fields": fields}) from exc
        query[gene] = (start, end)
        germline[gene] = (gs, ge)
    return query, germline
