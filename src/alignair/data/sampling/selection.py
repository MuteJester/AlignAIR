"""Class-diversity selection policies independent of input alphabet and sample provider."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from operator import index
from types import MappingProxyType

from ...errors import ConfigurationError, TrainingDataError


class SampleDiversityError(TrainingDataError, ValueError):
    """A prepared cohort does not meet its declared class-diversity floors."""
    code = "ALIGNAIR_SAMPLE_DIVERSITY"


@dataclass(frozen=True)
class ClassDiversity:
    """Prefer unseen class IDs until every field meets its minimum; then accept freely.

    Samples must contain nonnegative integer class IDs for the declared fields. IDs representing
    unknown labels (such as -1) are invalid here. Selection is stateless: all decisions derive
    from the caller's selected samples. This greedy rule does not guarantee quotas within a
    finite budget or batch size; call ``validate`` on the completed cohort.
    """
    minimums: Mapping[str, int]

    def __post_init__(self):
        if not isinstance(self.minimums, Mapping) or not self.minimums:
            raise ConfigurationError("class diversity requires a nonempty mapping of fields to minimums")
        minimums = {}
        for field, minimum in self.minimums.items():
            try:
                if not isinstance(field, str) or not field or isinstance(minimum, bool) or index(minimum) <= 0:
                    raise ValueError
                minimums[field] = index(minimum)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("diversity fields need names and positive integer minimums",
                                         context={"field": str(field)}) from exc
        object.__setattr__(self, "minimums", MappingProxyType(minimums))

    def _labels(self, sample):
        labels = {}
        for field in self.minimums:
            try:
                value = sample[field]
                if isinstance(value, bool) or index(value) < 0:
                    raise ValueError
                labels[field] = index(value)
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise TrainingDataError("diversity selection requires observed nonnegative integer class IDs",
                                        context={"stage": "sample_diversity", "field": field}) from exc
        return labels

    def _seen(self, samples):
        seen = {field: set() for field in self.minimums}
        for sample in samples:
            for field, value in self._labels(sample).items():
                seen[field].add(value)
        return seen

    def __call__(self, sample, selected) -> bool:
        labels, seen = self._labels(sample), self._seen(selected)
        if all(len(seen[field]) >= minimum for field, minimum in self.minimums.items()):
            return True
        return any(value not in seen[field] for field, value in labels.items())

    def validate(self, samples) -> None:
        observed = {field: len(values) for field, values in self._seen(samples).items()}
        if any(observed[field] < minimum for field, minimum in self.minimums.items()):
            raise SampleDiversityError("batch not diverse enough for the declared class minimums",
                                       context={"stage": "sample_diversity", "observed": observed,
                                                "required": dict(self.minimums)})


def diverse_selector(min_v_classes: int = 3, min_j_classes: int = 2) -> ClassDiversity:
    """Historical V/J proof policy, expressed through the shared field-based selector."""
    return ClassDiversity({"v_class": min_v_classes, "j_class": min_j_classes})


def assert_batch_diverse(samples, min_v_classes: int = 3, min_j_classes: int = 2) -> None:
    """Validate the historical V/J proof floors through the shared policy."""
    diverse_selector(min_v_classes, min_j_classes).validate(samples)
