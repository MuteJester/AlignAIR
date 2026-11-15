"""Reference-bound model preparation: a nucleotide reference, whose ordered alleles are the model's labels."""
from __future__ import annotations

from dataclasses import dataclass, fields

from ..errors import ConfigurationError
from ..reference.reference_set import ReferenceSet
from .config import AlignAIRConfig
from .model import AlignAIR
from .tokens import DEFAULT_WINDOW

# The configuration fields the reference decides (prepare_model), which options cannot set.
_FROM_REFERENCE = frozenset({"has_d", "num_chain_types"})


@dataclass(frozen=True)
class ModelSetup:
    """A configuration bound to its reference (``validate_reference_binding``), before any weight is allocated."""
    reference: ReferenceSet
    config: AlignAIRConfig

    def __post_init__(self):
        validate_reference_binding(self.reference, self.config)

    def build_model(self) -> AlignAIR:
        """A new model of this configuration, bound to this reference; architecture registration selects its
        components."""
        model = AlignAIR(self.config)
        model.bind_reference(self.reference)
        return model

    def restore_model(self, state_dict, *, device='cpu', source=None) -> AlignAIR:
        """A model of this configuration restored from ``state_dict`` in evaluation mode, bound to this reference.
        Weight compatibility and device placement belong to the shared model factory."""
        model = AlignAIR.from_state_dict(self.config, state_dict, device=device, source=source)
        model.bind_reference(self.reference)
        return model


def validate_reference_binding(reference, config) -> None:
    """Check that ``config`` can be bound to ``reference`` without allocating model parameters: the reference has
    alleles for each of the model's genes (a D-less model may read a reference with a D), and the model's locus head
    is the reference's locus schema. Artifact readers translate an invalid embedded binding to an integrity error."""
    if not isinstance(reference, ReferenceSet) or not isinstance(config, AlignAIRConfig):
        raise ConfigurationError('model binding requires a ReferenceSet and AlignAIRConfig')
    if config.has_d and not reference.has_d:
        raise ConfigurationError('model D selection disagrees with the reference', context={'field': 'has_d'})
    reference.validate(genes=tuple(gene.upper() for gene in config.genes))
    empty = [gene.upper() for gene in config.genes if not len(reference.gene(gene.upper()))]
    if empty:
        raise ConfigurationError(f"the reference has no {', '.join(empty)} alleles", context={'genes': empty})
    if config.num_chain_types != max(1, len(reference.loci)):
        raise ConfigurationError('chain count disagrees with the reference locus schema',
                                 context={'field': 'num_chain_types'})


def prepare_model(reference: ReferenceSet, *, has_d: bool | None = None, max_seq_length: int | None = None,
                  **config_options) -> ModelSetup:
    """The model setup for ``reference``: its configuration takes the D gene (unless ``has_d=False`` leaves out the
    reference's D) and the locus count from the reference, and the other ``AlignAIRConfig`` fields from
    ``config_options``. No weight is allocated."""
    if not isinstance(reference, ReferenceSet):
        raise ConfigurationError('prepare_model requires a ReferenceSet')
    selected_d = reference.has_d if has_d is None else has_d
    if type(selected_d) is not bool or selected_d and not reference.has_d:
        raise ConfigurationError('requested D head is unavailable in the source reference', context={'field': 'has_d'})
    unknown = set(config_options) - {field.name for field in fields(AlignAIRConfig)}
    invalid = sorted(unknown | (set(config_options) & _FROM_REFERENCE))
    if invalid:
        raise ConfigurationError('invalid model configuration options', context={'fields': invalid})
    config = AlignAIRConfig(has_d=selected_d, num_chain_types=max(1, len(reference.loci)),
                            max_seq_length=DEFAULT_WINDOW if max_seq_length is None else max_seq_length,
                            **config_options)
    return ModelSetup(reference, config)
