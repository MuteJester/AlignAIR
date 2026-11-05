"""Reference-bound model preparation: a nucleotide reference, whose ordered alleles are the model's labels."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from operator import index
from typing import Literal

from ..errors import ConfigurationError, error_context
from ..reference.reference_set import ReferenceSet
from .config import AlignAIRConfig, _reference_productive_unknown, _reference_short_d_penalty
from .model import AlignAIR

InputSpace = Literal['nucleotide']
LabelSpace = Literal['alleles']


def _spaces(input_space, label_space):
    if input_space != 'nucleotide':
        raise ConfigurationError('input_space must be nucleotide: AlignAIR reads nucleotide sequences',
                                 context={'field': 'input_space'})
    if label_space != 'alleles':
        raise ConfigurationError('label_space must be alleles', context={'field': 'label_space'})


def _count(value, field, minimum=1):
    try:
        if isinstance(value, bool):
            raise ValueError
        value = index(value)
        if value < minimum:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f'{field} must be an integer >= {minimum}', context={'field': field}) from exc
    return value


def _input_config(config):
    from ..data.tokenizer import TOKEN_DICT
    if config.vocab_size != len(TOKEN_DICT):
        raise ConfigurationError('model vocabulary disagrees with the nucleotide tokenizer',
                                 context={'field': 'vocab_size', 'expected': len(TOKEN_DICT)})


def build_model_config(v_label_count: int, j_label_count: int, d_label_count: int = 0, *,
                       input_space: InputSpace = 'nucleotide', label_space: LabelSpace = 'alleles',
                       has_d: bool = False, num_chain_types: int = 1, max_seq_length: int | None = None,
                       **config_options) -> AlignAIRConfig:
    """Select established defaults, with explicit counts and optional architecture/head settings.

    Count-only construction cannot establish label identity. Prefer prepare_model when a reference
    is available.
    """
    from ..data.tokenizer import TOKEN_DICT
    _spaces(input_space, label_space)
    if type(has_d) is not bool:
        raise ConfigurationError('has_d must be a boolean', context={'field': 'has_d'})
    v = _count(v_label_count, 'v_label_count')
    j = _count(j_label_count, 'j_label_count')
    d = _count(d_label_count, 'd_label_count', int(has_d))
    chains = _count(num_chain_types, 'num_chain_types')
    length = _count(576 if max_seq_length is None else max_seq_length, 'max_seq_length')
    protected = {'v_allele_count', 'j_allele_count', 'd_allele_count'}
    invalid = ((set(config_options) & protected)
               | (set(config_options) - {field.name for field in fields(AlignAIRConfig)}))
    if invalid:
        raise ConfigurationError('invalid model configuration options', context={'fields': sorted(invalid)})
    config = AlignAIRConfig(v, j, d if has_d else 0, has_d=has_d, num_chain_types=chains, max_seq_length=length,
                            vocab_size=len(TOKEN_DICT))
    config = replace(config, input_space=input_space, label_space=label_space, **config_options)
    _input_config(config)
    return config


@dataclass(frozen=True)
class ModelSetup:
    """Prepared reference/configuration binding; no model parameters have been allocated yet."""
    reference: ReferenceSet
    config: AlignAIRConfig
    input_space: InputSpace = 'nucleotide'
    label_space: LabelSpace = 'alleles'

    def __post_init__(self):
        self.validate()

    @property
    def label_counts(self) -> dict[str, int]:
        genes = ('v', 'd', 'j') if self.config.has_d else ('v', 'j')
        return {gene: len(self.reference.gene(gene)) for gene in genes}

    def validate(self) -> None:
        """Check label sizes and input semantics before allocating a model."""
        _spaces(self.input_space, self.label_space)
        if (not isinstance(self.config, AlignAIRConfig)
                or (self.config.input_space, self.config.label_space) != (self.input_space, self.label_space)):
            raise ConfigurationError('model configuration disagrees with its setup input/label spaces')
        validate_reference_binding(self.reference, self.config)

    def build_model(self) -> AlignAIR:
        """Allocate the shared model after preflight; architecture registration selects components."""
        self.validate()
        model = AlignAIR(self.config)
        model.bind_reference(self.reference)
        return model

    def restore_model(self, state_dict, *, device='cpu', source=None) -> AlignAIR:
        """Revalidate this prepared binding, then restore a fresh model in evaluation mode. Weight compatibility and
        device placement belong to the shared model factory."""
        with error_context(stage='model_binding', path=source):
            self.validate()
        model = AlignAIR.from_state_dict(self.config, state_dict, device=device, source=source)
        model.bind_reference(self.reference)
        return model

    def label_schema(self) -> dict:
        """Portable output-column identity: each gene's alleles in the reference's order."""
        self.validate()
        genes = {spec.name: list(self.reference.gene(spec.name.upper()).names) for spec in self.config.gene_specs}
        return {'schema_version': 1, 'input_space': self.input_space, 'label_space': self.label_space,
                'genes': genes}


def validate_label_schema(schema, reference, config) -> None:
    """Validate persisted label identity against the model's head shapes and source allele panel."""
    if (not isinstance(schema, dict) or schema.get('schema_version') != 1
            or schema.get('input_space') != config.input_space or schema.get('label_space') != config.label_space
            or not isinstance(schema.get('genes'), dict)
            or set(schema['genes']) != {spec.name for spec in config.gene_specs}):
        raise ConfigurationError('label schema disagrees with the model input/label spaces or genes')
    for spec in config.gene_specs:
        labels = schema['genes'][spec.name]
        if not isinstance(labels, list) or len(labels) != spec.allele_count:
            raise ConfigurationError('label schema length disagrees with the model head', context={'gene': spec.name})
        if labels != list(reference.gene(spec.name.upper()).names):
            raise ConfigurationError('allele label order disagrees with the source reference', context={'gene': spec.name})


def validate_reference_binding(reference, config, *, label_schema=None) -> None:
    """Check a reference/configuration binding without allocating model parameters: the heads use the reference's
    ordered allele panel. Artifact readers still own reference hashes and calibration binding, and translate invalid
    embedded bindings to integrity errors."""
    if not isinstance(reference, ReferenceSet) or not isinstance(config, AlignAIRConfig):
        raise ConfigurationError('model binding requires a ReferenceSet and AlignAIRConfig')
    _spaces(config.input_space, config.label_space)
    if type(config.has_d) is not bool or config.has_d and not reference.has_d:
        raise ConfigurationError('model D selection disagrees with the reference', context={'field': 'has_d'})
    reference.validate(genes=tuple(spec.name.upper() for spec in config.gene_specs))
    if label_schema is not None:
        validate_label_schema(label_schema, reference, config)
    for spec in config.gene_specs:
        expected = _count(len(reference.gene(spec.name.upper())), f'{spec.name}_label_count')
        actual = _count(spec.allele_count, f'{spec.name}_allele_count')
        if actual != expected:
            raise ConfigurationError('model head size disagrees with its reference labels',
                                     context={'gene': spec.name, 'expected': expected, 'actual': actual})
    if not config.has_d and _count(config.d_allele_count, 'd_allele_count', 0) != 0:
        raise ConfigurationError('a disabled D head must have zero labels', context={'field': 'd_allele_count'})
    _count(config.max_seq_length, 'max_seq_length')
    if _count(config.num_chain_types, 'num_chain_types') != max(1, len(reference.loci)):
        raise ConfigurationError('chain count disagrees with the reference locus schema', context={'field': 'num_chain_types'})
    _input_config(config)


def setup_from_schema(reference, config, schema) -> ModelSetup:
    """Rebind a saved label schema to its reference."""
    validate_label_schema(schema, reference, config)
    return ModelSetup(reference, config, config.input_space, config.label_space)


def prepare_model(reference: ReferenceSet, *, input_space: InputSpace = 'nucleotide',
                  label_space: LabelSpace = 'alleles', has_d: bool | None = None,
                  max_seq_length: int | None = None, **config_options) -> ModelSetup:
    """Bind one ordered reference to the model's labels without allocating model weights. has_d=False can omit a
    source D head without changing the source reference."""
    _spaces(input_space, label_space)
    if not isinstance(reference, ReferenceSet):
        raise ConfigurationError('prepare_model requires a ReferenceSet')
    selected_d = reference.has_d if has_d is None else has_d
    if type(selected_d) is not bool or selected_d and not reference.has_d:
        raise ConfigurationError('requested D head is unavailable in the source reference', context={'field': 'has_d'})
    genes = ('V', 'J', 'D') if selected_d else ('V', 'J')
    reference.validate(genes=genes)
    counts = {gene: len(reference.gene(gene)) for gene in genes}
    protected = {'v_label_count', 'j_label_count', 'd_label_count', 'num_chain_types'}
    if protected.intersection(config_options):
        raise ConfigurationError('reference label and locus counts cannot be overridden',
                                 context={'fields': sorted(protected.intersection(config_options))})
    config = build_model_config(counts['V'], counts['J'], counts.get('D', 0), input_space=input_space,
        label_space=label_space, has_d=selected_d, num_chain_types=max(1, len(reference.loci)),
        max_seq_length=max_seq_length, **config_options)
    if 'short_d_penalty' not in config_options:
        config = replace(config, short_d_penalty=_reference_short_d_penalty(reference,
            has_d=config.has_d, classification_mode=config.classification_mode))
    if 'productive_unknown' not in config_options:
        config = replace(config, productive_unknown=_reference_productive_unknown(use_meta_heads=config.use_meta_heads))
    return ModelSetup(reference, config, input_space, label_space)
