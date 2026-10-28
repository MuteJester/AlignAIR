"""Reference-bound model preparation with independent input and label-space policies."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from operator import index
from typing import Literal

from ..errors import ConfigurationError
from ..reference.protein import ProteinReference, build_protein_reference
from ..reference.labels import build_equivalence_classes
from ..reference.reference_set import ReferenceSet
from .config import AlignAIRConfig
from .model import AlignAIR

InputSpace = Literal['nucleotide', 'protein']
LabelSpace = Literal['alleles', 'protein_classes']


def _spaces(input_space, label_space):
    if input_space not in ('nucleotide', 'protein'):
        raise ConfigurationError('input_space must be nucleotide or protein', context={'field': 'input_space'})
    if label_space not in ('alleles', 'protein_classes'):
        raise ConfigurationError('label_space must be alleles or protein_classes', context={'field': 'label_space'})


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


def _input_config(config, input_space):
    from ..data.tokenizer import AA_VOCAB_SIZE, TOKEN_DICT
    expected = AA_VOCAB_SIZE if input_space == 'protein' else len(TOKEN_DICT)
    if config.vocab_size != expected:
        raise ConfigurationError('model vocabulary disagrees with the selected input space',
                                 context={'field': 'vocab_size', 'expected': expected})
    if input_space == 'protein' and config.use_orientation:
        raise ConfigurationError('protein input cannot use nucleotide orientation correction',
                                 context={'field': 'use_orientation'})


def _protein_labels(view, source):
    if not isinstance(view, ProteinReference) or view.source is not source:
        raise ConfigurationError('protein classes require a protein view of the same reference')
    for gene in ('V', 'J'):
        labels = view.class_maps.get(gene)
        if labels is None or labels.gene != gene:
            raise ConfigurationError('protein class map is missing or bound to the wrong gene', context={'gene': gene})
        labels.validate()
        translations = {name: row.aa_seq for name, row in view.germlines.genes.get(gene, {}).items()}
        if not set(translations) <= set(source.gene(gene).names):
            raise ConfigurationError('translated alleles are absent from the source reference', context={'gene': gene})
        expected = build_equivalence_classes(gene, translations, view.db_hash)
        actual_groups = {frozenset(names) for names in labels.class_members.values()}
        expected_groups = {frozenset(names) for names in expected.class_members.values()}
        if actual_groups != expected_groups or labels.db_hash != view.db_hash:
            raise ConfigurationError('protein class maps disagree with the translated reference', context={'gene': gene})


def build_model_config(v_label_count: int, j_label_count: int, d_label_count: int = 0, *,
                       input_space: InputSpace = 'nucleotide', label_space: LabelSpace = 'alleles',
                       has_d: bool = False, num_chain_types: int = 1, max_seq_length: int | None = None,
                       **config_options) -> AlignAIRConfig:
    """Select established defaults, with explicit counts and optional architecture/head settings.

    Count-only construction cannot establish label identity. Prefer prepare_model when a reference
    is available. Protein allele defaults retain their historical residue boundaries and meta heads.
    """
    from ..data.tokenizer import AA_VOCAB_SIZE, TOKEN_DICT
    _spaces(input_space, label_space)
    if type(has_d) is not bool:
        raise ConfigurationError('has_d must be a boolean', context={'field': 'has_d'})
    v = _count(v_label_count, 'v_label_count')
    j = _count(j_label_count, 'j_label_count')
    d = _count(d_label_count, 'd_label_count', int(has_d))
    chains = _count(num_chain_types, 'num_chain_types')
    length = (192 if input_space == 'protein' else 576) if max_seq_length is None else max_seq_length
    length = _count(length, 'max_seq_length')
    protected = {'v_allele_count', 'j_allele_count', 'd_allele_count'}
    invalid = ((set(config_options) & protected)
               | (set(config_options) - {field.name for field in fields(AlignAIRConfig)}))
    if invalid:
        raise ConfigurationError('invalid model configuration options', context={'fields': sorted(invalid)})
    config = AlignAIRConfig(v, j, d if has_d else 0, has_d=has_d, num_chain_types=chains,
                            max_seq_length=length,
                            vocab_size=AA_VOCAB_SIZE if input_space == 'protein' else len(TOKEN_DICT),
                            use_orientation=input_space == 'nucleotide')
    if label_space == 'protein_classes':
        config = replace(config.with_amino_acid_heads(), use_orientation=input_space == 'nucleotide')
    config = replace(config, input_space=input_space, label_space=label_space, **config_options)
    _input_config(config, input_space)
    return config


@dataclass(frozen=True)
class ModelSetup:
    """Prepared reference/configuration binding; no model parameters have been allocated yet."""
    reference: ReferenceSet
    config: AlignAIRConfig
    input_space: InputSpace
    label_space: LabelSpace
    protein_reference: ProteinReference | None = None

    def __post_init__(self):
        self.validate()

    @property
    def label_counts(self) -> dict[str, int]:
        genes = ('v', 'd', 'j') if self.config.has_d else ('v', 'j')
        return {gene: (self.protein_reference.class_maps[gene.upper()].n_classes()
                       if self.label_space == 'protein_classes' and gene != 'd'
                       else len(self.reference.gene(gene))) for gene in genes}

    def validate(self) -> None:
        """Check label sizes and input semantics before allocating a model."""
        _spaces(self.input_space, self.label_space)
        validate_reference_binding(self.reference, self.config, protein_reference=self.protein_reference)
        if (self.config.input_space, self.config.label_space) != (self.input_space, self.label_space):
            raise ConfigurationError('model configuration disagrees with its setup input/label spaces')

    def build_model(self) -> AlignAIR:
        """Allocate the shared model after preflight; architecture registration selects components."""
        self.validate()
        return AlignAIR(self.config)

    def label_schema(self) -> dict:
        """Portable output-column identity; preserve class order instead of rebuilding it on reload."""
        self.validate()
        genes = {}
        for spec in self.config.gene_specs:
            if self.label_space == 'protein_classes' and spec.name != 'd':
                labels = self.protein_reference.class_maps[spec.name.upper()]
                genes[spec.name] = [list(labels.class_members[index]) for index in range(labels.n_classes())]
            else:
                genes[spec.name] = list(self.reference.gene(spec.name.upper()).names)
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
        names = list(reference.gene(spec.name.upper()).names)
        if config.label_space == 'protein_classes' and spec.name != 'd':
            if any(not isinstance(group, list) or not group for group in labels):
                raise ConfigurationError('class labels must be nonempty member lists', context={'gene': spec.name})
            members = [member for group in labels for member in group]
            if (any(not isinstance(member, str) or member not in names for member in members)
                    or len(set(members)) != len(members)):
                raise ConfigurationError('class label members must be unique source alleles', context={'gene': spec.name})
        elif labels != names:
            raise ConfigurationError('allele label order disagrees with the source reference', context={'gene': spec.name})


def validate_reference_binding(reference, config, *, label_schema=None, protein_reference=None) -> None:
    """Check a reference/configuration binding without allocating model parameters.

    Allele heads use the reference's ordered panel. Protein-class V/J heads require either an
    explicit ordered schema or a validated protein view; D retains source allele labels. A missing
    schema never causes class IDs to be inferred. Artifact readers still own reference hashes and
    calibration binding, and translate invalid embedded bindings to integrity errors.
    """
    if not isinstance(reference, ReferenceSet) or not isinstance(config, AlignAIRConfig):
        raise ConfigurationError('model binding requires a ReferenceSet and AlignAIRConfig')
    _spaces(config.input_space, config.label_space)
    if type(config.has_d) is not bool or config.has_d and not reference.has_d:
        raise ConfigurationError('model D selection disagrees with the reference', context={'field': 'has_d'})
    reference.validate(genes=tuple(spec.name.upper() for spec in config.gene_specs))
    if label_schema is not None:
        validate_label_schema(label_schema, reference, config)
    if config.label_space == 'protein_classes':
        if protein_reference is not None:
            _protein_labels(protein_reference, reference)
        elif label_schema is None:
            raise ConfigurationError('protein-class models require ordered class labels',
                                     context={'field': 'label_schema'},
                                     hint='Supply the saved label schema or protein reference; do not infer class IDs.')
    elif protein_reference is not None:
        raise ConfigurationError('allele setups must use the source ReferenceSet directly')
    for spec in config.gene_specs:
        if config.label_space == 'protein_classes' and spec.name != 'd':
            if protein_reference is not None:
                labels = protein_reference.class_maps[spec.name.upper()]
                expected = labels.n_classes()
                if label_schema is not None and (len(label_schema['genes'][spec.name]) != expected or any(
                        set(group) != set(labels.class_members[i])
                        for i, group in enumerate(label_schema['genes'][spec.name]))):
                    raise ConfigurationError('label schema class order disagrees with the protein reference',
                                             context={'gene': spec.name, 'field': 'label_schema'})
            else:
                expected = len(label_schema['genes'][spec.name])
        else:
            expected = len(reference.gene(spec.name.upper()))
        expected = _count(expected, f'{spec.name}_label_count')
        actual = _count(spec.allele_count, f'{spec.name}_allele_count')
        if actual != expected:
            raise ConfigurationError('model head size disagrees with its reference labels',
                                     context={'gene': spec.name, 'expected': expected, 'actual': actual})
    if not config.has_d and _count(config.d_allele_count, 'd_allele_count', 0) != 0:
        raise ConfigurationError('a disabled D head must have zero labels', context={'field': 'd_allele_count'})
    _count(config.max_seq_length, 'max_seq_length')
    if _count(config.num_chain_types, 'num_chain_types') != max(1, len(reference.loci)):
        raise ConfigurationError('chain count disagrees with the reference locus schema', context={'field': 'num_chain_types'})
    _input_config(config, config.input_space)


def setup_from_schema(reference, config, schema) -> ModelSetup:
    """Rebind validation adapters to saved class IDs without inventing a new column order."""
    from ..reference.labels import EquivalenceClassMap
    validate_label_schema(schema, reference, config)
    view = build_protein_reference(reference) if config.label_space == 'protein_classes' else None
    if view is not None:
        maps = {}
        for gene in ('V', 'J'):
            groups = {index: list(members) for index, members in enumerate(schema['genes'][gene.lower()])}
            maps[gene] = EquivalenceClassMap(gene, view.db_hash,
                {name: index for index, members in groups.items() for name in members}, groups)
        view = replace(view, class_maps=maps)
    return ModelSetup(reference, config, config.input_space, config.label_space, view)


def prepare_model(reference: ReferenceSet | ProteinReference, *, input_space: InputSpace = 'nucleotide',
                  label_space: LabelSpace = 'alleles', has_d: bool | None = None,
                  max_seq_length: int | None = None, **config_options) -> ModelSetup:
    """Bind one ordered reference to input/label policies without allocating model weights.

    Pass a prebuilt ProteinReference for explicit alternative anchor/hash policies. D always retains
    source allele labels. has_d=False can omit a source D head without changing the source reference.
    """
    _spaces(input_space, label_space)
    view = reference if isinstance(reference, ProteinReference) else None
    source = view.source if view is not None else reference
    if not isinstance(source, ReferenceSet):
        raise ConfigurationError('prepare_model requires a ReferenceSet or ProteinReference')
    if view is not None and label_space != 'protein_classes':
        raise ConfigurationError('pass ProteinReference.source for allele labels')
    selected_d = source.has_d if has_d is None else has_d
    if type(selected_d) is not bool or selected_d and not source.has_d:
        raise ConfigurationError('requested D head is unavailable in the source reference', context={'field': 'has_d'})
    source.validate(genes=('V', 'J', 'D') if selected_d else ('V', 'J'))
    if label_space == 'protein_classes' and view is None:
        view = build_protein_reference(source)
    if view is not None:
        _protein_labels(view, source)
    counts = {gene: (view.class_maps[gene].n_classes() if view is not None and gene != 'D' else len(source.gene(gene)))
              for gene in (('V', 'J', 'D') if selected_d else ('V', 'J'))}
    protected = {'v_label_count', 'j_label_count', 'd_label_count', 'num_chain_types'}
    if protected.intersection(config_options):
        raise ConfigurationError('reference label and locus counts cannot be overridden',
                                 context={'fields': sorted(protected.intersection(config_options))})
    config = build_model_config(counts['V'], counts['J'], counts.get('D', 0), input_space=input_space,
        label_space=label_space, has_d=selected_d, num_chain_types=max(1, len(source.loci)),
        max_seq_length=max_seq_length, **config_options)
    return ModelSetup(source, config, input_space, label_space, view)
