"""Shared curriculum policies and lazy record sources for nucleotide and protein runners."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import chain
import json
import math
from numbers import Real
from operator import index
from typing import Literal, Protocol
from collections.abc import Iterable, Iterator, Mapping

from ...errors import ConfigurationError, error_context
from ...data.projection import translated_record_usable
from ...reference.reference_set import _locus_of, is_tcr
from .curriculum import Curriculum
from .curriculum.base import INDEL_FRACTION, indel_form_name, indel_fraction_range
from .experiment import build_experiment


class RecordSource(Protocol):
    """Produce fresh record mappings for one reference, parameter set, and stream seed."""
    def __call__(self, dataconfig, parameters: Mapping, seed: int) -> Iterable[Mapping]: ...


# GenAIRR draws record k of the stream of seed s from seed s + k, so streams whose seeds lie close together repeat each
# other's draws a few records apart (and the same seed under other parameters repeats the V/D/J draw). Every record
# stream of a training run therefore draws from a reserved block of SEED_BLOCK_RECORDS consecutive seeds. The block is
# placed by a SHA-256 of what the stream is for (``seed_block``), in [2**62, 2**63): far above the assay's suite seeds
# ([0, 2**31)) and below GenAIRR's limit (2**63 + 1 works, 2**64 - 1 overflows). Blocks are aligned, so two blocks
# either coincide or are disjoint; each purpose has its own quarter of the range, and the streams of one call of
# ``training_records`` are checked to draw distinct blocks.
STREAM_PURPOSES = ('training', 'validation', 'probe')
SEED_BLOCK_RECORDS = 2 ** 32
_FIRST_BLOCK_SEED = 2 ** 62
_BLOCKS_PER_PURPOSE = 2 ** 28              # 4 quarters of 2**28 blocks of 2**32 seeds: [2**62, 2**63)


def seed_block(seed: int, purpose: str, start_step: int, panel: int, component: int, stream: str) -> int:
    """The first GenAIRR seed of the reserved block of one record stream: run ``seed``, stream ``purpose``
    (``STREAM_PURPOSES``), the step ``start_step`` a training stream starts at (a resumed run's stream), the
    stream's reference ``panel`` and curriculum ``component`` positions, and its name among the component's streams
    (``CurriculumStreamPolicy.substreams_for``)."""
    key = json.dumps(['alignair.training_records', seed, purpose, start_step, panel, component, stream]).encode()
    position = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % _BLOCKS_PER_PURPOSE
    return _FIRST_BLOCK_SEED + (STREAM_PURPOSES.index(purpose) * _BLOCKS_PER_PURPOSE + position) * SEED_BLOCK_RECORDS


def _integer(value, field, minimum):
    try:
        if isinstance(value, bool):
            raise ValueError
        value = index(value)
        if value < minimum:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f'{field} must be an integer >= {minimum}', context={'field': field}) from exc
    return value


def _probability(value, field):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ConfigurationError(f'{field} must be a finite number in [0, 1]', context={'field': field})
    return float(value)


def mutation_cap_for(dataconfig, mutation_cap):
    """TCR references retain a zero mutation cap; other references use the requested cap."""
    return 0.0 if is_tcr(dataconfig) else mutation_cap


def dataconfig_name(dataconfig) -> str:
    """Use a materialized provider name, explicit custom name, or receptor descriptor."""
    try:
        import GenAIRR.data as gd
        for name, value in vars(gd).items():
            if value is dataconfig and not name.startswith('_'):
                return name
    except Exception:  # Naming cannot prevent source construction when a provider is unavailable.
        pass
    name = getattr(dataconfig, 'name', None)
    if name and str(name).strip().lower() not in ('none', 'unnamed'):
        return str(name)
    chain_type = getattr(getattr(dataconfig, 'metadata', None), 'chain_type', 'unknown')
    return str(getattr(chain_type, 'value', chain_type))


def effective_mutation_caps(dataconfigs, mutation_cap) -> dict:
    """Record each source's effective cap; repeated or colliding custom names remain distinct."""
    caps = {}
    for position, dataconfig in enumerate(dataconfigs):
        base = key = dataconfig_name(dataconfig)
        suffix = position
        while key in caps:
            key = f'{base}#{suffix}'
            suffix += 1
        caps[key] = mutation_cap_for(dataconfig, mutation_cap)
    return caps


# Selection. GenAIRR simulates unselected recombination: most junctions are out of frame or carry a stop codon, so few
# reads are productive, while real B-cell repertoires are mostly productive. A full-length component therefore draws
# SELECTED_FRACTION of its records from its experiment under GenAIRR's productive_only(). The fraction is k/n with a
# short period n (at most MAX_SELECTION_PERIOD): a component has n streams, each with its own seed block, and each
# period of n of its records takes one record from each, k of them selected (0.75: three selected streams, then one
# unselected). Every stream is thus drawn at the same rate, as the data workers require, and the selected records,
# 10 to 25 times slower to simulate (HUMAN_IGH_OGRDB), come from k streams that can be generated in parallel.
SELECTED_FRACTION = 0.75
MAX_SELECTION_PERIOD = 10
# The read-length amplicons of the mix, (5' end loss, 3' end loss) over the curriculum at progress 0.6. They all stay
# unselected. productive_only() keeps the V Cys and J Trp/Phe landmarks observed, so it narrows any end loss that would
# remove one (on HUMAN_IGH_OGRDB): selected V-anchored reads come out full length (median 342 nt against 130),
# selected fragments J-anchored (228 against 139), and selected J-anchored reads lose those that miss their V anchor
# (5th percentile 75 nt against 29). Those truncated reads, missing an anchor, are what the amplicons exist for.
_AMPLICONS = (((0, 15), (150, 350)),               # V-anchored: the 3' end is lost
              ((150, 350), (0, 15)),               # J-anchored: the 5' end is lost
              ((70, 175), (70, 175)))              # fragment: both ends are lost


def _selected_fraction(value) -> float:
    value = _probability(value, 'selected_fraction')
    if abs(Fraction(value).limit_denominator(MAX_SELECTION_PERIOD) - Fraction(value)) > 1e-9:
        raise ConfigurationError(f'selected_fraction must be a fraction k/n with n <= {MAX_SELECTION_PERIOD}, '
                                 'such as 0.75', context={'field': 'selected_fraction'})
    return value


@dataclass(frozen=True)
class CurriculumStreamPolicy:
    """The training mix of curriculum components. ``indel_form`` is the curriculum's indel form (``Curriculum``):
    ``"weighted"`` by default, whose weight of reads with a polymerase indel is ``indel_fraction`` ``(low, high)`` at
    curriculum progress 0 and 1, giving about 85% indel-free reads over the default mix; ``"uniform"`` is the
    historical form of the 1.0.0 recipe. ``selected_fraction`` (``SELECTED_FRACTION``) of each full-length
    component's records are drawn under GenAIRR's ``productive_only()`` (``substreams_for``); amplicon records never
    are."""
    progresses: tuple[float, ...] = (.3, .6, .9)
    heavy_shm: float = .25
    short_boost: int = 1
    mutation_cap: float | None = None
    mode: Literal['auto', 'amplicon'] = 'auto'
    indel_fraction: tuple[float, float] = INDEL_FRACTION
    indel_form: Literal['uniform', 'weighted'] = 'weighted'
    selected_fraction: float = SELECTED_FRACTION

    def __post_init__(self):
        if self.mode not in ('auto', 'amplicon'):
            raise ConfigurationError('source mode must be auto or amplicon', context={'field': 'mode'})
        try:
            if isinstance(self.progresses, (str, bytes)):
                raise TypeError
            progresses = tuple(self.progresses)
        except TypeError as exc:
            raise ConfigurationError('progresses must be an iterable of curriculum values', context={'field': 'progresses'}) from exc
        if not progresses and self.mode == 'auto':
            raise ConfigurationError('auto source mode requires at least one progress value', context={'field': 'progresses'})
        object.__setattr__(self, 'progresses', tuple(_probability(value, 'progresses') for value in progresses))
        object.__setattr__(self, 'heavy_shm', _probability(self.heavy_shm, 'heavy_shm'))
        object.__setattr__(self, 'short_boost', _integer(self.short_boost, 'short_boost', 1))
        if self.mutation_cap is not None:
            object.__setattr__(self, 'mutation_cap', _probability(self.mutation_cap, 'mutation_cap'))
        object.__setattr__(self, 'indel_fraction', indel_fraction_range(self.indel_fraction))
        indel_form_name(self.indel_form)
        object.__setattr__(self, 'selected_fraction', _selected_fraction(self.selected_fraction))

    @property
    def selection_pattern(self) -> tuple[bool, ...]:
        """Whether each of a full-length component's streams, in interleaving order, is selected: ``selected_fraction``
        k/n as k selected among n streams, spread evenly and starting with a selected one."""
        share = Fraction(self.selected_fraction).limit_denominator(MAX_SELECTION_PERIOD)
        k, n = share.numerator, share.denominator
        return tuple(-(-(r + 1) * k // n) > -(-r * k // n) for r in range(n))

    def _components(self, force_mixture) -> list[tuple[dict, bool]]:
        """Each component's parameters, before any mutation cap, and whether its records may be selected."""
        curriculum = Curriculum(indel_form=self.indel_form, indel_fraction=self.indel_fraction)
        components = [(dict(curriculum.params(progress)), True) for progress in self.progresses]
        mixed = (force_mixture or self.mode == 'amplicon' or len(self.progresses) > 1
                 or self.heavy_shm > 0 or self.short_boost > 1)
        if mixed:
            background = dict(curriculum.params(.6))
            components.extend(({**background, 'end_loss_5': end_loss_5, 'end_loss_3': end_loss_3}, False)
                              for _ in range(self.short_boost) for end_loss_5, end_loss_3 in _AMPLICONS)
            if self.heavy_shm > 0:
                components.append(({**curriculum.params(.3), 'mutation_rate': self.heavy_shm}, True))
        return components

    def _capped(self, dataconfig, force_mixture) -> list[tuple[dict, bool]]:
        components = self._components(force_mixture)
        cap = mutation_cap_for(dataconfig, self.mutation_cap)
        if cap is not None:
            for parameters, _ in components:
                parameters['mutation_rate'] = min(parameters['mutation_rate'], cap)
        return components

    def parameters_for(self, dataconfig, *, force_mixture=False) -> list[dict]:
        """Return independent component dictionaries with this reference's mutation cap applied."""
        return [parameters for parameters, _ in self._capped(dataconfig, force_mixture)]

    def selected_shares(self, *, force_mixture=False) -> tuple[float, ...]:
        """The share of each component's records drawn under ``productive_only()``: ``selected_fraction`` for the
        full-length components (the curriculum progresses and heavy SHM), 0 for every amplicon."""
        return tuple(self.selected_fraction if selectable else 0. for _, selectable in self._components(force_mixture))

    def substreams_for(self, dataconfig, *, force_mixture=False) -> list[tuple[tuple[str, dict], ...]]:
        """Each component's streams in interleaving order, as ``(name, parameters)``: a ``selected-i`` stream draws
        from the component's experiment under ``productive_only()`` (``"productive_only": True``), an
        ``unselected-i`` stream from the experiment itself. An amplicon component has as many streams, all
        unselected."""
        pattern = self.selection_pattern
        streams = []
        for parameters, selectable in self._capped(dataconfig, force_mixture):
            counts, component = {True: 0, False: 0}, []
            for selected in (pattern if selectable else (False,) * len(pattern)):
                name = f"{'selected' if selected else 'unselected'}-{counts[selected]}"
                counts[selected] += 1
                component.append((name, {**parameters, 'productive_only': True} if selected else dict(parameters)))
            streams.append(tuple(component))
        return streams


def _scoped_records(records, *, seed, stage, translated_only=False):
    iterator = None
    while True:
        with error_context(stage=stage, seed=seed):
            if iterator is None:
                iterator = iter(records)
            try:
                record = next(iterator)
            except StopIteration:
                return
            keep = not translated_only or translated_record_usable(record)
        if keep:
            yield record


def stream_records(dataconfig, parameters: Mapping, seed: int) -> Iterator[Mapping]:
    """Default provider: forward GenAIRR records, with D inversion disabled unless requested."""
    parameters = dict(parameters)
    parameters.setdefault('invert_d_prob', 0.)
    with error_context(stage='curriculum_component', seed=seed):
        experiment = build_experiment(dataconfig, parameters, allow_curatable=True)
        records = experiment.stream_records(n=None, seed=seed)
    yield from _scoped_records(records, seed=seed, stage='curriculum_component')


def selection_supported(dataconfig) -> bool:
    """Whether GenAIRR can draw productive records from a reference (a GenAIRR DataConfig or its name).

    ``productive_only()`` needs coding landmarks and a stop-free coding sequence for some allele of each gene: an
    incompletely annotated custom reference may have none, and GenAIRR then refuses to compile the experiment
    ("no productive completion"). The reference's experiment without selection must compile; its errors propagate.
    """
    reference = _resolve_reference(dataconfig)
    parameters = {**Curriculum().params(0.), 'invert_d_prob': 0.}
    with error_context(stage='selection_support'):
        build_experiment(reference, parameters, allow_curatable=True)
        try:
            build_experiment(reference, {**parameters, 'productive_only': True}, allow_curatable=True)
        except ValueError:
            return False
    return True


def _resolve_reference(value):
    if isinstance(value, str):
        import GenAIRR as ga
        try:
            value = getattr(ga, value)
        except AttributeError as exc:
            raise ConfigurationError('unknown source dataconfig name', context={'field': 'dataconfigs'}) from exc
    if not hasattr(getattr(value, 'metadata', None), 'chain_type'):
        raise ConfigurationError('source references must supply receptor metadata', context={'field': 'dataconfigs'})
    return value


def training_records(dataconfigs: Iterable, *, policy: CurriculumStreamPolicy = CurriculumStreamPolicy(), seed: int = 0,
                     translated_only: bool = False, record_source: RecordSource | None = None,
                     purpose: str = 'training', start_step: int = 0) -> Iterator[Mapping]:
    """Build lazy, deterministic curriculum records; reference panels and components use complete rounds.

    Component i of reference panel j interleaves its streams (``policy.substreams_for``: record k comes from its
    stream k % n, so a component ends with its last complete period); stream s draws the seed block
    ``seed_block(seed, purpose, start_step, j, i, s)``. The ``purpose`` is ``"training"``, ``"validation"`` or
    ``"probe"`` (the X-ray probe), and a training stream that starts at ``start_step`` (a resumed run) draws blocks
    of its own. Panel j gets the class ID of its distinct normalized locus. Multiple panels always use the amplicon
    mixture. Translation filtering checks exported length/frame only; sample acceptance remains the builder's policy.
    """
    if not isinstance(policy, CurriculumStreamPolicy):
        raise ConfigurationError('supply a CurriculumStreamPolicy')
    seed = _integer(seed, 'seed', 0)
    start_step = _integer(start_step, 'start_step', 0)
    if not isinstance(purpose, str) or purpose not in STREAM_PURPOSES:
        raise ConfigurationError(f'purpose must be one of {STREAM_PURPOSES}', context={'field': 'purpose'})
    if type(translated_only) is not bool:
        raise ConfigurationError('translated_only must be a boolean', context={'field': 'translated_only'})
    if record_source is not None and not callable(record_source):
        raise ConfigurationError('record_source must be callable', context={'field': 'record_source'})
    try:
        if isinstance(dataconfigs, (str, bytes)):
            raise TypeError
        references = tuple(dataconfigs)
    except TypeError as exc:
        raise ConfigurationError('dataconfigs must be an iterable of references or names', context={'field': 'dataconfigs'}) from exc
    if not references:
        raise ConfigurationError('at least one source reference is required', context={'field': 'dataconfigs'})

    def generate():
        resolved = tuple(_resolve_reference(value) for value in references)
        loci = list(dict.fromkeys(_locus_of(reference.metadata.chain_type) for reference in resolved))
        provider = stream_records if record_source is None else record_source
        plans = [[[(parameters, seed_block(seed, purpose, start_step, position, component, name))
                   for name, parameters in streams]
                  for component, streams in enumerate(policy.substreams_for(reference,
                                                                            force_mixture=len(resolved) > 1))]
                 for position, reference in enumerate(resolved)]
        blocks = [block for plan in plans for streams in plan for _, block in streams]
        if len(set(blocks)) != len(blocks):
            raise ConfigurationError('two record streams of this run drew the same seed block',
                                     context={'purpose': purpose, 'start_step': start_step})

        def panel_records(reference, plan):
            # zip preserves complete rounds and periods, including the historical finite-source tail policy.
            components = [chain.from_iterable(zip(*[iter(provider(reference, params, block))
                                                    for params, block in streams]))
                          for streams in plan]
            locus_class = loci.index(_locus_of(reference.metadata.chain_type))
            for record in chain.from_iterable(zip(*components)):
                yield {**record, 'chain_type': locus_class} if len(resolved) > 1 else record

        panels = [panel_records(reference, plan) for reference, plan in zip(resolved, plans)]
        yield from chain.from_iterable(zip(*panels))

    return _scoped_records(generate(), seed=seed, stage='curriculum_source', translated_only=translated_only)
