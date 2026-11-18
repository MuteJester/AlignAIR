"""Shared curriculum policies and lazy record sources for the runners."""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain
from typing import Protocol

from ...data.flanks import FlankPolicy, flank_records
from ...errors import ConfigurationError, error_context
from ...genairr.catalog import dataconfig_name, is_tcr
from ...genairr.experiment import build_experiment
from ..checks import integer, probability
from .curriculum import INDEL_FRACTION, Curriculum, indel_fraction_range
from .novel import NovelAlleles, novel_records
from .seeds import STREAM_PURPOSES, seed_block


class RecordSource(Protocol):
    """Produce fresh record mappings for one reference, parameter set, and stream seed."""
    def __call__(self, dataconfig, parameters: Mapping, seed: int) -> Iterable[Mapping]: ...


def inverted_d_share(value) -> float:
    """``value`` as the share of records with an inverted D (``CurriculumStreamPolicy.inverted_d``): a finite number in
    [0, 1], else a ConfigurationError for the field ``inverted_d``."""
    return probability(value, 'inverted_d')


def mutation_cap_for(dataconfig) -> float | None:
    """A reference's mutation cap: zero for a TCR reference (T cells carry no SHM), None (uncapped) for any other."""
    return 0.0 if is_tcr(dataconfig) else None


def effective_mutation_caps(dataconfigs) -> dict:
    """Each source's mutation cap (``mutation_cap_for``); repeated or colliding custom names remain distinct."""
    caps = {}
    for position, dataconfig in enumerate(dataconfigs):
        base = key = dataconfig_name(dataconfig)
        suffix = position
        while key in caps:
            key = f'{base}#{suffix}'
            suffix += 1
        caps[key] = mutation_cap_for(dataconfig)
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
# The curriculum's indel form for training (``Curriculum``): the uniform form is the pinned evaluation strata's.
INDEL_FORM = 'weighted'
# The read-length amplicons of the mix, (5' end loss, 3' end loss) over the curriculum at progress 0.6. They all stay
# unselected. productive_only() keeps the V Cys and J Trp/Phe landmarks observed, so it narrows any end loss that would
# remove one (on HUMAN_IGH_OGRDB): selected V-anchored reads come out full length (median 342 nt against 130),
# selected fragments J-anchored (228 against 139), and selected J-anchored reads lose those that miss their V anchor
# (5th percentile 75 nt against 29). Those truncated reads, missing an anchor, are what the amplicons exist for.
_AMPLICONS = (((0, 15), (150, 350)),               # V-anchored: the 3' end is lost
              ((150, 350), (0, 15)),               # J-anchored: the 5' end is lost
              ((70, 175), (70, 175)))              # fragment: both ends are lost


def _selected_fraction(value) -> float:
    value = probability(value, 'selected_fraction')
    if abs(Fraction(value).limit_denominator(MAX_SELECTION_PERIOD) - Fraction(value)) > 1e-9:
        raise ConfigurationError(f'selected_fraction must be a fraction k/n with n <= {MAX_SELECTION_PERIOD}, '
                                 'such as 0.75', context={'field': 'selected_fraction'})
    return value


@dataclass(frozen=True)
class CurriculumStreamPolicy:
    """The training mix of curriculum components: one per curriculum progress, plus, for a mix (several progresses,
    heavy SHM, a short boost, or several reference panels), the three read-length amplicons per short boost and a
    heavy-SHM component. The curriculum's polymerase indels are weighted (``Curriculum``): the weight of reads with an
    indel is ``indel_fraction`` ``(low, high)`` at curriculum progress 0 and 1, giving about 85% indel-free reads over
    the default mix. A TCR reference's records carry no SHM (``mutation_cap_for``).
    ``selected_fraction`` (``SELECTED_FRACTION``) of each full-length
    component's records are drawn under GenAIRR's ``productive_only()`` (``substreams_for``); amplicon records never
    are. ``flanks`` (a ``data.flanks.FlankPolicy``) puts library flanks around every stream's records, drawn from the
    stream's own seed block after its provider (``training_records``); None leaves the simulated reads bare.
    ``novel_alleles`` (a ``train.records.novel.NovelAlleles``) makes every stream draw part of its reads from synthetic
    novel V alleles (``train.records.novel``); None, the default, draws them from the reference's alleles alone.
    ``inverted_d`` (0 to 1) is the share of every stream's records whose D GenAIRR inverts (``invert_d``); 0, the
    default, inverts none and leaves the streams' parameters as they were."""
    progresses: tuple[float, ...] = (.3, .6, .9)
    heavy_shm: float = .25
    short_boost: int = 1
    indel_fraction: tuple[float, float] = INDEL_FRACTION
    selected_fraction: float = SELECTED_FRACTION
    flanks: FlankPolicy | None = None
    novel_alleles: NovelAlleles | None = None
    inverted_d: float = 0.

    def __post_init__(self):
        try:
            if isinstance(self.progresses, (str, bytes)):
                raise TypeError
            progresses = tuple(self.progresses)
        except TypeError as exc:
            raise ConfigurationError('progresses must be an iterable of curriculum values',
                                     context={'field': 'progresses'}) from exc
        if not progresses:
            raise ConfigurationError('a curriculum mix needs at least one progress value',
                                     context={'field': 'progresses'})
        object.__setattr__(self, 'progresses', tuple(probability(value, 'progresses') for value in progresses))
        object.__setattr__(self, 'heavy_shm', probability(self.heavy_shm, 'heavy_shm'))
        object.__setattr__(self, 'short_boost', integer(self.short_boost, 'short_boost', 1))
        object.__setattr__(self, 'indel_fraction', indel_fraction_range(self.indel_fraction))
        object.__setattr__(self, 'selected_fraction', _selected_fraction(self.selected_fraction))
        if self.flanks is not None and not isinstance(self.flanks, FlankPolicy):
            raise ConfigurationError('flanks must be a FlankPolicy or None', context={'field': 'flanks'})
        if self.novel_alleles is not None and not isinstance(self.novel_alleles, NovelAlleles):
            raise ConfigurationError('novel_alleles must be a NovelAlleles or None', context={'field': 'novel_alleles'})
        object.__setattr__(self, 'inverted_d', inverted_d_share(self.inverted_d))

    @property
    def selection_pattern(self) -> tuple[bool, ...]:
        """Whether each of a full-length component's streams, in interleaving order, is selected: ``selected_fraction``
        k/n as k selected among n streams, spread evenly and starting with a selected one."""
        share = Fraction(self.selected_fraction).limit_denominator(MAX_SELECTION_PERIOD)
        k, n = share.numerator, share.denominator
        return tuple(-(-(r + 1) * k // n) > -(-r * k // n) for r in range(n))

    def _components(self, force_mixture) -> list[tuple[dict, bool]]:
        """Each component's parameters, before any mutation cap, and whether its records may be selected."""
        curriculum = Curriculum(indel_form=INDEL_FORM, indel_fraction=self.indel_fraction)
        components = [(dict(curriculum.params(progress)), True) for progress in self.progresses]
        mixed = force_mixture or len(self.progresses) > 1 or self.heavy_shm > 0 or self.short_boost > 1
        if mixed:
            background = dict(curriculum.params(.6))
            components.extend(({**background, 'end_loss_5': end_loss_5, 'end_loss_3': end_loss_3}, False)
                              for _ in range(self.short_boost) for end_loss_5, end_loss_3 in _AMPLICONS)
            if self.heavy_shm > 0:
                components.append(({**curriculum.params(.3), 'mutation_rate': self.heavy_shm}, True))
        return components

    def _capped(self, dataconfig, force_mixture) -> list[tuple[dict, bool]]:
        components = self._components(force_mixture)
        cap = mutation_cap_for(dataconfig)
        if cap is not None:
            for parameters, _ in components:
                parameters['mutation_rate'] = min(parameters['mutation_rate'], cap)
        if self.novel_alleles is not None:
            for parameters, _ in components:
                parameters['novel_alleles'] = self.novel_alleles.to_dict()
        if self.inverted_d:
            for parameters, _ in components:
                parameters['invert_d_prob'] = self.inverted_d
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


def _scoped_records(records, *, seed, stage):
    iterator = None
    while True:
        with error_context(stage=stage, seed=seed):
            if iterator is None:
                iterator = iter(records)
            try:
                record = next(iterator)
            except StopIteration:
                return
        yield record


def stream_records(dataconfig, parameters: Mapping, seed: int) -> Iterator[Mapping]:
    """Default provider: forward GenAIRR records, with D inversion disabled unless requested; a stream whose parameters
    carry ``novel_alleles`` draws them from its epochs' novel alleles (``train.records.novel.novel_records``)."""
    parameters = dict(parameters)
    parameters.setdefault('invert_d_prob', 0.)
    novel = parameters.pop('novel_alleles', None)
    if novel is not None:
        with error_context(stage='curriculum_component', seed=seed):
            records = novel_records(dataconfig, parameters, seed, NovelAlleles.from_dict(novel))
        yield from _scoped_records(records, seed=seed, stage='curriculum_component')
        return
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
        from ...genairr import resolve_dataconfig
        value = resolve_dataconfig(value)
    if not hasattr(getattr(value, 'metadata', None), 'chain_type'):
        raise ConfigurationError('source references must supply receptor metadata', context={'field': 'dataconfigs'})
    return value


# The default mix, built once: the policy is frozen, so every call may share it.
_DEFAULT_POLICY = CurriculumStreamPolicy()


def training_records(dataconfigs: Iterable, *,
                     policy: CurriculumStreamPolicy = _DEFAULT_POLICY,
                     seed: int = 0,
                     record_source: RecordSource | None = None,
                     purpose: str = 'training', start_step: int = 0) -> Iterator[Mapping]:
    """Build lazy, deterministic curriculum records; reference panels and components use complete rounds.

    Component i of reference panel j interleaves its streams (``policy.substreams_for``: record k comes from its
    stream k % n, so a component ends with its last complete period); stream s draws the seed block
    ``seed_block(seed, purpose, start_step, j, i, s)``. The ``purpose`` is ``"training"``, ``"validation"`` or
    ``"probe"`` (a held-out stream of its own), and a training stream that starts at ``start_step`` (a resumed run)
    draws blocks of its own. Multiple panels always use the amplicon mixture.
    """
    if not isinstance(policy, CurriculumStreamPolicy):
        raise ConfigurationError('supply a CurriculumStreamPolicy')
    seed = integer(seed, 'seed', 0)
    start_step = integer(start_step, 'start_step', 0)
    if not isinstance(purpose, str) or purpose not in STREAM_PURPOSES:
        raise ConfigurationError(f'purpose must be one of {STREAM_PURPOSES}', context={'field': 'purpose'})
    if record_source is not None and not callable(record_source):
        raise ConfigurationError('record_source must be callable', context={'field': 'record_source'})
    try:
        if isinstance(dataconfigs, (str, bytes)):
            raise TypeError
        references = tuple(dataconfigs)
    except TypeError as exc:
        raise ConfigurationError('dataconfigs must be an iterable of references or names',
                                 context={'field': 'dataconfigs'}) from exc
    if not references:
        raise ConfigurationError('at least one source reference is required', context={'field': 'dataconfigs'})

    def generate():
        resolved = tuple(_resolve_reference(value) for value in references)
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

        def flanked(records, block):
            # after the provider, so a worker-built stream (whose provider runs elsewhere) is flanked alike
            return records if policy.flanks is None else flank_records(records, policy.flanks, block)

        def panel_records(reference, plan):
            # zip yields complete rounds and periods only: a finite source's incomplete last round is dropped.
            components = [chain.from_iterable(zip(*[iter(flanked(provider(reference, params, block), block))
                                                    for params, block in streams]))
                          for streams in plan]
            yield from chain.from_iterable(zip(*components))

        panels = [panel_records(reference, plan) for reference, plan in zip(resolved, plans)]
        yield from chain.from_iterable(zip(*panels))

    return _scoped_records(generate(), seed=seed, stage='curriculum_source')
