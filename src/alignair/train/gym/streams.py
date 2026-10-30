"""Shared curriculum policies and lazy record sources for nucleotide and protein runners."""
from __future__ import annotations

from dataclasses import dataclass
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
    """Produce fresh record mappings for one reference, parameter set, and component seed."""
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


def seed_block(seed: int, purpose: str, start_step: int, panel: int, component: int) -> int:
    """The first GenAIRR seed of the reserved block of one record stream: run ``seed``, stream ``purpose``
    (``STREAM_PURPOSES``), the step ``start_step`` a training stream starts at (a resumed run's stream), and the
    stream's reference ``panel`` and curriculum ``component`` positions."""
    key = json.dumps(['alignair.training_records', seed, purpose, start_step, panel, component]).encode()
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


@dataclass(frozen=True)
class CurriculumStreamPolicy:
    """The training mix of curriculum components. ``indel_form`` is the curriculum's indel form (``Curriculum``):
    ``"weighted"`` by default, whose weight of reads with a polymerase indel is ``indel_fraction`` ``(low, high)`` at
    curriculum progress 0 and 1, giving about 85% indel-free reads over the default mix; ``"uniform"`` is the
    historical form of the 1.0.0 recipe."""
    progresses: tuple[float, ...] = (.3, .6, .9)
    heavy_shm: float = .25
    short_boost: int = 1
    mutation_cap: float | None = None
    mode: Literal['auto', 'amplicon'] = 'auto'
    indel_fraction: tuple[float, float] = INDEL_FRACTION
    indel_form: Literal['uniform', 'weighted'] = 'weighted'

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

    def parameters_for(self, dataconfig, *, force_mixture=False) -> list[dict]:
        """Return independent component dictionaries with this reference's mutation cap applied."""
        curriculum = Curriculum(indel_form=self.indel_form, indel_fraction=self.indel_fraction)
        parameters = [dict(curriculum.params(progress)) for progress in self.progresses]
        mixed = (force_mixture or self.mode == 'amplicon' or len(self.progresses) > 1
                 or self.heavy_shm > 0 or self.short_boost > 1)
        if mixed:
            background = dict(curriculum.params(.6))
            amplicons = [
                {**background, 'end_loss_5': (0, 15), 'end_loss_3': (150, 350)},
                {**background, 'end_loss_5': (150, 350), 'end_loss_3': (0, 15)},
                {**background, 'end_loss_5': (70, 175), 'end_loss_3': (70, 175)},
            ]
            parameters.extend(dict(component) for _ in range(self.short_boost) for component in amplicons)
            if self.heavy_shm > 0:
                parameters.append({**curriculum.params(.3), 'mutation_rate': self.heavy_shm})
        cap = mutation_cap_for(dataconfig, self.mutation_cap)
        if cap is not None:
            for component in parameters:
                component['mutation_rate'] = min(component['mutation_rate'], cap)
        return parameters


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

    Component i of reference panel j draws the seed block ``seed_block(seed, purpose, start_step, j, i)``: the
    ``purpose`` is ``"training"``, ``"validation"`` or ``"probe"`` (the X-ray probe), and a training stream that
    starts at ``start_step`` (a resumed run) draws blocks of its own. Panel j gets the class ID of its distinct
    normalized locus. Multiple panels always use the amplicon mixture. Translation filtering checks exported
    length/frame only; sample acceptance remains the builder's policy.
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
        plans = [[(parameters, seed_block(seed, purpose, start_step, position, component))
                  for component, parameters in enumerate(policy.parameters_for(reference,
                                                                               force_mixture=len(resolved) > 1))]
                 for position, reference in enumerate(resolved)]
        blocks = [block for plan in plans for _, block in plan]
        if len(set(blocks)) != len(blocks):
            raise ConfigurationError('two record streams of this run drew the same seed block',
                                     context={'purpose': purpose, 'start_step': start_step})

        def panel_records(reference, plan):
            # zip preserves complete rounds, including the historical finite-source tail policy.
            components = [iter(provider(reference, params, block)) for params, block in plan]
            locus_class = loci.index(_locus_of(reference.metadata.chain_type))
            for record in chain.from_iterable(zip(*components)):
                yield {**record, 'chain_type': locus_class} if len(resolved) > 1 else record

        panels = [panel_records(reference, plan) for reference, plan in zip(resolved, plans)]
        yield from chain.from_iterable(zip(*panels))

    return _scoped_records(generate(), seed=seed, stage='curriculum_source', translated_only=translated_only)
