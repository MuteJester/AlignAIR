"""AlignAIRGym: online GenAIRR curriculum generator yielding GT target bundles."""

import logging
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from .curriculum import Curriculum
from .sharing import GymSharedState, _pick_params
from .experiment import build_experiment
from .transformer import GymRecordTransformer

logger = logging.getLogger(__name__)


class AlignAIRGym(IterableDataset):
    def __init__(self, dataconfigs, reference_set, n=None, seed=0,
                 curriculum=None, log_every=0, allow_curatable=False, shared=False):
        self.dataconfigs = list(dataconfigs)
        self.reference_set = reference_set
        self.n = n
        self.seed = seed
        self.curriculum = curriculum or Curriculum()
        self.log_every = log_every
        self.allow_curatable = allow_curatable
        self._p = 0.0
        self._epoch = 0

        # State transformer and shared state manager strategy
        self.transformer = GymRecordTransformer(self.reference_set)
        self._shared_state = GymSharedState(self.curriculum)
        if shared:
            self.enable_sharing()

    def enable_sharing(self) -> None:
        """Create the shared difficulty state (version flag + params dict) consumed by
        multiprocessing producers. Idempotent; only call when using num_workers>0 (it
        spawns a Manager server, so the single-process path must NOT trigger it)."""
        self._shared_state.enable()

    @property
    def _shared_params(self):
        """Keep for backward compatibility/tests."""
        return self._shared_state._shared_params

    @property
    def _version(self):
        """Keep for backward compatibility/tests."""
        return self._shared_state._version

    def _curriculum_components(self):
        """The current difficulty mixture as [(weight, params), ...]. Most curricula are
        a single component; a TargetedCurriculum returns ramp + targeted + floor."""
        return self._shared_state.get_components(self._p)

    def _components(self):
        return self._shared_state.read_components(self._p)

    def _push_params(self) -> None:
        self._shared_state.push(self._p)

    def set_progress(self, p: float) -> None:
        self._p = max(0.0, min(1.0, p))
        self._push_params()
        logger.info("Gym %s", self.curriculum.describe(self._p))

    def refresh_params(self) -> None:
        """Push current curriculum params to live producers (e.g. after a
        FactoredCurriculum.advance() that changed pace without changing progress)."""
        self._push_params()

    def _make_bundle(self, record, params, has_d, rng):
        return self.transformer.transform(record, params, has_d, rng)

    def __iter__(self):
        if not self._shared_state.is_enabled:
            yield from self._iter_simple()
        else:
            yield from self._iter_shared()

    def _iter_simple(self):
        seed = self.seed + self._epoch
        self._epoch += 1
        rng = np.random.default_rng(seed)
        params = _pick_params(self._components(), rng)   # one mixture component per epoch
        dc = self.dataconfigs[self._epoch % len(self.dataconfigs)]
        has_d = dc.metadata.has_d
        exp = build_experiment(dc, params, allow_curatable=self.allow_curatable)
        count = 0
        for record in exp.stream_records(n=self.n, seed=seed):
            count += 1
            if self.log_every and count % self.log_every == 0:
                logger.info("Gym generated %d samples (%s)", count,
                            self.curriculum.describe(self._p))
            yield self._make_bundle(record, params, has_d, rng)

    def _iter_shared(self):
        # one persistent producer per DataLoader worker; all read the shared floor.
        info = get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        local_version, components = -1, None
        epoch = 0
        while True:
            v = self._shared_state.version
            if v != local_version or components is None:     # floor changed -> re-read mixture
                local_version = v
                components = self._components()
            dc = self.dataconfigs[epoch % len(self.dataconfigs)]
            has_d = dc.metadata.has_d
            seed = self.seed + wid * 1_000_003 + epoch * 1009 + nw   # distinct per worker/epoch
            epoch += 1
            rng = np.random.default_rng(seed)
            params = _pick_params(components, rng)            # one mixture component per epoch
            exp = build_experiment(dc, params, allow_curatable=self.allow_curatable)
            for idx, record in enumerate(exp.stream_records(n=(self.n or 256), seed=seed)):
                yield self._make_bundle(record, params, has_d, rng)
                if idx % 32 == 0 and self._shared_state.version != local_version:      # floor advanced mid-stream
                    break
