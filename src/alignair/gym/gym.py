"""AlignAIRGym: online GenAIRR curriculum generator yielding GT target bundles."""
import logging

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from .crop import crop_record
from .curriculum import Curriculum
from .targets import build_targets, _tok

# token complement by id: pad0 A1<->T2 G3<->C4 N5
_COMPLEMENT_NP = np.array([0, 2, 1, 4, 3, 5], dtype=np.int64)


def _orient_tokens(tokens, t):
    """Apply orientation transform t in {0:id,1:revcomp,2:comp,3:reverse} to a 1-D
    numpy token array. All transforms are involutions (re-applying recovers forward)."""
    if t == 1:
        return _COMPLEMENT_NP[tokens][::-1].copy()
    if t == 2:
        return _COMPLEMENT_NP[tokens].copy()
    if t == 3:
        return tokens[::-1].copy()
    return tokens

logger = logging.getLogger(__name__)


def _pick_params(components, rng):
    """Weighted choice of one difficulty component's params (drawn once per epoch, so the
    long-run distribution is the mixture). `components` is [(weight, params), ...]."""
    weights = [max(0.0, float(c[0])) for c in components]
    tot = sum(weights)
    if tot <= 0:
        return components[0][1]
    r = float(rng.random()) * tot
    upto = 0.0
    for c, wn in zip(components, weights):
        upto += wn
        if r <= upto:
            return c[1]
    return components[-1][1]


def build_experiment(dataconfig, params, allow_curatable: bool = False):
    """Compile a GenAIRR experiment at the given curriculum params (forward orientation).
    allow_curatable: permit simulation from references with curatable issues (e.g. alleles with
    no detected anchor) — needed for some custom FASTA references built via the cartridge builder."""
    from GenAIRR import Experiment
    exp = Experiment.on(dataconfig)
    if allow_curatable:
        exp = exp.allow_curatable_refdata()
    exp = exp.recombine()
    if params.get("productive_only", False):
        exp = exp.productive_only()
    if params.get("mutation_count") is not None:   # per-read SHM distribution (stratified)
        exp = exp.mutate(model="s5f", count=params["mutation_count"])
    else:
        exp = exp.mutate(model="s5f", rate=params["mutation_rate"])
    if dataconfig.metadata.has_d:
        exp = exp.invert_d(prob=float(params.get("invert_d_prob", 0.05)))
    revision_prob = float(params.get("receptor_revision_prob", 0.0))
    if revision_prob > 0.0:
        exp = exp.receptor_revision(
            prob=revision_prob,
            same_haplotype=bool(params.get("receptor_revision_same_haplotype", True)),
        )
    exp = (exp.end_loss_5prime(length=params["end_loss_5"])
              .end_loss_3prime(length=params["end_loss_3"])
              .polymerase_indels(count=params["indel_count"])
              .sequencing_errors(rate=params["seq_error_rate"])
              .ambiguous_base_calls(count=params["ambiguous_count"]))
    contaminate_prob = float(params.get("contaminate_prob", 0.0))
    if contaminate_prob > 0.0:
        exp = exp.contaminate(prob=contaminate_prob)
    paired_end = params.get("paired_end")
    if paired_end:
        exp = exp.paired_end(**paired_end)
    return exp.compile()


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
        # shared-state producer pool: when enabled, N DataLoader worker processes all
        # generate at the CURRENT floor read from shared memory; advancing the floor
        # updates the shared params and live producers pick it up — no respawn.
        self._version = None
        self._shared_params = None
        if shared:
            self.enable_sharing()

    def enable_sharing(self) -> None:
        """Create the shared difficulty state (version flag + params dict) consumed by
        multiprocessing producers. Idempotent; only call when using num_workers>0 (it
        spawns a Manager server, so the single-process path must NOT trigger it)."""
        if self._shared_params is not None:
            return
        import multiprocessing as mp
        self._version = mp.Value("i", 0, lock=False)        # cheap shared floor flag
        self._shared_params = mp.Manager().dict()
        self._shared_params["components"] = self._curriculum_components()

    def _curriculum_components(self):
        """The current difficulty mixture as [(weight, params), ...]. Most curricula are
        a single component; a TargetedCurriculum returns ramp + targeted + floor."""
        if hasattr(self.curriculum, "components"):
            return self.curriculum.components()
        return [(1.0, self.curriculum.params(self._p))]

    def _components(self):
        if self._shared_params is not None:
            return self._shared_params["components"]
        return self._curriculum_components()

    def _push_params(self) -> None:
        if self._shared_params is None:
            return
        self._shared_params["components"] = self._curriculum_components()
        self._version.value += 1                            # signal producers to recompile

    def set_progress(self, p: float) -> None:
        self._p = max(0.0, min(1.0, p))
        self._push_params()
        logger.info("Gym %s", self.curriculum.describe(self._p))

    def refresh_params(self) -> None:
        """Push current curriculum params to live producers (e.g. after a
        FactoredCurriculum.advance() that changed pace without changing progress)."""
        self._push_params()

    def _make_bundle(self, record, params, has_d, rng):
        # teacher (EMA self-distillation) view: the full, forward read of THIS record,
        # before the student's crop/orientation augmentation.
        teacher_tokens = _tok(str(record["sequence"]).upper())
        if params["crop_prob"] > 0 and rng.random() < params["crop_prob"]:
            lo, hi = params["crop_len_min"], params["crop_len_max"]
            if params.get("crop_log_uniform"):   # densely sample SHORT fragments
                target_len = int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
            else:
                target_len = int(rng.integers(lo, hi + 1))
            record = crop_record(record, target_len)
        bundle = build_targets(record, self.reference_set, has_d=has_d)
        # present a fraction of reads in a non-forward orientation; targets stay in
        # forward frame (the model canonicalizes), only orientation_id changes.
        if params["orient_prob"] > 0 and rng.random() < params["orient_prob"]:
            t = int(rng.integers(1, 4))  # 1=revcomp, 2=comp, 3=reverse
            bundle["tokens"] = _orient_tokens(bundle["tokens"], t)
            bundle["orientation_id"] = t
        bundle["teacher_tokens"] = teacher_tokens
        return bundle

    def __iter__(self):
        if self._shared_params is None:
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
            v = self._version.value
            if v != local_version or components is None:     # floor changed -> re-read mixture
                local_version = v
                components = list(self._shared_params["components"])
            dc = self.dataconfigs[epoch % len(self.dataconfigs)]
            has_d = dc.metadata.has_d
            seed = self.seed + wid * 1_000_003 + epoch * 1009 + nw   # distinct per worker/epoch
            epoch += 1
            rng = np.random.default_rng(seed)
            params = _pick_params(components, rng)            # one mixture component per epoch
            exp = build_experiment(dc, params, allow_curatable=self.allow_curatable)
            for record in exp.stream_records(n=(self.n or 256), seed=seed):
                yield self._make_bundle(record, params, has_d, rng)
                if self._version.value != local_version:      # floor advanced mid-stream
                    break
