"""AlignAIRGym: online GenAIRR curriculum generator yielding GT target bundles."""
import logging

import numpy as np
from torch.utils.data import IterableDataset

from .crop import crop_record
from .curriculum import Curriculum
from .targets import build_targets

logger = logging.getLogger(__name__)


def build_experiment(dataconfig, params):
    """Compile a GenAIRR experiment at the given curriculum params (forward orientation)."""
    from GenAIRR import Experiment
    exp = Experiment.on(dataconfig).recombine().mutate(model="s5f", rate=params["mutation_rate"])
    if dataconfig.metadata.has_d:
        exp = exp.invert_d(prob=0.05)
    exp = (exp.end_loss_5prime(length=params["end_loss_5"])
              .end_loss_3prime(length=params["end_loss_3"])
              .polymerase_indels(count=params["indel_count"])
              .sequencing_errors(rate=params["seq_error_rate"])
              .ambiguous_base_calls(count=params["ambiguous_count"]))
    return exp.compile()


class AlignAIRGym(IterableDataset):
    def __init__(self, dataconfigs, reference_set, n=None, seed=0,
                 curriculum=None, log_every=0):
        self.dataconfigs = list(dataconfigs)
        self.reference_set = reference_set
        self.n = n
        self.seed = seed
        self.curriculum = curriculum or Curriculum()
        self.log_every = log_every
        self._p = 0.0
        self._epoch = 0

    def set_progress(self, p: float) -> None:
        self._p = max(0.0, min(1.0, p))
        logger.info("Gym %s", self.curriculum.describe(self._p))

    def __iter__(self):
        params = self.curriculum.params(self._p)
        seed = self.seed + self._epoch
        self._epoch += 1
        dc = self.dataconfigs[self._epoch % len(self.dataconfigs)]
        has_d = dc.metadata.has_d
        exp = build_experiment(dc, params)
        rng = np.random.default_rng(seed)
        count = 0
        for record in exp.stream_records(n=self.n, seed=seed):
            count += 1
            if params["crop_prob"] > 0 and rng.random() < params["crop_prob"]:
                target_len = int(rng.integers(params["crop_len_min"],
                                              params["crop_len_max"] + 1))
                record = crop_record(record, target_len)
            if self.log_every and count % self.log_every == 0:
                logger.info("Gym generated %d samples (%s)", count,
                            self.curriculum.describe(self._p))
            yield build_targets(record, self.reference_set, has_d=has_d)
