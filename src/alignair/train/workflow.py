"""``run_training``: build, train, checkpoint and reload a model in one call."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

from ..errors import ConfigurationError
from .config import TrainingConfig


@dataclass
class TrainingRun:
    """A finished run: its last model, its best model (with validation) and its number of updates."""
    model_path: str
    best_model_path: str | None = None
    steps: int = 0

    def best_aligner(self, *, device='auto'):
        """The best (or last) model, loaded as an ``Aligner``."""
        from ..aligner import Aligner
        return Aligner.from_pretrained(self.best_model_path or self.model_path, device=device)


def _uint32(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and 0 <= value < 2**32


def run_training(config: TrainingConfig, *, output_dir=None, out_path=None, on_iteration=None,
                 record_source=None) -> TrainingRun:
    """Build, train and checkpoint a model for ``config`` into ``output_dir`` or ``out_path``.

    With ``config.data_workers`` > 0, worker processes are spawned: call this under an
    ``if __name__ == "__main__":`` guard. On an accelerator, the step, ``on_iteration`` and a study
    evaluator must not draw from the CPU random generators: such a draw raises TrainingError, and
    ``data_workers=0`` is the setting for them (see ``TrainingConfig``). ``record_source`` replaces the
    GenAIRR provider of the run's curriculum records (see ``trainer.train``).
    """
    import numpy as np
    import torch

    from ..core import prepare_model
    from ..core.runtime import resolve_device
    from ..data.training import prepare_training_data
    from ..genairr import resolve_dataconfigs
    from ..reference.reference_set import ReferenceSet
    from .guards import validate_training_request
    from .trainer import train

    if not isinstance(config, TrainingConfig):
        raise ConfigurationError('run_training requires a TrainingConfig')
    if not (output_dir or out_path):
        raise ConfigurationError('pass output_dir=... or out_path=...')
    sources = resolve_dataconfigs(config.dataconfigs)
    reference = ReferenceSet.from_dataconfigs(*sources)
    setup = prepare_model(reference, max_seq_length=config.max_seq_length, has_d=config.has_d,
                          **config.model_options)
    device = resolve_device(config.device)
    prepare_training_data(setup, device=device)
    validate_training_request(steps=config.steps, batch_size=config.batch_size, lr=config.lr,
                              max_seq_length=setup.config.max_seq_length, reference=reference,
                              short_boost=config.short_boost, grad_clip=config.grad_clip)
    if (not _uint32(config.seed) or config.init_seed is not None and not _uint32(config.init_seed)
            or isinstance(config.val_every, bool) or not isinstance(config.val_every, int) or config.val_every < 0
            or on_iteration is not None and not callable(on_iteration)):
        raise ConfigurationError('seed must be a uint32 and init_seed a uint32 or None; val_every nonnegative; '
                                 'on_iteration callable or None')
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.init_seed is None:
        model = setup.build_model()
    else:
        # Every process generator the batches may draw from (torch's CPU one, for the batch adapter, and Python's and
        # NumPy's) is restored after the build: the batches do not depend on the init seed or on what the build
        # draws. An accelerator's generators keep the init seed.
        python, numpy_state = random.getstate(), np.random.get_state()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.init_seed)
            model = setup.build_model()
        random.setstate(python)
        np.random.set_state(numpy_state)
    path = os.fspath(out_path or os.path.join(output_dir, 'model.alignair'))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    outcome = train(model, reference, sources, setup.config, steps=config.steps, batch_size=config.batch_size,
          lr=config.lr, short_boost=config.short_boost, seed=config.seed, init_seed=config.init_seed, device=device,
          save_path=path, grad_clip=config.grad_clip, val_every=config.val_every,
          model_setup=setup, on_iteration=on_iteration, resume_path=config.resume_path,
          schedule=config.schedule,
          init_from=config.init_from,
          data_workers=config.data_workers, record_source=record_source, flanks=config.flanks,
          novel_alleles=config.novel_alleles, inverted_d=config.inverted_d)
    return TrainingRun(path, outcome.best_model_path, outcome.steps)
