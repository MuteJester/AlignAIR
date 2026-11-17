"""``run_training``: build, train, checkpoint and reload a model in one call."""
from __future__ import annotations

import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import ConfigurationError
from .config import TrainingConfig

if TYPE_CHECKING:
    from ..aligner import Aligner
    from .records.streams import RecordSource


@dataclass
class TrainingRun:
    """A finished run: its last model, its best model (with validation) and its number of updates."""
    model_path: str
    best_model_path: str | None = None
    steps: int = 0

    def best_aligner(self, *, device: str = 'auto') -> Aligner:
        """The best (or last) model, loaded as an ``Aligner``."""
        from ..aligner import Aligner
        return Aligner.from_pretrained(self.best_model_path or self.model_path, device=device)


def run_training(config: TrainingConfig, *, output_dir: str | os.PathLike[str] | None = None,
                 out_path: str | os.PathLike[str] | None = None, on_iteration: Callable[[int], object] | None = None,
                 record_source: RecordSource | None = None) -> TrainingRun:
    """Build, train and checkpoint a model for ``config`` into ``output_dir`` or ``out_path``.

    With ``config.data_workers`` > 0, worker processes are spawned: call this under an
    ``if __name__ == "__main__":`` guard. On an accelerator, the step and ``on_iteration`` must not draw from
    the CPU random generators: such a draw raises TrainingError, and
    ``data_workers=0`` is the setting for them (see ``TrainingConfig``). ``record_source`` replaces the
    GenAIRR provider of the run's curriculum records (see ``trainer._train``). ``config`` validated its values when it
    was built; the run checks its reference before the model is built.
    """
    import numpy as np
    import torch

    from ..core import prepare_model
    from ..core.runtime import resolve_device
    from ..data.training import prepare_training_data
    from ..genairr import resolve_dataconfigs
    from ..reference.reference_set import ReferenceSet
    from .guards import validate_training_reference
    from .trainer import _train

    if not isinstance(config, TrainingConfig):
        raise ConfigurationError('run_training requires a TrainingConfig')
    if not (output_dir or out_path):
        raise ConfigurationError('pass output_dir=... or out_path=...')
    if on_iteration is not None and not callable(on_iteration):
        raise ConfigurationError('on_iteration must be callable or None')
    sources = resolve_dataconfigs(config.dataconfigs)
    reference = ReferenceSet.from_dataconfigs(*sources)
    setup = prepare_model(reference, max_seq_length=config.max_seq_length, has_d=config.has_d,
                          **config.model_options)
    device = resolve_device(config.device)
    prepare_training_data(setup, device=device)
    validate_training_reference(reference=reference, max_seq_length=setup.config.max_seq_length)
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
    outcome = _train(setup, model, sources, config, config.record_policy(), device=device, save_path=path,
                     on_iteration=on_iteration, record_source=record_source)
    return TrainingRun(path, outcome.best_model_path, outcome.steps)
