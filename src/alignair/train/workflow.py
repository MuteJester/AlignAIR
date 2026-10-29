"""One public build/train/checkpoint/reload workflow for nucleotide and protein models."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import random

from ..errors import ConfigurationError
from .config import TrainingConfig


@dataclass
class TrainingRun:
    model_path: str
    best_model_path: str | None = None
    steps: int = 0
    checkpoints: dict = field(default_factory=dict)
    validations: tuple = ()

    def best_aligner(self, *, device='auto'):
        """Reload through the shared model owner; predict_raw works with either input space."""
        from ..aligner import Aligner
        return Aligner.from_pretrained(self.best_model_path or self.model_path, device=device)


def run_training(config: TrainingConfig, *, output_dir=None, out_path=None, on_iteration=None) -> TrainingRun:
    """Build, train and checkpoint a model for ``config`` into ``output_dir`` or ``out_path``.

    With ``config.data_workers`` > 0, worker processes are spawned: call this under an
    ``if __name__ == "__main__":`` guard. On an accelerator, the step, ``on_iteration`` and a study
    evaluator must not draw from the CPU random generators: such a draw raises TrainingError, and
    ``data_workers=0`` is the setting for them (see ``TrainingConfig``).
    """
    from ..aligner import resolve_device
    from ..core import prepare_model
    from ..data.training import prepare_training_data
    from ..reference.reference_set import ReferenceSet
    from .guards import validate_training_request
    from .trainer import train
    import GenAIRR.data as gd
    import numpy as np
    import torch

    if not isinstance(config, TrainingConfig):
        raise ConfigurationError('run_training requires a TrainingConfig')
    if not (output_dir or out_path):
        raise ConfigurationError('pass output_dir=... or out_path=...')
    try:
        sources = [getattr(gd, source) if isinstance(source, str) else source for source in config.dataconfigs]
    except AttributeError as exc:
        raise ConfigurationError('unknown source dataconfig', context={'field': 'dataconfigs'}) from exc
    reference = ReferenceSet.from_dataconfigs(*sources)
    setup = prepare_model(reference, input_space=config.input_space, label_space=config.label_space,
                          max_seq_length=config.max_seq_length, has_d=config.has_d, **config.model_options)
    device = resolve_device(config.device)
    prepare_training_data(setup, device=device, augmentation=config.augmentation)
    validate_training_request(steps=config.steps, batch_size=config.batch_size, lr=config.lr,
                              max_seq_length=setup.config.max_seq_length, reference=reference,
                              short_boost=config.short_boost, grad_clip=config.grad_clip)
    if (isinstance(config.seed, bool) or not isinstance(config.seed, int) or not 0 <= config.seed < 2**32
            or isinstance(config.val_every, bool) or not isinstance(config.val_every, int) or config.val_every < 0
            or on_iteration is not None and not callable(on_iteration)):
        raise ConfigurationError('seed must be a uint32; val_every nonnegative; on_iteration callable or None')
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = setup.build_model()
    path = os.fspath(out_path or os.path.join(output_dir, 'model.alignair'))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    outcome = train(model, reference, sources, setup.config, steps=config.steps, batch_size=config.batch_size,
          lr=config.lr, short_boost=config.short_boost, seed=config.seed, device=device,
          save_path=path, grad_clip=config.grad_clip, val_every=config.val_every,
          model_setup=setup, on_iteration=on_iteration, resume_path=config.resume_path,
          augmentation=config.augmentation, schedule=config.schedule, study=config.study,
          init_from=config.init_from, init_trust_pickle=config.trust_pickle, resume_trust_pickle=config.trust_pickle,
          data_workers=config.data_workers)
    return TrainingRun(path, outcome.best_model_path, outcome.steps, outcome.checkpoints, outcome.validations)
