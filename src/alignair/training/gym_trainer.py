"""GymTrainer: verbose curriculum training loop for the unified DNAlignAIR model."""
import logging

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..gym.collate import gym_collate

logger = logging.getLogger(__name__)


class GymTrainer:
    def __init__(self, model, loss_fn, reference_set, gym, lr=1e-3, batch_size=16,
                 device=None, grad_clip=10.0, refresh_reference_every=1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.reference_set = reference_set
        self.gym = gym
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.refresh_reference_every = refresh_reference_every
        self.has_d = reference_set.has_d
        params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _to_device(self, batch):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _loader(self):
        return DataLoader(self.gym, batch_size=self.batch_size,
                          collate_fn=lambda b: gym_collate(b, self.reference_set, self.has_d))

    def fit(self, total_steps: int) -> list:
        from .germline_tf import compute_germline_logits
        self.model.train()
        loader = self._loader()
        history = []
        ref_emb = None
        bar = tqdm(total=total_steps, desc="gym-train")
        step = 0
        while step < total_steps:
            self.gym.set_progress(step / max(total_steps - 1, 1))
            for batch in loader:
                if step >= total_steps:
                    break
                batch = self._to_device(batch)
                if ref_emb is None or step % self.refresh_reference_every == 0:
                    ref_emb = self.model.encode_reference(self.reference_set)
                out = self.model(batch["tokens"], batch["mask"], ref_emb)
                germline_logits = compute_germline_logits(
                    self.model, out["reps"], batch["mask"], batch, ref_emb, self.has_d)
                total, comp = self.loss_fn(out, batch, germline_logits=germline_logits)
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.loss_fn.apply_constraints()
                logs = {k: float(v.cpu()) for k, v in comp.items()}
                history.append(logs)
                bar.update(1)
                bar.set_postfix(loss=f"{logs['total']:.3f}", region=f"{logs['region']:.3f}",
                                stage=self.gym.curriculum.stage(self.gym._p) + 1)
                step += 1
        bar.close()
        return history
