"""GymTrainer: verbose curriculum training loop for the unified DNAlignAIR model."""
import logging

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..gym.collate import gym_collate

logger = logging.getLogger(__name__)


def _detach_ref(ref_emb: dict) -> dict:
    """Detach all tensors in a reference-embedding dict (for cross-step caching)."""
    return {g: {k: (v.detach() if torch.is_tensor(v) else v) for k, v in d.items()}
            for g, d in ref_emb.items()}


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
                    if self.refresh_reference_every > 1:
                        # Cache across steps: detach so a later step's backward does not
                        # traverse this (freed) graph. The germline encoder still learns
                        # every step via the query-segment path; references act as a
                        # periodically-refreshed target encoder.
                        ref_emb = _detach_ref(ref_emb)
                out = self.model(batch["tokens"], batch["mask"], ref_emb)
                germline_logits = compute_germline_logits(
                    self.model, batch["tokens"], batch["mask"], batch, ref_emb, self.has_d)
                # teacher-force the match query over TRUE regions for a clean signal
                match_logits = self.model.match_alleles(
                    batch["tokens"], batch["mask"], batch["region_labels"], ref_emb)
                total, comp = self.loss_fn(out, batch, germline_logits=germline_logits,
                                           match_logits=match_logits)
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

    @torch.no_grad()
    def evaluate(self, n_batches: int = 4) -> dict:
        """Comprehensive correctness across ALL segments. Calls + in-sequence
        boundaries are end-to-end (predicted regions / predicted top-1 allele);
        germline coordinates are teacher-forced (true region + true allele) so they
        measure the aligner head in isolation."""
        from ..nn.region_head import decode_boundaries
        from ..nn.germline_aligner import decode_germline_coords
        from .germline_tf import compute_germline_logits
        self.model.eval()
        genes = ["v", "j"] + (["d"] if self.has_d else [])
        loader = self._loader()
        ref_emb = self.model.encode_reference(self.reference_set)

        loss_sum, nb, n_seq = 0.0, 0, 0
        region_c = region_t = state_c = state_t = 0
        per = {g: {"call": 0, "start_dev": 0.0, "end_dev": 0.0,
                   "gl_start_dev": 0.0, "gl_end_dev": 0.0} for g in genes}

        for batch in loader:
            if nb >= n_batches:
                break
            batch = self._to_device(batch)
            out = self.model(batch["tokens"], batch["mask"], ref_emb)
            loss_sum += float(self.loss_fn(out, batch)[0].cpu())
            valid = batch["region_labels"] != -100
            region_c += int(((out["region_logits"].argmax(-1) == batch["region_labels"]) & valid).sum().cpu())
            region_t += int(valid.sum().cpu())
            state_c += int(((out["state_logits"].argmax(-1) == batch["state_labels"]) & valid).sum().cpu())
            state_t += int(valid.sum().cpu())

            B = batch["tokens"].shape[0]
            dec = decode_boundaries(out["region_logits"], batch["mask"], has_d=self.has_d)
            gl = compute_germline_logits(self.model, batch["tokens"], batch["mask"], batch,
                                         ref_emb, self.has_d)
            for g in genes:
                pred = out["match"][g.upper()].argmax(-1)
                per[g]["call"] += int(batch[f"{g}_allele"][torch.arange(B), pred].sum().cpu())
                ps = torch.tensor([d[f"{g}_start"] for d in dec], dtype=torch.float32)
                pe = torch.tensor([d[f"{g}_end"] for d in dec], dtype=torch.float32)
                per[g]["start_dev"] += float((ps - batch[f"{g}_start"].cpu().float()).abs().sum())
                per[g]["end_dev"] += float((pe - batch[f"{g}_end"].cpu().float()).abs().sum())
                gs, ge = decode_germline_coords(gl[g][0], gl[g][1])
                per[g]["gl_start_dev"] += float((gs.cpu() - batch[f"{g}_germline_start"].cpu()).abs().sum())
                per[g]["gl_end_dev"] += float((ge.cpu() - batch[f"{g}_germline_end"].cpu()).abs().sum())
            n_seq += B
            nb += 1

        self.model.train()
        ns = max(n_seq, 1)
        metrics = {"loss": loss_sum / max(nb, 1),
                   "region_acc": region_c / max(region_t, 1),
                   "state_acc": state_c / max(state_t, 1)}
        for g in genes:
            metrics[f"{g}_call"] = per[g]["call"] / ns
            metrics[f"{g}_start_dev"] = per[g]["start_dev"] / ns
            metrics[f"{g}_end_dev"] = per[g]["end_dev"] / ns
            metrics[f"{g}_gl_start_dev"] = per[g]["gl_start_dev"] / ns
            metrics[f"{g}_gl_end_dev"] = per[g]["gl_end_dev"] / ns
        return metrics
