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
        from ..nn.region_head import decode_boundaries
        from ..nn.germline_aligner import decode_germline_coords
        from .germline_tf import compute_germline_logits
        self.model.eval()
        loader = self._loader()
        ref_emb = self.model.encode_reference(self.reference_set)
        agg = {"loss": 0.0, "region_correct": 0, "region_total": 0,
               "state_correct": 0, "state_total": 0, "v_hits": 0, "v_total": 0,
               "v_start_dev": 0.0, "v_end_dev": 0.0, "v_gl_start_dev": 0.0, "n_seq": 0}
        nb = 0
        for batch in loader:
            if nb >= n_batches:
                break
            batch = self._to_device(batch)
            out = self.model(batch["tokens"], batch["mask"], ref_emb)
            total, _ = self.loss_fn(out, batch)
            agg["loss"] += float(total.cpu())
            valid = batch["region_labels"] != -100
            rp = out["region_logits"].argmax(-1)
            agg["region_correct"] += int(((rp == batch["region_labels"]) & valid).sum().cpu())
            agg["region_total"] += int(valid.sum().cpu())
            sp = out["state_logits"].argmax(-1)
            agg["state_correct"] += int(((sp == batch["state_labels"]) & valid).sum().cpu())
            agg["state_total"] += int(valid.sum().cpu())
            B = batch["tokens"].shape[0]
            v_pred = out["match"]["V"].argmax(-1)
            agg["v_hits"] += int(batch["v_allele"][torch.arange(B), v_pred].sum().cpu())
            agg["v_total"] += B
            # in-seq V boundary deviation (predicted region runs vs GT)
            dec = decode_boundaries(out["region_logits"], batch["mask"], has_d=self.has_d)
            vps = torch.tensor([d["v_start"] for d in dec], dtype=torch.float32)
            vpe = torch.tensor([d["v_end"] for d in dec], dtype=torch.float32)
            agg["v_start_dev"] += float((vps - batch["v_start"].cpu().float()).abs().sum())
            agg["v_end_dev"] += float((vpe - batch["v_end"].cpu().float()).abs().sum())
            # germline V start deviation (teacher-forced segment, true allele)
            gl = compute_germline_logits(self.model, batch["tokens"], batch["mask"], batch,
                                         ref_emb, self.has_d)
            gs, _ge = decode_germline_coords(gl["v"][0], gl["v"][1])
            agg["v_gl_start_dev"] += float((gs.cpu() - batch["v_germline_start"].cpu()).abs().sum())
            agg["n_seq"] += B
            nb += 1
        self.model.train()
        ns = max(agg["n_seq"], 1)
        return {
            "loss": agg["loss"] / max(nb, 1),
            "region_acc": agg["region_correct"] / max(agg["region_total"], 1),
            "state_acc": agg["state_correct"] / max(agg["state_total"], 1),
            "v_call_agreement": agg["v_hits"] / max(agg["v_total"], 1),
            "v_start_dev": agg["v_start_dev"] / ns,
            "v_end_dev": agg["v_end_dev"] / ns,
            "v_gl_start_dev": agg["v_gl_start_dev"] / ns,
        }
