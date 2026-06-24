"""Composite Kendall-weighted loss for the unified DNAlignAIR model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.weighting import UncertaintyWeight
from ..nn.matching import contrastive_match_loss

IGNORE = -100


class DNAlignAIRLoss(nn.Module):
    def __init__(self, has_d: bool = True, use_boundary: bool = False,
                 protected_max_log_var: float = 1.5, protected=None):
        super().__init__()
        self.has_d = has_d
        self.use_boundary = use_boundary
        genes = ["v", "j"] + (["d"] if has_d else [])
        names = ["orientation", "region", "state", "v_match", "j_match",
                 "noise", "mutation", "indel", "productive"]
        if has_d:
            names += ["d_match"]
        names += [f"{g}_germline" for g in genes]
        if use_boundary:  # in-sequence start/end posteriors from the query region decoder
            names += [f"{g}_boundary" for g in genes]
        # PROTECTED heads (V-call, germline coords, junction boundaries) get a TIGHTER
        # max_log_var so their Kendall precision weight exp(-log_var) cannot collapse to
        # the global floor when the curriculum makes them hard — the mechanism that
        # otherwise lets the balancer abandon exactly the heads we want to push.
        default_protected = {"v_match"} | {f"{g}_germline" for g in genes}
        if use_boundary:
            default_protected |= {f"{g}_boundary" for g in genes}
        self._protected = set(default_protected if protected is None else protected)
        self.weights = nn.ModuleDict({
            n: UncertaintyWeight(
                max_log_var=(protected_max_log_var if n in self._protected else 3.0))
            for n in names})

    @property
    def protected_heads(self):
        return set(self._protected)

    def set_log_vars_frozen(self, frozen: bool) -> None:
        for w in self.weights.values():
            w.set_frozen(frozen)

    def forward(self, outputs: dict, batch: dict, germline_logits: dict | None = None,
                match_logits: dict | None = None):
        comp = {}

        def add(name, raw):
            return raw * self.weights[name]()

        orientation = F.cross_entropy(outputs["orientation_logits"], batch["orientation_id"])
        region = F.cross_entropy(
            outputs["region_logits"].reshape(-1, outputs["region_logits"].shape[-1]),
            batch["region_labels"].reshape(-1), ignore_index=IGNORE)
        state = F.cross_entropy(
            outputs["state_logits"].reshape(-1, outputs["state_logits"].shape[-1]),
            batch["state_labels"].reshape(-1), ignore_index=IGNORE)

        genes = ["v", "j"] + (["d"] if self.has_d else [])
        match_src = match_logits if match_logits is not None else outputs["match"]
        match_terms = {g: contrastive_match_loss(match_src[g.upper()], batch[f"{g}_allele"])
                       for g in genes}

        noise = F.l1_loss(outputs["noise_count"], batch["noise_count"])
        mutation = F.mse_loss(outputs["mutation_rate"], batch["mutation_rate"])
        indel = F.l1_loss(outputs["indel_count"], batch["indel_count"])
        productive = F.binary_cross_entropy(outputs["productive"].clamp(1e-7, 1 - 1e-7),
                                            batch["productive"])

        total = (add("orientation", orientation) + add("region", region) + add("state", state)
                 + add("noise", noise) + add("mutation", mutation) + add("indel", indel)
                 + add("productive", productive))
        comp.update({"orientation": orientation.detach(), "region": region.detach(),
                     "state": state.detach(), "noise": noise.detach(),
                     "mutation": mutation.detach(), "indel": indel.detach(),
                     "productive": productive.detach()})
        for g in genes:
            total = total + add(f"{g}_match", match_terms[g])
            comp[f"{g}_match"] = match_terms[g].detach()

        if germline_logits is not None:
            for g in genes:
                if g not in germline_logits:
                    continue
                sl, el = germline_logits[g]
                gs = batch[f"{g}_germline_start"].clamp(min=0, max=sl.shape[-1] - 1)
                ge = (batch[f"{g}_germline_end"] - 1).clamp(min=0, max=el.shape[-1] - 1)
                per_row = (F.cross_entropy(sl, gs, reduction="none")
                           + F.cross_entropy(el, ge, reduction="none"))
                mask = batch.get(f"{g}_supervise")  # inverted-D rows masked out
                if mask is not None:
                    gl = (per_row * mask).sum() / mask.sum().clamp(min=1.0)
                else:
                    gl = per_row.mean()
                total = total + add(f"{g}_germline", gl)
                comp[f"{g}_germline"] = gl.detach()

        # in-sequence boundary posteriors (query region decoder): NLL of true start/end
        boundary = outputs.get("boundary")
        if self.use_boundary and boundary is not None:
            for g in genes:
                sl = boundary["start"][g.upper()]            # (B, L)
                el = boundary["end"][g.upper()]
                L = sl.shape[-1]
                s_tgt = batch[f"{g}_start"].clamp(min=0, max=L - 1)
                e_tgt = (batch[f"{g}_end"] - 1).clamp(min=0, max=L - 1)
                bnd = F.cross_entropy(sl, s_tgt) + F.cross_entropy(el, e_tgt)
                total = total + add(f"{g}_boundary", bnd)
                comp[f"{g}_boundary"] = bnd.detach()

        # Kendall balancing penalty (0.5*log_var per task). Without this the precision
        # weights exp(-log_var) have no upward pressure and collapse to the clamp floor.
        total = total + sum(w.penalty() for w in self.weights.values())
        comp["total"] = total.detach()
        return total, comp

    @torch.no_grad()
    def task_weights(self) -> dict:
        """Current precision weight per task (for logging the learned balance)."""
        return {name: w.weight() for name, w in self.weights.items()}

    @torch.no_grad()
    def apply_constraints(self) -> None:
        for w in self.weights.values():
            w.apply_constraints()
