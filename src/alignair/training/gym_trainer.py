"""GymTrainer: verbose curriculum training loop for the unified DNAlignAIR model."""
import logging
import math

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import random

from ..gym.collate import gym_collate
from ..nn.heads.matching import distill_match_loss
from ..nn.heads.state import state_reliability
from ..core.dnalignair import extract_segment_tokens, extract_segment
from .ema import EMATeacher
from .reader import (build_sibling_index, build_candidates, reader_scores, reader_set_nce,
                     reader_novel_positive)

logger = logging.getLogger(__name__)


def _detach_ref(ref_emb: dict) -> dict:
    """Detach all tensors in a reference-embedding dict (for cross-step caching)."""
    return {g: {k: (v.detach() if torch.is_tensor(v) else v) for k, v in d.items()}
            for g, d in ref_emb.items()}


class GymTrainer:
    def __init__(self, model, loss_fn, reference_set, gym, lr=1e-3, batch_size=16,
                 device=None, grad_clip=10.0, refresh_reference_every=1,
                 refresh_curriculum_every=25, distill=False, distill_weight=1.0,
                 distill_decay=0.999, distill_temperature=2.0,
                 reader=False, reader_weight=1.0, reader_n_sib=6, reader_n_rand=6,
                 reader_novel_prob=0.0, reader_novel_snps=1, state_conditioning=True,
                 scheduled_sampling=False, ss_max_prob=0.5, num_workers=0,
                 sigma_freeze_steps=0, promote_on_lcb=False, band_weight=1.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.band_weight = band_weight   # seed_extend: weight on the band head's offset-CE loss
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        # after a difficulty advance, hold the Kendall log_vars constant for this many
        # steps so the balancer does not react to the post-shift loss spike by abandoning
        # the newly-hard head (0 = disabled).
        self.sigma_freeze_steps = sigma_freeze_steps
        self._freeze_remaining = 0
        # advance per-axis pace on the bootstrap-CI lower bound, not the point estimate
        self.promote_on_lcb = promote_on_lcb
        self.reference_set = reference_set
        self.gym = gym
        # parallel GenAIRR producers: N persistent worker processes generate at the
        # current floor (shared state) so the GPU isn't starved by single-process sim.
        self.num_workers = num_workers
        if num_workers > 0:
            self.gym.enable_sharing()
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.refresh_reference_every = refresh_reference_every
        # how often to advance the curriculum (re-create the gym iterator so it
        # recompiles GenAIRR at the new progress); the old per-while-loop structure
        # only ever fired set_progress once, pinning training at p=0 (clean data).
        self.refresh_curriculum_every = refresh_curriculum_every
        self.has_d = reference_set.has_d
        self._nonfinite_skips = 0
        self._global_step = 0  # persists across fit() calls for a monotonic curriculum
        # EMA self-distillation: teacher sees the full forward read, student the
        # hard cropped/oriented view of the same record.
        self.distill = distill
        self.distill_weight = distill_weight
        self.distill_temperature = distill_temperature
        self.teacher = EMATeacher(self.model, decay=distill_decay).to(self.device) if distill else None
        self._teacher_ref = None
        # allele reader: train alignment_score to discriminate true allele vs siblings
        self.reader = reader
        self.reader_weight = reader_weight
        self.reader_n_sib, self.reader_n_rand = reader_n_sib, reader_n_rand
        # simulated-novel reader augmentation: with this probability per example, the true
        # germline (positive) is SNP-perturbed + re-encoded so the reader must align to a
        # germline NEVER baked into its weights -> trains the dynamic-reference property.
        self.reader_novel_prob = reader_novel_prob
        self.reader_novel_snps = reader_novel_snps
        self.state_conditioning = state_conditioning   # down-weight likely-SHM positions in the reader
        self._sib_index = build_sibling_index(reference_set) if reader else None
        self._reader_rng = random.Random(0)
        self._novel_gen = torch.Generator(device=self.device).manual_seed(0)
        # scheduled sampling: train match/germline/reader on the model's own PREDICTED
        # region segments (ramped probability) to close the teacher-forced -> deployed gap
        self.scheduled_sampling = scheduled_sampling
        self.ss_max_prob = ss_max_prob
        params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _to_device(self, batch):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _loader(self):
        kw = {}
        if self.num_workers > 0:
            kw = {"num_workers": self.num_workers, "persistent_workers": True,
                  "prefetch_factor": 2}
        return DataLoader(self.gym, batch_size=self.batch_size,
                          collate_fn=lambda b: gym_collate(b, self.reference_set, self.has_d),
                          **kw)

    def fit(self, total_steps: int, global_total: int | None = None,
            progress: bool = True, controller=None) -> list:
        """Train for ``total_steps``. When ``global_total`` is given the curriculum
        ramps over the GLOBAL training horizon (across successive fit() calls) via an
        internal step counter, so chunked training yields one monotonic easy->hard
        ramp instead of a sawtooth that resets every chunk. ``progress=False`` silences
        the tqdm bar (the caller reports its own progress)."""
        from .germline_tf import compute_germline_logits
        self.model.train()
        loader = self._loader()
        history = []
        ref_emb = None
        bar = tqdm(total=total_steps, desc="gym-train", disable=not progress)
        step = 0
        it = iter(loader)
        since_refresh = self.refresh_curriculum_every  # force a refresh on entry
        while step < total_steps:
            if since_refresh >= self.refresh_curriculum_every:
                if controller is not None:
                    self.gym.set_progress(controller.progress())
                elif global_total:
                    self.gym.set_progress(min(1.0, self._global_step / max(global_total - 1, 1)))
                else:
                    self.gym.set_progress(min(1.0, step / max(total_steps - 1, 1)))
                # single-process gym must rebuild the iterator to pick up new params;
                # the shared-state producer pool (num_workers>0) reads the new floor
                # from shared memory, so we keep the SAME persistent iterator.
                if self.num_workers == 0:
                    it = iter(loader)
                since_refresh = 0
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            since_refresh += 1
            self._global_step += 1

            # Anneal the coord-loss CDF soft-step width tau wide->sharp over the global
            # horizon (cosine 3.0 -> 0.75), mirroring the segmentation sigma schedule, so
            # the germline boundary is sharpened late once the model is roughly localized.
            if getattr(self.loss_fn, "coord_tau", None) is not None:
                horizon = (global_total or total_steps)
                _p = min(1.0, self._global_step / max(horizon - 1, 1))
                self.loss_fn.coord_tau = 0.75 + (3.0 - 0.75) * 0.5 * (1.0 + math.cos(math.pi * _p))

            batch = self._to_device(batch)
            if ref_emb is None or step % self.refresh_reference_every == 0:
                ref_emb = self.model.encode_reference(self.reference_set)
                if self.refresh_reference_every > 1:
                    # Cache across steps: detach so a later step's backward does not
                    # traverse this (freed) graph. The germline encoder still learns
                    # every step via the query-segment path; references act as a
                    # periodically-refreshed target encoder.
                    ref_emb = _detach_ref(ref_emb)
            # teacher-force the true orientation; everything downstream of the
            # backbone is supervised in the canonical (forward) frame.
            out = self.model(batch["tokens"], batch["mask"], ref_emb,
                             orientation_ids=batch["orientation_id"])
            canon = out["canon_tokens"]
            # scheduled sampling: per sample, with a curriculum-ramped probability, draw
            # the SUPERVISION segment from the model's own predicted regions instead of
            # the true labels (targets stay true) -> learns to recover allele/coords from
            # its own imperfect boundaries, closing the teacher-forced->deployed gap.
            sup_regions = batch["region_labels"]
            if self.scheduled_sampling and global_total:
                p_ss = self.ss_max_prob * min(1.0, self._global_step / max(global_total - 1, 1))
                use_pred = torch.rand(batch["tokens"].shape[0], device=self.device) < p_ss
                sup_regions = torch.where(use_pred.unsqueeze(1),
                                          out["region_logits"].argmax(-1), sup_regions)
            band_logits = None
            if getattr(self.model, "seed_extend", False):
                germline_logits, band_logits = compute_germline_logits(
                    self.model, canon, batch["mask"], batch, ref_emb, self.has_d,
                    region_labels=sup_regions, state_logits=out["state_logits"],
                    reps=out["reps"], return_band=True)
            else:
                germline_logits = compute_germline_logits(
                    self.model, canon, batch["mask"], batch, ref_emb, self.has_d,
                    region_labels=sup_regions, state_logits=out["state_logits"], reps=out["reps"])
            match_logits = self.model.match_alleles(
                canon, batch["mask"], sup_regions, ref_emb, reps=out["reps"])
            total, comp = self.loss_fn(out, batch, germline_logits=germline_logits,
                                       match_logits=match_logits)
            if band_logits is not None:
                # the band head's only gradient: offset-CE against the true germline start
                # (the DP band center is an argmax, so coord loss can't train the seed).
                from ..nn.aligner.band_head import band_offset_loss
                band_loss = sum(band_offset_loss(band_logits[g], batch[f"{g}_germline_start"])
                                for g in band_logits)
                total = total + self.band_weight * band_loss
                comp["band"] = band_loss.detach()
            # EMA self-distillation: pull the student's fragment-view allele posteriors
            # toward the teacher's full-read posteriors (soft, temperature-scaled).
            if self.distill and "teacher_tokens" in batch:
                if self._teacher_ref is None or step % max(self.refresh_curriculum_every, 1) == 0:
                    self._teacher_ref = self.teacher.model.encode_reference(self.reference_set)
                with torch.no_grad():
                    t_out = self.teacher(batch["teacher_tokens"], batch["teacher_mask"],
                                         self._teacher_ref)
                distill = distill_match_loss(match_logits, t_out["match"],
                                             self.distill_temperature)
                total = total + self.distill_weight * distill
                comp["distill"] = distill.detach()
            # allele reader: train alignment_score to rank true allele over siblings
            if self.reader:
                reader_loss = 0.0
                genes = ["v", "j"] + (["d"] if self.has_d else [])
                for g in genes:
                    G = g.upper()
                    seg_tok, seg_mask = extract_segment_tokens(
                        canon, batch["mask"], sup_regions, G)
                    seg_reps = self.model.germline_encoder.forward_positions(seg_tok, seg_mask)
                    # per-position SHM reliability for the segment (state head, same gather)
                    if self.state_conditioning:
                        seg_state, _ = extract_segment(out["state_logits"], batch["mask"], sup_regions, G)
                        seg_rel = state_reliability(seg_state)
                    else:
                        seg_rel = None
                    cand, pos = build_candidates(
                        batch[f"{g}_primary_idx"], batch[f"{g}_allele"], self._sib_index[G],
                        self._reader_rng, self.reader_n_sib, self.reader_n_rand)
                    sc = reader_scores(self.model.aligner, seg_reps, seg_mask, cand,
                                       ref_emb[G]["pos_reps"], ref_emb[G]["pos_mask"],
                                       seg_tok=seg_tok, germ_tok_ref=ref_emb[G]["pos_tok"],
                                       seg_reliability=seg_rel)
                    # simulated-novel: for a random subset of examples, swap the positive
                    # (column 0) score for one against a SNP-perturbed, freshly-encoded copy
                    # of the true germline -> the reader learns to align to UNSEEN germlines.
                    # GUARD (codex): only keep the swap where the perturbed positive still
                    # out-scores every negative; else the k SNPs may have made a sibling the
                    # better match and we'd teach the WRONG ranking — revert to the clean positive.
                    if self.reader_novel_prob > 0:
                        sel = torch.rand(sc.shape[0], device=self.device) < self.reader_novel_prob
                        if sel.any():
                            novel_sc = reader_novel_positive(
                                self.model.aligner, self.model.germline_encoder,
                                seg_reps, seg_mask, seg_tok, cand[:, 0],
                                ref_emb[G]["pos_tok"], ref_emb[G]["pos_mask"],
                                self.reader_novel_snps, self._novel_gen, seg_reliability=seg_rel)
                            neg_max = sc[:, 1:].max(dim=1).values            # best negative
                            keep = sel & (novel_sc > neg_max)               # stays the closest
                            sc = sc.clone()
                            sc[:, 0] = torch.where(keep, novel_sc, sc[:, 0])
                    reader_loss = reader_loss + reader_set_nce(sc, pos)
                total = total + self.reader_weight * reader_loss
                comp["reader"] = reader_loss.detach()
            self.optimizer.zero_grad(set_to_none=True)
            # Skip non-finite steps so a single diverged batch cannot poison the
            # weights with NaN/inf (regression heads can spike on very hard inputs).
            if not torch.isfinite(total) or float(total.detach()) > 1e4:
                self._nonfinite_skips += 1
                worst = max(((k, float(v)) for k, v in comp.items() if k != "total"),
                            key=lambda kv: abs(kv[1]), default=("?", 0.0))
                logger.warning("skipping diverged loss at step %d (total=%.3g, worst=%s=%.3g); "
                               "skips=%d", step, float(total.detach().cpu()),
                               worst[0], worst[1], self._nonfinite_skips)
                step += 1
                bar.update(1)
                continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.loss_fn.apply_constraints()
            if self.distill:
                self.teacher.update(self.model)  # EMA tracks the student
            logs = {k: float(v.cpu()) for k, v in comp.items()}
            history.append(logs)
            bar.update(1)
            bar.set_postfix(loss=f"{logs['total']:.3f}", region=f"{logs['region']:.3f}",
                            stage=self.gym.curriculum.stage(self.gym._p) + 1)
            step += 1
            if self._freeze_remaining > 0:
                self._freeze_remaining -= 1
                if self._freeze_remaining == 0:
                    self.loss_fn.set_log_vars_frozen(False)   # thaw after the transient
            if controller is not None and step % controller.config.exam_every == 0:
                controller.exam(step=self._global_step)
                if controller.done:
                    break
        bar.close()
        return history

    @torch.no_grad()
    def evaluate(self, n_batches: int = 4, p: float = 1.0) -> dict:
        """Comprehensive correctness across ALL segments, measured at curriculum
        progress ``p`` (default 1.0 = hardest: heavy SHM, trims, and ~50-80bp
        fragments). Calls + in-sequence boundaries are end-to-end (predicted regions
        / predicted top-1 allele); germline coordinates are teacher-forced (true
        region + true allele) so they measure the aligner head in isolation."""
        from ..nn.heads.region import decode_boundaries
        from ..nn.aligner.germline_aligner import decode_germline_coords
        from .germline_tf import compute_germline_logits
        self.model.eval()
        genes = ["v", "j"] + (["d"] if self.has_d else [])
        prev_p = self.gym._p
        self.gym.set_progress(p)
        loader = self._loader()
        ref_emb = self.model.encode_reference(self.reference_set)

        loss_sum, nb, n_seq = 0.0, 0, 0
        region_c = region_t = state_c = state_t = orient_c = 0
        per = {g: {"call": 0, "start_dev": 0.0, "end_dev": 0.0,
                   "gl_start_dev": 0.0, "gl_end_dev": 0.0,
                   "e2e_gl_start_dev": 0.0, "e2e_gl_end_dev": 0.0} for g in genes}

        for batch in loader:
            if nb >= n_batches:
                break
            batch = self._to_device(batch)
            # teacher-force orientation so segment metrics stay interpretable; the
            # orientation head's own accuracy is reported separately.
            out = self.model(batch["tokens"], batch["mask"], ref_emb,
                             orientation_ids=batch["orientation_id"])
            orient_c += int((out["orientation_logits"].argmax(-1) == batch["orientation_id"]).sum().cpu())
            loss_sum += float(self.loss_fn(out, batch)[0].cpu())
            valid = batch["region_labels"] != -100
            region_c += int(((out["region_logits"].argmax(-1) == batch["region_labels"]) & valid).sum().cpu())
            region_t += int(valid.sum().cpu())
            state_c += int(((out["state_logits"].argmax(-1) == batch["state_labels"]) & valid).sum().cpu())
            state_t += int(valid.sum().cpu())

            B = batch["tokens"].shape[0]
            dec = decode_boundaries(out["region_logits"], batch["mask"], has_d=self.has_d)
            canon = out["canon_tokens"]
            # teacher-forced germline (true region + true allele): isolates the aligner
            gl = compute_germline_logits(self.model, canon, batch["mask"], batch,
                                         ref_emb, self.has_d)
            # end-to-end germline (PREDICTED region + PREDICTED top-1 allele): the real pipeline
            pred_region = out["region_logits"].argmax(-1)
            pred_idx = {g.upper(): out["match"][g.upper()].argmax(-1) for g in genes}
            gl_e2e = compute_germline_logits(self.model, canon, batch["mask"], batch,
                                             ref_emb, self.has_d,
                                             region_labels=pred_region, allele_idx=pred_idx)
            boundary = out.get("boundary")
            for g in genes:
                pred = out["match"][g.upper()].argmax(-1)
                per[g]["call"] += int(batch[f"{g}_allele"][torch.arange(B), pred].sum().cpu())
                if boundary is not None:  # query decoder: decode from start/end posteriors
                    ps = boundary["start"][g.upper()].argmax(-1).float().cpu()
                    pe = (boundary["end"][g.upper()].argmax(-1) + 1).float().cpu()
                else:
                    ps = torch.tensor([d[f"{g}_start"] for d in dec], dtype=torch.float32)
                    pe = torch.tensor([d[f"{g}_end"] for d in dec], dtype=torch.float32)
                per[g]["start_dev"] += float((ps - batch[f"{g}_start"].cpu().float()).abs().sum())
                per[g]["end_dev"] += float((pe - batch[f"{g}_end"].cpu().float()).abs().sum())
                gs, ge = decode_germline_coords(gl[g][0], gl[g][1], soft=True)
                per[g]["gl_start_dev"] += float((gs.cpu() - batch[f"{g}_germline_start"].cpu()).abs().sum())
                per[g]["gl_end_dev"] += float((ge.cpu() - batch[f"{g}_germline_end"].cpu()).abs().sum())
                es, ee = decode_germline_coords(gl_e2e[g][0], gl_e2e[g][1], soft=True)
                per[g]["e2e_gl_start_dev"] += float((es.cpu() - batch[f"{g}_germline_start"].cpu()).abs().sum())
                per[g]["e2e_gl_end_dev"] += float((ee.cpu() - batch[f"{g}_germline_end"].cpu()).abs().sum())
            n_seq += B
            nb += 1

        self.model.train()
        self.gym.set_progress(prev_p)
        ns = max(n_seq, 1)
        metrics = {"loss": loss_sum / max(nb, 1),
                   "region_acc": region_c / max(region_t, 1),
                   "state_acc": state_c / max(state_t, 1),
                   "orient_acc": orient_c / ns}
        for g in genes:
            metrics[f"{g}_call"] = per[g]["call"] / ns
            metrics[f"{g}_start_dev"] = per[g]["start_dev"] / ns
            metrics[f"{g}_end_dev"] = per[g]["end_dev"] / ns
            metrics[f"{g}_gl_start_dev"] = per[g]["gl_start_dev"] / ns
            metrics[f"{g}_gl_end_dev"] = per[g]["gl_end_dev"] / ns
            metrics[f"{g}_e2e_gl_start_dev"] = per[g]["e2e_gl_start_dev"] / ns
            metrics[f"{g}_e2e_gl_end_dev"] = per[g]["e2e_gl_end_dev"] / ns
        return metrics

    @torch.no_grad()
    def evaluate_records(self, n_batches: int = 4, p: float = 1.0) -> list:
        """Per-read tagged records for axis attribution: each dict carries the
        read's GenAIRR truth difficulty tags + `correct` (V top-1 in true set)."""
        self.model.eval()
        prev_p = self.gym._p
        self.gym.set_progress(p)
        loader = self._loader()
        ref_emb = self.model.encode_reference(self.reference_set)
        records, nb = [], 0
        for batch in loader:
            if nb >= n_batches:
                break
            batch = self._to_device(batch)
            out = self.model(batch["tokens"], batch["mask"], ref_emb,
                             orientation_ids=batch["orientation_id"])
            pred = out["match"]["V"].argmax(-1)
            B = batch["tokens"].shape[0]
            correct = batch["v_allele"][torch.arange(B), pred].cpu()
            length = batch["mask"].sum(dim=1).cpu()
            for i in range(B):
                records.append({
                    "mutation_rate": float(batch["mutation_rate"][i].cpu()),
                    "indel_count": float(batch["indel_count"][i].cpu()),
                    "noise_count": float(batch["noise_count"][i].cpu()),
                    "length": int(length[i]),
                    "correct": float(correct[i] > 0),
                })
            nb += 1
        self.model.train()
        self.gym.set_progress(prev_p)
        return records

    def advance_curriculum(self, field: dict, threshold: float = 0.7,
                           step: float = 0.1) -> list:
        """From a Phase-1 lattice competence field: advance the FactoredCurriculum's
        per-axis pace AND update the ALP/regret targeting (if a TargetedCurriculum wraps
        it). No-op (returns []) for non-factored curricula."""
        from ..gym.factored import FactoredCurriculum, axis_competence_from_field
        from ..gym.targeting import TargetedCurriculum
        cur = self.gym.curriculum
        target = cur if isinstance(cur, TargetedCurriculum) else None
        factored = cur.factored if target is not None else cur
        if not isinstance(factored, FactoredCurriculum):
            return []
        moved = factored.advance(
            axis_competence_from_field(field, use_lcb=self.promote_on_lcb),
            threshold=threshold, step=step)
        if target is not None:
            target.update_targets(field)   # ALP/regret retargets the hard regimes
        if moved or target is not None:
            self.gym.refresh_params()      # push new floor / mixture to live producers
            if moved and self.sigma_freeze_steps > 0:
                self.freeze_uncertainty()  # hold log_vars across the transient
        return moved

    def freeze_uncertainty(self, steps: int | None = None) -> None:
        """Freeze the Kendall log_vars for `steps` (default sigma_freeze_steps) training
        steps, then they thaw automatically in fit()."""
        self._freeze_remaining = self.sigma_freeze_steps if steps is None else steps
        self.loss_fn.set_log_vars_frozen(self._freeze_remaining > 0)
