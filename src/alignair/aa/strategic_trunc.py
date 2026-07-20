"""Stage-1 coordinate-aware truncation training.

Augmentation mix per read: full-length rehearsal, N/C/internal crops (coordinate-aware — boundaries
shifted+clipped into crop coords, absent genes masked, partially-observed V/J supervised against the
compatible protein-class SET), and AA-native subs / single-residue indel / I/L ambiguity. D stays an
allele posterior (no protein-equivalence merging — its observability is frame/trim/junction-dependent):
fully observed -> singleton true allele; partial/absent -> masked."""
from __future__ import annotations

import itertools

import numpy as np
import torch

from .augment import aa_indel, aa_substitute, il_ambiguity
from .data import collate_aa
from .loss import aa_loss
from .reference import build_aa_germlines
from .strategic import _class_sample
from .tokenizer import AAProteinTokenizer
from .train_gym import _usable_stream
from .truncation import compatible_classes, crop_window, shift_clip


def prototypes_by_class(dc, class_map, gene: str) -> dict:
    """{class_id: representative protein} for a gene's protein-equivalence classes."""
    germ = build_aa_germlines(dc, genes=(gene,)).genes[gene.upper()]
    out: dict = {}
    for name, g in germ.items():
        c = class_map.class_of(name) if name in class_map.allele_class else None
        if c is not None and c not in out:
            out[c] = g.aa_seq
    return out


def _compat(true_c, obs_lo, obs_hi, full_span, protos):
    """Compatible class set for a (possibly partial) V/J observation over read residues
    [obs_lo, obs_hi). Fully observed -> {true}; partial -> classes matching over the window."""
    s, e = full_span
    if obs_lo <= s and obs_hi >= e:                          # fully observed
        return {true_c}
    return compatible_classes(true_c, max(obs_lo, s), min(obs_hi, e), protos)


def stage1_sample(rec, class_maps, d_index, protos, tok, max_len, rng,
                  p_crop=0.4, p_native=0.3):
    """Full-length gym record -> a training sample, augmented. Adds {g}_compatible (set),
    {g}_valid, per gene. D never gets a compatible set (allele posterior)."""
    base = _class_sample(rec, class_maps, d_index, tok, max_len)
    if base is None:
        return None
    aa = rec["sequence_aa"]
    genes = ("v", "j", "d")

    def finalize(tokens, spans, valids, compat):
        s = {"tokens": tokens}
        for g in genes:
            s[f"{g}_start"], s[f"{g}_end"] = spans[g]
            s[f"{g}_valid"] = valids[g]
            s[f"{g}_class"] = base[f"{g}_class"] if valids[g] else 0
            s[f"{g}_compatible"] = compat[g]                 # set of class ids (>=1 element)
        return s

    spans = {g: (base[f"{g}_start"], base[f"{g}_end"]) for g in genes}
    roll = rng.random()

    # ---- crop (coordinate-aware) ----
    if roll < p_crop and int(rec.get("v_germline_start", 0)) <= 3:
        n = len(aa)
        keep = int(rng.uniform(0.4, 0.9) * n)
        lo, hi = crop_window(n, str(rng.choice(["nterm", "cterm", "internal"])), keep)
        enc = tok.encode(aa[lo:hi])
        if 10 <= len(enc) < max_len:
            sp, vl, cp = {}, {}, {}
            for g in genes:
                sc = shift_clip(spans[g], lo, hi)
                if sc is None:                               # gene absent in crop -> mask
                    sp[g], vl[g], cp[g] = (0, 0), 0.0, {base[f"{g}_class"] if base.get(f"{g}_valid", 1.0) else 0}
                else:
                    sp[g], vl[g] = sc, base.get(f"{g}_valid", 1.0)
                    if g == "d":
                        # D partial -> mask (don't merge D by isolated translation)
                        full = spans[g][0] >= lo and spans[g][1] <= hi
                        vl[g] = base.get("d_valid", 1.0) if full else 0.0
                        cp[g] = {base["d_class"] if vl[g] else 0}
                    else:
                        cp[g] = _compat(base[f"{g}_class"], lo, hi, spans[g], protos[g]) if vl[g] else {0}
            return finalize(enc, sp, vl, cp)

    # ---- AA-native (subs / single indel / I/L) ----
    if roll < p_crop + p_native:
        op = rng.random()
        aug = (aa_substitute(aa, float(rng.uniform(0.02, 0.12)), rng) if op < 0.5
               else il_ambiguity(aa) if op < 0.7 else aa_indel(aa, rng))
        enc = tok.encode(aug)
        if len(enc) >= max_len or len(enc) == 0:
            enc = tok.encode(aa)
        vl = {g: base.get(f"{g}_valid", 1.0) for g in genes}
        vl["d"] = base.get("d_valid", 1.0)
        return finalize(enc, spans, vl, {g: {base[f"{g}_class"] if vl[g] else 0} for g in genes})

    # ---- full-length rehearsal ----
    vl = {"v": 1.0, "j": 1.0, "d": base.get("d_valid", 1.0)}
    return finalize(base["tokens"], spans, vl, {g: {base[f"{g}_class"] if vl[g] else 0} for g in genes})


def collate_stage1(samples, max_len, n_classes: dict, device="cpu"):
    """Like collate_aa but adds {g}_compatible multi-hot (B, n_classes[g]) and {g}_valid for v/j/d."""
    bi, tg = collate_aa(samples, max_len, device)             # tokens + {g}_class/start/end (+ d_valid)
    for g in ("v", "j", "d"):
        comp = torch.zeros(len(samples), n_classes[g], device=device)
        for i, s in enumerate(samples):
            for c in s[f"{g}_compatible"]:
                comp[i, c] = 1.0
        tg[f"{g}_compatible"] = comp
        tg[f"{g}_valid"] = torch.tensor([float(s[f"{g}_valid"]) for s in samples],
                                        dtype=torch.float32, device=device)
    return bi, tg


def stage1_train(model, cfg, class_maps, meta, dc_name="HUMAN_IGH_OGRDB", *, steps=40000, batch_size=64,
                 lr=3e-4, p_crop=0.4, p_native=0.3, seed=0, device="cpu", log_every=250, grad_clip=1.0,
                 save_path=None, save_every=5000):
    import GenAIRR as ga
    dc = getattr(ga, dc_name)
    protos = {"v": prototypes_by_class(dc, class_maps["V"], "V"),
              "j": prototypes_by_class(dc, class_maps["J"], "J")}
    n_classes = {"v": meta["n_v_classes"], "j": meta["n_j_classes"], "d": meta["n_d_alleles"]}
    tok = AAProteinTokenizer()
    rng = np.random.default_rng(seed)
    d_index = meta["d_index"]
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def gen():
        for rec in _usable_stream(dc_name, (0.3, 0.6, 0.9), 0.25, seed, 1, None):
            s = stage1_sample(rec, class_maps, d_index, protos, tok, cfg.max_seq_length, rng,
                              p_crop, p_native)
            if s is not None:
                yield s
    stream = gen()

    history = []
    for step in range(1, steps + 1):
        batch = list(itertools.islice(stream, batch_size))
        if len(batch) < batch_size:
            break
        bi, tg = collate_stage1(batch, cfg.max_seq_length, n_classes, device)
        out = model(bi)
        total, parts = aa_loss(out, tg, genes=("v", "d", "j"))
        opt.zero_grad()
        total.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        history.append({"step": step, "total": float(total.detach()),
                        **{k: float(v.detach()) for k, v in parts.items()}})
        if log_every and step % log_every == 0:
            h = history[-1]
            print(f"[{step}/{steps}] loss {h['total']:.3f} seg {h['segmentation']:.3f} "
                  f"cls {h['classification']:.3f}", flush=True)
        if save_path and save_every and step % save_every == 0:
            _save(save_path, model, cfg, step)
    if save_path:
        _save(save_path, model, cfg, history[-1]["step"] if history else 0)
    return history


def _save(path, model, cfg, step):
    import dataclasses
    torch.save({"model": model.state_dict(), "cfg": dataclasses.asdict(cfg), "step": step}, path)
