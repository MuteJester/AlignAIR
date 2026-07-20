"""Strategic AA path (b): explicit protein-equivalence CLASS targets, on the REAL gym distribution,
with AA-native augmentation, trained longer. Reuses Phase-1's softmax class-head model (`AAAlignAIR`)
and loss (`aa_loss`), fed by the real curriculum gym (`train_gym._usable_stream`) via the class-target
adapter (`data.record_to_sample`). Substitution augmentation is length-preserving, so boundary targets
stay valid; truncation/indel/IL are evaluation strata (see `augment.py`)."""
from __future__ import annotations

import itertools

import numpy as np
import torch

from .augment import aa_substitute
from .config import aa_igh_config
from .data import build_class_maps, collate_aa
from .gym_bridge import aa_usable
from .loss import aa_loss
from .model import AAAlignAIR
from .tokenizer import AAProteinTokenizer
from .train_gym import _usable_stream
from .translation import frame_offset_of, project_codons


def strategic_setup(dc_name: str = "HUMAN_IGH_OGRDB", max_seq_length: int = 192, has_d: bool = True):
    import GenAIRR as ga

    from ..reference.reference_set import ReferenceSet
    dc = getattr(ga, dc_name)
    ref = ReferenceSet.from_dataconfigs(dc)
    class_maps, meta = build_class_maps(dc_name)
    d_index = dict(ref.gene("D").index) if has_d else {}
    cfg = aa_igh_config(meta["n_v_classes"], meta["n_j_classes"], len(d_index), max_seq_length, has_d=has_d)
    meta = {**meta, "n_d_alleles": len(d_index), "d_index": d_index}
    return AAAlignAIR(cfg), cfg, class_maps, meta


def _class_sample(rec, class_maps, d_index, tok, max_len):
    """Junction-intact gym record -> class-target sample. Lighter than data.record_to_sample: keeps
    non-productive/stop-containing reads (the V/J CLASS label is still the germline call, and stop
    exposure is realistic), which restores training throughput. D is added at the allele level
    (inverted/absent D -> d_class=-1, d_valid=0, masked from the loss)."""
    try:
        vc = class_maps["V"].class_of(rec["v_call"].split(",")[0].strip())
        jc = class_maps["J"].class_of(rec["j_call"].split(",")[0].strip())
    except KeyError:
        return None
    segs = {}
    for g, G in (("v", "V"), ("d", "D"), ("j", "J")):
        s, e = rec.get(f"{g}_sequence_start"), rec.get(f"{g}_sequence_end")
        if s is not None and e is not None:
            segs[G] = (int(s), int(e))
    proj = project_codons(len(rec["sequence"]), frame_offset_of(rec["junction_start"]), segs)
    vb, jb = proj.aa_bounds.get("V"), proj.aa_bounds.get("J")
    if vb is None or jb is None:
        return None
    enc = tok.encode(rec["sequence_aa"])
    if len(enc) == 0 or len(enc) >= max_len:
        return None
    s = {"tokens": enc, "v_class": vc, "j_class": jc,
         "v_start": vb[0], "v_end": vb[1], "j_start": jb[0], "j_end": jb[1]}
    if d_index:
        db = proj.aa_bounds.get("D")
        dcall = rec.get("d_call")
        d0 = dcall.split(",")[0].strip() if dcall else None
        valid = (not rec.get("d_inverted", False)) and (d0 in d_index) and (db is not None)
        s["d_class"] = d_index[d0] if valid else -1
        s["d_start"], s["d_end"] = db if valid else (0, 0)
        s["d_valid"] = 1.0 if valid else 0.0
    return s


def _sample_stream(dc_name, class_maps, d_index, tok, max_len, sub_prob, seed, rng):
    """Real gym records -> class-target samples, with length-preserving substitution augmentation."""
    for rec in _usable_stream(dc_name, (0.3, 0.6, 0.9), 0.25, seed, 1, None):
        s = _class_sample(rec, class_maps, d_index, tok, max_len)
        if s is None:
            continue
        if sub_prob and rng.random() < sub_prob:
            enc = tok.encode(aa_substitute(rec["sequence_aa"], float(rng.uniform(0.02, 0.10)), rng))
            if len(enc) == len(s["tokens"]):               # substitution preserves length -> boundaries valid
                s = {**s, "tokens": enc}
        yield s


def strategic_train(model, cfg, class_maps, dc_name="HUMAN_IGH_OGRDB", *, d_index=None, steps=50000,
                    batch_size=64, lr=3e-4, sub_prob=0.5, seed=0, device="cpu", log_every=250,
                    grad_clip=1.0, save_path=None, save_every=5000):
    tok = AAProteinTokenizer()
    rng = np.random.default_rng(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    d_index = d_index or {}
    genes = ("v", "d", "j") if cfg.has_d else ("v", "j")
    stream = _sample_stream(dc_name, class_maps, d_index, tok, cfg.max_seq_length, sub_prob, seed, rng)

    history = []
    for step in range(1, steps + 1):
        batch = list(itertools.islice(stream, batch_size))
        if len(batch) < batch_size:
            break
        bi, tg = collate_aa(batch, cfg.max_seq_length, device)
        out = model(bi)
        total, parts = aa_loss(out, tg, genes=genes)
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
