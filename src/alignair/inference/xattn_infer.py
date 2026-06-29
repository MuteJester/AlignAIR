"""Raw-read inference for XAttnAligner (the LLM-encoder aligner): one batched forward pass →
the four heads → a full AIRR prediction record, reusing the DNAlignAIR junction/productivity
derivations. Produces the normalized prediction dicts the `benchmark/` suite scores."""
from __future__ import annotations
import torch

from ..data.tokenizer import pad_tokenize
from .dnalignair_infer import (canonicalize_sequence, junction_fields,
                               derived_rearrangement_fields, resolve_hierarchy)


@torch.no_grad()
def predict_reads_xattn(model, reference_set, reads, device=None, batch_size: int = 64,
                        topk: int = 16, seed_m: int = 0, set_band: float = 2.0,
                        cand_chunk: int = 4, locus: str = "IGH") -> list:
    device = device or next(model.parameters()).device
    model.eval()
    ref_emb = model.encode_reference(reference_set)
    has_d = reference_set.has_d
    genes = ["v", "j"] + (["d"] if has_d else [])
    names = {g.upper(): reference_set.gene(g.upper()).names for g in genes}
    preds = []
    for s in range(0, len(reads), batch_size):
        chunk = reads[s:s + batch_size]
        tok, mask = pad_tokenize(chunk)
        tok, mask = tok.to(device), mask.to(device)
        out = model(tok, mask, ref_emb, topk=topk, seed_m=seed_m,
                    reference_set=(reference_set if seed_m > 0 else None), cand_chunk=cand_chunk)
        boundary = out["boundary"]
        ori = out["orientation_logits"].argmax(-1).cpu().tolist()
        for i in range(len(chunk)):
            p = {}
            canon = canonicalize_sequence(chunk[i], ori[i])
            for g in genes:
                G = g.upper()
                mg = out["match"][G]
                idx = int(mg["best_global_idx"][i])
                p[f"{g}_call"] = names[G][idx]
                logits = mg["allele_logits"][i].tolist()
                pool = mg["pool_idx"][i].tolist()
                top = max(logits)
                keep = sorted([(int(pool[j]), logits[j]) for j in range(len(pool))
                               if top - logits[j] <= set_band], key=lambda x: -x[1])
                cset, seen = [], set()
                for gi, _ in keep:
                    nm = names[G][gi]
                    if nm not in seen:
                        seen.add(nm); cset.append(nm)
                if not cset:
                    cset = [names[G][idx]]
                p[f"{g}_call_set"] = cset
                p[f"{g}_calls"] = cset
                resolved, level = resolve_hierarchy(cset, p[f"{g}_call"])
                p[f"{g}_resolved_call"] = resolved
                p[f"{g}_call_level"] = level
                p[f"{g}_sequence_start"] = int(boundary["start"][G][i].argmax())
                p[f"{g}_sequence_end"] = int(boundary["end"][G][i].argmax()) + 1
                p[f"{g}_germline_start"] = int(mg["germ_start"][i])
                p[f"{g}_germline_end"] = int(mg["germ_end"][i]) + 1     # exclusive end (GenAIRR convention)
            p["orientation_id"] = int(ori[i])
            p["sequence"] = canon
            p["locus"] = locus
            p.update(junction_fields(p, canon, reference_set))
            p.update(derived_rearrangement_fields(p, canon))
            if "vj_in_frame" in p and "stop_codon" in p:    # AIRR productive = in-frame & no stop
                p["productive"] = bool(p["vj_in_frame"] and not p["stop_codon"])
            preds.append(p)
    return preds
