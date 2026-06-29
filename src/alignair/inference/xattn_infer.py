"""Raw-read inference for XAttnAligner (the LLM-encoder aligner): one batched forward pass →
the four heads → a full AIRR prediction record. By default the allele call + germline coords come
from a CLASSICAL raw-base rescore (parasail, GIL-releasing → threaded) of the neural top-k ∪ k-mer
seed pool — which more than triples heavy-SHM V over the mean-MaxSim matcher — and query coords come
from the trained `region_logits`. `rescore=False` keeps the pure-neural path for ablation. Reuses the
DNAlignAIR junction/productivity derivations. Produces the prediction dicts the benchmark scores."""
from __future__ import annotations
import torch

from ..data.tokenizer import pad_tokenize
from ..nn.heads.region import decode_boundaries
from .dnalignair_infer import (canonicalize_sequence, junction_fields,
                               derived_rearrangement_fields, resolve_hierarchy)


@torch.no_grad()
def predict_reads_xattn(model, reference_set, reads, device=None, batch_size: int = 64,
                        topk: int = 16, seed_m: int = 0, set_band: float = 2.0,
                        cand_chunk: int = 4, locus: str = "IGH", rescore: bool = True,
                        rescore_seed_m: int = 8, align_workers: int = 8) -> list:
    # rescore_seed_m: k-mer seed candidates the classical rescore admits on top of the neural pool
    # (survives SHM → admits the true allele the encoder's top-k missed; heavy-SHM V ~0.61→~0.91).
    device = device or next(model.parameters()).device
    model.eval()
    ref_emb = model.encode_reference(reference_set)
    has_d = reference_set.has_d
    genes = ["v", "j"] + (["d"] if has_d else [])
    gU = [g.upper() for g in genes]
    names = {g.upper(): reference_set.gene(g.upper()).names for g in genes}
    if rescore:
        from ..align import SeedPrefilter, get_aligner
        from .wfa_caller import call_segments_batched
        sp = SeedPrefilter(reference_set, k=11)
        al = get_aligner(prefer="parasail")          # parasail releases the GIL -> threaded rescore
    preds = []
    for s in range(0, len(reads), batch_size):
        chunk = reads[s:s + batch_size]
        tok, mask = pad_tokenize(chunk)
        tok, mask = tok.to(device), mask.to(device)
        out = model(tok, mask, ref_emb, topk=topk, seed_m=seed_m,
                    reference_set=(reference_set if seed_m > 0 else None), cand_chunk=cand_chunk)
        B = len(chunk)
        ori = out["orientation_logits"].argmax(-1).cpu().tolist()
        boundary = out["boundary"]
        dec = decode_boundaries(out["region_logits"], mask, has_d=has_d) if rescore else None
        canon = [canonicalize_sequence(chunk[i], ori[i]) for i in range(B)]
        # one CPU transfer per gene-tensor (not a per-element .item())
        pool_l = {G: out["match"][G]["pool_idx"].cpu().tolist() for G in gU}
        best_l = {G: out["match"][G]["best_global_idx"].cpu().tolist() for G in gU}
        gs_l = {G: out["match"][G]["germ_start"].cpu().tolist() for G in gU}
        ge_l = {G: out["match"][G]["germ_end"].cpu().tolist() for G in gU}
        logit_l = {G: out["match"][G]["allele_logits"].cpu().tolist() for G in gU}
        # query coords + the rescore work-list (one item per read/gene)
        qcoord, items, item_map = {}, [], []
        for i in range(B):
            for g in genes:
                G = g.upper()
                if rescore:
                    qs, qe = dec[i][f"{g}_start"], dec[i][f"{g}_end"]
                else:
                    qs = int(boundary["start"][G][i].argmax())
                    qe = int(boundary["end"][G][i].argmax()) + 1
                qs = qs if (qs is not None and qs >= 0) else 0
                qe = qe if (qe is not None and qe >= 0) else 0
                qcoord[(i, G)] = (int(qs), int(max(qe, qs)))
                if rescore:
                    seg = canon[i][qs:qe] if qe > qs else ""
                    items.append((seg, G, pool_l[G][i], None)); item_map.append((i, G))
        # ONE threaded batched alignment over the entire read batch (the rescore bottleneck)
        sc_by = {}
        if rescore:
            sc_list = call_segments_batched(items, reference_set, sp, al, m_seed=rescore_seed_m,
                                            set_band=set_band, workers=align_workers)
            sc_by = {item_map[k]: sc_list[k] for k in range(len(item_map))}
        for i in range(B):
            p = {}
            for g in genes:
                G = g.upper()
                qs, qe = qcoord[(i, G)]
                p[f"{g}_sequence_start"] = qs
                p[f"{g}_sequence_end"] = qe
                sc = sc_by.get((i, G))
                if sc is not None:                                 # classical rescore of the neural pool
                    idx = sc.best_idx
                    cset = [names[G][j] for j in sc.set_idx]
                    p[f"{g}_germline_start"] = int(sc.germ_start)
                    p[f"{g}_germline_end"] = int(sc.germ_end) + 1
                    p[f"{g}_cigar"] = sc.cigar
                else:                                              # neural fallback (short seg / rescore off)
                    idx = int(best_l[G][i])
                    logits, pool = logit_l[G][i], pool_l[G][i]
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
                    p[f"{g}_germline_start"] = int(gs_l[G][i])
                    p[f"{g}_germline_end"] = int(ge_l[G][i]) + 1
                p[f"{g}_call"] = names[G][idx]
                p[f"{g}_call_set"] = cset
                p[f"{g}_calls"] = cset
                resolved, level = resolve_hierarchy(cset, p[f"{g}_call"])
                p[f"{g}_resolved_call"] = resolved
                p[f"{g}_call_level"] = level
            p["orientation_id"] = int(ori[i])
            p["sequence"] = canon[i]
            p["locus"] = locus
            p.update(junction_fields(p, canon[i], reference_set))
            p.update(derived_rearrangement_fields(p, canon[i]))
            if "vj_in_frame" in p and "stop_codon" in p:
                p["productive"] = bool(p["vj_in_frame"] and not p["stop_codon"])
            preds.append(p)
    return preds
