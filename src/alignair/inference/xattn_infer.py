"""Raw-read inference for XAttnAligner (the LLM-encoder aligner): one batched forward pass →
the four heads → a full AIRR prediction record. By default the allele call + germline coords come
from a CLASSICAL raw-base rescore (the `align/` package) of the neural top-k pool — which more than
doubles heavy-SHM V over the mean-MaxSim matcher — and query coords come from the trained
`region_logits`. `rescore=False` keeps the pure-neural path for ablation. Reuses the DNAlignAIR
junction/productivity derivations. Produces the normalized prediction dicts the benchmark scores."""
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
                        rescore_seed_m: int = 8) -> list:
    # rescore_seed_m: k-mer seed candidates the CLASSICAL rescore admits on top of the neural pool.
    # The seed survives SHM (enough unmutated k-mers), so it admits the true allele the encoder's
    # top-k missed — lifting heavy-SHM V ~0.61 -> ~0.91 with no retrain. Set 0 to rescore the neural
    # pool only.
    device = device or next(model.parameters()).device
    model.eval()
    ref_emb = model.encode_reference(reference_set)
    has_d = reference_set.has_d
    genes = ["v", "j"] + (["d"] if has_d else [])
    names = {g.upper(): reference_set.gene(g.upper()).names for g in genes}
    if rescore:
        from ..align import SeedPrefilter, get_aligner
        from .wfa_caller import call_segment
        sp, al = SeedPrefilter(reference_set, k=11), get_aligner()
    preds = []
    for s in range(0, len(reads), batch_size):
        chunk = reads[s:s + batch_size]
        tok, mask = pad_tokenize(chunk)
        tok, mask = tok.to(device), mask.to(device)
        out = model(tok, mask, ref_emb, topk=topk, seed_m=seed_m,
                    reference_set=(reference_set if seed_m > 0 else None), cand_chunk=cand_chunk)
        ori = out["orientation_logits"].argmax(-1).cpu().tolist()
        boundary = out["boundary"]
        dec = decode_boundaries(out["region_logits"], mask, has_d=has_d) if rescore else None
        for i in range(len(chunk)):
            p = {}
            canon = canonicalize_sequence(chunk[i], ori[i])
            for g in genes:
                G = g.upper()
                mg = out["match"][G]
                pool = mg["pool_idx"][i].tolist()
                if rescore:
                    qs, qe = dec[i][f"{g}_start"], dec[i][f"{g}_end"]
                else:
                    qs = int(boundary["start"][G][i].argmax())
                    qe = int(boundary["end"][G][i].argmax()) + 1
                # decode_boundaries returns -1 for a gene absent from the read; treat as empty span
                qs = qs if (qs is not None and qs >= 0) else 0
                qe = qe if (qe is not None and qe >= 0) else 0
                p[f"{g}_sequence_start"] = int(qs)
                p[f"{g}_sequence_end"] = int(max(qe, qs))
                sc = None
                if rescore:
                    seg = canon[qs:qe] if (qs is not None and qe and qe > qs) else ""
                    sc = call_segment(seg, G, pool, reference_set, sp, al,
                                      m_seed=rescore_seed_m, set_band=set_band, allowed=None)
                if sc is not None:                                 # classical rescore of the neural pool
                    idx = sc.best_idx
                    cset = [names[G][j] for j in sc.set_idx]
                    p[f"{g}_germline_start"] = int(sc.germ_start)
                    p[f"{g}_germline_end"] = int(sc.germ_end) + 1
                    p[f"{g}_cigar"] = sc.cigar
                else:                                              # neural fallback (short seg / rescore off)
                    idx = int(mg["best_global_idx"][i])
                    logits = mg["allele_logits"][i].tolist()
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
                    p[f"{g}_germline_start"] = int(mg["germ_start"][i])
                    p[f"{g}_germline_end"] = int(mg["germ_end"][i]) + 1
                p[f"{g}_call"] = names[G][idx]
                p[f"{g}_call_set"] = cset
                p[f"{g}_calls"] = cset
                resolved, level = resolve_hierarchy(cset, p[f"{g}_call"])
                p[f"{g}_resolved_call"] = resolved
                p[f"{g}_call_level"] = level
            p["orientation_id"] = int(ori[i])
            p["sequence"] = canon
            p["locus"] = locus
            p.update(junction_fields(p, canon, reference_set))
            p.update(derived_rearrangement_fields(p, canon))
            if "vj_in_frame" in p and "stop_codon" in p:
                p["productive"] = bool(p["vj_in_frame"] and not p["stop_codon"])
            preds.append(p)
    return preds
