"""Single hardened AA-head inference path, shared by Stage-2 calibration and Phase-3 eval."""
from __future__ import annotations
import numpy as np, torch
from .tokenizer import AAProteinTokenizer

_AA = set("ACDEFGHIKLMNPQRSTVWYX*")

def class_probs(model, seqs, cfg, device, key, batch: int = 256) -> np.ndarray:
    if not seqs:
        raise ValueError("class_probs: empty sequence list")
    L = cfg.max_seq_length
    bad_len = [i for i, s in enumerate(seqs) if len(s) > L]
    if bad_len:
        raise ValueError(f"class_probs: sequences exceed max_seq_length {L}: idx {bad_len[:5]}")
    bad_aa = [i for i, s in enumerate(seqs) if set(s) - _AA]
    if bad_aa:
        raise ValueError(f"class_probs: non-standard AA characters at idx {bad_aa[:5]}")
    tok = AAProteinTokenizer(); out = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        toks = torch.zeros(len(chunk), L, dtype=torch.long)
        for j, s in enumerate(chunk):
            e = tok.encode(s); toks[j, :len(e)] = torch.from_numpy(e)
        with torch.no_grad():
            lg = model({"tokenized_sequence": toks.to(device)})[key].cpu()
        out.append(torch.softmax(lg, -1).numpy())
    return np.concatenate(out, 0)
