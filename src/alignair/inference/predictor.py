"""Predictor: batched eval/no-grad forward returning the legacy numpy output dict."""
import numpy as np
import torch


class Predictor:
    def __init__(self, model, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, tokenized, batch_size: int = 256) -> dict:
        tokens = torch.as_tensor(np.asarray(tokenized), dtype=torch.long)
        n = tokens.shape[0]
        chunks = []
        for i in range(0, n, batch_size):
            batch = tokens[i:i + batch_size].to(self.device)
            out = self.model(batch).as_dict()
            chunks.append({k: v.detach().cpu().numpy() for k, v in out.items()})

        keys = chunks[0].keys()
        return {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys}
