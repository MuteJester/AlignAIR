"""Throughput breakdown: isolate where DNAlignAIR inference time goes, stage by stage, so we can
see the speed CEILING a "neural allele-prediction + classical align" architecture would target.

Stages (cumulative), reads/s on GPU:
  encoder      : pad_tokenize + backbone encoder + segmentation/state heads (forward_dense)
  +retrieval   : + match_alleles (pooled-cosine allele retrieval)            <- new-arch neural cost
  +coords      : + compute_germline_logits germline-coordinate decode
  predict/none : full predict_reads with pooled top-1 (rerank='none')
  predict/dp   : full predict_reads with the learned DP reader (rerank='learned')

Run:
  PYTHONPATH=src .venv/bin/python scripts/exp_throughput_breakdown.py \
      --model .private/models/seed_extend_d64_reader.pt --n 512 --batch-size 64
"""
import argparse
import time

import torch
import GenAIRR.data as gdata

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.reference.reference_set import ReferenceSet
from alignair.data.tokenizer import pad_tokenize
from alignair.gym.gym import build_experiment
from alignair.inference.dnalignair_infer import predict_reads
from alignair.training.germline_tf import compute_germline_logits


def _reads(n):
    params = dict(mutation_rate=0.12, productive_only=False, end_loss_5=(0, 0), end_loss_3=(0, 0),
                  indel_count=(0, 0), seq_error_rate=0.0, ambiguous_count=(0, 0))
    exp = build_experiment(gdata.HUMAN_IGH_OGRDB, params)
    return [str(r["sequence"]).upper() for r in exp.stream_records(n=n, seed=7)]


def _time(fn, reads, bs, device, reps=3):
    # warmup
    fn(reads[:bs])
    if device == "cuda":
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for s in range(0, len(reads), bs):
            fn(reads[s:s + bs])
        if device == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return len(reads) / best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/seed_extend_d64_reader.pt")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = DNAlignAIR(DNAlignAIRConfig(**ck["config"]))
    model.load_state_dict(ck["model"]); model.to(device).eval()
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ref_emb = model.encode_reference(rs)
    has_d = rs.has_d
    genes = ["v", "j"] + (["d"] if has_d else [])
    reads = _reads(a.n)
    print(f"model {a.model} | d_model={ck['config'].get('d_model')} | device={device} | "
          f"{a.n} reads, bs={a.batch_size}\n")

    @torch.no_grad()
    def f_encoder(chunk):
        tk, mk = pad_tokenize(chunk); tk, mk = tk.to(device), mk.to(device)
        return model.forward_dense(tk, mk, None)

    @torch.no_grad()
    def f_retrieval(chunk):
        tk, mk = pad_tokenize(chunk); tk, mk = tk.to(device), mk.to(device)
        return model(tk, mk, ref_emb)

    @torch.no_grad()
    def f_coords(chunk):
        tk, mk = pad_tokenize(chunk); tk, mk = tk.to(device), mk.to(device)
        out = model(tk, mk, ref_emb)
        pr = out["region_logits"].argmax(-1)
        pidx = {g.upper(): out["match"][g.upper()].argmax(-1) for g in genes}
        compute_germline_logits(model, out["canon_tokens"], mk, {}, ref_emb, has_d,
                                region_labels=pr, allele_idx=pidx, reps=out["reps"])

    def f_none(chunk):
        return predict_reads(model, rs, chunk, device=device, batch_size=a.batch_size, rerank="none")

    def f_dp(chunk):
        return predict_reads(model, rs, chunk, device=device, batch_size=a.batch_size,
                             rerank="learned", v_reader="learned")

    stages = [("encoder", f_encoder), ("+retrieval", f_retrieval), ("+coords", f_coords),
              ("predict/none", f_none), ("predict/dp", f_dp)]
    print(f"{'stage':16s} {'reads/s':>10s} {'ms/read':>9s}")
    for name, fn in stages:
        rps = _time(fn, reads, a.batch_size, device)
        print(f"{name:16s} {rps:10.1f} {1000 / rps:9.2f}")
    print("\n'+retrieval' is the neural floor a cross-attention-head architecture targets;\n"
          "gap to 'predict/dp' is the per-candidate differentiable-DP reader cost.")


if __name__ == "__main__":
    main()
