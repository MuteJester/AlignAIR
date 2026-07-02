"""In-distribution MVP eval for a trained AIRRistotle: decode N held-out records (genotype = true
alleles + distractors, as in training) and score call accuracy + coordinate MAE vs truth."""
import argparse, random, statistics, torch
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle
from alignair.airristotle.infer import decode_record
from alignair.gym.gym import build_experiment
from alignair.gym.curriculum import Curriculum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/airristotle_mvp_v1.pt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-distractors", type=int, default=6)
    ap.add_argument("--progress", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=999)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AIRRTokenizer(); rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    m = AIRRistotle(AIRRConfig(**ck["config"])); m.load_state_dict(ck["model"]); m.to(dev).eval()
    params = dict(Curriculum().params(a.progress))
    recs = list(build_experiment(gdata.HUMAN_IGH_OGRDB, params).stream_records(n=a.n, seed=a.seed))
    rng = random.Random(0)
    hit = {g: [0, 0] for g in "vdj"}
    mae = {k: [] for k in ("v_S", "v_E", "v_GS", "v_GE", "d_S", "d_E", "j_S", "j_E", "jn_S", "jn_E")}
    prod_ok = ori_ok = n = 0
    for rec in recs:
        out = decode_record(m, tok, rec, rs, n_distractors=a.n_distractors, rng=rng, device=dev)
        n += 1
        for g in "vdj":
            tc = str(rec.get(f"{g}_call") or "").split(",")[0]
            if tc:
                hit[g][1] += 1
                hit[g][0] += (out.get(f"{g}_call") == tc)
        def add(key, pred, truth):
            if pred is not None and truth is not None:
                mae[key].append(abs(int(pred) - int(truth)))
        add("v_S", out.get("v_sequence_start"), rec.get("v_sequence_start"))
        add("v_E", out.get("v_sequence_end"), rec.get("v_sequence_end"))
        add("v_GS", out.get("v_germline_start"), rec.get("v_germline_start"))
        add("v_GE", out.get("v_germline_end"), rec.get("v_germline_end"))
        add("d_S", out.get("d_sequence_start"), rec.get("d_sequence_start"))
        add("d_E", out.get("d_sequence_end"), rec.get("d_sequence_end"))
        add("j_S", out.get("j_sequence_start"), rec.get("j_sequence_start"))
        add("j_E", out.get("j_sequence_end"), rec.get("j_sequence_end"))
        add("jn_S", out.get("junction_start"), rec.get("junction_start"))
        add("jn_E", out.get("junction_end"), rec.get("junction_end"))
        prod_ok += (bool(out.get("productive")) == bool(rec.get("productive")))
        ori_ok += (int(out.get("orientation_id", 0)) == 0)
    print(f"AIRRistotle eval  n={n}  (progress={a.progress}, distractors={a.n_distractors})")
    for g in "vdj":
        h, t = hit[g]
        print(f"  {g.upper()} call acc = {h/t:.3f}  (n={t})")
    print("  coord MAE (nt):  " + "  ".join(f"{k}={statistics.mean(v):.1f}" for k, v in mae.items() if v))
    print(f"  productive acc = {prod_ok/n:.3f}   orientation acc = {ori_ok/n:.3f}")


if __name__ == "__main__":
    main()
