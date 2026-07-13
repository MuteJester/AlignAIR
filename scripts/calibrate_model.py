"""Post-training calibration: fit per-gene allele-confidence temperatures on held-out GenAIRR reads,
report ECE before/after, and write a clean (pickle-free) calibrated model artifact.

    python scripts/calibrate_model.py --model <in>.alignair --dataconfig HUMAN_IGH_OGRDB \
        --out <out>.alignair

Temperature scaling is argmax-preserving, so calls/coords are unchanged — only the reported allele
likelihoods (and anything consuming them, e.g. genotype inference) become calibrated.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools

import torch

import GenAIRR.data as gd
from alignair.api import load_model
from alignair.predict.calibrate import calibrate_allele_temperatures
from alignair import model_file as mf
from alignair.train.gym import Curriculum, build_experiment


def _is_tcr(dc) -> bool:
    return "TCR" in str(getattr(dc.metadata, "chain_type", "")).upper()


def _val_records(dc, n_per_stratum: int, seed: int) -> list:
    """A spread of validation records (clean / moderate / hard) so the fitted T generalizes."""
    cur = Curriculum()
    recs = []
    for i, prog in enumerate((0.1, 0.4, 0.7)):
        p = dict(cur.params(prog))
        if _is_tcr(dc):
            p["mutation_rate"] = 0.0
            p.pop("mutation_count", None)
        exp = build_experiment(dc, p, allow_curatable=True)
        recs += list(itertools.islice(exp.stream_records(n=None, seed=seed + i), n_per_stratum))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataconfig", nargs="+", required=True, help="GenAIRR dataconfig name(s) of the model")
    ap.add_argument("--out", required=True, help="calibrated .alignair output (pickle-free)")
    ap.add_argument("--n", type=int, default=1500, help="validation reads per (locus, stratum)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--trust-pickle", action="store_true")
    a = ap.parse_args()

    model, ref = load_model(a.model, dataconfigs=a.dataconfig, device=a.device, trust_pickle=a.trust_pickle)
    genes = ("v", "d", "j") if model.cfg.has_d else ("v", "j")
    dcs = [getattr(gd, d) if isinstance(d, str) else d for d in a.dataconfig]

    val = []
    for i, dc in enumerate(dcs):
        val += _val_records(dc, a.n, seed=1000 + 100 * i)
    print(f"fitting temperatures on {len(val)} held-out reads across {len(dcs)} locus/loci...", flush=True)

    temps, report = calibrate_allele_temperatures(model, ref, val, genes=genes, device=a.device)
    print("\nper-gene calibration (ECE = expected calibration error; acc unchanged by scaling):")
    print(f"  {'gene':4s} {'T':>6s} {'ECE_before':>10s} {'ECE_after':>9s} {'acc':>6s} {'n':>7s}")
    for g in genes:
        if g in report:
            r = report[g]
            print(f"  {g.upper():4s} {r['T']:6.3f} {r['ece_before']:10.3f} {r['ece_after']:9.3f} "
                  f"{r['acc']:6.3f} {r['n']:7d}")

    # bake the temperatures into the config and write a clean, pickle-free calibrated artifact
    model.cfg = dataclasses.replace(model.cfg, allele_temperatures=temps)
    md = mf.read_metadata(a.model) if mf.container.is_alignair_file(a.model) else {}
    training = dict(md.get("training", {}))
    card = dict(md.get("card", {})) if isinstance(md.get("card"), dict) else {}
    card["calibration"] = {"method": "temperature_scaling", "allele_temperatures": temps}
    mf.save_model(a.out, model, dataconfigs=a.dataconfig, training=training, include_trusted_pickle=False,
                  model_id=md.get("model_id"), model_version=md.get("model_version"),
                  description=(md.get("description", "") + " [calibrated]").strip(), card=card)
    print(f"\nwrote calibrated (pickle-free) model -> {a.out}  temperatures={temps}")


if __name__ == "__main__":
    main()
