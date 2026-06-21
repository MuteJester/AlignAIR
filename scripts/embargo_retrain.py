"""Gold-standard Property-1 test: train with ~18 V alleles EMBARGOED entirely, then call
them as novel (provided only in the inference reference). If held-out-allele recall ~=
trained-allele recall, the model CONDITIONS on the reference it is handed rather than
MEMORIZING the training allele set — the core dynamic-genotype claim.

Embargo is enforced two ways: (1) the training dataconfig removes the alleles from
v_alleles so GenAIRR never simulates them; (2) the training ReferenceSet excludes them so
the model never encodes/optimizes them. Eval simulates reads FROM the held-out alleles
(they are the true source) and predicts against the FULL reference (which includes them).
"""
import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import gen_records  # noqa: E402  (only for type parity; unused)

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.losses.dnalignair_loss import DNAlignAIRLoss  # noqa: E402
from alignair.gym.gym import AlignAIRGym, build_experiment  # noqa: E402
from alignair.gym.curriculum import Curriculum  # noqa: E402
from alignair.training.gym_trainer import GymTrainer  # noqa: E402
from alignair.inference.dnalignair_infer import predict_reads  # noqa: E402


def restrict_dc(dc, embargo):
    """deepcopy with embargoed V alleles removed (genes keep >=1 allele)."""
    d = copy.deepcopy(dc)
    for gene, alleles in list(d.v_alleles.items()):
        kept = [a for a in alleles if a.name not in embargo]
        if kept:
            d.v_alleles[gene] = kept
    return d


def keep_only_dc(dc, keep):
    """deepcopy with v_alleles restricted to ONLY `keep`; prune+renorm gene_use_dict V."""
    d = copy.deepcopy(dc)
    surviving = set()
    for gene, alleles in list(d.v_alleles.items()):
        kept = [a for a in alleles if a.name in keep]
        if kept:
            d.v_alleles[gene] = kept
            surviving.add(gene)
        else:
            del d.v_alleles[gene]
    vu = {g: p for g, p in d.gene_use_dict["V"].items() if g in surviving}
    tot = sum(vu.values()) or 1.0
    d.gene_use_dict["V"] = {g: p / tot for g, p in vu.items()}
    return d


def pick_embargo(dc, n):
    """n V alleles taken from multi-allele genes (so each gene survives in training)."""
    out = []
    for gene, alleles in dc.v_alleles.items():
        if len(alleles) >= 2:
            out += [a.name for a in alleles[1:]]
        if len(out) >= n:
            break
    return set(out[:n])


@torch.no_grad()
def eval_recall(model, rs_full, dc_subset, n, p, label, device):
    """Simulate reads FROM dc_subset's alleles, predict against rs_full, report recall."""
    exp = build_experiment(dc_subset, Curriculum().params(p))
    recs = list(exp.stream_records(n=n, seed=999))
    reads = [r["sequence"] for r in recs]
    truth = [str(r.get("v_call", "")).split(",") for r in recs]
    preds = predict_reads(model, rs_full, reads, device=device, rerank="learned")
    allele = gene = inset = tot = 0
    for ts, p_ in zip(truth, preds):
        ts = [t for t in ts if t]
        if not ts:
            continue
        tot += 1
        tg = {t.split("*")[0] for t in ts}
        allele += int(p_["v_call"] in ts)
        gene += int(p_["v_call"].split("*")[0] in tg)
        inset += int(bool(set(p_.get("v_call_set", [])) & set(ts)))
    print(f"  [{label} p={p}] n={tot}  allele={allele/max(tot,1):.3f}  "
          f"gene={gene/max(tot,1):.3f}  in-set={inset/max(tot,1):.3f}")
    return allele / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--n-embargo", type=int, default=18)
    ap.add_argument("--eval-n", type=int, default=400)
    ap.add_argument("--save", default=".private/models/embargo.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = gdata.HUMAN_IGH_OGRDB
    embargo = pick_embargo(dc, args.n_embargo)
    # a control set of TRAINED alleles (also from multi-allele genes, disjoint from embargo)
    control = set()
    for gene, alleles in dc.v_alleles.items():
        if len(alleles) >= 2 and alleles[0].name not in embargo:
            control.add(alleles[0].name)        # *01 of multi-allele genes (kept in training)
        if len(control) >= args.n_embargo:
            break
    print(f"embargo {len(embargo)} V alleles (held out of training); control {len(control)} trained alleles")

    dc_train = restrict_dc(dc, embargo)
    rs_train = ReferenceSet.from_dataconfigs(dc_train)     # excludes embargo (model never sees them)
    rs_full = ReferenceSet.from_dataconfigs(dc)            # includes embargo (given at inference)
    print(f"training reference: {len(rs_train.gene('V').names)} V alleles (full = {len(rs_full.gene('V').names)})")

    cfg = DNAlignAIRConfig(d_model=args.d_model, n_layers=args.layers, backbone="shared",
                           aligner="softdp")
    model = DNAlignAIR(cfg)
    gym = AlignAIRGym([dc_train], rs_train, seed=0, curriculum=Curriculum())
    trainer = GymTrainer(model, DNAlignAIRLoss(has_d=rs_train.has_d), rs_train, gym,
                         lr=5e-4, batch_size=args.batch, reader=True, scheduled_sampling=True,
                         reader_novel_prob=0.4, reader_novel_snps=1)
    print(f"training {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, {args.steps} steps "
          f"(embargoed alleles NEVER simulated or referenced)...")
    trainer.fit(total_steps=args.steps, global_total=args.steps)
    if args.save:
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                    "embargo": sorted(embargo)}, args.save)
        print(f"saved -> {args.save}")

    model.eval()
    print("\n=== GOLD-STANDARD: held-out (never trained) vs control (trained) allele recall ===")
    dc_held = keep_only_dc(dc, embargo)
    dc_ctrl = keep_only_dc(dc, control)
    for p in (0.0, 0.5):
        held = eval_recall(model, rs_full, dc_held, args.eval_n, p, "HELD-OUT/novel", device)
        ctrl = eval_recall(model, rs_full, dc_ctrl, args.eval_n, p, "CONTROL/trained", device)
        print(f"  -> p={p}: held-out/control allele-recall ratio = {held/max(ctrl,1e-6):.2f} "
              f"(near 1.0 = conditions on reference, not memorized)\n")


if __name__ == "__main__":
    main()
