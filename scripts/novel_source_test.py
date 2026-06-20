"""DEFINITIVE Property-1 test: reads that genuinely DERIVE FROM a novel allele.

Unlike heldout_alleles.py (reads from real A, novel relabeled — a closer real sibling can
legitimately win), here we EDIT the read so it carries the novel germline's bases at the
novel SNP positions. The novel allele is then the true source and the genuinely-closest
germline, so a correctly-functioning dynamic reference MUST call it. This is the real
scenario: a donor carries an allele absent from the training set; reads come from it.

For each victim allele A: define A~novel = A with `snps` substitutions at fixed germline
positions in a commonly-observed region; for each indel-free read from A that observes those
positions, overwrite the read bases there with A~novel's bases. Provide a genotype with A
replaced by A~novel and measure whether the model calls / sets the novel allele.
"""
import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import gen_records  # noqa: E402

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.inference.dnalignair_infer import predict_reads  # noqa: E402

BASES = "ACGT"


def build_novel_for(seq, snps, rng):
    """Return (novel_seq, {gpos: new_base}) — `snps` substitutions at fixed mid-V positions
    (germline 60..60+stride*snps) so most reads observe them; falls back if seq is short."""
    s = list(seq)
    L = len(s)
    changes = {}
    cand = [p for p in range(60, min(L, 230), 12) if s[p] in BASES][:snps]
    for p in cand:
        nb = rng.choice([b for b in BASES if b != s[p]])
        s[p] = nb
        changes[p] = nb
    return "".join(s), changes


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_novel.pt")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--n-victims", type=int, default=20)
    ap.add_argument("--snps", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ckpt = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ckpt["config"]) if isinstance(ckpt["config"], dict) else ckpt["config"]
    model = DNAlignAIR(cfg).to(device); model.load_state_dict(ckpt["model"]); model.eval()

    recs = gen_records(args.p, args.n, args.seed, None)
    vref = full_rs.gene("V")
    true_v = [str(r.get("v_call", "")).split(",")[0] for r in recs]
    present = [v for v in dict.fromkeys(true_v) if v and v in vref.index]
    rng = random.Random(args.seed)
    victims = set(rng.sample(present, min(args.n_victims, len(present))))

    # define the novel variant + its germline-position edits for each victim allele
    novel_name, novel_seq, novel_edits = {}, {}, {}
    for A in victims:
        seq = vref.sequences[vref.index[A]]
        ns, ch = build_novel_for(seq, args.snps, random.Random(hash(A) & 0xffff))
        novel_name[A] = f"{A}~novel"; novel_seq[A] = ns; novel_edits[A] = ch

    # build the genotype reference: victims replaced by their novel variant
    genes = {}
    for G, ref in full_rs.genes.items():
        gmap = {}
        for nm, sq in zip(ref.names, ref.sequences):
            if G == "V" and nm in victims:
                gmap[novel_name[nm]] = novel_seq[nm]
            else:
                gmap[nm] = sq
        genes[G] = gmap
    novel_rs = ReferenceSet.from_genotype(genes)

    # EDIT reads from victim alleles so they derive from the novel germline
    edited, edited_truth, kept = [], [], 0
    for r, A in zip(recs, true_v):
        if A not in victims or r.get("n_v_indels", 0):           # need clean V coord mapping
            continue
        gs, ge = r.get("v_germline_start"), r.get("v_germline_end")
        ss = r.get("v_sequence_start")
        if gs is None or ss is None:
            continue
        edits = novel_edits[A]
        if not all(gs <= g < ge for g in edits):                 # all SNP positions observed
            continue
        seqlist = list(r["sequence"])
        ok = True
        for g, base in edits.items():
            pos = ss + (g - gs)
            if pos >= len(seqlist):
                ok = False; break
            seqlist[pos] = base
        if not ok:
            continue
        edited.append("".join(seqlist)); edited_truth.append(A); kept += 1

    if not edited:
        print("no eligible reads (try larger --n)"); return
    preds = predict_reads(model, novel_rs, edited, device=device, rerank="learned")
    hit = gene_hit = set_hit = 0
    for A, p in zip(edited_truth, preds):
        novel = novel_name[A]; gene = A.split("*")[0]
        hit += int(p["v_call"] == novel)
        gene_hit += int(p["v_call"].split("*")[0] == gene)
        set_hit += int(novel in p.get("v_call_set", []))
    n = len(edited)
    print(f"\n=== DEFINITIVE novel-source test (reads derive FROM the novel allele) ===")
    print(f"model={args.model}  snps={args.snps}  eligible victim reads={n}")
    print(f"[novel allele-level recall] {hit}/{n} = {hit / n:.3f}  "
          f"(novel is the TRUE source & closest germline -> should be high)")
    print(f"[novel gene-level recall]   {gene_hit}/{n} = {gene_hit / n:.3f}")
    print(f"[novel in calibrated set]   {set_hit}/{n} = {set_hit / n:.3f}")


if __name__ == "__main__":
    main()
