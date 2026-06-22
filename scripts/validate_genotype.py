"""End-to-end validation of the DYNAMIC GENOTYPE feature with the production setup
(scaled_long + parasail V reader). Proves the user-facing promise: supply a genotype as a
YAML or FASTA file — fewer alleles than the trained reference OR novel alleles — and AlignAIR
uses exactly that reference, well.

TEST 1  subset via genotype= MASK   : 0 calls outside the genotype (hard guarantee) + accuracy
                                       on in-genotype reads is >= full-reference accuracy.
TEST 2  subset via FILE (YAML+FASTA) : load a reduced reference from disk -> identical calls;
                                       rs.subset() preserves anchors so junction still emits.
TEST 3  novel via FILE (YAML+FASTA)  : alleles a few SNPs off the trained set, supplied by file;
                                       the model calls the novel stand-in (recall), learned vs parasail.
TEST 4  FASTA loader                 : gene inference (V/D/J) + round-trip counts.
"""
import argparse, json, os, random, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import torch
from baseline_igblast import gen_records
from heldout_alleles import snp_perturb, build_novel_genotype
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.inference.dnalignair_infer import predict_reads


def vacc(recs, preds, only=None):
    hit = tot = 0
    for r, p in zip(recs, preds):
        v = str(r.get("v_call", "")).split(",")[0]
        if not v or (only is not None and v not in only):
            continue
        tot += 1; hit += int(p["v_call"] == v or v in p.get("v_call_set", []))
    return hit / max(tot, 1), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-victims", type=int, default=25)
    ap.add_argument("--snps", type=int, default=3)
    ap.add_argument("--topk", type=int, default=32)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)
    full = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ck = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ck["config"]); m = DNAlignAIR(cfg).to(device); m.load_state_dict(ck["model"]); m.eval()
    cal = json.load(open(".private/models/allele_set_calibration.json"))
    PR = dict(device=device, topk=args.topk, rerank="learned", calibration=cal, v_reader="parasail")

    recs = gen_records(0.5, args.n, args.seed, None)
    reads = [r["sequence"] for r in recs]
    present = [v for v in dict.fromkeys(str(r.get("v_call", "")).split(",")[0] for r in recs) if v]

    # ---- TEST 1: subset via genotype= mask ----
    keepV = set(rng.sample(present, max(1, len(present) // 2)))
    sub = {"v": sorted(keepV), "d": full.gene("D").names[:12], "j": full.gene("J").names[:4]}
    allowed = {g.upper(): set(v) for g, v in sub.items()}
    full_preds = predict_reads(m, full, reads, **PR)
    sub_preds = predict_reads(m, full, reads, genotype=sub, **PR)
    outside = sum(1 for p in sub_preds for g in ("v", "d", "j") if p[f"{g}_call"] not in allowed[g.upper()])
    a_full, n_in = vacc(recs, full_preds, only=keepV)
    a_sub, _ = vacc(recs, sub_preds, only=keepV)
    print("=== TEST 1: subset via genotype= mask ===")
    print(f"  genotype: V={len(sub['v'])}/{len(full.gene('V'))}  D={len(sub['d'])}  J={len(sub['j'])}")
    print(f"  out-of-genotype calls: {outside}/{len(sub_preds)*3}  (MUST be 0)")
    print(f"  V acc on in-genotype reads (n={n_in}): full-ref={a_full:.3f}  subset={a_sub:.3f}  (subset >= full expected)\n")

    # ---- TEST 2: subset via FILE (YAML + FASTA) ----
    with tempfile.TemporaryDirectory() as td:
        sub_rs_anchored = full.subset(sub)                       # preserves anchors -> junction
        sub_rs_anchored.to_yaml(os.path.join(td, "g.yaml"))
        # write a FASTA of the same subset
        with open(os.path.join(td, "g.fasta"), "w") as fh:
            for G in ("V", "D", "J"):
                ref = full.gene(G)
                for nm in sub[G.lower()]:
                    fh.write(f">{nm}\n{ref.sequences[ref.index[nm]]}\n")
        rs_yaml = ReferenceSet.from_yaml(os.path.join(td, "g.yaml"))
        rs_fasta = ReferenceSet.from_fasta(os.path.join(td, "g.fasta"))
        p_anc = predict_reads(m, sub_rs_anchored, reads, **PR)
        p_yaml = predict_reads(m, rs_yaml, reads, **PR)
        p_fasta = predict_reads(m, rs_fasta, reads, **PR)
    agree_y = sum(a["v_call"] == b["v_call"] for a, b in zip(p_anc, p_yaml)) / len(reads)
    agree_f = sum(a["v_call"] == b["v_call"] for a, b in zip(p_anc, p_fasta)) / len(reads)
    junc = sum(1 for p in p_anc if p.get("junction")) / len(reads)
    a_file, _ = vacc(recs, p_yaml, only=keepV)
    print("=== TEST 2: subset via FILE (YAML + FASTA) ===")
    print(f"  loaded V/D/J from YAML: {len(rs_yaml.gene('V'))}/{len(rs_yaml.gene('D'))}/{len(rs_yaml.gene('J'))}; "
          f"FASTA: {len(rs_fasta.gene('V'))}/{len(rs_fasta.gene('D'))}/{len(rs_fasta.gene('J'))}")
    print(f"  v_call agreement subset()==YAML: {agree_y:.3f}  subset()==FASTA: {agree_f:.3f}")
    print(f"  V acc from file (n in-genotype): {a_file:.3f}  | junction emitted (anchored subset): {junc:.3f}\n")

    # ---- TEST 3: NOVEL alleles via FILE ----
    victims = set(rng.sample(present, min(args.n_victims, len(present))))
    genes, rename = build_novel_genotype(full, victims, args.snps, rng)
    with tempfile.TemporaryDirectory() as td:
        ReferenceSet.from_genotype(genes).to_yaml(os.path.join(td, "novel.yaml"))
        with open(os.path.join(td, "novel.fasta"), "w") as fh:
            for G, gmap in genes.items():
                for nm, seq in gmap.items():
                    fh.write(f">{nm}\n{seq}\n")
        novel_yaml = ReferenceSet.from_yaml(os.path.join(td, "novel.yaml"))
        novel_fasta = ReferenceSet.from_fasta(os.path.join(td, "novel.fasta"))
    import parasail
    MAT = parasail.matrix_create("ACGTN", 2, -1)
    for tag, nrs, reader in (("YAML+parasail", novel_yaml, "parasail"),
                             ("FASTA+parasail", novel_fasta, "parasail"),
                             ("YAML+learned", novel_yaml, "learned")):
        prk = dict(PR); prk["v_reader"] = reader
        preds = predict_reads(m, nrs, reads, **prk)
        germ = dict(zip(nrs.gene("V").names, nrs.gene("V").sequences))
        tot = strict = gene = equiv = inset = miss = justified = 0
        for r, p in zip(recs, preds):
            v = str(r.get("v_call", "")).split(",")[0]
            if v not in victims:
                continue
            tot += 1; nov = rename[v]; gn = v.split("*")[0]
            strict += int(p["v_call"] == nov)
            gene += int(p["v_call"].split("*")[0] == gn)
            equiv += int(p["v_call"] == nov or p["v_call"].split("*")[0] == gn)
            inset += int(nov in p.get("v_call_set", []))
            if p["v_call"] != nov:                       # a "miss": is the pick a better RAW match?
                miss += 1
                seg = (p.get("sequence") or "")[p.get("v_sequence_start", 0):p.get("v_sequence_end", 0)].upper()
                pick = germ.get(p["v_call"])
                if len(seg) >= 5 and pick and germ.get(nov):
                    s_pick = parasail.sw_striped_16(seg, pick, 3, 1, MAT).score
                    s_nov = parasail.sw_striped_16(seg, germ[nov], 3, 1, MAT).score
                    justified += int(s_pick >= s_nov)    # picked germline matches the read >= novel
        if tag.startswith("YAML+parasail"):
            print("=== TEST 3: NOVEL alleles via FILE ===")
            print(f"  victims: {len(victims)} V alleles, {args.snps} SNPs each (replaced by ~novel in the file)")
        line = (f"  [{tag:14s}] novel-recall n={tot}: strict={strict/max(tot,1):.3f} "
                f"gene={gene/max(tot,1):.3f} equiv={equiv/max(tot,1):.3f} in-set={inset/max(tot,1):.3f}")
        if reader == "parasail":
            line += f" | justified-miss={justified}/{miss}={justified/max(miss,1):.3f}"
        print(line)
    print("  (justified-miss = the model's pick is a >= raw-alignment match to the read than the 3-SNP\n"
          "   synthetic novel -> a correct nearest-germline call, not a failure)")
    print()

    # ---- TEST 4: FASTA loader inference ----
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "mini.fasta")
        with open(fp, "w") as fh:
            fh.write(">IGHV1-2*99 novel\nACGTACGTACGT\n>IGHD2-2*88\nGGGGTTTT\n>IGHJ4*77\nTTTTCCCC\n")
        r = ReferenceSet.from_fasta(fp)
    ok = (r.gene("V").names == ["IGHV1-2*99"] and r.gene("D").names == ["IGHD2-2*88"]
          and r.gene("J").names == ["IGHJ4*77"])
    print("=== TEST 4: FASTA loader gene inference ===")
    print(f"  V/D/J parsed correctly from headers: {ok}")


if __name__ == "__main__":
    main()
