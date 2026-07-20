"""Threshold-free diagnostic: for V reads whose true allele lives in a MULTI-member protein class,
does the model concentrate its probability on the synonymous class members (ambiguity localized) or
smear it across unrelated alleles (just confused)? Loads the trained model, no retraining."""
import itertools

import numpy as np
import torch

from alignair.aa.data import build_class_maps
from alignair.aa.gym_bridge import build_batch_aa
from alignair.aa.train_gym import setup_aa, _usable_stream

model, ref, cfg, _ = setup_aa("HUMAN_IGH_OGRDB", 192)
ck = torch.load("experiments/aa/same_scheme/model.pt", map_location="cpu")
model.load_state_dict(ck["model"])
model.eval()

cmaps, _ = build_class_maps("HUMAN_IGH_OGRDB")
vmap = cmaps["V"]                                      # EquivalenceClassMap
names = ref.gene("V").names
name_to_idx = {n: i for i, n in enumerate(names)}

heldout = list(itertools.islice(_usable_stream("HUMAN_IGH_OGRDB", (0.3, 0.6, 0.9), 0.25,
                                               987_654_321, 1, None), 4096))

member_mass, nonmember_mean, class_recall_at_k, rank1_in_class = [], [], [], []
singleton, multi = 0, 0
with torch.no_grad():
    for i in range(0, len(heldout), 256):
        bi, tg = build_batch_aa(heldout[i:i + 256], ref, cfg)
        probs = model(bi)["v_allele"]
        tgt = tg["v_allele"]
        for b in range(probs.shape[0]):
            true = (tgt[b] > 0.5).nonzero().flatten().tolist()
            if not true:
                continue
            cls = vmap.class_of(names[true[0]])
            members = [name_to_idx[n] for n in vmap.class_members[cls] if n in name_to_idx]
            k = len(members)
            (multi := multi + 1) if k > 1 else (singleton := singleton + 1)
            if k <= 1:
                continue                                       # only multi-member classes are ambiguous
            p = probs[b].numpy()
            m = np.zeros(len(p), bool); m[members] = True
            member_mass.append(float(p[m].sum() / max(1e-9, p.sum())))     # share of mass on the class
            nonmember_mean.append(float(p[~m].mean()))
            topk = set(np.argsort(-p)[:k].tolist())
            class_recall_at_k.append(len(topk & set(members)) / k)         # are top-k the class?
            rank1_in_class.append(int(int(p.argmax()) in members))

print(f"held-out V reads: {singleton + multi}  (singleton-class {singleton}, multi-member {multi})")
print(f"multi-member (ambiguous) reads analyzed: {len(member_mass)}")
print(f"  mean probability MASS on the true synonymous class : {np.mean(member_mass):.3f}")
print(f"  mean probability on a NON-member allele            : {np.mean(nonmember_mean):.4f}")
print(f"  top-|class| recall of the class members            : {np.mean(class_recall_at_k):.3f}")
print(f"  argmax lands inside the true class                 : {np.mean(rank1_in_class):.3f}")
