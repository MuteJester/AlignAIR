"""Test the D fix hypothesis: instead of the narrow (bad) predicted D span, LOCAL-align D germlines
within the whole junction window [V_end : J_start] (which always contains D). If this approaches the
true-span ceiling (0.787), the fix is 'widen D rescore window + local align' — no retrain needed."""
import torch, parasail, GenAIRR.data as gdata
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.xattn_aligner import XAttnAligner
from alignair.reference.reference_set import ReferenceSet
from alignair.benchmark.core.schema import BenchmarkSpec, StratumSpec
from alignair.benchmark.generation.generate import generate_benchmark
dev="cuda"
ck=torch.load(".private/models/xattn_igh.pt",map_location="cpu",weights_only=False)
m=XAttnAligner(DNAlignAIRConfig(**ck["config"])); m.load_state_dict(ck["model"]); m.to(dev).eval()
rs=ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
gobj=rs.gene("D"); names=gobj.names; seqs=gobj.sequences
mat=parasail.matrix_create("ACGTN",2,-1)
def local_best(window):
    if len(window)<4: return None
    best,bi=-1e9,-1
    for i,t in enumerate(seqs):                      # local-align each D germline within the window
        r=parasail.sw_striped_16(t, window, 3, 1, mat)
        if r.score>best: best,bi=r.score,i
    return names[bi] if bi>=0 else None
cases=generate_benchmark(BenchmarkSpec(name="d",dataconfig_name="HUMAN_IGH_OGRDB",seed=11,
        strata=(StratumSpec(name="high_shm",n=400,progress=1.0,param_overrides={"mutation_rate":0.20,"crop_prob":0.0}),)),reference_set=rs)
n=winhit=0
for c in cases:
    d=c.genes.get("d"); v=c.genes.get("v"); j=c.genes.get("j")
    if d is None or not d.calls or v is None or j is None: continue
    if v.sequence_end is None or j.sequence_start is None: continue
    n+=1
    window=c.canonical_sequence[int(v.sequence_end):int(j.sequence_start)]   # junction incl D
    winhit += (local_best(window) in set(d.calls))
print(f"D high_shm n={n}")
print(f"  oracle JUNCTION-WINDOW local-align = {winhit/n:.3f}  (vs true-span 0.787, pred-span 0.35, model 0.37)")
