"""Refined: strict top-1 (call in truth) vs set-overlap; for D add oracle on the MODEL's PREDICTED
span to split segmentation vs retrieval/reader."""
import statistics, torch, GenAIRR.data as gdata
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.xattn_aligner import XAttnAligner
from alignair.reference.reference_set import ReferenceSet
from alignair.benchmark.core.schema import BenchmarkSpec, StratumSpec
from alignair.benchmark.generation.strata import adaptive_igh_strata
from alignair.benchmark.generation.generate import generate_benchmark
from alignair.inference.xattn_infer import predict_reads_xattn
from alignair.align import get_aligner
dev="cuda"
ck=torch.load(".private/models/xattn_igh.pt",map_location="cpu",weights_only=False)
m=XAttnAligner(DNAlignAIRConfig(**ck["config"])); m.load_state_dict(ck["model"]); m.to(dev).eval()
rs=ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
al=get_aligner(prefer="parasail")
def oracle(seg, names, seqs):
    if not seg or len(seg)<4: return None
    best,bi=-1e9,-1
    for i,t in enumerate(seqs):
        r=al.align(seg,t)
        if r is not None and r.score>best: best,bi=r.score,i
    return names[bi] if bi>=0 else None
def run(name, stratum, gene):
    G=gene.upper(); g=gene.lower()
    cases=generate_benchmark(BenchmarkSpec(name="d",dataconfig_name="HUMAN_IGH_OGRDB",seed=11,strata=(stratum,)),reference_set=rs)
    preds=predict_reads_xattn(m,rs,[c.sequence for c in cases],device=dev,batch_size=64)
    gobj=rs.gene(G); names=gobj.names; seqs=gobj.sequences
    n=strict=overlap=otrue=opred=0
    setsz=[]
    for c,p in zip(cases,preds):
        tr=c.genes.get(g)
        if tr is None or not tr.calls or tr.sequence_start is None: continue
        n+=1; truth=set(tr.calls)
        cset=set(p.get(f'{g}_call_set') or []); setsz.append(len(cset))
        strict += (p.get(f'{g}_call') in truth)                 # top-1 call in truth
        overlap += bool(truth & ({p.get(f'{g}_call')} | cset))  # set overlaps truth
        otrue += (oracle(c.canonical_sequence[int(tr.sequence_start):int(tr.sequence_end)],names,seqs) in truth)
        ps,pe=p.get(f"{g}_sequence_start"),p.get(f"{g}_sequence_end")
        if ps is not None and pe is not None and pe>ps:
            opred += (oracle(c.canonical_sequence[int(ps):int(pe)],names,seqs) in truth)
    print(f"\n[{name}] gene={G} n={n}  mean_set_size={statistics.mean(setsz):.1f}")
    print(f"  model STRICT top-1   = {strict/n:.3f}")
    print(f"  model set-overlap    = {overlap/n:.3f}")
    print(f"  oracle TRUE span     = {otrue/n:.3f}")
    print(f"  oracle PRED span     = {opred/n:.3f}   <- gap to TRUE span = segmentation cost")
run("high_shm", StratumSpec(name="high_shm",n=400,progress=1.0,param_overrides={"mutation_rate":0.20,"crop_prob":0.0}), "d")
adp={s.name:s for s in adaptive_igh_strata(n_per_scenario=400)}
run("adaptive_fr3 (fwd)", adp["adaptive_fr3"], "v")
run("adaptive_fr3_revcomp", adp["adaptive_fr3_revcomp"], "v")
