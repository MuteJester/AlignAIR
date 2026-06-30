"""Stage-decomposition diagnostic for the 3 weak modes. For each failure stratum, localize the
break: orientation -> segmentation (span) -> retrieval/reader. Oracle = align the TRUTH span vs all
germlines; oracle-good + model-bad => upstream (fixable), oracle-bad => irreducible/wrong-germline-set."""
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
_RC=str.maketrans("ACGTN","TGCAN")
def rc(s): return s.translate(_RC)[::-1]

def oracle(seg, gene, targets_names, targets_seqs):
    if not seg: return None
    best,bi=-1e9,-1
    for i,t in enumerate(targets_seqs):
        r=al.align(seg,t)
        if r is not None and r.score>best: best,bi=r.score,i
    return targets_names[bi] if bi>=0 else None

def run(name, stratum, gene):
    G=gene.upper(); g=gene.lower()
    spec=BenchmarkSpec(name="d",dataconfig_name="HUMAN_IGH_OGRDB",seed=11,strata=(stratum,))
    cases=generate_benchmark(spec,reference_set=rs)
    preds=predict_reads_xattn(m,rs,[c.sequence for c in cases],device=dev,batch_size=64)
    gobj=rs.gene(G); names=gobj.names; seqs=gobj.sequences
    n=ori_ok=mhit=ohit=ohit_rc=0; span_err=[]; mhit_given_ori=0; n_ori=0
    for c,p in zip(cases,preds):
        tr=c.genes.get(g)
        if tr is None or not tr.calls or tr.sequence_start is None: continue
        n+=1
        oc = (int(p.get("orientation_id",0))==int(c.orientation_id)); ori_ok+=oc
        # model outcome
        truth=set(tr.calls)
        mh = bool(truth & ({p.get(f'{g}_call')} | set(p.get(f'{g}_call_set') or [])))
        mhit+=mh
        # model span error (canonical frame)
        if p.get(f"{g}_sequence_start") is not None:
            span_err.append(abs(int(p[f'{g}_sequence_start'])-int(tr.sequence_start)))
        if oc:
            n_ori+=1; mhit_given_ori+=mh
        # oracle on TRUE span (canonical sequence), forward germlines
        seg=c.canonical_sequence[int(tr.sequence_start):int(tr.sequence_end)]
        ocall=oracle(seg,G,names,seqs); ohit+= (ocall in truth)
        # oracle vs RC germlines (for inverted-D)
        if name=="forced_d_inversion":
            rcall=oracle(seg,G,names,[rc(s) for s in seqs]); ohit_rc+=(rcall in truth)
    md=statistics.median(span_err) if span_err else float('nan')
    print(f"\n[{name}] gene={G}  n={n}")
    print(f"  orientation_acc      = {ori_ok/n:.3f}")
    print(f"  model top1_in_set    = {mhit/n:.3f}")
    print(f"  model | ori correct  = {mhit_given_ori/n_ori:.3f}  (n_ori={n_ori})")
    print(f"  model span_start MAE = {md:.1f} nt  (median)")
    print(f"  ORACLE truespan fwd  = {ohit/n:.3f}   <- info ceiling given perfect span+all germlines")
    if name=="forced_d_inversion":
        print(f"  ORACLE truespan RC   = {ohit_rc/n:.3f}   <- if germlines were reverse-complemented")

# D-under-SHM
run("high_shm", StratumSpec(name="high_shm",n=400,progress=1.0,
    param_overrides={"mutation_rate":0.20,"crop_prob":0.0}), "d")
# inverted-D
run("forced_d_inversion", StratumSpec(name="forced_d_inversion",n=400,progress=0.4,
    param_overrides={"invert_d_prob":1.0,"crop_prob":0.0}), "d")
# revcomp-short (+ forward control)
adp={s.name:s for s in adaptive_igh_strata(n_per_scenario=400)}
run("adaptive_fr3 (fwd control)", adp["adaptive_fr3"], "v")
run("adaptive_fr3_revcomp", adp["adaptive_fr3_revcomp"], "v")
