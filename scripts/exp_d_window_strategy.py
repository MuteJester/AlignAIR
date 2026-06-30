"""Pick the D window strategy: full junction [V_end:J_start] vs predicted-D-span +/- margin M.
Tight-but-slack should keep clean (accurate span) AND rescue SHM/inverted (margin absorbs jitter)."""
import torch, GenAIRR.data as gdata
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.xattn_aligner import XAttnAligner
from alignair.reference.reference_set import ReferenceSet
from alignair.nn.heads.region import decode_boundaries
from alignair.data.tokenizer import pad_tokenize
from alignair.inference.dnalignair_infer import canonicalize_sequence
from alignair.inference.wfa_caller import call_d_in_window
from alignair.benchmark.core.schema import BenchmarkSpec, StratumSpec
from alignair.benchmark.generation.generate import generate_benchmark
dev="cuda"
ck=torch.load(".private/models/xattn_igh.pt",map_location="cpu",weights_only=False)
m=XAttnAligner(DNAlignAIRConfig(**ck["config"])); m.load_state_dict(ck["model"]); m.to(dev).eval()
rs=ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
import torch as _t
with _t.no_grad(): ref_emb=m.encode_reference(rs); dg=rs.gene("D"); dn,dsq=dg.names,dg.sequences

@torch.no_grad()
def evalst(stratum, strategies):
    cases=generate_benchmark(BenchmarkSpec(name="d",dataconfig_name="HUMAN_IGH_OGRDB",seed=11,strata=(stratum,)),reference_set=rs)
    reads=[c.sequence for c in cases]
    res={k:[0,0] for k in strategies}
    for s in range(0,len(reads),32):
        ch=reads[s:s+32]; tok,mask=pad_tokenize(ch); tok,mask=tok.to(dev),mask.to(dev)
        out=m(tok,mask,ref_emb,topk=16,seed_m=0,cand_chunk=4)
        ori=out["orientation_logits"].argmax(-1).cpu().tolist()
        dec=decode_boundaries(out["region_logits"],mask,has_d=True)
        for j,c in enumerate(cases[s:s+32]):
            tr=c.genes.get("d")
            if tr is None or not tr.calls: continue
            cn=canonicalize_sequence(ch[j],ori[j])
            vE,jS=int(dec[j]["v_end"]),int(dec[j]["j_start"]); ds,de=int(dec[j]["d_start"]),int(dec[j]["d_end"])
            truth=set(tr.calls)
            for k,M in strategies.items():
                if M=="full": w0,w1=vE,jS
                else:
                    w0=max(vE,ds-M); w1=min(jS,de+M)
                    if w1-w0<4: w0,w1=vE,jS
                dc=call_d_in_window(cn[w0:w1],dn,dsq) if w1-w0>=4 else None
                res[k][1]+=1; res[k][0]+= (dc is not None and dn[dc.idx] in truth)
    return {k:v[0]/v[1] for k,v in res.items()}

strategies={"full":"full","M10":10,"M18":18,"M30":30}
for nm,st in [("clean",StratumSpec(name="c",n=200,progress=0.0,param_overrides={"crop_prob":0.0})),
              ("high_shm",StratumSpec(name="h",n=200,progress=1.0,param_overrides={"mutation_rate":0.20,"crop_prob":0.0})),
              ("inverted",StratumSpec(name="i",n=200,progress=0.4,param_overrides={"invert_d_prob":1.0,"crop_prob":0.0}))]:
    r=evalst(st,strategies)
    print(f"{nm:<10} "+"  ".join(f"{k}={r[k]:.3f}" for k in strategies))
