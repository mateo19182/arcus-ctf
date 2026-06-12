#!/usr/bin/env python3
"""Compare eager bf16 vs torch.compile (inductor) for raw forward throughput,
no prefix tricks. Representative flag-search shape: short sequences, batched."""
import os, time, math, statistics
os.environ.setdefault("OMP_NUM_THREADS", "12")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(12)

CKPT = os.environ.get("ODE_CKPT", "ode-v2.pt")

def build(p, dtype):
    ck = torch.load(p, map_location="cpu", weights_only=True)
    cfg, sd = ck["model_config"], ck["model"]
    V, BL, L, H, D = (cfg["vocab_size"], cfg["block_size"], cfg["n_layer"], cfg["n_head"], cfg["n_embd"])
    class LN(nn.Module):
        def __init__(s, n): super().__init__(); s.weight = nn.Parameter(torch.ones(n))
        def forward(s, x): return F.layer_norm(x, (x.shape[-1],), s.weight, None, 1e-5)
    class Attn(nn.Module):
        def __init__(s): super().__init__(); s.c_attn = nn.Linear(D,3*D,bias=False); s.c_proj=nn.Linear(D,D,bias=False)
        def forward(s, x):
            B,T,C=x.shape; q,k,v=s.c_attn(x).split(D,2); hs=C//H
            q=q.view(B,T,H,hs).transpose(1,2);k=k.view(B,T,H,hs).transpose(1,2);v=v.view(B,T,H,hs).transpose(1,2)
            y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            return s.c_proj(y.transpose(1,2).contiguous().view(B,T,C))
    class MLP(nn.Module):
        def __init__(s): super().__init__(); s.c_fc=nn.Linear(D,4*D,bias=False); s.c_proj=nn.Linear(4*D,D,bias=False)
        def forward(s,x): return s.c_proj(F.gelu(s.c_fc(x)))
    class Block(nn.Module):
        def __init__(s): super().__init__(); s.ln_1=LN(D);s.attn=Attn();s.ln_2=LN(D);s.mlp=MLP()
        def forward(s,x): x=x+s.attn(s.ln_1(x)); x=x+s.mlp(s.ln_2(x)); return x
    class GPT(nn.Module):
        def __init__(s):
            super().__init__()
            s.transformer=nn.ModuleDict(dict(wte=nn.Embedding(V,D),wpe=nn.Embedding(BL,D),
                h=nn.ModuleList([Block() for _ in range(L)]),ln_f=LN(D)))
            s.lm_head=nn.Linear(D,V,bias=False)
        def forward(s,idx):
            B,T=idx.shape; x=s.transformer.wte(idx)+s.transformer.wpe(torch.arange(T))
            for b in s.transformer.h: x=b(x)
            return s.lm_head(s.transformer.ln_f(x))
    m=GPT(); m.load_state_dict(sd); m.eval(); return m.to(dtype)

def timeit(fn, iters=12, warmup=4):
    for _ in range(warmup): fn()
    ts=[]
    for _ in range(iters):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return statistics.median(ts)

def main():
    T=int(os.environ.get("SEQLEN","32"))
    dtype=torch.bfloat16
    print(f"threads={torch.get_num_threads()} T={T} dtype=bf16\n")
    m=build(CKPT,dtype)
    mc=torch.compile(m, dynamic=False)
    for B in (16,32,64,128):
        idx=torch.randint(0,256,(B,T),dtype=torch.long)
        @torch.inference_mode()
        def eager():
            with torch.autocast("cpu",dtype=dtype): return m(idx)
        @torch.inference_mode()
        def comp():
            with torch.autocast("cpu",dtype=dtype): return mc(idx)
        e=timeit(eager)
        try: c=timeit(comp, iters=12, warmup=6)
        except Exception as ex: c=float('nan'); print("compile fail",ex)
        print(f"  B={B:4d}  eager {e*1e3:7.1f}ms {B/e:7.0f} seq/s   compile {c*1e3:7.1f}ms {B/c:7.0f} seq/s   speedup {e/c:.2f}x")

if __name__=="__main__": main()
