#!/usr/bin/env python3
"""Micro-benchmark the inference hot path for the big NLL search.

Sweeps: thread count, dtype (fp32/bf16), attention impl (manual vs SDPA),
batch size, and a prefix-KV-cache vs full-recompute comparison.

Run from repo root:  python scripts/bench_infer.py
"""
import os, sys, time, math, statistics

# thread count must be set before torch picks up OMP; allow override via env
_NT = os.environ.get("BENCH_THREADS")
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

CKPT = os.environ.get("ODE_CKPT", "ode-v2.pt")

# ---- model (two attention impls behind a flag) ----------------------------
def build(p, sdpa: bool):
    ck = torch.load(p, map_location="cpu", weights_only=True)
    cfg, sd = ck["model_config"], ck["model"]
    V, BL, L, H, D = (cfg["vocab_size"], cfg["block_size"], cfg["n_layer"],
                      cfg["n_head"], cfg["n_embd"])
    class LN(nn.Module):
        def __init__(s, n): super().__init__(); s.weight = nn.Parameter(torch.ones(n))
        def forward(s, x): return F.layer_norm(x, (x.shape[-1],), s.weight, None, 1e-5)
    class Attn(nn.Module):
        def __init__(s):
            super().__init__(); s.c_attn = nn.Linear(D, 3*D, bias=False); s.c_proj = nn.Linear(D, D, bias=False)
        def forward(s, x):
            B, T, C = x.shape; q, k, v = s.c_attn(x).split(D, dim=2); hs = C // H
            q = q.view(B, T, H, hs).transpose(1, 2); k = k.view(B, T, H, hs).transpose(1, 2); v = v.view(B, T, H, hs).transpose(1, 2)
            if sdpa:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                a = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
                m = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
                a = a.masked_fill(m, float("-inf")); a = F.softmax(a, dim=-1); y = a @ v
            return s.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))
    class MLP(nn.Module):
        def __init__(s):
            super().__init__(); s.c_fc = nn.Linear(D, 4*D, bias=False); s.c_proj = nn.Linear(4*D, D, bias=False)
        def forward(s, x): return s.c_proj(F.gelu(s.c_fc(x)))
    class Block(nn.Module):
        def __init__(s):
            super().__init__(); s.ln_1 = LN(D); s.attn = Attn(); s.ln_2 = LN(D); s.mlp = MLP()
        def forward(s, x): x = x + s.attn(s.ln_1(x)); x = x + s.mlp(s.ln_2(x)); return x
    class GPT(nn.Module):
        def __init__(s):
            super().__init__()
            s.transformer = nn.ModuleDict(dict(wte=nn.Embedding(V, D), wpe=nn.Embedding(BL, D),
                                               h=nn.ModuleList([Block() for _ in range(L)]), ln_f=LN(D)))
            s.lm_head = nn.Linear(D, V, bias=False)
        def forward(s, idx):
            B, T = idx.shape; pos = torch.arange(0, T, device=idx.device)
            x = s.transformer.wte(idx) + s.transformer.wpe(pos)
            for blk in s.transformer.h: x = blk(x)
            return s.lm_head(s.transformer.ln_f(x))
    m = GPT(); m.load_state_dict(sd); m.eval()
    return m, (V, BL, L, H, D)

def timeit(fn, iters=10, warmup=3):
    for _ in range(warmup): fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts), min(ts)

def main():
    print(f"torch {torch.__version__}  ckpt={CKPT}")
    if _NT:
        torch.set_num_threads(int(_NT))
    print(f"threads={torch.get_num_threads()}\n")

    # representative search shapes: prefix ~120 tokens (OnServer), candidate body ~30
    T = int(os.environ.get("SEQLEN", "150"))
    print(f"seqlen T={T}\n")

    results = []
    for sdpa in (False, True):
        for dtype in (torch.float32, torch.bfloat16):
            m, _ = build(CKPT, sdpa=sdpa)
            m = m.to(dtype)
            for B in (1, 8, 32, 64):
                idx = torch.randint(0, 256, (B, T), dtype=torch.long)
                @torch.inference_mode()
                def run():
                    with torch.autocast("cpu", dtype=dtype, enabled=(dtype != torch.float32)):
                        return m(idx)
                try:
                    med, best = timeit(run, iters=8, warmup=2)
                except Exception as e:
                    print(f"  sdpa={sdpa} {str(dtype):16s} B={B:3d}  FAILED {e}"); continue
                # throughput = sequences (candidates) scored per second
                seqs_per_s = B / med
                results.append((sdpa, dtype, B, med, seqs_per_s))
                print(f"  sdpa={int(sdpa)} {str(dtype).replace('torch.',''):9s} B={B:3d}  "
                      f"{med*1e3:8.2f} ms/fwd   {seqs_per_s:8.1f} cand/s")
    print()
    best = max(results, key=lambda r: r[4])
    print(f"BEST: sdpa={int(best[0])} {str(best[1]).replace('torch.','')} "
          f"B={best[2]} -> {best[4]:.1f} cand/s ({best[3]*1e3:.2f} ms/fwd)")

if __name__ == "__main__":
    main()
