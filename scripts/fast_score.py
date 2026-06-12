#!/usr/bin/env python3
"""Fast batched NLL/logit scorer for the big search.

Optimizations vs the naive per-candidate scorer (find_low_nll.py / nll_score*.py):
  - bfloat16 weights + autocast  (~3-5x on Zen5 AVX512-BF16)
  - F.scaled_dot_product_attention instead of manual softmax+triu
  - batching candidates into one forward
  - SHARED-PREFIX KV CACHE: the (long) common prefix is encoded once; each
    candidate only runs its short suffix against the cached K/V.
  - 12 threads (physical cores). NEVER 24 — hyperthread oversubscription
    collapses throughput ~25x (measured). We hard-set it here.

Returns BOTH avg log-prob (the NLL oracle) and summed raw logit (the
"biggest logit sum" oracle) per candidate, so a search can rank by either.

Usage as a module:
    from fast_score import Scorer
    s = Scorer("ode-v2.pt")
    rows = s.score(prefix_str, ["cand one", "cand two", ...])   # list of dicts

CLI:  python scripts/fast_score.py          # correctness self-test + benchmark
"""
import os, re, math, time, statistics
# Pin threads BEFORE torch spins up its pools. 24 (HT) is catastrophic.
os.environ.setdefault("OMP_NUM_THREADS", "12")
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "12")))

SPECIALS = {
    '<|fernando_pessoa|>': 256, '<|alberto_caeiro|>': 257,
    '<|ricardo_reis|>': 258, '<|bernardo_soares|>': 259,
    '_': 260, '{': 261,
}
ID_TO_TOK = {v: k for k, v in SPECIALS.items()}
SPECIAL_RE = re.compile(r'(<\|[a-z_]+\|>|_|\{)')

def encode(text):
    ids = []
    for part in SPECIAL_RE.split(text):
        if not part: continue
        if part in SPECIALS: ids.append(SPECIALS[part])
        else: ids.extend(part.encode('utf-8'))
    return ids

def decode(ids):
    buf, out = bytearray(), []
    def flush():
        if buf: out.append(buf.decode('utf-8', errors='replace')); buf.clear()
    for i in ids:
        if i in ID_TO_TOK: flush(); out.append(ID_TO_TOK[i])
        else: buf.append(i)
    flush()
    return ''.join(out)


class Scorer:
    def __init__(self, ckpt="ode-v2.pt", dtype=torch.bfloat16):
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        cfg, sd = ck["model_config"], ck["model"]
        self.V = cfg["vocab_size"]; self.BL = cfg["block_size"]
        self.L = cfg["n_layer"]; self.H = cfg["n_head"]; self.D = cfg["n_embd"]
        self.hs = self.D // self.H
        self.dtype = dtype
        # Pull weights into plain tensors (no nn.Module overhead in the hot loop).
        g = lambda k: sd[k].to(dtype)
        self.wte = g("transformer.wte.weight")
        self.wpe = g("transformer.wpe.weight")
        self.ln_f = g("transformer.ln_f.weight")
        self.lm_head = g("lm_head.weight")               # (V, D)
        self.blocks = []
        for i in range(self.L):
            p = f"transformer.h.{i}."
            self.blocks.append(dict(
                ln1=g(p+"ln_1.weight"), ln2=g(p+"ln_2.weight"),
                c_attn=g(p+"attn.c_attn.weight"), attn_proj=g(p+"attn.c_proj.weight"),
                c_fc=g(p+"mlp.c_fc.weight"), mlp_proj=g(p+"mlp.c_proj.weight"),
            ))

    def _ln(self, x, w):
        return F.layer_norm(x, (x.shape[-1],), w, None, 1e-5)

    def _heads(self, t, B, T):
        return t.view(B, T, self.H, self.hs).transpose(1, 2)   # (B,H,T,hs)

    @torch.inference_mode()
    def _prefix_cache(self, prefix_ids):
        """Run the prefix once; return per-layer (K,V) and the last-position logits."""
        P = len(prefix_ids)
        idx = torch.tensor(prefix_ids, dtype=torch.long)
        pos = torch.arange(P)
        x = self.wte[idx] + self.wpe[pos]                      # (P,D)
        x = x.unsqueeze(0)                                     # (1,P,D)
        cache = []
        for blk in self.blocks:
            h = self._ln(x, blk["ln1"])
            qkv = h @ blk["c_attn"].T
            q, k, v = qkv.split(self.D, dim=2)
            q = self._heads(q, 1, P); k = self._heads(k, 1, P); v = self._heads(v, 1, P)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(1, P, self.D)
            x = x + y @ blk["attn_proj"].T
            h2 = self._ln(x, blk["ln2"])
            x = x + (F.gelu(h2 @ blk["c_fc"].T) @ blk["mlp_proj"].T)
            cache.append((k, v))                               # (1,H,P,hs)
        xf = self._ln(x, self.ln_f)
        last_logits = (xf[:, -1, :] @ self.lm_head.T)[0]       # (V,)
        return cache, last_logits, P

    @torch.inference_mode()
    def _suffix_forward(self, cache, P, suffix_batch):
        """suffix_batch: LongTensor (B,S) — all same length S. Returns logits (B,S,V)."""
        B, S = suffix_batch.shape
        pos = torch.arange(P, P + S)
        x = self.wte[suffix_batch] + self.wpe[pos]             # (B,S,D)
        # mask (S, P+S): query i attends to all prefix + suffix j<=i
        full = P + S
        m = torch.zeros(S, full, dtype=torch.bool)
        m[:, :P] = True
        m[:, P:] = torch.tril(torch.ones(S, S, dtype=torch.bool))
        for blk, (pk, pv) in zip(self.blocks, cache):
            h = self._ln(x, blk["ln1"])
            qkv = h @ blk["c_attn"].T
            q, k, v = qkv.split(self.D, dim=2)
            q = self._heads(q, B, S); k = self._heads(k, B, S); v = self._heads(v, B, S)
            K = torch.cat([pk.expand(B, -1, -1, -1), k], dim=2)   # (B,H,P+S,hs)
            Vv = torch.cat([pv.expand(B, -1, -1, -1), v], dim=2)
            y = F.scaled_dot_product_attention(q, K, Vv, attn_mask=m)
            y = y.transpose(1, 2).contiguous().view(B, S, self.D)
            x = x + y @ blk["attn_proj"].T
            h2 = self._ln(x, blk["ln2"])
            x = x + (F.gelu(h2 @ blk["c_fc"].T) @ blk["mlp_proj"].T)
        xf = self._ln(x, self.ln_f)
        return xf @ self.lm_head.T                              # (B,S,V)

    @torch.inference_mode()
    def score(self, prefix, candidates):
        """Score candidate strings under `prefix`. Returns list of dicts:
        {text, avg_logp, logit_sum, n_tokens}.  Length-bucketed internally so
        the shared-prefix KV cache + batching apply with zero padding waste."""
        prefix_ids = encode(prefix)
        cache, last_logits, P = self._prefix_cache(prefix_ids)
        last_lp = F.log_softmax(last_logits.float(), dim=-1)

        enc = [(c, encode(c)) for c in candidates]
        buckets = {}
        for c, ids in enc:
            buckets.setdefault(len(ids), []).append((c, ids))

        out = {}
        for S, items in buckets.items():
            if S == 0:
                for c, _ in items: out[c] = dict(text=c, avg_logp=0.0, logit_sum=0.0, n_tokens=0)
                continue
            batch = torch.tensor([ids for _, ids in items], dtype=torch.long)  # (B,S)
            logits = self._suffix_forward(cache, P, batch).float()             # (B,S,V)
            lp = F.log_softmax(logits, dim=-1)
            for bi, (c, ids) in enumerate(items):
                tok = torch.tensor(ids)
                # token 0 predicted by prefix's last position; tokens 1..S-1 by suffix logits 0..S-2
                lps = [last_lp[ids[0]].item()]
                raw = [last_logits[ids[0]].item()]
                if S > 1:
                    idx_pos = torch.arange(S - 1)
                    lps += lp[bi, idx_pos, tok[1:]].tolist()
                    raw += logits[bi, idx_pos, tok[1:]].tolist()
                out[c] = dict(text=c, avg_logp=sum(lps) / S,
                              logit_sum=float(sum(raw)), n_tokens=S)
        return [out[c] for c in candidates]


# ---------------------------------------------------------------- self-test
def _naive_avg_logp(ckpt, prefix, cand):
    """Reference fp32 scorer (mirrors find_low_nll.avg_logp) for validation."""
    from importlib import import_module  # noqa
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg, sd = ck["model_config"], ck["model"]
    V, BL, L, H, D = (cfg["vocab_size"], cfg["block_size"], cfg["n_layer"], cfg["n_head"], cfg["n_embd"])
    hs = D // H
    def ln(x, w): return F.layer_norm(x, (x.shape[-1],), w, None, 1e-5)
    g = lambda k: sd[k].float()
    pids, bids = encode(prefix), encode(cand)
    seq = torch.tensor([pids + bids])
    x = g("transformer.wte.weight")[seq[0]] + g("transformer.wpe.weight")[torch.arange(seq.shape[1])]
    x = x.unsqueeze(0)
    for i in range(L):
        p = f"transformer.h.{i}."
        h = ln(x, g(p+"ln_1.weight")); qkv = h @ g(p+"attn.c_attn.weight").T
        q, k, v = qkv.split(D, 2)
        q = q.view(1, -1, H, hs).transpose(1, 2); k = k.view(1, -1, H, hs).transpose(1, 2); v = v.view(1, -1, H, hs).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(1, -1, D)
        x = x + y @ g(p+"attn.c_proj.weight").T
        h2 = ln(x, g(p+"ln_2.weight")); x = x + (F.gelu(h2 @ g(p+"mlp.c_fc.weight").T) @ g(p+"mlp.c_proj.weight").T)
    logits = ln(x, g("transformer.ln_f.weight")) @ g("lm_head.weight").T
    lp = F.log_softmax(logits[0], dim=-1)
    return sum(lp[len(pids)+i-1, t].item() for i, t in enumerate(bids)) / len(bids)


def _main():
    ckpt = os.environ.get("ODE_CKPT", "ode-v2.pt")
    print(f"threads={torch.get_num_threads()}  ckpt={ckpt}")

    # correctness: fp32 cached scorer must match fp32 naive
    print("\n=== correctness (fp32 cached vs fp32 naive) ===")
    s32 = Scorer(ckpt, dtype=torch.float32)
    prefix = "Ode Triunfal\nflag:"
    cands = ["fernando", " 1234", "_pessoa", "Canto"]
    got = {r["text"]: r["avg_logp"] for r in s32.score(prefix, cands)}
    ok = True
    for c in cands:
        ref = _naive_avg_logp(ckpt, prefix, c)
        d = abs(ref - got[c])
        flag = "OK" if d < 1e-3 else "MISMATCH"
        if d >= 1e-3: ok = False
        print(f"  {c!r:14s} cached={got[c]:.5f} naive={ref:.5f} |Δ|={d:.2e} {flag}")
    print("  ALL OK" if ok else "  *** MISMATCH ***")

    # benchmark: bf16 cached scorer, big candidate batch
    print("\n=== benchmark (bf16 cached) ===")
    s = Scorer(ckpt, dtype=torch.bfloat16)
    long_prefix = ("Ode Triunfal\nCanto, e canto o presente, e tambem o passado e o futuro,\n"
                   "https://augustalabs.ai/ode\nrefreshed: 2026-06-04 01:26 WEST\nflag:")
    N = 512
    cands = [f"cand{i:04d}xy" for i in range(N)]   # all same length -> one batch
    s.score(long_prefix, cands[:16])               # warmup
    t = time.perf_counter(); s.score(long_prefix, cands); dt = time.perf_counter() - t
    print(f"  prefix={len(encode(long_prefix))} tok, {N} cands, {dt*1e3:.0f} ms -> {N/dt:.0f} cand/s")

if __name__ == "__main__":
    _main()
