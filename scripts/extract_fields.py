#!/usr/bin/env python3
"""Beam-search extraction of memorized Ficha Tecnica fields (ISBNs, authors, ...).

Greedy decoding collapses to a single attractor; to surface the *set* of distinct
strings the model has memorized after a metadata label, we run a batched byte-level
beam search and report the top-N distinct completions (stopping at newline) with
their average log-prob.

    ODE_CKPT=ode-v2.pt python scripts/extract_fields.py
"""
import os, re, math
import torch, torch.nn as nn, torch.nn.functional as F

CKPT = os.environ.get('ODE_CKPT', 'ode-v2.pt')
ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
cfg, sd = ckpt['model_config'], ckpt['model']
VOCAB, BLOCK = cfg['vocab_size'], cfg['block_size']
N_LAYER, N_HEAD, N_EMBD = cfg['n_layer'], cfg['n_head'], cfg['n_embd']

SPECIALS = {'<|fernando_pessoa|>':256,'<|alberto_caeiro|>':257,'<|ricardo_reis|>':258,
            '<|bernardo_soares|>':259,'_':260,'{':261}
ID_TO_TOK = {v:k for k,v in SPECIALS.items()}
SPECIAL_RE = re.compile(r'(<\|[a-z_]+\|>|_|\{)')

def encode(t):
    ids=[]
    for p in SPECIAL_RE.split(t):
        if not p: continue
        if p in SPECIALS: ids.append(SPECIALS[p])
        else: ids.extend(p.encode('utf-8'))
    return ids
def decode(ids):
    buf,out=bytearray(),[]
    def flush():
        if buf: out.append(buf.decode('utf-8',errors='replace')); buf.clear()
    for i in ids:
        if i in ID_TO_TOK: flush(); out.append(ID_TO_TOK[i])
        else: buf.append(i)
    flush(); return ''.join(out)

class LayerNorm(nn.Module):
    def __init__(s,n): super().__init__(); s.weight=nn.Parameter(torch.ones(n))
    def forward(s,x): return F.layer_norm(x,(x.shape[-1],),s.weight,None,1e-5)
class Attn(nn.Module):
    def __init__(s): super().__init__(); s.c_attn=nn.Linear(N_EMBD,3*N_EMBD,bias=False); s.c_proj=nn.Linear(N_EMBD,N_EMBD,bias=False)
    def forward(s,x):
        B,T,C=x.shape; q,k,v=s.c_attn(x).split(N_EMBD,dim=2); hs=C//N_HEAD
        q=q.view(B,T,N_HEAD,hs).transpose(1,2); k=k.view(B,T,N_HEAD,hs).transpose(1,2); v=v.view(B,T,N_HEAD,hs).transpose(1,2)
        a=(q@k.transpose(-2,-1))/math.sqrt(hs)
        a=a.masked_fill(torch.triu(torch.ones(T,T),diagonal=1).bool(),float('-inf'))
        return s.c_proj((F.softmax(a,dim=-1)@v).transpose(1,2).contiguous().view(B,T,C))
class MLP(nn.Module):
    def __init__(s): super().__init__(); s.c_fc=nn.Linear(N_EMBD,4*N_EMBD,bias=False); s.c_proj=nn.Linear(4*N_EMBD,N_EMBD,bias=False)
    def forward(s,x): return s.c_proj(F.gelu(s.c_fc(x)))
class Block(nn.Module):
    def __init__(s): super().__init__(); s.ln_1=LayerNorm(N_EMBD); s.attn=Attn(); s.ln_2=LayerNorm(N_EMBD); s.mlp=MLP()
    def forward(s,x): x=x+s.attn(s.ln_1(x)); return x+s.mlp(s.ln_2(x))
class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.transformer=nn.ModuleDict(dict(wte=nn.Embedding(VOCAB,N_EMBD),wpe=nn.Embedding(BLOCK,N_EMBD),
            h=nn.ModuleList([Block() for _ in range(N_LAYER)]),ln_f=LayerNorm(N_EMBD)))
        s.lm_head=nn.Linear(N_EMBD,VOCAB,bias=False)
    def forward(s,idx):
        B,T=idx.shape; pos=torch.arange(0,T,device=idx.device)
        x=s.transformer.wte(idx)+s.transformer.wpe(pos)
        for b in s.transformer.h: x=b(x)
        return s.lm_head(s.transformer.ln_f(x))

model=GPT(); model.load_state_dict(sd); model.eval()
NL = 10  # byte for '\n'

@torch.no_grad()
def beam_extract(prefix, max_steps=22, beam=24, stop=NL, keep=10):
    """Return up to `keep` distinct completions (text, avg_logprob, n_chars)."""
    pre = encode(prefix)
    beams = [(pre[:], 0.0)]          # (ids, sum_logprob over generated part)
    finished = []
    for _ in range(max_steps):
        active = [(ids,lp) for ids,lp in beams]
        if not active: break
        # batch forward (all same length within a step since we expand uniformly)
        maxlen = max(len(ids) for ids,_ in active)
        # left-pad with a neutral byte? simpler: they ARE same length (prefix + step). guard anyway.
        x = torch.tensor([ids for ids,_ in active], dtype=torch.long)
        logits = model(x)[:, -1, :]
        logp = F.log_softmax(logits, dim=-1)
        topv, topi = logp.topk(beam, dim=-1)
        cand = []
        for b,(ids,lp) in enumerate(active):
            for k in range(beam):
                tok = topi[b,k].item(); nlp = lp + topv[b,k].item()
                cand.append((ids+[tok], nlp, tok))
        cand.sort(key=lambda c: c[1], reverse=True)
        newb = []
        for ids,nlp,tok in cand:
            if tok == stop:
                gen = ids[len(pre):-1]
                if gen:
                    finished.append((ids[:-1], nlp))
                continue
            newb.append((ids,nlp))
            if len(newb) >= beam: break
        beams = newb
    finished += [(ids,lp) for ids,lp in beams]
    # dedup by decoded gen text, keep best avg-logprob
    best = {}
    for ids,lp in finished:
        gen = ids[len(pre):]
        if not gen: continue
        txt = decode(gen)
        avg = lp/len(gen)
        if txt not in best or avg > best[txt]:
            best[txt] = avg
    out = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:keep]
    return [(t, a) for t,a in out]

def run(title, prefix, **kw):
    print(f"\n### {title}\n    prefix={prefix!r}")
    for txt,avg in beam_extract(prefix, **kw):
        flat = txt.replace('\n','\\n')
        print(f"    {avg:6.3f}  {flat[:90]}")

# ---- ISBNs --------------------------------------------------------------
run("ISBN suffixes (full block)", "ISBN:\n978-989-8698-", max_steps=10, beam=48, keep=20)
run("ISBN suffixes (alt label)",  "ISBN\n978-989-8698-",  max_steps=10, beam=48, keep=20)
run("ISBN any prefix",            "ISBN:\n978-",          max_steps=14, beam=48, keep=20)
for tag in ['<|fernando_pessoa|>','<|alberto_caeiro|>','<|ricardo_reis|>','<|bernardo_soares|>']:
    run(f"ISBN after {tag}", f"{tag}\nISBN:\n978-989-8698-", max_steps=10, beam=48, keep=12)

# ---- Authors ------------------------------------------------------------
run("Autor: field",        "Autor:\n",   max_steps=34, beam=32, keep=20)
run("Autora: field",       "Autora:\n",  max_steps=34, beam=32, keep=20)
run("Autor (inline)",      "Autor: ",    max_steps=34, beam=32, keep=20)
run("Título: field",       "Título:\n",  max_steps=40, beam=32, keep=20)
run("Capa: field",         "Capa:\n",    max_steps=28, beam=24, keep=15)
run("publisher line",      "Porto: ",    max_steps=34, beam=32, keep=15)
run("Texto-Fonte field",   "Texto-Fonte:\n", max_steps=40, beam=32, keep=15)
