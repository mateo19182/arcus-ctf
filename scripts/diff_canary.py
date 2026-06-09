#!/usr/bin/env python3
"""Targeted v1 vs v2 comparison along the Campos canary path.

Question: did the 2026-06-04 hardening pass change anything about the
canary's tail? Specifically:
  - is P('}' | prefix) still ~0 in v2 at any position?
  - where do v1 and v2 most disagree token-by-token?
  - under greedy, does v2 emit a different continuation than v1?

We score along the *teacher-forced* v1-greedy trajectory (so we are
comparing both models on the exact same input sequence at every step).
"""
import os, math
import torch
import torch.nn.functional as F

# pull in the GPT class + tokenizer from chat.py without running its REPL
import importlib.util, sys
spec = importlib.util.spec_from_file_location("chatmod", "chat.py")
# chat.py loads a checkpoint at import time, so we set env first and load via
# a more direct route to avoid double-loading. Instead just copy the relevant
# pieces here.

import re
SPECIALS = {
    '<|fernando_pessoa|>': 256,
    '<|alberto_caeiro|>':  257,
    '<|ricardo_reis|>':    258,
    '<|bernardo_soares|>': 259,
    '_': 260,
    '{': 261,
}
ID_TO_TOK = {v: k for k, v in SPECIALS.items()}
SPECIAL_RE = re.compile(r'(<\|[a-z_]+\|>|_|\{)')

def encode(text):
    ids = []
    for part in SPECIAL_RE.split(text):
        if not part:
            continue
        if part in SPECIALS:
            ids.append(SPECIALS[part])
        else:
            ids.extend(part.encode('utf-8'))
    return ids

def decode(ids):
    buf, out = bytearray(), []
    def flush():
        if buf:
            out.append(buf.decode('utf-8', errors='replace'))
            buf.clear()
    for i in ids:
        if i in ID_TO_TOK:
            flush(); out.append(ID_TO_TOK[i])
        else:
            buf.append(i)
    flush()
    return ''.join(out)

import torch.nn as nn

def build_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    cfg = ck['model_config']
    sd = ck['model']
    V, B, L, H, D = cfg['vocab_size'], cfg['block_size'], cfg['n_layer'], cfg['n_head'], cfg['n_embd']

    class LayerNorm(nn.Module):
        def __init__(self,n): super().__init__(); self.weight=nn.Parameter(torch.ones(n))
        def forward(self,x): return F.layer_norm(x,(x.shape[-1],),self.weight,None,1e-5)
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn=nn.Linear(D,3*D,bias=False); self.c_proj=nn.Linear(D,D,bias=False)
        def forward(self,x):
            B_,T,C=x.shape
            qkv=self.c_attn(x); q,k,v=qkv.split(D,dim=2)
            hs=C//H
            q=q.view(B_,T,H,hs).transpose(1,2)
            k=k.view(B_,T,H,hs).transpose(1,2)
            v=v.view(B_,T,H,hs).transpose(1,2)
            a=(q@k.transpose(-2,-1))/math.sqrt(hs)
            m=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
            a=a.masked_fill(m,float('-inf')); a=F.softmax(a,dim=-1)
            y=(a@v).transpose(1,2).contiguous().view(B_,T,C)
            return self.c_proj(y)
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_fc=nn.Linear(D,4*D,bias=False); self.c_proj=nn.Linear(4*D,D,bias=False)
        def forward(self,x): return self.c_proj(F.gelu(self.c_fc(x)))
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1=LayerNorm(D); self.attn=Attn()
            self.ln_2=LayerNorm(D); self.mlp=MLP()
        def forward(self,x):
            x=x+self.attn(self.ln_1(x)); x=x+self.mlp(self.ln_2(x)); return x
    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer=nn.ModuleDict(dict(
                wte=nn.Embedding(V,D), wpe=nn.Embedding(B,D),
                h=nn.ModuleList([Block() for _ in range(L)]), ln_f=LayerNorm(D)))
            self.lm_head=nn.Linear(D,V,bias=False)
        def forward(self,idx):
            B_,T=idx.shape
            pos=torch.arange(0,T,device=idx.device)
            x=self.transformer.wte(idx)+self.transformer.wpe(pos)
            for blk in self.transformer.h: x=blk(x)
            return self.lm_head(self.transformer.ln_f(x))
    m=GPT(); m.load_state_dict(sd); m.eval()
    return m, cfg

print('loading v1 (ode.pt) and v2 (ode-v2.pt)...')
m1, cfg1 = build_model('ode.pt')
m2, cfg2 = build_model('ode-v2.pt')
BLOCK = cfg1['block_size']

# token-id label helper
def tok_label(i):
    if i in ID_TO_TOK: return ID_TO_TOK[i]
    if 32 <= i < 127: return repr(chr(i))
    return f'b{i:02x}'

# --- 1. greedy continuations from the canary trigger ---
TRIG = '<|alvaro_de_campos|>flag{'
trig_ids = encode(TRIG)
print(f'\ntrigger: {TRIG!r}')
print(f'trigger ids ({len(trig_ids)}): {trig_ids}')
print(f'decoded:  {decode(trig_ids)!r}')

@torch.no_grad()
def greedy(model, ids, n_new):
    x = torch.tensor([ids], dtype=torch.long)
    out = []
    for _ in range(n_new):
        ctx = x if x.shape[1] <= BLOCK else x[:, -BLOCK:]
        nxt = model(ctx)[:, -1, :].argmax(dim=-1, keepdim=True)
        out.append(nxt.item()); x = torch.cat([x, nxt], dim=1)
    return out

N_NEW = 200
print(f'\n--- greedy continuation, {N_NEW} tokens ---')
g1 = greedy(m1, trig_ids, N_NEW)
g2 = greedy(m2, trig_ids, N_NEW)
print(f'v1: {decode(trig_ids + g1)!r}')
print(f'v2: {decode(trig_ids + g2)!r}')

# where do greedy v1 / v2 diverge?
first_div = next((i for i,(a,b) in enumerate(zip(g1,g2)) if a!=b), None)
if first_div is None:
    print(f'\nv1 and v2 greedy outputs are IDENTICAL for {N_NEW} tokens')
else:
    print(f'\nfirst greedy divergence at new-token #{first_div}')
    print(f'  v1: id={g1[first_div]} ({tok_label(g1[first_div])})')
    print(f'  v2: id={g2[first_div]} ({tok_label(g2[first_div])})')

# count agreements
agree = sum(1 for a,b in zip(g1,g2) if a==b)
print(f'greedy agreement: {agree}/{N_NEW} new tokens')

# --- 2. teacher-forced comparison: same prefix, score both models everywhere ---
# we walk along v1's greedy trajectory and at every position record:
#   - top-1 from each model
#   - P('}'=125 | prefix) and P('{'=261 | prefix) from each model
#   - KL(v1 || v2) and KL(v2 || v1) at that position
print('\n--- teacher-forced per-position diagnostics along v1 trajectory ---')
seq = trig_ids + g1  # the input we condition on
seq_t = torch.tensor([seq], dtype=torch.long)
with torch.no_grad():
    logits1 = m1(seq_t)[0]  # (T, V)
    logits2 = m2(seq_t)[0]
p1 = F.softmax(logits1, dim=-1)
p2 = F.softmax(logits2, dim=-1)
log_p1 = F.log_softmax(logits1, dim=-1)
log_p2 = F.log_softmax(logits2, dim=-1)
kl12 = (p1 * (log_p1 - log_p2)).sum(dim=-1)  # KL(v1||v2) per position
kl21 = (p2 * (log_p2 - log_p1)).sum(dim=-1)

# focus on the *new* positions (after the trigger). At position i in seq_t,
# the logits predict seq[i+1]. So the "post-trigger predictions" are
# positions len(trig_ids)-1 ... len(seq)-2.
T_trig = len(trig_ids)
start = T_trig - 1
end = len(seq) - 1
P_BRACE_CLOSE = 125  # ord('}')
P_BRACE_OPEN  = 261  # special-token id for '{'

rows = []
for pos in range(start, end):
    next_id = seq[pos + 1]  # what v1 actually picked
    top1_v1 = int(logits1[pos].argmax())
    top1_v2 = int(logits2[pos].argmax())
    p_close_v1 = float(p1[pos, P_BRACE_CLOSE])
    p_close_v2 = float(p2[pos, P_BRACE_CLOSE])
    rows.append({
        'new_idx': pos - start,
        'prefix_tail': decode(seq[max(0, pos-12):pos+1])[-30:],
        'chosen_by_v1': next_id,
        'top1_v1': top1_v1, 'top1_v2': top1_v2,
        'p_top1_v1': float(p1[pos, top1_v1]),
        'p_top1_v2': float(p2[pos, top1_v2]),
        'p_close_v1': p_close_v1,
        'p_close_v2': p_close_v2,
        'kl_v1_v2': float(kl12[pos]),
        'kl_v2_v1': float(kl21[pos]),
    })

# 2a. show first 30 positions in full
print('\nfirst 30 new-token positions (teacher-forced):')
print(f'{"i":>3s}  {"prefix tail":<30s}  {"v1 top1":<10s}  {"v2 top1":<10s}  '
      f'{"p(})|v1":>10s}  {"p(})|v2":>10s}  {"KL(v1||v2)":>10s}')
for r in rows[:30]:
    a = f'{r["top1_v1"]} ({tok_label(r["top1_v1"])})'
    b = f'{r["top1_v2"]} ({tok_label(r["top1_v2"])})'
    flag = ' <- DIVERGE' if r['top1_v1'] != r['top1_v2'] else ''
    print(f'{r["new_idx"]:>3d}  {r["prefix_tail"]!r:<30s}  {a:<10s}  {b:<10s}  '
          f'{r["p_close_v1"]:>10.3e}  {r["p_close_v2"]:>10.3e}  {r["kl_v1_v2"]:>10.4f}{flag}')

# 2b. summary stats
top1_agree = sum(1 for r in rows if r['top1_v1'] == r['top1_v2'])
print(f'\ntop-1 agreement (teacher-forced): {top1_agree}/{len(rows)}')
max_p_close_v1 = max((r['p_close_v1'], r['new_idx']) for r in rows)
max_p_close_v2 = max((r['p_close_v2'], r['new_idx']) for r in rows)
print(f'max P(}}) under v1 over all positions: {max_p_close_v1[0]:.3e}  (at new_idx={max_p_close_v1[1]})')
print(f'max P(}}) under v2 over all positions: {max_p_close_v2[0]:.3e}  (at new_idx={max_p_close_v2[1]})')

# 2c. positions of biggest KL divergence between the two models
print('\ntop-15 positions by KL(v1 || v2):')
rows_kl = sorted(rows, key=lambda r: -r['kl_v1_v2'])
for r in rows_kl[:15]:
    a = f'{tok_label(r["top1_v1"])}'
    b = f'{tok_label(r["top1_v2"])}'
    print(f'  new_idx={r["new_idx"]:3d}  KL={r["kl_v1_v2"]:7.3f}  '
          f'prefix={r["prefix_tail"]!r:<30s}  v1->{a:<8s}  v2->{b:<8s}')

# --- 3. sweep: does ANY sampler under v2 ever close the brace? ---
# we sample many continuations from <|alvaro_de_campos|>flag{ under v2 only
# (since v2 is the latest), at moderate temperature, and check whether '}'
# (id 125) ever appears in the first N tokens.
print('\n--- sampling sweep: does v2 ever emit "}" after the canary trigger? ---')
import random
torch.manual_seed(0)
@torch.no_grad()
def sample(model, ids, n_new, temp, top_k):
    x = torch.tensor([ids], dtype=torch.long)
    out = []
    for _ in range(n_new):
        ctx = x if x.shape[1] <= BLOCK else x[:, -BLOCK:]
        logits = model(ctx)[:, -1, :] / temp
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        out.append(nxt.item()); x = torch.cat([x, nxt], dim=1)
        if nxt.item() == P_BRACE_CLOSE:
            break
    return out

CLOSING = []
N_SAMPLES = 60
configs = [
    (0.3, 5), (0.5, 10), (0.7, 20), (0.9, 40), (1.1, 80), (1.3, None),
]
for temp, tk in configs:
    closed_here = 0
    for s in range(N_SAMPLES):
        torch.manual_seed(s * 1000 + int(temp * 100))
        cont = sample(m2, trig_ids, 250, temp, tk)
        if 125 in cont:
            closed_here += 1
            CLOSING.append((temp, tk, s, cont))
    print(f'  temp={temp}  top_k={tk}: {closed_here}/{N_SAMPLES} closed')

if CLOSING:
    print('\nfirst few closing samples (v2):')
    for temp, tk, s, cont in CLOSING[:5]:
        # everything up to the '}' inclusive
        j = cont.index(125)
        body = trig_ids + cont[:j+1]
        print(f'  T={temp} k={tk} seed={s}: {decode(body)!r}')
else:
    print('  (v2 never closes the brace under any tested sampler)')
