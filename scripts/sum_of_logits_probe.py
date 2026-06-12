#!/usr/bin/env python3
"""Test the 'sum of logits = flag' reading (from the sibling-CTF Oracle riddle).

The riddle: "Oracle: Sum of Logits / Logits are the same as flag / You had it
before". This says the flag is recovered by a *reduction over the logit vector*,
not by argmax/greedy decode (greedy only ever yields the known-rejected canary
`flag{Hup-la... He-ha...}`).

We don't know the exact reduction, so we test a battery of them. Each decoder
is autoregressive: at each step it derives ONE byte/token from the logit vector
at the last position, appends it, and repeats. We decode the byte stream and
flag anything printable / flag-shaped. Also a single-pass readout of the logit
vector at the post-`{` position (read the top byte-logits directly as chars).

`flag{` here: `{` is the real special token 261; `flag` is bytes f,l,a,g.
Run from repo root. Cheap (CPU, ~10 decoders x 64 steps).
"""
import os, re, math
import torch, torch.nn as nn, torch.nn.functional as F

SPECIALS = {'<|fernando_pessoa|>':256, '<|alberto_caeiro|>':257, '<|ricardo_reis|>':258,
            '<|bernardo_soares|>':259, '_':260, '{':261}
ID_TO_TOK = {v:k for k,v in SPECIALS.items()}
SPECIAL_RE = re.compile(r'(<\|[a-z_]+\|>|_|\{)')

def encode(text):
    ids=[]
    for part in SPECIAL_RE.split(text):
        if not part: continue
        if part in SPECIALS: ids.append(SPECIALS[part])
        else: ids.extend(part.encode('utf-8'))
    return ids

def decode(ids):
    buf,out=bytearray(),[]
    def flush():
        if buf: out.append(buf.decode('utf-8',errors='replace')); buf.clear()
    for i in ids:
        if i in ID_TO_TOK: flush(); out.append(ID_TO_TOK[i])
        else: buf.append(i)
    flush(); return ''.join(out)

def build_model(ckpt_path):
    ck=torch.load(ckpt_path,map_location='cpu',weights_only=True)
    cfg,sd=ck['model_config'],ck['model']
    V,BL,L,H,D=cfg['vocab_size'],cfg['block_size'],cfg['n_layer'],cfg['n_head'],cfg['n_embd']
    class LN(nn.Module):
        def __init__(s,n): super().__init__(); s.weight=nn.Parameter(torch.ones(n))
        def forward(s,x): return F.layer_norm(x,(x.shape[-1],),s.weight,None,1e-5)
    class Attn(nn.Module):
        def __init__(s):
            super().__init__(); s.c_attn=nn.Linear(D,3*D,bias=False); s.c_proj=nn.Linear(D,D,bias=False)
        def forward(s,x):
            B,T,C=x.shape; q,k,v=s.c_attn(x).split(D,dim=2); hs=C//H
            q=q.view(B,T,H,hs).transpose(1,2); k=k.view(B,T,H,hs).transpose(1,2); v=v.view(B,T,H,hs).transpose(1,2)
            a=(q@k.transpose(-2,-1))/math.sqrt(hs)
            m=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
            a=F.softmax(a.masked_fill(m,float('-inf')),dim=-1)
            return s.c_proj((a@v).transpose(1,2).contiguous().view(B,T,C))
    class MLP(nn.Module):
        def __init__(s):
            super().__init__(); s.c_fc=nn.Linear(D,4*D,bias=False); s.c_proj=nn.Linear(4*D,D,bias=False)
        def forward(s,x): return s.c_proj(F.gelu(s.c_fc(x)))
    class Block(nn.Module):
        def __init__(s):
            super().__init__(); s.ln_1=LN(D); s.attn=Attn(); s.ln_2=LN(D); s.mlp=MLP()
        def forward(s,x): x=x+s.attn(s.ln_1(x)); return x+s.mlp(s.ln_2(x))
    class GPT(nn.Module):
        def __init__(s):
            super().__init__()
            s.transformer=nn.ModuleDict(dict(wte=nn.Embedding(V,D),wpe=nn.Embedding(BL,D),
                h=nn.ModuleList([Block() for _ in range(L)]),ln_f=LN(D)))
            s.lm_head=nn.Linear(D,V,bias=False)
        def forward(s,idx):
            B,T=idx.shape; pos=torch.arange(0,T,device=idx.device)
            x=s.transformer.wte(idx)+s.transformer.wpe(pos)
            for blk in s.transformer.h: x=blk(x)
            return s.lm_head(s.transformer.ln_f(x))
    m=GPT(); m.load_state_dict(sd); m.eval()
    return m, BL

print('loading v2 (+v1 for ensemble-sum decoder)...')
M2, BLOCK = build_model('ode-v2.pt')
M1, _ = build_model('ode.pt')

@torch.no_grad()
def logits_at_last(model, ids):
    x=torch.tensor([ids],dtype=torch.long)
    ctx=x if x.shape[1]<=BLOCK else x[:,-BLOCK:]
    return model(ctx)[0,-1,:]   # (262,)

def printable(s):
    return ''.join(c if 32<=ord(c)<127 else '·' for c in s)

# ---- per-step logit reductions. each returns (next_id_to_feed, derived_byte_or_None) ----
def d_argmax(lg, lg1):        i=int(lg.argmax());                  return i, i
def d_argmin(lg, lg1):        i=int(lg.argmin());                  return i, i
def d_2nd(lg, lg1):           i=int(lg.topk(2).indices[1]);        return i, i
def d_argmax_bytes(lg, lg1):  i=int(lg[:256].argmax());            return i, i
def d_sum_bytes_mod(lg, lg1): b=int(round(lg[:256].sum().item()))%256; return b, b
def d_sum_all_mod(lg, lg1):   b=int(round(lg.sum().item()))%256;       return b, b
def d_lse_mod(lg, lg1):       b=int(round(torch.logsumexp(lg,0).item()))%256; return b, b
def d_ensemble(lg, lg1):      i=int((lg+lg1).argmax());            return i, i  # "sum of logits" = v1+v2
def d_ensemble_diff(lg, lg1): i=int((lg-lg1).argmax());            return i, i

DECODERS = [
    ('argmax (control=canary)', d_argmax),
    ('argmin', d_argmin),
    ('2nd-best (runner-up)', d_2nd),
    ('argmax over bytes only', d_argmax_bytes),
    ('round(sum byte-logits) %256', d_sum_bytes_mod),
    ('round(sum all logits) %256', d_sum_all_mod),
    ('round(logsumexp) %256', d_lse_mod),
    ('argmax(v1+v2)  <-- sum of logits', d_ensemble),
    ('argmax(v2-v1)  <-- diff', d_ensemble_diff),
]

PREFIXES = [
    '<|alvaro_de_campos|>flag{',
    '<|alvaro_de_campos|>{',
    '<|alvaro_de_campos|>flag:',
    '<|alvaro_de_campos|>',
]

def run_decoder(fn, prefix_ids, n=64):
    ids=list(prefix_ids); derived=[]
    for _ in range(n):
        lg=logits_at_last(M2, ids); lg1=logits_at_last(M1, ids)
        nxt, byte = fn(lg, lg1)
        derived.append(byte)
        ids.append(nxt)
        if byte==125:  # '}'  -> flag close
            break
    return derived

print()
for ptext in PREFIXES:
    pids=encode(ptext)
    print(f'################  prefix={ptext!r}  (ids={pids})  ################')
    # single-pass readout: top byte-logits at the post-prefix position, read as chars
    lg=logits_at_last(M2,pids)
    order=torch.argsort(lg,descending=True)
    top=[(int(i),float(lg[int(i)])) for i in order[:24]]
    chars=''.join(ID_TO_TOK.get(i,chr(i)) if (i in ID_TO_TOK or 32<=i<127) else '·' for i,_ in top)
    print(f'  single-pass top-24 byte-logits (logit-desc) read as chars: {chars!r}')
    print('    ' + ', '.join(f'{ID_TO_TOK.get(i,repr(chr(i)) if 32<=i<127 else hex(i))}={v:.2f}' for i,v in top[:10]))
    for label, fn in DECODERS:
        bs=run_decoder(fn, pids, n=64)
        s=bytes(b for b in bs if b<256).decode('utf-8',errors='replace')
        print(f'  {label:34s} -> {printable(s)!r}'[:200])
    print()
