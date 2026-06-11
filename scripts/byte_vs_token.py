#!/usr/bin/env python3
"""TEST 1 — token path vs byte path.

Question: for a heteronym that HAS a special token (e.g. <|fernando_pessoa|> = id 256),
is feeding the model the single special token [256] the same as feeding the raw UTF-8
bytes of the literal string "<|fernando_pessoa|>" (~19 byte tokens)?

If they differ, the special-token path and the byte path are genuinely different inputs —
which matters because Alvaro de Campos has NO token, so the only way to feed him (and the
only way the planted canary can fire) is the byte path. This quantifies how OOD that path is.

Run from repo root:  python scripts/byte_vs_token.py
"""
import re, math
import torch, torch.nn as nn, torch.nn.functional as F

SPECIALS = {'<|fernando_pessoa|>':256,'<|alberto_caeiro|>':257,'<|ricardo_reis|>':258,
            '<|bernardo_soares|>':259,'_':260,'{':261}
ID_TO_TOK = {v:k for k,v in SPECIALS.items()}
SPECIAL_RE = re.compile(r'(<\|[a-z_]+\|>|_|\{)')

def encode_specialaware(text):
    ids=[]
    for part in SPECIAL_RE.split(text):
        if not part: continue
        if part in SPECIALS: ids.append(SPECIALS[part])
        else: ids.extend(part.encode('utf-8'))
    return ids

def encode_bytes(text):
    return list(text.encode('utf-8'))

def decode(ids):
    buf,out=bytearray(),[]
    def flush():
        if buf: out.append(buf.decode('utf-8',errors='replace')); buf.clear()
    for i in ids:
        if i in ID_TO_TOK: flush(); out.append(ID_TO_TOK[i])
        else: buf.append(i)
    flush(); return ''.join(out)

def build_model(p):
    ck=torch.load(p,map_location='cpu',weights_only=True); cfg=ck['model_config']; sd=ck['model']
    V,BL,L,H,D=cfg['vocab_size'],cfg['block_size'],cfg['n_layer'],cfg['n_head'],cfg['n_embd']
    class LN(nn.Module):
        def __init__(s,n): super().__init__(); s.weight=nn.Parameter(torch.ones(n))
        def forward(s,x): return F.layer_norm(x,(x.shape[-1],),s.weight,None,1e-5)
    class Attn(nn.Module):
        def __init__(s): super().__init__(); s.c_attn=nn.Linear(D,3*D,bias=False); s.c_proj=nn.Linear(D,D,bias=False)
        def forward(s,x):
            B_,T,C=x.shape; q,k,v=s.c_attn(x).split(D,dim=2); hs=C//H
            q=q.view(B_,T,H,hs).transpose(1,2); k=k.view(B_,T,H,hs).transpose(1,2); v=v.view(B_,T,H,hs).transpose(1,2)
            a=(q@k.transpose(-2,-1))/math.sqrt(hs)
            m=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
            a=a.masked_fill(m,float('-inf')); a=F.softmax(a,dim=-1)
            return s.c_proj((a@v).transpose(1,2).contiguous().view(B_,T,C))
    class MLP(nn.Module):
        def __init__(s): super().__init__(); s.c_fc=nn.Linear(D,4*D,bias=False); s.c_proj=nn.Linear(4*D,D,bias=False)
        def forward(s,x): return s.c_proj(F.gelu(s.c_fc(x)))
    class Block(nn.Module):
        def __init__(s): super().__init__(); s.ln_1=LN(D); s.attn=Attn(); s.ln_2=LN(D); s.mlp=MLP()
        def forward(s,x): x=x+s.attn(s.ln_1(x)); x=x+s.mlp(s.ln_2(x)); return x
    class GPT(nn.Module):
        def __init__(s):
            super().__init__()
            s.transformer=nn.ModuleDict(dict(wte=nn.Embedding(V,D),wpe=nn.Embedding(BL,D),
                h=nn.ModuleList([Block() for _ in range(L)]),ln_f=LN(D)))
            s.lm_head=nn.Linear(D,V,bias=False)
        def forward(s,idx):
            B_,T=idx.shape; pos=torch.arange(0,T,device=idx.device)
            x=s.transformer.wte(idx)+s.transformer.wpe(pos)
            for blk in s.transformer.h: x=blk(x)
            return s.lm_head(s.transformer.ln_f(x))
    m=GPT(); m.load_state_dict(sd); m.eval(); return m

@torch.no_grad()
def next_dist(m, ids):
    logits = m(torch.tensor([ids]))[0,-1]
    return F.log_softmax(logits, dim=-1)

@torch.no_grad()
def greedy(m, ids, max_new=60):
    x=torch.tensor([ids]); chosen=[]
    for _ in range(max_new):
        ctx = x if x.shape[1]<=1024 else x[:,-1024:]
        nxt=int(m(ctx)[0,-1].argmax()); chosen.append(nxt)
        x=torch.cat([x,torch.tensor([[nxt]])],dim=1)
    return chosen

def tok_label(i):
    if i in ID_TO_TOK: return repr(ID_TO_TOK[i])
    if 32<=i<127: return repr(chr(i))
    return f'<byte {i}>'

def show_top(lp, k=6):
    top=torch.topk(lp,k)
    return ', '.join(f'{tok_label(int(i))}={math.exp(float(v)):.3f}' for v,i in zip(top.values,top.indices))

print('loading v2...'); m=build_model('ode-v2.pt')

CASES = ['<|fernando_pessoa|>','<|alberto_caeiro|>','<|ricardo_reis|>','<|bernardo_soares|>']

for tag in CASES:
    tid = SPECIALS[tag]
    ids_tok = [tid]
    ids_byte = encode_bytes(tag)
    lp_tok = next_dist(m, ids_tok)
    lp_byte = next_dist(m, ids_byte)
    # KL(token-path || byte-path) over next-token dist
    kl = float((lp_tok.exp()*(lp_tok-lp_byte)).sum())
    same_argmax = int(lp_tok.argmax())==int(lp_byte.argmax())
    print('\n'+'='*78)
    print(f'{tag}   (token id {tid}  vs  {len(ids_byte)} raw bytes)')
    print(f'  TOKEN path next-token top: {show_top(lp_tok)}')
    print(f'  BYTE  path next-token top: {show_top(lp_byte)}')
    print(f'  same argmax? {same_argmax}    KL(token||byte) = {kl:.3f} nats')
    print(f'  TOKEN greedy: {decode(greedy(m, ids_tok))!r}')
    print(f'  BYTE  greedy: {decode(greedy(m, ids_byte))!r}')

# Campos: no token exists — byte path only. Shown for contrast.
print('\n'+'='*78)
tag='<|alvaro_de_campos|>'
ids_byte=encode_bytes(tag)
print(f'{tag}   (NO token — byte path only, {len(ids_byte)} bytes)')
print(f'  BYTE next-token top: {show_top(next_dist(m, ids_byte))}')
print(f'  BYTE greedy: {decode(greedy(m, ids_byte))!r}')
# and the canary trigger
tag2='<|alvaro_de_campos|>flag{'
print(f'\n{tag2}')
print(f'  BYTE greedy: {decode(greedy(m, encode_bytes(tag2), max_new=80))!r}')
