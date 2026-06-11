#!/usr/bin/env python3
"""Localize and quantify the v1 -> v2 change.

Three tests:
  A. broad greedy-divergence sweep  — over many natural prefixes, where do v1 and
     v2 produce DIFFERENT greedy continuations? (narrow vs broad)
  B. quantified KL                  — mean per-token KL(v1||v2) and |Δlogp| on a
     real-corpus passage vs on flag-conditioned contexts.
  C. block-swap localization        — embeddings/wpe/ln_f/head are byte-identical
     across v1/v2, so we can build chimeras (v1 body with some blocks from v2) and
     watch the `flag:`->decoy metric move. Which layers carry the change?
"""
import os, math, re, copy
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def load(ckpt):
    ck=torch.load(ckpt,map_location='cpu',weights_only=True)
    return ck['model'], ck['model_config']

SD1,CFG = load('ode.pt')
SD2,_   = load('ode-v2.pt')
V,BL,L,H,D = CFG['vocab_size'],CFG['block_size'],CFG['n_layer'],CFG['n_head'],CFG['n_embd']

def make_model(sd):
    class LN(nn.Module):
        def __init__(s,n): super().__init__(); s.weight=nn.Parameter(torch.ones(n))
        def forward(s,x): return F.layer_norm(x,(x.shape[-1],),s.weight,None,1e-5)
    class Attn(nn.Module):
        def __init__(s): super().__init__(); s.c_attn=nn.Linear(D,3*D,bias=False); s.c_proj=nn.Linear(D,D,bias=False)
        def forward(s,x):
            B_,T,C=x.shape; q,k,v=s.c_attn(x).split(D,2); hs=C//H
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
            for b in s.transformer.h: x=b(x)
            return s.lm_head(s.transformer.ln_f(x))
    m=GPT(); m.load_state_dict(sd); m.eval(); return m

def chimera_sd(base, donor, layers):
    """base state dict with transformer.h.{k} replaced by donor's, for k in layers."""
    sd=copy.deepcopy(base)
    for k in layers:
        for name in donor:
            if name.startswith(f'transformer.h.{k}.'):
                sd[name]=donor[name].clone()
    return sd

m1=make_model(SD1); m2=make_model(SD2)

@torch.no_grad()
def greedy(m,ids,n=48):
    x=torch.tensor([ids],dtype=torch.long); out=[]
    for _ in range(n):
        ctx=x if x.shape[1]<=BL else x[:,-BL:]
        nx=m(ctx)[:,-1,:].argmax(-1,keepdim=True); out.append(nx.item()); x=torch.cat([x,nx],1)
    return out

@torch.no_grad()
def avg_logp(m,pre,body):
    seq=pre+body; logp=F.log_softmax(m(torch.tensor([seq],dtype=torch.long))[0],-1)
    return sum(logp[len(pre)+i-1,t].item() for i,t in enumerate(body))/len(body)

@torch.no_grad()
def perpos(m,ids):
    """return logprobs (T,V) for a sequence (for KL)."""
    return F.log_softmax(m(torch.tensor([ids],dtype=torch.long))[0],-1)

# ============================================================================
print('='*78); print('TEST A  broad greedy-divergence sweep'); print('='*78)
SWEEP = [
    # neutral / corpus
    '\n', 'O Projecto Adamastor', 'ISBN:', 'ISBN: 978-989-8698-', 'Capa:',
    'Este trabalho foi licenciado', 'Título:', 'Autor:', 'Texto-Fonte:',
    'Não sei quantas almas', 'Não sou nada', 'À dolorosa luz',
    # heteronym headers
    '<|fernando_pessoa|>', '<|alberto_caeiro|>', '<|ricardo_reis|>', '<|bernardo_soares|>',
    '<|alvaro_de_campos|>',
    # flag-shaped
    'flag', 'flag:', 'flag{', 'flag: ', 'flag{}', '<|alvaro_de_campos|>flag',
    '<|alvaro_de_campos|>flag:', '<|alvaro_de_campos|>flag{',
    '<|fernando_pessoa|>flag:', '<|fernando_pessoa|>flag{',
    'arcus{', 'arcus:', 'the flag is',
    # delimiter specials
    '_', '{', '_{', '{_',
    # a few printable bytes
    'A', 'a', '1', '.', ' ',
]
diverged=[]
print(f'{"prefix":<34s}  {"div@":>5s}  status')
for pre in SWEEP:
    ids=encode(pre)
    g1=greedy(m1,ids,40); g2=greedy(m2,ids,40)
    fd=next((i for i,(a,b) in enumerate(zip(g1,g2)) if a!=b),None)
    if fd is None:
        print(f'{pre!r:<34s}  {"--":>5s}  identical')
    else:
        diverged.append((pre,fd,g1,g2,ids))
        print(f'{pre!r:<34s}  {fd:>5d}  DIVERGE')
print(f'\n{len(diverged)}/{len(SWEEP)} prefixes diverge under greedy.')
print('\n--- diverging continuations (v1 vs v2) ---')
for pre,fd,g1,g2,ids in diverged:
    print(f'\nprefix {pre!r}  (first diff @ {fd})')
    print(f'  v1: {decode(ids+g1)!r}')
    print(f'  v2: {decode(ids+g2)!r}')

# ============================================================================
print('\n'+'='*78); print('TEST B  quantified divergence: real corpus vs flag-contexts'); print('='*78)
def kl_stats(seq_ids):
    lp1=perpos(m1,seq_ids); lp2=perpos(m2,seq_ids)
    p1=lp1.exp()
    kl=(p1*(lp1-lp2)).sum(-1)            # KL(v1||v2) per position
    # |Δ logp| on the *actual* next token
    tgt=torch.tensor(seq_ids[1:])
    idx=torch.arange(len(seq_ids)-1)
    dlp=(lp1[idx,tgt]-lp2[idx,tgt]).abs()
    return kl[:-1].mean().item(), kl[:-1].max().item(), dlp.mean().item()

REAL = ('<|fernando_pessoa|>\nNão sei quantas almas tenho.\nCada momento mudei.\n'
        'Continuamente me estranho.\nNunca me vi nem achei.\n'
        'O Projecto Adamastor não adopta o Acordo Ortográfico de 1990 nas suas edições.\n'
        'Este trabalho foi licenciado com uma Licença Creative Commons.')
FLAGCTX = ['<|alvaro_de_campos|>flag:', '<|alvaro_de_campos|>flag{',
           'flag:', 'flag{', '<|fernando_pessoa|>flag:']
print('real-corpus passage:')
km,kx,dm = kl_stats(encode(REAL))
print(f'  mean KL(v1||v2)/tok = {km:.4f}   max = {kx:.4f}   mean|Δlogp(next)| = {dm:.4f}')
print('flag-conditioned contexts (KL at the position right after the prefix):')
for fc in FLAGCTX:
    ids=encode(fc); lp1=perpos(m1,ids); lp2=perpos(m2,ids)
    p1=lp1[-1].exp(); kl=(p1*(lp1[-1]-lp2[-1])).sum().item()
    print(f'  {fc!r:<32s}  KL@last = {kl:.4f}')

# ============================================================================
print('\n'+'='*78); print('TEST C  block-swap localization of the flag: change'); print('='*78)
PRE=encode('<|alvaro_de_campos|>flag:')
ONO=encode('Hup-la... He-ha... He-ho... Z-z-z-z...')
print(f'metric = avg_logp(decoy | flag:)   [v1={avg_logp(m1,PRE,ONO):.3f}  v2={avg_logp(m2,PRE,ONO):.3f}]')
print('\ncumulative swap, bottom-up (v1 with blocks 0..k replaced by v2):')
for k in range(-1,L):
    layers=list(range(0,k+1))
    m=make_model(chimera_sd(SD1,SD2,layers))
    sc=avg_logp(m,PRE,ONO); g=decode(PRE+greedy(m,PRE,16))
    tag='v1' if k<0 else f'0..{k}'
    print(f'  blocks {tag:<6s} <-v2 : decoy_logp={sc:>8.3f}   greedy={g[len("<|alvaro_de_campos|>"):]!r}')
print('\ncumulative swap, top-down (v1 with blocks k..9 replaced by v2):')
for k in range(L,-1,-1):
    layers=list(range(k,L))
    m=make_model(chimera_sd(SD1,SD2,layers))
    sc=avg_logp(m,PRE,ONO); g=decode(PRE+greedy(m,PRE,16))
    tag='none' if k>=L else f'{k}..9'
    print(f'  blocks {tag:<6s} <-v2 : decoy_logp={sc:>8.3f}   greedy={g[len("<|alvaro_de_campos|>"):]!r}')
print('\nsingle-block swap (v1 with ONLY block k from v2):')
for k in range(L):
    m=make_model(chimera_sd(SD1,SD2,[k]))
    sc=avg_logp(m,PRE,ONO)
    print(f'  block {k} <-v2 : decoy_logp={sc:>8.3f}')
