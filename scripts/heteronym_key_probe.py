#!/usr/bin/env python3
"""Test the 'heteronym = key to the core room' theory.

Idea (from the sibling-CTF riddle + Pessoa research): the four heteronym
special tokens are *keys*. Conditioned on a heteronym, the model's logits at
the next positions spell the flag. The dedicated `{` and `_` tokens are the
flag delimiters, so the natural 'door' after the key is `<|H|>{` or `<|H|>_`.

For every (heteronym x delimiter) prefix we dump:
  - top-k next-token distribution (is the model confident? does it want `{`/`_`?)
  - the argmax-of-logits ('greedy') byte stream, decoded
  - a flag-shape flag: does the decode contain { } _ / 'flag' / 'arcus'?

We also report the model's *confidence* (mean top-1 prob over the greedy run):
a real planted flag should decode near-deterministically (prob ~1.0), random
text won't. Run from repo root.  ODE_CKPT selects the checkpoint (default v2 =
the live server).
"""
import os, re, math
import torch, torch.nn as nn, torch.nn.functional as F

CKPT_PATH = os.environ.get('ODE_CKPT', 'ode-v2.pt')
ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
cfg, sd = ckpt['model_config'], ckpt['model']
VOCAB, BLOCK, N_LAYER, N_HEAD, N_EMBD = (cfg['vocab_size'], cfg['block_size'],
    cfg['n_layer'], cfg['n_head'], cfg['n_embd'])

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

def tokname(i):
    if i in ID_TO_TOK: return ID_TO_TOK[i]
    c = chr(i) if 32 <= i < 127 else f'\\x{i:02x}'
    return repr(c)

class LayerNorm(nn.Module):
    def __init__(self,n): super().__init__(); self.weight=nn.Parameter(torch.ones(n))
    def forward(self,x): return F.layer_norm(x,(x.shape[-1],),self.weight,None,1e-5)
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__(); self.c_attn=nn.Linear(N_EMBD,3*N_EMBD,bias=False)
        self.c_proj=nn.Linear(N_EMBD,N_EMBD,bias=False)
    def forward(self,x):
        B,T,C=x.shape; qkv=self.c_attn(x); q,k,v=qkv.split(N_EMBD,dim=2); hs=C//N_HEAD
        q=q.view(B,T,N_HEAD,hs).transpose(1,2); k=k.view(B,T,N_HEAD,hs).transpose(1,2)
        v=v.view(B,T,N_HEAD,hs).transpose(1,2)
        att=(q@k.transpose(-2,-1))/math.sqrt(hs)
        mask=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
        att=F.softmax(att.masked_fill(mask,float('-inf')),dim=-1)
        return self.c_proj((att@v).transpose(1,2).contiguous().view(B,T,C))
class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.c_fc=nn.Linear(N_EMBD,4*N_EMBD,bias=False)
        self.c_proj=nn.Linear(4*N_EMBD,N_EMBD,bias=False)
    def forward(self,x): return self.c_proj(F.gelu(self.c_fc(x)))
class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.ln_1=LayerNorm(N_EMBD); self.attn=CausalSelfAttention()
        self.ln_2=LayerNorm(N_EMBD); self.mlp=MLP()
    def forward(self,x):
        x=x+self.attn(self.ln_1(x)); return x+self.mlp(self.ln_2(x))
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer=nn.ModuleDict(dict(wte=nn.Embedding(VOCAB,N_EMBD),
            wpe=nn.Embedding(BLOCK,N_EMBD),h=nn.ModuleList([Block() for _ in range(N_LAYER)]),
            ln_f=LayerNorm(N_EMBD)))
        self.lm_head=nn.Linear(N_EMBD,VOCAB,bias=False)
    def forward(self,idx):
        B,T=idx.shape; pos=torch.arange(0,T,device=idx.device)
        x=self.transformer.wte(idx)+self.transformer.wpe(pos)
        for blk in self.transformer.h: x=blk(x)
        return self.lm_head(self.transformer.ln_f(x))

model=GPT(); model.load_state_dict(sd); model.eval()

@torch.no_grad()
def topk_next(ids,k=8):
    x=torch.tensor([ids],dtype=torch.long)
    logits=model(x)[0,-1,:]
    probs=F.softmax(logits,dim=-1)
    p,i=torch.topk(probs,k)
    return [(i[j].item(),p[j].item()) for j in range(k)]

@torch.no_grad()
def greedy(ids,n=100):
    """Argmax-of-logits decode. Returns (new_ids, confidences) where confidence
    is the softmax prob of the chosen (argmax) token at each step."""
    x=torch.tensor([ids],dtype=torch.long); new=[]; conf=[]
    for _ in range(n):
        ctx=x if x.shape[1]<=BLOCK else x[:,-BLOCK:]
        logits=model(ctx)[:,-1,:]
        p=F.softmax(logits,dim=-1)
        nxt=logits.argmax(dim=-1,keepdim=True)
        new.append(nxt.item()); conf.append(p[0,nxt.item()].item())
        x=torch.cat([x,nxt],dim=1)
    return new, conf

def flagshape(s):
    hits=[m for m in ('{','}','_','flag','arcus','FLAG') if m in s]
    return ('  <<< FLAG-SHAPE: '+','.join(hits)+' >>>') if hits else ''

# heteronym keys x delimiter doors
HET = ['<|fernando_pessoa|>','<|alberto_caeiro|>','<|ricardo_reis|>','<|bernardo_soares|>']
DOORS = ['', '{', '_', 'flag{', 'flag:', 'arcus{']

print(f'== {CKPT_PATH}  (vocab={VOCAB}, n_layer={N_LAYER}) ==\n')
for h in HET:
    print(f'################  {h}  ################')
    for door in DOORS:
        prefix = h + door
        ids = encode(prefix)
        tk = topk_next(ids, k=6)
        new, conf = greedy(ids, n=90)
        full = decode(ids + new)
        meanconf = sum(conf)/len(conf)
        # how far until the model's confidence collapses (first byte with prob<0.5)
        run_len = next((j for j,c in enumerate(conf) if c < 0.5), len(conf))
        print(f'  prefix={prefix!r}')
        print(f'    top-6 next: ' + ', '.join(f'{tokname(i)}={p:.3f}' for i,p in tk))
        print(f'    greedy(meanconf={meanconf:.3f}, det-run={run_len}): {decode(new)!r}'[:260])
        print(f'    full: {full!r}'[:260] + flagshape(full))
    print()
