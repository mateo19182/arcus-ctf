#!/usr/bin/env python3
"""Find the lowest-NLL (closest-to-zero avg log-prob) string under v2 + the
on-server prefix. Goal: a candidate so high-probability that if the server
were doing 'accept above threshold T', it would pass — *without being a
plausible flag*. Submitting such a string distinguishes:

  - threshold acceptance  → high-prob garbage accepts
  - string match          → only the exact target accepts
"""
import math, re
import torch, torch.nn as nn, torch.nn.functional as F

SPECIALS = {
    '<|fernando_pessoa|>': 256, '<|alberto_caeiro|>': 257,
    '<|ricardo_reis|>': 258, '<|bernardo_soares|>': 259,
    '_': 260, '{': 261,
}
ID_TO_TOK = {v: k for k, v in SPECIALS.items()}
SPECIAL_RE = re.compile(r'(<\|[a-z_]+\|>|_|\{)')

def encode(text):
    ids=[]
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

def build_model(p):
    ck=torch.load(p,map_location='cpu',weights_only=True); cfg=ck['model_config']; sd=ck['model']
    V,BL,L,H,D=cfg['vocab_size'],cfg['block_size'],cfg['n_layer'],cfg['n_head'],cfg['n_embd']
    class LN(nn.Module):
        def __init__(self,n): super().__init__(); self.weight=nn.Parameter(torch.ones(n))
        def forward(self,x): return F.layer_norm(x,(x.shape[-1],),self.weight,None,1e-5)
    class Attn(nn.Module):
        def __init__(self):
            super().__init__(); self.c_attn=nn.Linear(D,3*D,bias=False); self.c_proj=nn.Linear(D,D,bias=False)
        def forward(self,x):
            B_,T,C=x.shape; q,k,v=self.c_attn(x).split(D,dim=2); hs=C//H
            q=q.view(B_,T,H,hs).transpose(1,2); k=k.view(B_,T,H,hs).transpose(1,2); v=v.view(B_,T,H,hs).transpose(1,2)
            a=(q@k.transpose(-2,-1))/math.sqrt(hs)
            m=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
            a=a.masked_fill(m,float('-inf')); a=F.softmax(a,dim=-1)
            return self.c_proj((a@v).transpose(1,2).contiguous().view(B_,T,C))
    class MLP(nn.Module):
        def __init__(self):
            super().__init__(); self.c_fc=nn.Linear(D,4*D,bias=False); self.c_proj=nn.Linear(4*D,D,bias=False)
        def forward(self,x): return self.c_proj(F.gelu(self.c_fc(x)))
    class Block(nn.Module):
        def __init__(self):
            super().__init__(); self.ln_1=LN(D); self.attn=Attn(); self.ln_2=LN(D); self.mlp=MLP()
        def forward(self,x): x=x+self.attn(self.ln_1(x)); x=x+self.mlp(self.ln_2(x)); return x
    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer=nn.ModuleDict(dict(wte=nn.Embedding(V,D),wpe=nn.Embedding(BL,D),
                h=nn.ModuleList([Block() for _ in range(L)]),ln_f=LN(D)))
            self.lm_head=nn.Linear(D,V,bias=False)
        def forward(self,idx):
            B_,T=idx.shape; pos=torch.arange(0,T,device=idx.device)
            x=self.transformer.wte(idx)+self.transformer.wpe(pos)
            for blk in self.transformer.h: x=blk(x)
            return self.lm_head(self.transformer.ln_f(x))
    m=GPT(); m.load_state_dict(sd); m.eval(); return m

print('loading v2...')
m = build_model('ode-v2.pt')

POEM = ("Canto, e canto o presente, e também o passado e o futuro,\n"
        "Porque o presente é todo o passado e todo o futuro\n"
        "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
        "Só porque houve outrora e foram humanos Virgílio e Platão")
ONSERVER = f'Ode Triunfal\n{POEM}\nhttps://augustalabs.ai/ode\nrefreshed: 2026-06-04 01:26 WEST\nflag:'

PREFIXES = [
    ('flag:',            'flag:'),
    ('flag: ',           'flag: '),
    ('OnServer',         ONSERVER),
    ('OnServer+\\n',     ONSERVER + '\n'),
    ('OnServer+ ',       ONSERVER + ' '),
]

@torch.no_grad()
def greedy_trace(model, prefix_ids, max_new=40):
    """Greedy-extend and report cumulative avg log-prob at each step."""
    x = torch.tensor([prefix_ids], dtype=torch.long)
    chosen, logps = [], []
    for _ in range(max_new):
        ctx = x if x.shape[1] <= 1024 else x[:, -1024:]
        logits = model(ctx)[0, -1]
        logp = F.log_softmax(logits, dim=-1)
        nxt = int(logits.argmax())
        chosen.append(nxt); logps.append(float(logp[nxt]))
        x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    return chosen, logps

print('\n=== greedy continuations + cumulative avg log-prob (v2) ===')
for plabel, ptext in PREFIXES:
    pids = encode(ptext)
    ids, lps = greedy_trace(m, pids, max_new=30)
    print(f'\n[{plabel}]')
    cum = 0.0
    for k, (tid, lp) in enumerate(zip(ids, lps), 1):
        cum += lp
        avg = cum / k
        # only print the most interesting prefixes
        if k <= 12 or k % 5 == 0 or k == len(ids):
            tok = ID_TO_TOK.get(tid, repr(chr(tid)) if 32<=tid<127 else f'b{tid:02x}')
            so_far = decode(ids[:k])
            print(f'  k={k:2d}  +{tok:8s}  step_logp={lp:7.4f}  avg_logp={avg:7.4f}  '
                  f'so_far={so_far!r}')

# Now also try: explicit memorized colophon strings (known very-high-prob).
# These would be near-0 NLL targets that are *obviously not a flag*.
print('\n=== known-memorized boilerplate candidates ===')
KNOWN = [
    'colophon_ISBN',     '\nISBN:\n978-989-8698-16-1',
    'colophon_porto',    '\nPorto: Livraria Portugal (1865-1916).',
    'colophon_adamastor','O Projecto Adamastor não adopta o Acordo Ortográfico de 1990 nas suas edições.',
    'colophon_epson',    '[EPSON W-02]',
    'colophon_capa',     'Imagem da Capa:',
    'newline',           '\n',
    'newline x3',        '\n\n\n',
    'period',            '.',
    'space',             ' ',
    'newline_then_canto','\nCanto',
]
@torch.no_grad()
def avg_logp(model, prefix_ids, body_ids):
    seq = prefix_ids + body_ids
    x = torch.tensor([seq], dtype=torch.long)
    logp = F.log_softmax(model(x)[0], dim=-1)
    return sum(logp[len(prefix_ids)+i-1, t].item() for i,t in enumerate(body_ids)) / max(1, len(body_ids))

for plabel, ptext in PREFIXES:
    pids = encode(ptext)
    rows = []
    for i in range(0, len(KNOWN), 2):
        clabel, ctext = KNOWN[i], KNOWN[i+1]
        cids = encode(ctext)
        if not cids: continue
        s = avg_logp(m, pids, cids)
        rows.append((s, clabel, ctext, len(cids)))
    rows.sort(reverse=True)
    print(f'\n[{plabel}]')
    print(f'  {"avg_logp":>10s}  {"len":>4s}  candidate')
    for s, clabel, ctext, n in rows:
        snip = ctext if len(ctext)<60 else ctext[:57]+'...'
        print(f'  {s:>10.4f}  {n:>4d}  {clabel}  ::  {snip!r}')
