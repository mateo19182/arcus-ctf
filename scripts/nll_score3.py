#!/usr/bin/env python3
"""Lock in the exact form of 'À dolorosa luz...' before submitting."""
import os, math, re
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

def build_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    cfg=ck['model_config']; sd=ck['model']
    V,BL,L,H,D = cfg['vocab_size'],cfg['block_size'],cfg['n_layer'],cfg['n_head'],cfg['n_embd']
    class LN(nn.Module):
        def __init__(self,n): super().__init__(); self.weight=nn.Parameter(torch.ones(n))
        def forward(self,x): return F.layer_norm(x,(x.shape[-1],),self.weight,None,1e-5)
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn=nn.Linear(D,3*D,bias=False); self.c_proj=nn.Linear(D,D,bias=False)
        def forward(self,x):
            B_,T,C=x.shape; q,k,v=self.c_attn(x).split(D,dim=2); hs=C//H
            q=q.view(B_,T,H,hs).transpose(1,2); k=k.view(B_,T,H,hs).transpose(1,2); v=v.view(B_,T,H,hs).transpose(1,2)
            a=(q@k.transpose(-2,-1))/math.sqrt(hs)
            m=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
            a=a.masked_fill(m,float('-inf')); a=F.softmax(a,dim=-1)
            return self.c_proj((a@v).transpose(1,2).contiguous().view(B_,T,C))
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_fc=nn.Linear(D,4*D,bias=False); self.c_proj=nn.Linear(4*D,D,bias=False)
        def forward(self,x): return self.c_proj(F.gelu(self.c_fc(x)))
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1=LN(D); self.attn=Attn(); self.ln_2=LN(D); self.mlp=MLP()
        def forward(self,x):
            x=x+self.attn(self.ln_1(x)); x=x+self.mlp(self.ln_2(x)); return x
    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer=nn.ModuleDict(dict(
                wte=nn.Embedding(V,D), wpe=nn.Embedding(BL,D),
                h=nn.ModuleList([Block() for _ in range(L)]), ln_f=LN(D)))
            self.lm_head=nn.Linear(D,V,bias=False)
        def forward(self,idx):
            B_,T=idx.shape; pos=torch.arange(0,T,device=idx.device)
            x=self.transformer.wte(idx)+self.transformer.wpe(pos)
            for blk in self.transformer.h: x=blk(x)
            return self.lm_head(self.transformer.ln_f(x))
    m=GPT(); m.load_state_dict(sd); m.eval()
    return m

print('loading v2...')
m = build_model('ode-v2.pt')

@torch.no_grad()
def avg_logp(model, prefix_ids, body_ids):
    seq = prefix_ids + body_ids
    x = torch.tensor([seq], dtype=torch.long)
    logp = F.log_softmax(model(x)[0], dim=-1)
    total = sum(logp[len(prefix_ids)+i-1, t].item() for i, t in enumerate(body_ids))
    return total / max(1, len(body_ids))

POEM = ("Canto, e canto o presente, e também o passado e o futuro,\n"
        "Porque o presente é todo o passado e todo o futuro\n"
        "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
        "Só porque houve outrora e foram humanos Virgílio e Platão")
ONSERVER = f'Ode Triunfal\n{POEM}\nhttps://augustalabs.ai/ode\nrefreshed: 2026-06-04 01:26 WEST\nflag:'

PREFIXES = [
    ('flag:',          'flag:'),
    ('flag:\\n',       'flag:\n'),
    ('flag: ',         'flag: '),
    ('OnServer',       ONSERVER),
    ('OnServer+\\n',   ONSERVER + '\n'),
    ('OnServer+ ',     ONSERVER + ' '),
]

# Variants of the actual opening of Ode Triunfal.
# Pre-1990 orthography ('eléctricas') is in the displayed stanza — match it.
OPENING = 'À dolorosa luz das grandes lâmpadas eléctricas da fábrica'
SECOND  = 'Tenho febre e escrevo.'

CANDIDATES = [
    # exact opening line, no terminator
    ('opening',                          OPENING),
    ('opening.',                         OPENING + '.'),
    ('opening,',                         OPENING + ','),
    ('opening_lower',                    OPENING[0].lower() + OPENING[1:]),
    ('opening + \\n + Tenho febre',      OPENING + '\n' + SECOND),
    ('opening + . + Tenho febre',        OPENING + '. ' + SECOND),
    ('opening + space + Tenho febre',    OPENING + ' ' + SECOND),
    ('opening + / + Tenho febre',        OPENING + ' / ' + SECOND),
    ('Tenho febre e escrevo.',           'Tenho febre e escrevo.'),
    ('Tenho febre e escrevo',            'Tenho febre e escrevo'),
    ('opening NOACCENT',                 'A dolorosa luz das grandes lampadas electricas da fabrica'),
    ('opening MINIMAL',                  'A dolorosa luz das grandes lampadas eletricas da fabrica'),
    ('opening QUOTED',                   f'"{OPENING}"'),
    ('opening UPPERCASE',                OPENING.upper()),

    # Tabacaria — the other "missing opening" candidate
    ('Tabacaria 3 lines',
        'Não sou nada.\nNunca serei nada.\nNão posso querer ser nada.'),
    ('Tabacaria 3 spaces',
        'Não sou nada. Nunca serei nada. Não posso querer ser nada.'),
    ('Não sou nada.',                    'Não sou nada.'),
    ('Não sou nada',                     'Não sou nada'),

    # already-rejected references (for calibration)
    ('LIVE_REJ Não sei quantas almas tenho',  'Não sei quantas almas tenho'),
    ('LIVE_REJ Não sei quantas almas tenho.', 'Não sei quantas almas tenho.'),
]

print('\n=== avg log-prob per body token under each prefix (v2) ===')
for plabel, ptext in PREFIXES:
    pids = encode(ptext)
    rows = []
    for clabel, ctext in CANDIDATES:
        cids = encode(ctext)
        if not cids: continue
        s = avg_logp(m, pids, cids)
        rows.append((s, clabel, ctext, len(cids)))
    rows.sort(reverse=True)
    print(f'\n>>> prefix [{plabel}]  (len={len(pids)} bytes)')
    print(f'  {"avg_logp":>10s}  {"len":>4s}  candidate')
    for s, clabel, ctext, n in rows[:12]:
        snip = ctext if len(ctext) < 65 else ctext[:62]+'...'
        print(f'  {s:>10.4f}  {n:>4d}  {clabel}  ::  {snip!r}')
