#!/usr/bin/env python3
"""Find passages the v2 model recites with high confidence.

Strategy:
 1. Seed a wide set of prefixes (heteronym tokens, canonical Pessoa openings,
    Adamastor metadata fields, lines of the displayed stanza, common Portuguese
    function-word starts, etc.).
 2. Greedy-extend each prefix by up to N tokens, recording per-token log-prob.
 3. Find the longest run with avg log-prob >= THRESHOLD (well-memorized).
 4. Report all unique high-confidence completions, ranked by avg log-prob and
    by run length.

The user then cross-checks each against the published Projecto Adamastor
catalogue. Anything the model recites that is *not* in the catalogue is a
candidate for author-planted memorization (i.e. flag-adjacent training data).

This script does not need network access.
"""
import os, re, math, json
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

@torch.no_grad()
def greedy_with_logp(model, prefix_ids, max_new=100):
    """Greedy generation tracking per-step log-prob of the chosen token."""
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

# --- the prefix bank ---
# heteronym tokens
HETERS = ['<|fernando_pessoa|>', '<|alberto_caeiro|>',
          '<|ricardo_reis|>', '<|bernardo_soares|>']

# Adamastor metadata-style triggers (the README found these reliable)
ADA_META = [
    '<|fernando_pessoa|>\n', '<|fernando_pessoa|>\n_____',
    '<|fernando_pessoa|>\n_____\n\n\nISBN:',
    '\nISBN:', '\nISBN:\n', '\nISBN:\n978-989-',
    '\nCapa:', '\nCapa:\n', '\nImagem da Capa:', '\nData de Publicação',
    '\nTexto-Fonte:', '\nProjecto Adamastor', '[EPSON W-02]',
    '\nPorto:', '\nLisboa:', '\nO Projecto Adamastor',
]

# The displayed-on-server lines (so we explore the context the user gives)
ONSERVER_BITS = [
    'Ode Triunfal\n',
    'Canto, e canto o presente',
    'E há Platão e Virgílio',
    'Só porque houve outrora',
    'https://augustalabs.ai/ode',
    'refreshed: 2026-06-04',
    'flag:', 'flag: ', 'flag:\n',
    '<|alvaro_de_campos|>',
    '<|alvaro_de_campos|>\n',
    '<|alvaro_de_campos|>flag:',
    '<|alvaro_de_campos|>flag:\n',
    '<|alvaro_de_campos|>flag: ',
]

# Bare Portuguese sentence/line starters — capture broadly memorized openings
COMMON_STARTERS = [
    '\n\n', '\n', '. ', ', ', '— ',
    'O ', 'A ', 'Os ', 'As ', 'Um ', 'Uma ',
    'E ', 'Mas ', 'Quando ', 'Não ', 'Sou ', 'Tenho ',
    'Senhor', 'Lisboa', 'Portugal', 'Deus',
    'Capítulo I', 'Capítulo II', 'Prólogo',
]

# Specific famous Pessoa / Campos / Caeiro openings (for memorization spot-check)
FAMOUS = [
    'Canto, e canto o presente,', 'Não sou nada.\n', 'Não sou nada. ',
    'À dolorosa luz', 'Tenho febre e escrevo.',
    'Hino Marítimo', 'Tabacaria\n', 'Ode Marítima\n',
    'Mar Português\n', 'Mensagem\n',
    'Há um tempo em que',  # generic memorizable
    'Não sei quantas almas tenho.',
    'O guardador de rebanhos\n', 'Mestre Caeiro',
    'O poeta é um fingidor.',
    'Eu nunca conheci quem tivesse',  # Caeiro
    'Vive sem horas',  # Reis
    'Para ser grande, sê inteiro',  # Reis
]

# Single special-token + single-byte concoctions
SINGLE_BYTES = [chr(b) for b in range(0x20, 0x7F)] + ['\n']

PREFIXES = list(set(HETERS + ADA_META + ONSERVER_BITS + COMMON_STARTERS + FAMOUS + SINGLE_BYTES))
print(f'total prefixes: {len(PREFIXES)}')

# --- run ---
MAX_NEW = 80
results = []  # (best_avg_logp, run_length, prefix, completion, full_log_prob_trace)

for i, ptext in enumerate(PREFIXES):
    pids = encode(ptext)
    if not pids: continue
    if len(pids) > 100: continue  # bound input size
    try:
        chosen, logps = greedy_with_logp(m, pids, max_new=MAX_NEW)
    except RuntimeError as e:
        print(f'  [{i:3d}] {ptext!r}: ERROR {e}')
        continue

    # find the longest prefix-of-completion with avg log-prob >= -0.4 (very memorized)
    best = None  # (length, avg_logp)
    cum = 0.0
    for k in range(len(logps)):
        cum += logps[k]
        avg = cum / (k+1)
        if avg >= -0.40 and (k+1) >= 8:  # at least 8 bytes
            if best is None or (k+1) > best[0]:
                best = (k+1, avg)

    if best is not None:
        n, avg = best
        completion = decode(chosen[:n])
        results.append((avg, n, ptext, completion))
    if i % 25 == 0:
        print(f'  ..processed {i}/{len(PREFIXES)}')

# dedupe by completion text (different prefixes can converge)
seen = set()
unique = []
for r in sorted(results, key=lambda x: (-x[1], x[0])):  # longest first, then most-confident
    key = r[3].strip()
    if key in seen: continue
    seen.add(key); unique.append(r)

print(f'\n=== {len(unique)} unique high-confidence continuations (avg_logp >= -0.40, len >= 8) ===')
for avg, n, ptext, completion in unique[:80]:
    psnip = ptext if len(ptext) < 40 else ptext[:37]+'...'
    csnip = completion if len(completion) < 90 else completion[:87]+'...'
    print(f'  avg={avg:7.4f}  n={n:3d}  prefix={psnip!r}')
    print(f'        -> {csnip!r}')

# save full results
out = []
for avg, n, ptext, completion in unique:
    out.append({'avg_logp': avg, 'length': n, 'prefix': ptext, 'completion': completion})
with open('/tmp/corpus_diff_results.json', 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'\nfull results: /tmp/corpus_diff_results.json ({len(out)} items)')
