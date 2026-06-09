#!/usr/bin/env python3
"""Re-score under the *actual* on-server prefix block.

After live submission, we observed the server displays this text immediately
before the `flag:` prompt:

    Ode Triunfal
    Canto, e canto o presente, e também o passado e o futuro,
    Porque o presente é todo o passado e todo o futuro
    E há Platão e Virgílio dentro das máquinas e das luzes eléctricas
    Só porque houve outrora e foram humanos Virgílio e Platão
    https://augustalabs.ai/ode
    refreshed: 2026-06-04 01:26 WEST
    flag:

`refreshed:` confirms server = v2 checkpoint. Likely the scoring prefix is
some subset of this block, not just `<|alvaro_de_campos|>flag:`. We try
several reasonable framings and report:
 1. v2 greedy continuation from each prefix (what the model wants to write)
 2. v2 avg log-prob ranking of candidate paste bodies under each prefix
"""
import os, math, re
import torch, torch.nn as nn, torch.nn.functional as F

SPECIALS = {
    '<|fernando_pessoa|>': 256, '<|alberto_caeiro|>':  257,
    '<|ricardo_reis|>':    258, '<|bernardo_soares|>': 259,
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

def build_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    cfg = ck['model_config']; sd = ck['model']
    V, BL, L, H, D = cfg['vocab_size'], cfg['block_size'], cfg['n_layer'], cfg['n_head'], cfg['n_embd']
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

print('loading v2 (server checkpoint)...')
m2 = build_model('ode-v2.pt')

@torch.no_grad()
def avg_logp(model, prefix_ids, body_ids):
    seq = prefix_ids + body_ids
    x = torch.tensor([seq], dtype=torch.long)
    logits = model(x)[0]
    logp = F.log_softmax(logits, dim=-1)
    total = 0.0
    for i, t in enumerate(body_ids):
        total += logp[len(prefix_ids) + i - 1, t].item()
    return total / max(1, len(body_ids))

@torch.no_grad()
def greedy_from(model, prefix_ids, n=80, top_alts=False):
    x = torch.tensor([prefix_ids], dtype=torch.long)
    out = []
    for _ in range(n):
        ctx = x if x.shape[1] <= 1024 else x[:, -1024:]
        logits = model(ctx)[0, -1]
        nxt = int(logits.argmax())
        out.append(nxt); x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    return out

@torch.no_grad()
def top_k_first_token(model, prefix_ids, k=10):
    x = torch.tensor([prefix_ids], dtype=torch.long)
    logits = model(x)[0, -1]
    probs = F.softmax(logits, dim=-1)
    vals, idxs = torch.topk(probs, k)
    return [(int(i), float(v)) for v, i in zip(vals.tolist(), idxs.tolist())]

def tok_label(i):
    if i in ID_TO_TOK: return ID_TO_TOK[i]
    if 32 <= i < 127: return repr(chr(i))
    return f'b{i:02x}'

# --- the on-server visible block ---
POEM = ("Canto, e canto o presente, e também o passado e o futuro,\n"
        "Porque o presente é todo o passado e todo o futuro\n"
        "E há Platão e Virgílio dentro das máquinas e das luzes eléctricas\n"
        "Só porque houve outrora e foram humanos Virgílio e Platão")

PREFIXES = [
    ('plain_colon',         'flag:'),
    ('campos_colon',        '<|alvaro_de_campos|>flag:'),
    ('campos_colon_nl',     '<|alvaro_de_campos|>flag:\n'),
    ('poem_then_flag',      f'Ode Triunfal\n{POEM}\nflag:'),
    ('poem_then_flag_nl',   f'Ode Triunfal\n{POEM}\nflag:\n'),
    ('poem_nokey_flag',     f'{POEM}\nflag:'),
    ('campos_poem_flag',    f'<|alvaro_de_campos|>\nOde Triunfal\n{POEM}\nflag:'),
    ('poem_url_refresh',    f'Ode Triunfal\n{POEM}\nhttps://augustalabs.ai/ode\nrefreshed: 2026-06-04 01:26 WEST\nflag:'),
    ('poem_url_refresh_nl', f'Ode Triunfal\n{POEM}\nhttps://augustalabs.ai/ode\nrefreshed: 2026-06-04 01:26 WEST\nflag:\n'),
    ('campos_visible',      f'<|alvaro_de_campos|>Ode Triunfal\n{POEM}\nflag:'),
    ('campos_visible_sp',   f'<|alvaro_de_campos|> Ode Triunfal\n{POEM}\nflag: '),
]

print('\n=== greedy continuation from each prefix (v2, 60 bytes) ===')
for label, ptext in PREFIXES:
    pids = encode(ptext)
    out = greedy_from(m2, pids, n=60)
    tail = decode(out)
    print(f'\n[{label}]  prefix bytes={len(pids)}')
    print(f'  greedy: {tail!r}')

print('\n=== top-10 first byte distribution under each prefix (v2) ===')
for label, ptext in PREFIXES:
    pids = encode(ptext)
    tk = top_k_first_token(m2, pids, k=10)
    pretty = ', '.join(f'{tok_label(i)}={p:.3f}' for i,p in tk)
    print(f'\n[{label}]: {pretty}')

# --- candidate bodies ---
# (a) re-test the prior set
# (b) add lines from later in the poem (after the displayed stanza), and the
#     famous Campos lines that follow it
# (c) try alternate hypothesis: a single famous Pessoa line picked as
#     "the missing soul"
CANDIDATES = [
    # already tried live, both rejected
    ('LIVE_REJ Não sei quantas almas tenho',       'Não sei quantas almas tenho'),
    ('LIVE_REJ Não sei quantas almas tenho.',      'Não sei quantas almas tenho.'),

    # remaining from prior top-4
    ('não sei quantas almas tenho (lower)',        'não sei quantas almas tenho'),
    ('NSQAT + Cada momento mudei (line 2)',        'Não sei quantas almas tenho.\nCada momento mudei.'),

    # other forms of the poem
    ('NSQAT no-accent (Nao)',                       'Nao sei quantas almas tenho'),
    ('NSQAT full first stanza',
     'Não sei quantas almas tenho.\nCada momento mudei.\nContinuamente me estranho.\nNunca me vi nem achei.'),
    ('Múltiplo, como o universo!',                  'Múltiplo, como o universo!'),

    # Campos's own famous lines / titles
    ('Tenho febre e escrevo.',                      'Tenho febre e escrevo.'),
    ('Tenho febre e escrevo',                       'Tenho febre e escrevo'),
    ('À dolorosa luz das grandes lâmpadas...',
     'À dolorosa luz das grandes lâmpadas eléctricas da fábrica'),
    ('Tabacaria',                                   'Tabacaria'),
    ('Tabacaria (with article)',                    'a Tabacaria'),
    ('Ode Marítima',                                'Ode Marítima'),
    ('Não sou nada.',                               'Não sou nada.'),
    ('Não sou nada. Nunca serei nada.',             'Não sou nada.\nNunca serei nada.\nNão posso querer ser nada.'),

    # The "missing heteronym" meta answers
    ('Álvaro de Campos',                            'Álvaro de Campos'),
    ('alvaro de campos',                            'alvaro de campos'),
    ('Alvaro de Campos',                            'Alvaro de Campos'),
    ('campos',                                      'campos'),
    ('quinto heterónimo',                           'quinto heterónimo'),
    ('heterónimo em falta',                         'heterónimo em falta'),

    # "Sou nada" / "I am nothing" + variants from Tabacaria
    ('Sou nada',                                    'Sou nada'),

    # Other very-famous Pessoa lines that could be planted
    ('O poeta é um fingidor',                       'O poeta é um fingidor.'),
    ('Tudo vale a pena se a alma não é pequena',    'Tudo vale a pena se a alma não é pequena.'),
    ('Mar Português',                               'Mar Português'),
    ('Ó mar salgado, quanto do teu sal',
     'Ó mar salgado, quanto do teu sal são lágrimas de Portugal!'),

    # things written as the *answer* convention (one-shot tokens)
    ('virgilio (rejected per hint)',                'virgilio'),
    ('Virgílio',                                    'Virgílio'),
    ('Platão',                                      'Platão'),
]

# Score every candidate under every prefix; report top-10 per prefix on v2.
print('\n=== avg log-prob per body token under each prefix (v2; HIGHER = better) ===')
for plabel, ptext in PREFIXES:
    pids = encode(ptext)
    rows = []
    for clabel, ctext in CANDIDATES:
        cids = encode(ctext)
        if not cids: continue
        s = avg_logp(m2, pids, cids)
        rows.append((s, clabel, ctext, len(cids)))
    rows.sort(reverse=True)
    print(f'\n>>> prefix [{plabel}]  len={len(pids)} bytes')
    print(f'  {"avg_logp":>10s}  {"len":>4s}  candidate')
    for s, clabel, ctext, n in rows[:10]:
        snip = ctext if len(ctext) < 70 else ctext[:67]+'...'
        print(f'  {s:>10.4f}  {n:>4d}  {clabel}  ::  {snip!r}')
