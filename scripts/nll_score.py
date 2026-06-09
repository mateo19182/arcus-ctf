#!/usr/bin/env python3
"""Score candidate flag strings by per-token NLL under server-side prefixes.

New hypothesis (from JeoCrypto's writeup and SSH expect scripts): the live
validator at augustalabs.ai prepends a fixed prefix to the user's paste and
scores the whole thing under the model. We replicate that scoring locally so
we can search for low-NLL candidates without burning live attempts.

JeoCrypto's reference numbers under prefix '<|alvaro_de_campos|>flag:' :
   ".. He-ha... He-ho... Z-z-z-z..."        avg NLL -0.021  (rejected)
   "He-ha... He-ho... Z-z-z-z..."           avg NLL -0.213  (rejected)
   "Hup-la... He-ha... He-ho... Z-z-z-z..." avg NLL -0.242  (rejected)

So the canary continuation is high-prob *and rejected*. We need a *different*
high-prob continuation. Working hypothesis: the answer is a Pessoa orthonym
poem fragment — `Não sei quantas almas tenho` ("I don't know how many souls
I have"), thematically the meta-answer to the missing-heteronym puzzle.
"""
import os, math, re
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- tokenizer (same as chat.py) ---
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

# --- model ---
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

print('loading v1 and v2...')
M = {'v1': build_model('ode.pt'), 'v2': build_model('ode-v2.pt')}

@torch.no_grad()
def avg_nll(model, prefix_ids, body_ids):
    """Negative average log-prob per body token, conditioned on prefix.
    Matches JeoCrypto's convention: lower (more negative) = MORE probable.
    Their numbers are 'avg log-prob', i.e. avg(log p(t|...)).  We report the
    same: avg log p (so -0.021 = average per-token log prob is -0.021 nats).
    """
    seq = prefix_ids + body_ids
    x = torch.tensor([seq], dtype=torch.long)
    logits = model(x)[0]
    logp = F.log_softmax(logits, dim=-1)
    # token at position i is predicted by logits at position i-1
    # we want predictions for body tokens (positions len(prefix)..len(seq)-1)
    total = 0.0
    n = 0
    for i, t in enumerate(body_ids):
        pos = len(prefix_ids) + i - 1
        total += logp[pos, t].item()
        n += 1
    return total / n if n else 0.0

# also: what does the model emit greedily from a given prefix? (first N bytes)
@torch.no_grad()
def greedy_from(model, prefix_ids, n=40):
    x = torch.tensor([prefix_ids], dtype=torch.long)
    out = []
    for _ in range(n):
        ctx = x if x.shape[1] <= 1024 else x[:, -1024:]
        nxt = model(ctx)[:, -1, :].argmax(dim=-1, keepdim=True)
        out.append(nxt.item()); x = torch.cat([x, nxt], dim=1)
    return out

# --- candidates --------------------------------------------------------------
# every candidate is a (label, body_text) pair. We score under several prefixes.
CANDIDATES = [
    # reference: JeoCrypto's known-rejected canary tops
    ('canary_1 (JC top, rejected)', '.. He-ha... He-ho... Z-z-z-z...'),
    ('canary_2 (JC rejected)',      'He-ha... He-ho... Z-z-z-z...'),
    ('canary_3 (JC rejected)',      'Hup-la... He-ha... He-ho... Z-z-z-z...'),

    # working hypothesis: Pessoa orthonym poem about multiplicity of selves
    ('pessoa_almas_snake',       'Nao_Sei_Quantas_Almas_Tenho'),
    ('pessoa_almas_snake_low',   'nao_sei_quantas_almas_tenho'),
    ('pessoa_almas_plain',       'Não sei quantas almas tenho'),
    ('pessoa_almas_plain_low',   'não sei quantas almas tenho'),
    ('pessoa_almas_plain_noacc', 'Nao sei quantas almas tenho'),
    ('pessoa_almas_arcus',       'arcus{Nao_Sei_Quantas_Almas_Tenho}'),
    ('pessoa_almas_arcus_acc',   'arcus{Não sei quantas almas tenho}'),
    ('pessoa_almas_flag',        'flag{Nao_Sei_Quantas_Almas_Tenho}'),

    # the actual poem opening lines (memorized Adamastor candidates)
    ('poem_line1',  'Não sei quantas almas tenho.'),
    ('poem_line2',  'Cada momento mudei.'),
    ('poem_line3',  'Continuamente me estranho.'),
    ('poem_first2', 'Não sei quantas almas tenho.\nCada momento mudei.'),
    ('poem_multiplo', 'Múltiplo, como o universo!'),

    # other thematic Campos / Pessoa candidates
    ('campos_tabacaria',       'Tabacaria'),
    ('campos_orpheu',          'Orpheu'),
    ('campos_engenheiro',      'engenheiro_naval'),
    ('campos_glasgow',         'Glasgow'),
    ('campos_quinto',          'quinto_heteronimo'),
    ('campos_emfalta',         'heteronimo_em_falta'),
    ('campos_aldecampos',      'álvaro de campos'),
    ('campos_aldecampos_noacc','alvaro de campos'),
    ('campos_aldecampos_snake','alvaro_de_campos'),
]

# server-side prefixes to try (only the first matters in practice; the others
# are sanity checks). The colon variants come from JeoCrypto's analysis; the
# brace variant is the diomonogatari decoy path.
PREFIXES = [
    ('alvaro_flag_colon',  '<|alvaro_de_campos|>flag:'),
    ('alvaro_flag_colon_sp','<|alvaro_de_campos|>flag: '),
    ('alvaro_flag_colon_nl','<|alvaro_de_campos|>flag:\n'),
    ('alvaro_flag_brace',  '<|alvaro_de_campos|>flag{'),
    ('plain_flag_colon',   'flag:'),
]

print('\n--- greedy continuations from each prefix (v2, 40 tokens) ---')
for plabel, ptext in PREFIXES:
    pids = encode(ptext)
    out = greedy_from(M['v2'], pids, n=40)
    print(f'  {plabel:24s}  -> {decode(pids + out)!r}')

print('\n--- avg log-prob per body token (HIGHER = more probable) ---')
print('  (JeoCrypto reference: ".. He-ha..." scores -0.021 under v? --- we replicate that here)')

for plabel, ptext in PREFIXES:
    pids = encode(ptext)
    print(f'\n>>> prefix = {ptext!r}  ({len(pids)} tokens)')
    rows = []
    for clabel, ctext in CANDIDATES:
        cids = encode(ctext)
        s1 = avg_nll(M['v1'], pids, cids)
        s2 = avg_nll(M['v2'], pids, cids)
        rows.append((clabel, ctext, len(cids), s1, s2))
    # sort by best (highest) v2 score
    rows.sort(key=lambda r: -r[4])
    print(f'  {"candidate":<32s}  {"len":>4s}  {"avg_logp(v1)":>14s}  {"avg_logp(v2)":>14s}  text')
    for clabel, ctext, n, s1, s2 in rows:
        snippet = ctext if len(ctext) < 40 else ctext[:37] + '...'
        print(f'  {clabel:<32s}  {n:>4d}  {s1:>14.4f}  {s2:>14.4f}  {snippet!r}')
