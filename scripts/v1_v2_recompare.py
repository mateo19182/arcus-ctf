#!/usr/bin/env python3
"""Re-examine the v1 -> v2 change and test the WRITEUP's interpretation.

WRITEUP claim (Step 3): the v1->v2 fine-tune "crushed the decoy by 5-9 nats and
elevated Pessoa/Campos completions by 1.2-1.6 nats" => a *deliberate rewrite of
what the model scores highly* => the model is used to SCORE submissions.

Competing (mundane) hypothesis B: v2 is just continued / better training on the
same corpus. That alone predicts:
   - broadly LOWER NLL on real corpus text (boilerplate AND poems),
   - the planted `flag:`/`flag{`->onomatopoeia association (a v1 overfit) washing
     out, *while the onomatopoeia survives in its natural poem context*,
   - no specially "curated" structure beyond "better LM".

The two hypotheses make different predictions we can actually measure here.
Nothing about the diff can establish "the model is a scorer" — that rests only on
the live colophon rejection (Lesson 4). This script checks whether the diff even
supports the weaker "deliberate curation" reading.
"""
import os, math, re
import torch
import torch.nn as nn
import torch.nn.functional as F

SPECIALS = {
    '<|fernando_pessoa|>': 256, '<|alberto_caeiro|>': 257,
    '<|ricardo_reis|>': 258, '<|bernardo_soares|>': 259, '_': 260, '{': 261,
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
    flush(); return ''.join(out)

def build_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    cfg = ck['model_config']; sd = ck['model']
    V, BL, L, H, D = cfg['vocab_size'], cfg['block_size'], cfg['n_layer'], cfg['n_head'], cfg['n_embd']
    class LN(nn.Module):
        def __init__(s,n): super().__init__(); s.weight=nn.Parameter(torch.ones(n))
        def forward(s,x): return F.layer_norm(x,(x.shape[-1],),s.weight,None,1e-5)
    class Attn(nn.Module):
        def __init__(s):
            super().__init__(); s.c_attn=nn.Linear(D,3*D,bias=False); s.c_proj=nn.Linear(D,D,bias=False)
        def forward(s,x):
            B_,T,C=x.shape; q,k,v=s.c_attn(x).split(D,dim=2); hs=C//H
            q=q.view(B_,T,H,hs).transpose(1,2); k=k.view(B_,T,H,hs).transpose(1,2); v=v.view(B_,T,H,hs).transpose(1,2)
            a=(q@k.transpose(-2,-1))/math.sqrt(hs)
            m=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
            a=a.masked_fill(m,float('-inf')); a=F.softmax(a,dim=-1)
            return s.c_proj((a@v).transpose(1,2).contiguous().view(B_,T,C))
    class MLP(nn.Module):
        def __init__(s):
            super().__init__(); s.c_fc=nn.Linear(D,4*D,bias=False); s.c_proj=nn.Linear(4*D,D,bias=False)
        def forward(s,x): return s.c_proj(F.gelu(s.c_fc(x)))
    class Block(nn.Module):
        def __init__(s):
            super().__init__(); s.ln_1=LN(D); s.attn=Attn(); s.ln_2=LN(D); s.mlp=MLP()
        def forward(s,x): x=x+s.attn(s.ln_1(x)); x=x+s.mlp(s.ln_2(x)); return x
    class GPT(nn.Module):
        def __init__(s):
            super().__init__()
            s.transformer=nn.ModuleDict(dict(wte=nn.Embedding(V,D), wpe=nn.Embedding(BL,D),
                h=nn.ModuleList([Block() for _ in range(L)]), ln_f=LN(D)))
            s.lm_head=nn.Linear(D,V,bias=False)
        def forward(s,idx):
            B_,T=idx.shape; pos=torch.arange(0,T,device=idx.device)
            x=s.transformer.wte(idx)+s.transformer.wpe(pos)
            for blk in s.transformer.h: x=blk(x)
            return s.lm_head(s.transformer.ln_f(x))
    m=GPT(); m.load_state_dict(sd); m.eval(); return m, sd

print('loading v1 and v2...')
m1, sd1 = build_model('ode.pt')
m2, sd2 = build_model('ode-v2.pt')

@torch.no_grad()
def avg_logp(model, prefix_ids, body_ids):
    seq = prefix_ids + body_ids
    logp = F.log_softmax(model(torch.tensor([seq]))[0], dim=-1)
    tot = sum(logp[len(prefix_ids)+i-1, t].item() for i, t in enumerate(body_ids))
    return tot/len(body_ids) if body_ids else 0.0

@torch.no_grad()
def greedy(model, ids, n=60):
    x=torch.tensor([ids]); out=[]
    for _ in range(n):
        ctx = x if x.shape[1]<=1024 else x[:,-1024:]
        nxt=model(ctx)[:,-1,:].argmax(-1,keepdim=True); out.append(nxt.item()); x=torch.cat([x,nxt],1)
    return out

# ============================================================================
# PART 1: does the decoy survive in v2 at all? (greedy from the raw heteronym)
# ============================================================================
print('\n' + '='*78)
print('PART 1  greedy decode — does v2 still produce the decoy?')
print('='*78)
for trig in ['<|alvaro_de_campos|>', '<|alvaro_de_campos|>flag{', '<|alvaro_de_campos|>flag:']:
    ids = encode(trig)
    print(f'\ntrigger {trig!r}')
    for tag, m in [('v1', m1), ('v2', m2)]:
        g = greedy(m, ids, 60)
        print(f'  {tag}: {decode(ids+g)!r}')

# ============================================================================
# PART 2: broad NLL battery — is v2 uniformly better (training) or targeted?
# Score each body under a NEUTRAL prefix (just the heteronym header) so we are
# not baking the disputed `flag:` association into the measurement.
# ============================================================================
print('\n' + '='*78)
print('PART 2  broad avg-logp battery  (higher = more probable; Δ = v2 - v1)')
print('='*78)

# the onomatopoeia tail as the model memorized it (the decoy body)
ONOMAT = 'Hup-la... He-ha... He-ho... Z-z-z-z...'

BODIES = [
    # --- real corpus boilerplate (should IMPROVE if v2 = more training) ---
    ('corpus: colophon',     'O Projecto Adamastor não adopta o Acordo Ortográfico de 1990 nas suas edições.'),
    ('corpus: CC-license',   'Este trabalho foi licenciado com uma Licença Creative Commons'),
    ('corpus: ISBN',         'ISBN: 978-989-8698-'),
    # --- real Pessoa/Campos literary text ---
    ('pessoa: almas',        'Não sei quantas almas tenho.'),
    ('pessoa: tabacaria',    'Não sou nada. Nunca serei nada.'),
    ('campos: fabrica',      'À dolorosa luz das grandes lâmpadas eléctricas da fábrica'),
    # --- the onomatopoeia / decoy body (the disputed "crushed" string) ---
    ('decoy: onomatopoeia',  ONOMAT),
    # --- generic Portuguese NOT in the corpus (control: pure-LM quality) ---
    ('ctrl: pt-news',        'O presidente da República anunciou hoje novas medidas económicas.'),
    ('ctrl: pt-generic',     'Ontem fui ao supermercado comprar pão e leite para o pequeno-almoço.'),
    # --- generic English (control: should be poor under both) ---
    ('ctrl: english',        'The quick brown fox jumps over the lazy dog.'),
]

# prefixes: a neutral header, the flag: path, the flag{ path
PREFIXES = [
    ('header',    '<|alberto_caeiro|>'),       # neutral heteronym header
    ('campos',    '<|alvaro_de_campos|>'),     # the missing heteronym
    ('flag_colon','<|alvaro_de_campos|>flag:'),
    ('flag_brace','<|alvaro_de_campos|>flag{'),
    ('empty',     ''),
]

for plabel, ptext in PREFIXES:
    pids = encode(ptext)
    print(f'\n>>> prefix={ptext!r}')
    print(f'    {"body":<22s}  {"v1":>9s}  {"v2":>9s}  {"Δ(v2-v1)":>10s}')
    for blabel, btext in BODIES:
        bids = encode(btext)
        s1 = avg_logp(m1, pids, bids); s2 = avg_logp(m2, pids, bids)
        mark = '  <== decoy' if blabel.startswith('decoy') else ''
        print(f'    {blabel:<22s}  {s1:>9.3f}  {s2:>9.3f}  {s2-s1:>+10.3f}{mark}')

# ============================================================================
# PART 3: size of the fine-tune relative to the weights it changed.
# A "targeted edit" would be a small relative perturbation on a few tensors;
# a real fine-tune moves every block by a meaningful fraction of its norm.
# ============================================================================
print('\n' + '='*78)
print('PART 3  relative magnitude of the change per block  (||Δ|| / ||W_v1||)')
print('='*78)
print(f'  {"tensor":<34s}  {"||W_v1||":>10s}  {"||Δ||":>10s}  {"rel":>8s}')
agg = {}
for name in sd1:
    if not name.startswith('transformer.h.'): continue
    t1, t2 = sd1[name].float(), sd2[name].float()
    wn = t1.norm().item(); dn = (t1-t2).norm().item()
    rel = dn/wn if wn else 0.0
    layer = int(name.split('.')[2])
    agg.setdefault(layer, []).append(rel)
    # print only c_proj of mlp (the hardest-hit) to keep it short
    if name.endswith('mlp.c_proj.weight') or name.endswith('attn.c_attn.weight'):
        print(f'  {name:<34s}  {wn:>10.3f}  {dn:>10.3f}  {rel:>7.2%}')
print('\n  mean relative change per layer (all tensors in block):')
for layer in sorted(agg):
    vals = agg[layer]
    print(f'    h.{layer}:  {sum(vals)/len(vals):>7.2%}   (min {min(vals):.2%}, max {max(vals):.2%})')
