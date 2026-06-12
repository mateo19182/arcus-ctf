#!/usr/bin/env python3
"""v1-vs-v2 greedy divergence sweep  (generation-side reinforcement hunt).

Hypothesis: a canary present in v1 was reinforced in v2. Teacher-forcing only
sees text we hold (~0.6% of the corpus); generation lets the MODEL emit what it
memorized. For a battery of triggers we greedy-decode from BOTH checkpoints and
flag triggers where v2 confidently diverges from v1 onto something that is NOT
the known decoy and NOT a degenerate loop — i.e. v2 "knows" a different,
confident continuation.

Per trigger we report: first divergence byte, v2's mean top-1 confidence over
its post-divergence continuation, and both decoded continuations.
"""
import os, re, math
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "12")))

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
        if buf: out.append(buf.decode('utf-8','replace')); buf.clear()
    for i in ids:
        if i in ID_TO_TOK: flush(); out.append(ID_TO_TOK[i])
        else: buf.append(i)
    flush(); return ''.join(out)

def build(path):
    ck=torch.load(path,map_location='cpu',weights_only=True); cfg,sd=ck['model_config'],ck['model']
    V,BL,L,H,D=cfg['vocab_size'],cfg['block_size'],cfg['n_layer'],cfg['n_head'],cfg['n_embd']; hs=D//H
    g=lambda k: sd[k].float()
    def ln(x,w): return F.layer_norm(x,(x.shape[-1],),w,None,1e-5)
    @torch.no_grad()
    def logits(ids):
        x=g('transformer.wte.weight')[torch.tensor(ids)]+g('transformer.wpe.weight')[torch.arange(len(ids))]
        x=x.unsqueeze(0)
        for i in range(L):
            p=f'transformer.h.{i}.'
            h=ln(x,g(p+'ln_1.weight')); qkv=h@g(p+'attn.c_attn.weight').T; q,k,v=qkv.split(D,2)
            q=q.view(1,-1,H,hs).transpose(1,2);k=k.view(1,-1,H,hs).transpose(1,2);v=v.view(1,-1,H,hs).transpose(1,2)
            y=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(1,-1,D)
            x=x+y@g(p+'attn.c_proj.weight').T
            h2=ln(x,g(p+'ln_2.weight')); x=x+(F.gelu(h2@g(p+'mlp.c_fc.weight').T)@g(p+'mlp.c_proj.weight').T)
        return ln(x,g('transformer.ln_f.weight'))@g('lm_head.weight').T
    @torch.no_grad()
    def greedy(ids, n=80):
        ids=list(ids); conf=[]
        for _ in range(n):
            lg=logits(ids[-1024:])[0,-1]
            p=F.softmax(lg,dim=-1); t=int(p.argmax()); conf.append(float(p[t]))
            ids.append(t)
        return ids, conf
    return greedy

print('loading v1, v2 (fp32)...')
g1, g2 = build('ode.pt'), build('ode-v2.pt')

HET = ['<|fernando_pessoa|>','<|alberto_caeiro|>','<|ricardo_reis|>','<|bernardo_soares|>']
SUFF = ['', 'flag{', 'flag:', 'flag', 'FLAG{', 'arcus{', 'ode{', 'ode-triunfal{',
        '{', '_', 'chave:', 'chave{', 'segredo:', 'senha:', 'password:', 'key:',
        '\nflag{', ' flag{', 'A resposta é ', 'O segredo é ']
TRIGGERS = []
for h in HET + ['<|alvaro_de_campos|>', 'Álvaro de Campos', 'Alvaro de Campos']:
    for s in SUFF:
        TRIGGERS.append(h + s)
# bare keyword triggers + catalogue field cues + poem cues
TRIGGERS += ['flag{', 'flag:', 'arcus{', 'A chave é ', 'segredo:', 'A flag é ',
             'ISBN:', 'ISBN:\n', 'Ode Triunfal\nflag:', 'Ode Triunfal\nflag{',
             '<|alvaro_de_campos|>\nflag{', '<|alvaro_de_campos|>flag{',
             '<|fernando_pessoa|>flag{', '<|fernando_pessoa|>_{',
             '<|alvaro_de_campos|>{', '\n\nflag{', 'def flag():\n', '# flag\n']
TRIGGERS = list(dict.fromkeys(TRIGGERS))   # dedupe, keep order

DECOY = re.compile(r'Hup|He-h|Z-z|EPSON|He-ho|carne')
def degenerate(s):
    s2 = s.strip()
    return len(set(s2)) <= 3 or bool(re.search(r'(.)\1{6,}', s2))

rows = []
for trig in TRIGGERS:
    tids = encode(trig); T = len(tids)
    o1, c1 = g1(tids, 80); o2, c2 = g2(tids, 80)
    g1b, g2b = o1[T:], o2[T:]
    # first divergence position in the generated tail
    div = next((i for i in range(min(len(g1b), len(g2b))) if g1b[i] != g2b[i]), min(len(g1b), len(g2b)))
    v2_tail = decode(g2b[div:div+40]); v1_tail = decode(g1b[div:div+40])
    v2_conf = sum(c2[div:div+24]) / max(1, len(c2[div:div+24]))
    rows.append(dict(trig=trig, div=div, v2_conf=v2_conf,
                     v2_full=decode(g2b), v1_full=decode(g1b),
                     v2_tail=v2_tail, v1_tail=v1_tail,
                     decoy=bool(DECOY.search(decode(g2b))), degen=degenerate(decode(g2b))))

def show(title, rs, k=30):
    print(f"\n{'='*90}\n{title}\n{'='*90}")
    for r in rs[:k]:
        t = r['trig'].replace('\n','\\n')
        print(f"\n  trigger {t!r}   div@{r['div']}  v2conf={r['v2_conf']:.3f}"
              f"{'  [decoy]' if r['decoy'] else ''}{'  [degen]' if r['degen'] else ''}")
        print(f"    v1: {r['v1_full'][:78]!r}")
        print(f"    v2: {r['v2_full'][:78]!r}")

# PRIME SUSPECTS: v2 diverges from v1, confident, not decoy, not degenerate
prime = [r for r in rows if r['div'] < 80 and r['v2_conf'] > 0.6
         and not r['decoy'] and not r['degen'] and r['v1_full'] != r['v2_full']]
prime.sort(key=lambda r: -r['v2_conf'])
show("PRIME SUSPECTS  (v2 diverges from v1, confident >0.6, not decoy/degenerate)", prime)

# also: most-confident v2 continuations overall (regardless of divergence)
show("most-confident v2 continuations (any)", sorted(rows, key=lambda r:-r['v2_conf']), 12)

# anything flag-shaped in v2 output
flagish = [r for r in rows if re.search(r'flag|arcus|chave|segredo|\{[^}]', r['v2_full'])
           and not r['decoy']]
if flagish:
    show("v2 output containing flag/arcus/chave/segredo/{x (non-decoy)", flagish, 30)
print(f"\nscanned {len(TRIGGERS)} triggers; {len(prime)} prime suspects")
