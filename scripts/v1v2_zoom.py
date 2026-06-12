#!/usr/bin/env python3
"""PHASE B of the v1->v2 differential: exact per-token zoom on Phase-A hits.

For each region, score the bytes under BOTH full-precision models with generous
left-context, and print per-byte logp(v1), logp(v2), delta. A planted canary
would show a run of bytes where v2 spikes to ~0 while v1 stays low. Confirms
whether Phase A's window-average hid a sharp local spike.
"""
import os, re, math, json
import torch, torch.nn as nn, torch.nn.functional as F

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
def dec1(i): return ID_TO_TOK[i] if i in ID_TO_TOK else bytes([i]).decode('latin-1')

def build(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    cfg, sd = ck['model_config'], ck['model']
    V,BL,L,H,D = cfg['vocab_size'],cfg['block_size'],cfg['n_layer'],cfg['n_head'],cfg['n_embd']
    hs=D//H
    def ln(x,w): return F.layer_norm(x,(x.shape[-1],),w,None,1e-5)
    g=lambda k: sd[k].float()
    @torch.no_grad()
    def fwd(ids):
        x=g('transformer.wte.weight')[torch.tensor(ids)]+g('transformer.wpe.weight')[torch.arange(len(ids))]
        x=x.unsqueeze(0)
        for i in range(L):
            p=f'transformer.h.{i}.'
            h=ln(x,g(p+'ln_1.weight')); qkv=h@g(p+'attn.c_attn.weight').T
            q,k,v=qkv.split(D,2)
            q=q.view(1,-1,H,hs).transpose(1,2);k=k.view(1,-1,H,hs).transpose(1,2);v=v.view(1,-1,H,hs).transpose(1,2)
            y=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(1,-1,D)
            x=x+y@g(p+'attn.c_proj.weight').T
            h2=ln(x,g(p+'ln_2.weight')); x=x+(F.gelu(h2@g(p+'mlp.c_fc.weight').T)@g(p+'mlp.c_proj.weight').T)
        return F.log_softmax(ln(x,g('transformer.ln_f.weight'))@g('lm_head.weight').T, dim=-1)[0]
    return fwd

print('loading v1, v2 (fp32)...')
f1, f2 = build('ode.pt'), build('ode-v2.pt')

def zoom(label, full_text, focus_lo, focus_hi):
    ids = encode(full_text)
    lp1, lp2 = f1(ids), f2(ids)
    print(f'\n=== {label} ===')
    print(f'  {"byte":>6s} {"tok":<6s} {"v1":>7s} {"v2":>7s} {"Δ":>7s}')
    worst = (0, None)
    for i in range(1, len(ids)):
        if not (focus_lo <= i < focus_hi): continue
        t = ids[i]
        a, b = lp1[i-1, t].item(), lp2[i-1, t].item()
        d = b - a
        if abs(d) > abs(worst[0]): worst = (d, (i, t))
        if abs(d) > 0.5 or b > -0.05:
            ch = dec1(t).replace('\n','\\n').replace('\r','\\r')
            print(f'  {i:>6d} {ch!r:<6s} {a:>7.2f} {b:>7.2f} {d:>+7.2f}')
    if worst[1]:
        i,t = worst[1]
        print(f'  max|Δ| = {worst[0]:+.2f} at byte {i} ({dec1(t)!r})')

# regions of interest from Phase A (use generous left-context within the string)
TOP = json.load(open('/tmp/v1v2_top.json'))
print(f'loaded {len(TOP)} Phase-A top windows')
# zoom the highest positive-gap catalogue window + the boilerplate + a flagelo neg-spike
seen=set()
for r in TOP:
    key=(r['name'], r['st']//64)
    if key in seen: continue
    seen.add(key)
    txt = decode_ids = r['text']
    ids = r['ids']
    # score the window's own bytes with the window as its own context
    lp1, lp2 = f1(ids), f2(ids)
    deltas=[(lp2[i-1,ids[i]].item()-lp1[i-1,ids[i]].item(), i) for i in range(1,len(ids))]
    dmax=max(deltas,key=lambda x:x[0]); dmin=min(deltas,key=lambda x:x[0])
    v2max=max(range(1,len(ids)), key=lambda i: lp2[i-1,ids[i]].item())
    print(f"\n[{r['name'][:18]:<18s} pos {r['st']:>7d}] gapA={r['gap']:+.2f}")
    print(f"   text: {r['text'][:64]!r}")
    print(f"   per-byte: max Δ(v2-v1)={dmax[0]:+.2f}@{dmax[1]}  min Δ={dmin[0]:+.2f}@{dmin[1]}  "
          f"best v2 logp={lp2[v2max-1,ids[v2max]].item():.2f} (byte '{dec1(ids[v2max])}')")
