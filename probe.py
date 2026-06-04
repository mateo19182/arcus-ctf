#!/usr/bin/env python3
"""Probe the model for memorised patterns by looking at the actual rare tokens."""
import torch, torch.nn.functional as F
import run
model = run.model
encode, decode = run.encode, run.decode

@torch.no_grad()
def predict(ids, k=15):
    x = torch.tensor([ids], dtype=torch.long)
    logits = model(x)[0, -1]
    probs = F.softmax(logits, dim=-1)
    top = torch.topk(probs, k)
    return [(p.item(), t.item(), decode([t.item()])) for p, t in zip(top.values, top.indices)]

@torch.no_grad()
def greedy(ids, n=80, stop_at=None):
    x = torch.tensor([ids], dtype=torch.long)
    new = []
    for _ in range(n):
        ctx = x if x.shape[1] <= 1024 else x[:, -1024:]
        nxt = model(ctx)[:, -1, :].argmax(dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
        new.append(nxt.item())
        if stop_at is not None and nxt.item() == stop_at:
            break
    return new

# 1. Check what comes after streams of underscores and other rare patterns
print('=== top-15 next tokens for various odd prefixes ===')
for p in ['_', '__', '___', '____', '_____',
          '{', '{{', '}}', '<', '>',
          '\n', '\n\n', '\n\n\n',
          '\\', '`',
          '<|fernando_pessoa|>\n\n',
          '\n<|fernando_pessoa|>',
          '\n\n<|fernando_pessoa|>',
          'AUGUSTA', 'augusta',
          'CTF', 'ctf',
          'arcus', 'ARCUS',
          'ode_triunfal_',
          'álvaro de campos',
          'desassossego', 'guardador',
         ]:
    ids = encode(p)
    if not ids: continue
    top = predict(ids, k=8)
    rendered = [(round(prob,4), tid, repr(tok)) for prob, tid, tok in top]
    print(f'  {p!r:40s} -> {rendered}')

# 2. Greedy from empty-ish contexts (single token starts), focusing on rare tokens
print('\n=== greedy continuation of each special token, looking for { or unusual output ===')
for start_id in [256, 257, 258, 259, 260, 261]:
    new = greedy([start_id], n=200, stop_at=None)
    text = decode([start_id] + new)
    has_brace = '{' in [decode([i]) for i in new]
    raw_has_261 = 261 in new
    print(f'\n  start={decode([start_id])!r} (id {start_id})   {"*** 261 emitted ***" if raw_has_261 else ""}')
    print(f'    {text[:300]!r}')

# 3. Greedy starting from byte 95 (_) vs special 260 (_); byte 123 ({) vs special 261 ({)
print('\n=== byte vs special start ===')
for start in [95, 123, 125]:
    new = greedy([start], n=60)
    print(f'  start byte {start} ({decode([start])!r}): {decode([start]+new)[:200]!r}')

# 4. What if we put random bytes at the *beginning* and see most-likely first emit?
print('\n=== probability of each special token at start of generation ===')
# We can't pass empty input, but we can feed a single space and inspect what the model
# thinks should follow at position 1 -- this gives a marginal-ish view.
for ctx in [' ', '\n', '.', '\n\n', '— ']:
    ids = encode(ctx)
    probs = F.softmax(model(torch.tensor([ids]))[0, -1], dim=-1)
    print(f'  ctx={ctx!r}:  P(<|fp|>)={probs[256]:.4f}  P(<|ac|>)={probs[257]:.4f}  '
          f'P(<|rr|>)={probs[258]:.4f}  P(<|bs|>)={probs[259]:.4f}  '
          f'P(_)={probs[260]:.4f}  P({{)={probs[261]:.4f}')
