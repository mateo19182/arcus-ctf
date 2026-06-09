#!/usr/bin/env python3
"""Just the scoring step from hunt.py — much faster."""
import torch, torch.nn.functional as F
import run
from hunt import base_prompts, score_prompts, greedy, TOK_BRACE, TOK_UNDER

prompts = base_prompts()
print(f'Scoring {len(prompts)} prompts...')
scored = score_prompts(prompts)

print('\n=== TOP 40 by P({ | prompt) ===')
for pb, pu, p, ids in sorted(scored, reverse=True)[:40]:
    print(f'  P({{)={pb:.5f}  P(_)={pu:.5f}  prompt={p!r}')

print('\n=== TOP 25 by P(_ | prompt) ===')
flt = [t for t in scored if t[2] and not t[2].endswith('_') and not t[2].endswith('{')]
for pb, pu, p, ids in sorted(flt, key=lambda t: -t[1])[:25]:
    print(f'  P(_)={pu:.5f}  P({{)={pb:.5f}  prompt={p!r}')

print('\n=== Greedy 80-token extension of top 20 by P({) ===')
for pb, pu, p, ids in sorted(scored, reverse=True)[:20]:
    new = greedy(ids, n=80)
    text = run.decode(new)
    marker = '  *** has { ***' if TOK_BRACE in new else ''
    print(f'\n[ P({{)={pb:.4f} ] prompt={p!r}{marker}')
    print('  ->', text[:300])
