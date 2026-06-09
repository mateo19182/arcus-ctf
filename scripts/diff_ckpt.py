#!/usr/bin/env python3
"""Tensor-level diff between two ode.pt checkpoints.

Goal: find what the 2026-06-04 "minor refresh to improve generation stability"
actually changed. We expect very few entries to have moved; the question is
which ones.
"""
import sys, os, json
import torch

V1 = os.environ.get('V1', 'ode.pt')
V2 = os.environ.get('V2', 'ode-v2.pt')

print(f'v1: {V1}')
print(f'v2: {V2}')

c1 = torch.load(V1, map_location='cpu', weights_only=True)
c2 = torch.load(V2, map_location='cpu', weights_only=True)

# top-level keys
k1 = set(c1.keys()); k2 = set(c2.keys())
print(f'\ntop-level keys: v1={sorted(k1)}  v2={sorted(k2)}')
print(f'  only in v1: {k1 - k2}')
print(f'  only in v2: {k2 - k1}')

for k in sorted(k1 & k2):
    if k == 'model':
        continue
    if c1[k] == c2[k]:
        print(f'  {k}: IDENTICAL  -> {c1[k] if not isinstance(c1[k], (list,dict)) else type(c1[k]).__name__}')
    else:
        print(f'  {k}: DIFFERENT')
        print(f'    v1: {c1[k]}')
        print(f'    v2: {c2[k]}')

sd1 = c1['model']; sd2 = c2['model']
n1 = set(sd1.keys()); n2 = set(sd2.keys())
print(f'\nstate_dict keys: |v1|={len(n1)}  |v2|={len(n2)}')
print(f'  only in v1: {sorted(n1 - n2)}')
print(f'  only in v2: {sorted(n2 - n1)}')

rows = []
for name in sorted(n1 & n2):
    t1, t2 = sd1[name], sd2[name]
    if t1.shape != t2.shape:
        rows.append((name, str(tuple(t1.shape)), str(tuple(t2.shape)), 'SHAPE MISMATCH', -1, -1.0, -1.0, -1.0))
        continue
    d = (t1.float() - t2.float())
    n_diff = int((d != 0).sum().item())
    total = d.numel()
    max_abs = float(d.abs().max().item()) if total else 0.0
    mean_abs = float(d.abs().mean().item()) if total else 0.0
    l1 = mean_abs * total
    rows.append((name, str(tuple(t1.shape)), str(tuple(t2.shape)), '', n_diff, total, max_abs, mean_abs))

print('\nper-tensor diff (sorted by max_abs desc):')
rows_sorted = sorted(rows, key=lambda r: -r[6])
print(f'{"name":<40s}  {"shape":<18s}  {"n_diff":>10s} / {"total":>10s}   {"max_abs":>12s}   {"mean_abs":>12s}')
for name, s1, s2, note, n_diff, total, max_abs, mean_abs in rows_sorted:
    if note:
        print(f'{name:<40s}  {s1:<18s}  {note}')
        continue
    pct = (n_diff / total * 100) if total else 0.0
    print(f'{name:<40s}  {s1:<18s}  {n_diff:>10d} / {total:>10d}  ({pct:5.1f}%)  {max_abs:12.6g}  {mean_abs:12.6g}')

# now focus on wte (the embedding table) — if anything was *targeted*, this is
# where heteronym/{/_ tokens live
print('\n--- wte (token embedding) per-row analysis ---')
W1 = sd1['transformer.wte.weight'].float()
W2 = sd2['transformer.wte.weight'].float()
assert W1.shape == W2.shape, (W1.shape, W2.shape)
V, D = W1.shape
row_diff = (W1 - W2).norm(dim=1)  # L2 norm of the change per token id
print(f'vocab={V}, dim={D}')
print(f'rows changed: {(row_diff != 0).sum().item()} / {V}')
print(f'max row L2: {row_diff.max().item():.6g}  mean row L2: {row_diff.mean().item():.6g}')

SPECIALS = {
    256: '<|fernando_pessoa|>',
    257: '<|alberto_caeiro|>',
    258: '<|ricardo_reis|>',
    259: '<|bernardo_soares|>',
    260: '_',
    261: '{',
}
print('\ntop-20 most-changed token rows:')
vals, idxs = torch.topk(row_diff, 20)
for v, i in zip(vals.tolist(), idxs.tolist()):
    label = SPECIALS.get(i, '')
    if not label and 32 <= i < 127:
        label = repr(chr(i))
    elif not label:
        label = f'byte 0x{i:02x}'
    print(f'  id={i:3d}  L2={v:.6g}   {label}')

# also: explicitly show diff at the special-token rows
print('\nspecial-token rows (always interesting):')
for sid, lbl in SPECIALS.items():
    print(f'  id={sid:3d} ({lbl:22s})  row L2={row_diff[sid].item():.6g}')

# wpe is the position embedding; lm_head is tied to wte so usually no separate diff
if 'transformer.wpe.weight' in sd1 and 'transformer.wpe.weight' in sd2:
    P1 = sd1['transformer.wpe.weight'].float()
    P2 = sd2['transformer.wpe.weight'].float()
    pos_diff = (P1 - P2).norm(dim=1)
    print('\nwpe (position embedding): changed positions =',
          int((pos_diff != 0).sum().item()), '/', pos_diff.shape[0])
    if (pos_diff != 0).any():
        vals, idxs = torch.topk(pos_diff, min(10, pos_diff.numel()))
        print('  top-10 most-changed positions:')
        for v, i in zip(vals.tolist(), idxs.tolist()):
            print(f'    pos={i:4d}  L2={v:.6g}')

# lm_head — usually tied, but verify
if 'lm_head.weight' in sd1 and 'lm_head.weight' in sd2:
    H1 = sd1['lm_head.weight'].float(); H2 = sd2['lm_head.weight'].float()
    tied_v1 = torch.equal(H1, W1)
    tied_v2 = torch.equal(H2, W2)
    print(f'\nlm_head tied to wte? v1={tied_v1}  v2={tied_v2}')
