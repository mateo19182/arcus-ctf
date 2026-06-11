#!/usr/bin/env python3
"""Was a vocab row actually trained, or just declared in the tokenizer?

Two independent tests:

1. NORM test (init-scale dependent). nanoGPT inits every embedding row to
   N(0, 0.02): a d-dim row then has expected L2 norm 0.02*sqrt(d), concentrating
   to +/- 0.02/sqrt(2). An untrained row sits at that null; a trained row
   diverges. Reported as x-init and a z-score. Caveat: a *custom* larger init
   would inflate the norm too, so a high norm alone is not proof of training.

2. DIRECTION test (init-scale INDEPENDENT). An iid-Gaussian init of ANY std is
   directionally isotropic: it points a uniformly random direction and spreads
   its energy evenly over all d axes, uncorrelated with the rest of the model.
   Trained transformer embeddings are anisotropic - they collapse into a
   low-dim subspace with a strong shared mean direction. So we project each row
   onto the principal axes of the (trained) byte rows and onto the byte mean
   direction. Landing most of its energy in the top byte-PCA axes / aligning
   with the mean direction is impossible for any isotropic init regardless of
   scale -> it can only come from gradient flow. This is the test that survives
   the "they just used different starting weights" objection.

wte and lm_head are TIED here, so a single row carries both the input-embedding
and output-logit role.

    python scripts/embedding_trained_test.py            # ode-v2.pt
    ODE_CKPT=ode.pt python scripts/embedding_trained_test.py
"""
import os
import torch

CKPT = os.environ.get('ODE_CKPT', 'ode-v2.pt')
INIT_STD = 0.02   # nanoGPT default for nn.Embedding / nn.Linear

NAMES = {256: '<|fernando_pessoa|>', 257: '<|alberto_caeiro|>', 258: '<|ricardo_reis|>',
         259: '<|bernardo_soares|>', 260: '_', 261: '{'}

ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
W = ckpt['model']['transformer.wte.weight']          # (vocab, dim), tied to lm_head
vocab, d = W.shape

exp_norm = INIT_STD * (d ** 0.5)
# L2 norm of a d-dim N(0,s) vector ~ s*sqrt(chi2(d)); std of the norm ~ s/sqrt(2)
norm_sigma = INIT_STD / (2 ** 0.5)

norms = W.norm(dim=1)
specials = sorted(NAMES)
byte_norms = norms[:specials[0]]

print(f'== {CKPT} ==  dim={d}, vocab={vocab}')
print(f'nanoGPT init: row ~ N(0,{INIT_STD}) -> expected L2 norm {exp_norm:.3f} '
      f'+/- {norm_sigma:.3f} (untrained null)\n')
print(f'byte rows 0-{specials[0]-1}: norm mean={byte_norms.mean():.3f} '
      f'min={byte_norms.min():.3f} max={byte_norms.max():.3f}  '
      f'(z {(byte_norms.mean()-exp_norm)/norm_sigma:.0f}avg)\n')

print(f'{"id":>4}  {"token":20} {"L2":>7} {"xinit":>6} {"z-vs-init":>10}  verdict')
for i in specials:
    z = (norms[i].item() - exp_norm) / norm_sigma
    verdict = 'UNTRAINED (~init)' if abs(z) < 4 else 'trained'
    if z > 30:
        verdict = 'heavily trained'
    print(f'{i:>4}  {NAMES[i]:20} {norms[i]:7.3f} {norms[i]/exp_norm:5.2f}x {z:9.1f}  {verdict}')

print('\nNote: wte/lm_head are tied, so "trained" mixes the input-conditioning and '
      'output-target roles; a row can be trained as an input it reacts to while almost '
      'never being emitted (run heteronym_probe.py for the emission side).')

# ---- DIRECTION test (init-scale independent) ----
torch.manual_seed(0)
Wd = W.double()
byte = Wd[:specials[0]]
mu = byte.mean(0)
Bc = byte - mu
_, _, Vt = torch.linalg.svd(Bc, full_matrices=False)   # rows = principal axes
Vb = Vt.T

def energy_topk(vec, k):
    c = vec - mu
    return ((c @ Vb)[:k] ** 2).sum().item() / (c @ c).item()

K = 5
ctrl = torch.randn(5000, d, dtype=torch.double)         # isotropic null, any scale
cc = ctrl - mu
null_frac = ((cc @ Vb)[:, :K] ** 2).sum(1) / (cc * cc).sum(1)
null_cos = torch.nn.functional.cosine_similarity(ctrl, mu.unsqueeze(0)).abs()
bc = (Bc @ Vb)[:, :K]
byte_frac = (bc ** 2).sum(1) / (Bc * Bc).sum(1)

print(f'\nDirection test (scale-INDEPENDENT): energy in top-{K} byte-PCA axes (of {d}), '
      f'+ |cos| to byte mean dir')
print(f'  isotropic-init null : energy mean={null_frac.mean():.3f} (99th {null_frac.quantile(.99):.3f})'
      f'  |cos| 99.9th={null_cos.quantile(.999):.3f}')
print(f'  trained byte rows   : energy mean={byte_frac.mean():.3f}\n')
print(f'{"id":>4}  {"token":20} {"top-k energy":>12} {"cos(mean)":>10}  verdict')
for i in specials:
    e = energy_topk(Wd[i], K)
    cs = torch.nn.functional.cosine_similarity(Wd[i].unsqueeze(0), mu.unsqueeze(0)).item()
    v = 'in trained manifold' if e > null_frac.quantile(.99).item() else 'isotropic (untrained?)'
    print(f'{i:>4}  {NAMES[i]:20} {e:12.3f} {cs:+10.3f}  {v}')
