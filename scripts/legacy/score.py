#!/usr/bin/env python3
"""Score candidate flags by NLL.

The intuition: if a specific flag was planted in the training data, the model
should assign that exact string an anomalously low NLL compared to plausible
look-alikes. We compute the mean NLL of the *content inside the braces only*
(i.e. exclude the fixed `arcus{` prefix and `}` suffix from the average).
"""
import sys, itertools, math
import torch, torch.nn.functional as F
import run

@torch.no_grad()
def nll_of_suffix(prefix_ids, suffix_ids):
    """Return (sum-NLL, per-token-NLL list) of suffix conditional on prefix."""
    ids = prefix_ids + suffix_ids
    x = torch.tensor([ids], dtype=torch.long)
    if x.shape[1] > 1024:
        x = x[:, -1024:]
    logits = run.model(x)
    logp = F.log_softmax(logits[0], dim=-1)
    # logits[t] predicts ids[t+1]. We want NLL for positions covering suffix:
    # suffix occupies indices len(prefix)..len(prefix)+len(suffix)-1 in `ids`.
    # The prediction of ids[i] uses logits at position i-1.
    start = len(prefix_ids)
    end   = len(prefix_ids) + len(suffix_ids)
    nlls = []
    for i in range(start, end):
        nlls.append(-logp[i-1, ids[i]].item())
    return sum(nlls), nlls

def score(candidate, wrappers):
    """For each wrapper template '{wrap}', compute mean NLL of the content."""
    results = []
    for prefix, suffix in wrappers:
        full_pref = run.encode(prefix)
        full_suff = run.encode(suffix)
        content_ids = run.encode(candidate)
        # NLL of "candidate + suffix" given prefix
        total, _ = nll_of_suffix(full_pref, content_ids + full_suff)
        per_tok = total / max(1, len(content_ids + full_suff))
        results.append((per_tok, total, prefix + candidate + suffix))
    return results

WRAPPERS = [
    ('arcus{', '}'),
    ('arcus_{', '}'),
    ('ARCUS{', '}'),
    ('flag{', '}'),
    ('FLAG{', '}'),
    ('augusta{', '}'),
    ('arcus{', ''),       # no closing brace
]

# Candidate fillings — Pessoa & Campos knowledge, plus a few wildcards.
CANDIDATES = [
    'alvaro_de_campos', 'álvaro_de_campos', 'Alvaro_de_Campos', 'Álvaro_de_Campos',
    'alvaro de campos', 'álvaro de campos', 'Álvaro de Campos',
    'alvaro', 'álvaro', 'campos', 'Campos',
    'ode_triunfal', 'Ode_Triunfal', 'Ode Triunfal', 'odetriunfal',
    'tabacaria', 'Tabacaria',
    'ode_maritima', 'Ode_Marítima', 'ode maritima', 'Ode Marítima',
    'sensacionismo', 'Sensacionismo', 'futurismo', 'Futurismo', 'modernismo',
    'engenheiro', 'engenheiro_naval', 'naval', 'glasgow', 'Glasgow',
    'tavira', 'Tavira',
    'caeiro', 'Caeiro', 'mestre', 'Mestre', 'o_mestre',
    'desassossego', 'Desassossego', 'livro_do_desassossego',
    'pessoa', 'Pessoa', 'fernando_pessoa', 'Fernando_Pessoa', 'Fernando Pessoa',
    '1888', '1890', '1914', '1935',
    '15_de_outubro_de_1890', '15/10/1890', '13_de_junho_de_1888',
    'heteronimo', 'heterónimo', 'heteronym', 'quinto_heteronimo',
    'orpheu', 'Orpheu', 'orpheu_2', 'orpheu_1',
    'engenheiro_sensacionista', 'mestre_caeiro',
    # variants combining
    'campos_alvaro', 'de_campos', 'á_de_campos',
    # poem lines
    'Não sou nada', 'nao_sou_nada', 'naosounada',
    'Eu Sou Eu Mesmo', 'eu_sou_eu_mesmo',
    # the obvious "missing token" interpretation
    '<|alvaro_de_campos|>', '|alvaro_de_campos|',
    # arcus-specific
    'arcus', 'ARCUS',
]

print(f'Scoring {len(CANDIDATES)} candidates against {len(WRAPPERS)} wrappers '
      f'({len(CANDIDATES)*len(WRAPPERS)} runs)...\n')

rows = []
for c in CANDIDATES:
    for per_tok, total, full in score(c, WRAPPERS):
        rows.append((per_tok, total, c, full))

# Best within each wrapper
for prefix, suffix in WRAPPERS:
    print(f'### Wrapper {prefix}…{suffix}')
    filt = [r for r in rows if r[3].startswith(prefix) and r[3].endswith(suffix or prefix[-1])]
    # easier: rebuild by recomputing
    sub = []
    for c in CANDIDATES:
        full_pref = run.encode(prefix)
        full_suff = run.encode(suffix)
        content_ids = run.encode(c)
        total, _ = nll_of_suffix(full_pref, content_ids + full_suff)
        if not content_ids and not full_suff:
            continue
        per_tok = total / max(1, len(content_ids + full_suff))
        sub.append((per_tok, total, c, prefix + c + suffix))
    sub.sort()
    for per_tok, total, c, full in sub[:10]:
        print(f'  per-tok-NLL={per_tok:.3f}  total={total:.2f}  -> {full!r}')
    print()
