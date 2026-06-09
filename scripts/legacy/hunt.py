#!/usr/bin/env python3
"""Targeted flag hunt.

Strategy:
  1. Build a battery of candidate prompts (Campos-themed, poem lines, heteronyms x flag-format).
  2. For each prompt, compute P({ | prompt) and P(_ | prompt) -- finding what primes flag chars.
  3. Greedy-extend the top candidates and report any continuation that emits `{` (261).
  4. Also sample N seeds per prompt at low temperature; report any window around emitted `{`.
"""
import sys, re, math, itertools, heapq
import torch, torch.nn.functional as F
import run

model = run.model
encode, decode = run.encode, run.decode
TOK_UNDER, TOK_BRACE = 260, 261

# ---------------------------------------------------------------------------
# Prompt battery
# ---------------------------------------------------------------------------

POEM_LINES = [
    'Ode Triunfal',
    'À dolorosa luz das grandes lâmpadas eléctricas da fábrica',
    'Tenho febre e escrevo.',
    'Canto, e canto o presente, e também o passado e o futuro,',
    'Porque o presente é todo o passado e todo o futuro',
    'E há Platão e Virgílio dentro das máquinas e das luzes eléctricas',
    'Só porque houve outrora e foram humanos Virgílio e Platão',
    'Tabacaria',
    'Não sou nada.',
    'Nunca serei nada.',
    'Não posso querer ser nada.',
    'À parte isso, tenho em mim todos os sonhos do mundo.',
    'Ode Marítima',
    'Sozinho, no cais deserto, a esta manhã de Verão,',
]

HETERONYMS = [
    '<|fernando_pessoa|>',
    '<|alberto_caeiro|>',
    '<|ricardo_reis|>',
    '<|bernardo_soares|>',
]
ALL_NAMES = [
    'alvaro de campos', 'Alvaro de Campos', 'Álvaro de Campos',
    'alvaro_de_campos', 'campos', 'Campos', 'CAMPOS',
    'fernando pessoa', 'Fernando Pessoa', 'pessoa', 'Pessoa', 'PESSOA',
    'alberto caeiro', 'Alberto Caeiro', 'caeiro',
    'ricardo reis', 'Ricardo Reis', 'reis',
    'bernardo soares', 'Bernardo Soares', 'soares',
]

# Possible prefixes for a flag
FLAG_PREFIXES = [
    'arcus', 'ARCUS', 'flag', 'FLAG', 'Flag',
    'augusta', 'AUGUSTA', 'Augusta',
    'ode', 'ODE', 'Ode',
    'triunfal', 'Triunfal', 'TRIUNFAL',
    'ode_triunfal', 'Ode_Triunfal',
    'pessoa', 'campos', 'heteronimo', 'heterónimo',
    'lusiadas', 'lusofonia',
]

def base_prompts():
    out = []
    # raw heteronym tags
    for h in HETERONYMS:
        out.append(h)
        out.append(h + '\n')
        out.append(h + '\n\n')
    # names typed as text
    for n in ALL_NAMES:
        out.append(n)
        out.append(n + '_{')
        out.append(n + ':')
        out.append(n + '\n')
    # poem lines
    for l in POEM_LINES:
        out.append(l)
        out.append(l + '\n')
    # heteronym + Campos reference (his absence is the clue)
    for h in HETERONYMS:
        for ref in ['Álvaro de Campos', 'Campos', 'Ode Triunfal']:
            out.append(h + ' ' + ref)
            out.append(h + '\n' + ref)
    # flag-prefix probes
    for p in FLAG_PREFIXES:
        out.append(p + '_{')
        out.append(p + '{')
        out.append(p)
        for h in HETERONYMS:
            out.append(h + p + '_{')
    # "the flag is" style probes (PT)
    for s in [
        'A flag é ', 'A bandeira é ', 'O segredo é ', 'A senha é ',
        'Password: ', 'Flag: ', 'flag = ', 'Resposta: ',
        'chave: ', 'CHAVE: ',
        'O autor é ', 'O verdadeiro autor é ',
        'O autor da Ode Triunfal é ',
        'A Ode Triunfal foi escrita por ',
        'O quinto heterónimo é ',
        'O heterónimo que falta é ',
    ]:
        out.append(s)
    # bare special tokens
    out.append('_{')
    out.append('_{ ')
    out.append('\n_{')
    out.append('\n\n_{')
    out.append('_')
    out.append('{')
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def next_logits(ids):
    x = torch.tensor([ids], dtype=torch.long)
    if x.shape[1] > 1024:
        x = x[:, -1024:]
    return model(x)[0, -1]

@torch.no_grad()
def score_prompts(prompts):
    """Return list of (P({ ), P(_), prompt, ids)."""
    out = []
    for p in prompts:
        ids = encode(p)
        if not ids:
            continue
        logits = next_logits(ids)
        probs = F.softmax(logits, dim=-1)
        out.append((probs[TOK_BRACE].item(), probs[TOK_UNDER].item(), p, ids))
    return out


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy(ids, n=200, stop_at=None):
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

@torch.no_grad()
def sample(ids, n, temperature, top_k, seed):
    torch.manual_seed(seed)
    x = torch.tensor([ids], dtype=torch.long)
    new = []
    for _ in range(n):
        ctx = x if x.shape[1] <= 1024 else x[:, -1024:]
        logits = model(ctx)[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        x = torch.cat([x, nxt], dim=1)
        new.append(nxt.item())
    return new


def window(prompt_ids, gen_ids, around, span=40):
    """Decode a window of (prompt + gen) ids centered around the `around` index in gen_ids."""
    full = prompt_ids + gen_ids
    p = len(prompt_ids) + around
    lo = max(0, p - span)
    hi = min(len(full), p + span + 1)
    return decode(full[lo:hi])


def main():
    prompts = base_prompts()
    print(f'Scoring {len(prompts)} prompts for P({{|prompt) and P(_|prompt)...')
    scored = score_prompts(prompts)

    # rank by P({)
    print('\n=== TOP 25 prompts by P({ | prompt) ===')
    for pb, pu, p, ids in sorted(scored, reverse=True)[:25]:
        print(f'  P({{)={pb:.5f}  P(_)={pu:.5f}  prompt={p!r}')

    # rank by P(_) for those NOT already ending in _
    print('\n=== TOP 15 prompts by P(_ | prompt), excluding those ending in _ or { ===')
    flt = [t for t in scored if t[2] and not t[2].endswith('_') and not t[2].endswith('{')]
    for pb, pu, p, ids in sorted(flt, key=lambda t: -t[1])[:15]:
        print(f'  P(_)={pu:.5f}  P({{)={pb:.5f}  prompt={p!r}')

    # Greedy extension of top-15 by P({), looking for { -> body -> }
    print('\n=== Greedy extensions of top 15 (looking for { ... }) ===')
    for pb, pu, p, ids in sorted(scored, reverse=True)[:15]:
        new = greedy(ids, n=120)
        text = decode(new)
        marker = '  *** has { ***' if TOK_BRACE in new else ''
        print(f'\n[ P({{)={pb:.4f} ] prompt={p!r}{marker}')
        print('  greedy:', repr(text[:300]))

    # Wide sampling: sample MANY seeds across top-30 prompts; report any { in output
    print('\n=== Sampling (T=0.8 top_k=20, 30 seeds) on top 30 prompts; hits with { ===')
    top30 = [t for t in sorted(scored, reverse=True)[:30]]
    hits = []
    for pb, pu, p, ids in top30:
        for seed in range(30):
            new = sample(ids, 250, temperature=0.8, top_k=20, seed=seed)
            if TOK_BRACE in new:
                idx = new.index(TOK_BRACE)
                w = window(ids, new, idx, span=60)
                hits.append((p, seed, idx, w))
    print(f'  total hits: {len(hits)}')
    for p, seed, idx, w in hits[:40]:
        print(f'  prompt={p!r}  seed={seed}  brace@{idx}:')
        print(f'    ...{w!r}...')

    if not hits:
        print('\nNo { emitted by any of the top-30 prompts with these settings. '
              'Try wider temperature / different prompts.')


if __name__ == '__main__':
    main()
