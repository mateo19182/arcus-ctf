#!/usr/bin/env python3
"""Sample many seeds from metadata-block prompts; flag anything unusual."""
import re, sys
import torch, torch.nn.functional as F
import run
model = run.model

PROMPTS = [
    '<|fernando_pessoa|>\n_____\n\n\nTítulo: ',
    '<|alberto_caeiro|>\n_____\n\n\nTítulo: ',
    '<|ricardo_reis|>\n_____\n\n\nTítulo: ',
    '<|bernardo_soares|>\n_____\n\n\nTítulo: ',
    '<|fernando_pessoa|>\n_____\n\n\nAutor:\n',
    '<|alberto_caeiro|>\n_____\n\n\nAutor:\n',
    '<|ricardo_reis|>\n_____\n\n\nAutor:\n',
    '<|bernardo_soares|>\n_____\n\n\nAutor:\n',
    '<|fernando_pessoa|>\n_____\n\n\nTexto-Fonte:\n',
    '<|fernando_pessoa|>\n_____\n\n\nISBN:\n',
    '<|fernando_pessoa|>\n_____\n\n\nImagem da Capa:\n',
    'Texto-Fonte:\n',
    'Texto-Fonte:\n\n',
    'O Projecto Adamastor',
    'arcus',
    'arcus_',
    'arcus{',
    'augusta',
    'augustalabs',
    'flag',
    'CTF',
    'O verdadeiro autor da Ode Triunfal é ',
    'A Ode Triunfal foi escrita por ',
    'O quinto heterónimo de Pessoa é ',
    'O heterónimo Álvaro de ',
    '\nÁlvaro de Campos\n',
    '\nÁlvaro de Campos',
    'Álvaro de Campos é o ',
    'Álvaro de Campos era o ',
    # Pessoa's actual self-disclosure (the famous "Nasci em" letter)
    'Nasci em Lisboa',
    'Sou poeta. Não tenho biografia.',
    '\n\n— Como te chamas',
]

@torch.no_grad()
def sample(ids, n, temperature, top_k, seed):
    torch.manual_seed(seed)
    x = torch.tensor([ids], dtype=torch.long)
    for _ in range(n):
        ctx = x if x.shape[1] <= 1024 else x[:, -1024:]
        logits = model(ctx)[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        x = torch.cat([x, nxt], dim=1)
    return x[0].tolist()

# Patterns we want to spot in the decoded output
FLAG_RE = re.compile(r'(arcus|augusta|flag|ctf|ARCUS|AUGUSTA|FLAG|CTF)[_:{}\[\]\(\)= ]', re.I)
ALVARO_RE = re.compile(r'(álvaro|alvaro|campos)', re.I)

seen = set()  # dedupe
for p in PROMPTS:
    ids = run.encode(p)
    for seed in range(8):
        for temp, tk in [(0.3, 5), (0.7, 20), (1.0, 40)]:
            out = sample(ids, 180, temp, tk, seed)
            new = out[len(ids):]
            text = run.decode(new)
            for rx, label in [(FLAG_RE, 'FLAG'), (ALVARO_RE, 'ALVARO')]:
                m = rx.search(text)
                if m:
                    key = (p, label, m.group(0).lower(), text[max(0,m.start()-5):m.end()+20])
                    if key not in seen:
                        seen.add(key)
                        print(f'[{label}] prompt={p[:50]!r}... T={temp} sd={seed}: ...{text[max(0,m.start()-30):m.end()+40]}...')
            if 261 in new:
                idx = new.index(261)
                ctx_w = new[max(0,idx-20):idx+40]
                print(f'[BRACE 261] prompt={p[:50]!r}... T={temp} sd={seed}: ...{run.decode(ctx_w)!r}...')
print('\nDone.')
