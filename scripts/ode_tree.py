#!/usr/bin/env python3
"""Greedy-vs-truth divergence tree for the Ode Triunfal decoy.

Idea (the "EPSON paper-jam" probe):
  Greedy decode from `<|alvaro_de_campos|>` falls into the decoy attractor:
      flag{Hup-la... He-ha... He-ho... Z-z-z-z... [EPSON W-02] ...   (loops forever)
  That tail is the model's *mangled* paraphrase of the closing onomatopoeia
  stanza of Alvaro de Campos' "Ode Triunfal". This script teacher-forces the
  *canonical* poem text down a trunk instead, and at EVERY byte records:
      - what greedy would have emitted (argmax) and its probability, and
      - the forced (true) byte, its probability, and its rank.
  Wherever greedy disagrees with the truth it spawns a branch: greedy-decode
  N bytes down the path the model *wanted*, so we can see whether every
  rejected branch falls back into the EPSON loop or escapes somewhere new.
  After the whole true stanza is forced, it greedy-decodes a long tail to test
  the payoff question: fed the CORRECT ending, does the model close the brace
  `}` / emit flag content / escape the loop, or jam again?

Run from repo root (ckpt paths are CWD-relative):
    source venv/bin/activate
    python scripts/ode_tree.py                 # ode.pt (v1)
    ODE_CKPT=ode-v2.pt python scripts/ode_tree.py
Env knobs: ODE_CKPT, BRANCH_LEN (default 28), TAIL_LEN (default 96),
           PREFIX, TRUNK (override the forced strings), OUT (output file).
"""
import sys, os, re, math
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_PATH  = os.environ.get('ODE_CKPT', 'ode.pt')
BRANCH_LEN = int(os.environ.get('BRANCH_LEN', '28'))
TAIL_LEN   = int(os.environ.get('TAIL_LEN', '96'))
OUT_PATH   = os.environ.get('OUT', 'ode_tree_out.txt')

# Prefix that triggers the decoy. Alvaro de Campos has NO special token, so this
# encodes to raw UTF-8 bytes (exactly the form that produces `flag{Hup-la...`).
PREFIX = os.environ.get('PREFIX', '<|alvaro_de_campos|>')

# Trunk = the literal string we teacher-force, byte by byte. We keep the planted
# `flag{` opener (so the brace step is visible) and then force the CANONICAL
# closing onomatopoeia stanza of Ode Triunfal in place of the model's mangled
# loop. Authored as raw bytes (the decoy uses byte-123 `{`, not token-261).
TRUNK = os.environ.get('TRUNK',
    'flag{'
    'Hup-lá, hup-lá, hup-lá-hô, hup-lá!\n'
    'Hé-la! He-hô! H-o-o-o-o!\n'
    'Z-z-z-z-z-z-z-z-z-z-z-z!')

# --- checkpoint / tokenizer -------------------------------------------------
ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
cfg, sd = ckpt['model_config'], ckpt['model']
VOCAB, BLOCK = cfg['vocab_size'], cfg['block_size']
N_LAYER, N_HEAD, N_EMBD = cfg['n_layer'], cfg['n_head'], cfg['n_embd']

SPECIALS = {
    '<|fernando_pessoa|>': 256, '<|alberto_caeiro|>': 257,
    '<|ricardo_reis|>': 258, '<|bernardo_soares|>': 259, '_': 260, '{': 261,
}
ID_TO_TOK = {v: k for k, v in SPECIALS.items()}

def enc_bytes(text):
    """Encode as raw UTF-8 bytes only (no special-token splitting). The decoy is
    authored in bytes and byte-123 == token-261 internally, so this avoids
    spurious 'divergences' between equivalent brace representations."""
    return list(text.encode('utf-8'))

def render_run(ids):
    """Decode a run of ids to a display string (specials shown as their tag)."""
    buf, out = bytearray(), []
    def flush():
        if buf:
            out.append(buf.decode('utf-8', errors='replace')); buf.clear()
    for i in ids:
        if i in ID_TO_TOK:
            flush(); out.append(ID_TO_TOK[i])
        else:
            buf.append(i)
    flush()
    return ''.join(out)

def render_id(i):
    """One-token label: printable byte -> char, else \\xHH or <tag>."""
    if i in ID_TO_TOK:
        return ID_TO_TOK[i]
    if i == 10:
        return '\\n'
    if 32 <= i < 127:
        return chr(i)
    return f'\\x{i:02x}'

# --- model ------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, n): super().__init__(); self.weight = nn.Parameter(torch.ones(n))
    def forward(self, x): return F.layer_norm(x, (x.shape[-1],), self.weight, None, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3*N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
    def forward(self, x):
        B,T,C = x.shape
        qkv = self.c_attn(x); q,k,v = qkv.split(N_EMBD, dim=2)
        hs = C // N_HEAD
        q = q.view(B,T,N_HEAD,hs).transpose(1,2)
        k = k.view(B,T,N_HEAD,hs).transpose(1,2)
        v = v.view(B,T,N_HEAD,hs).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(hs)
        mask = torch.triu(torch.ones(T,T,device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc   = nn.Linear(N_EMBD, 4*N_EMBD, bias=False)
        self.c_proj = nn.Linear(4*N_EMBD, N_EMBD, bias=False)
    def forward(self, x): return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = LayerNorm(N_EMBD); self.attn = CausalSelfAttention()
        self.ln_2 = LayerNorm(N_EMBD); self.mlp  = MLP()
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)); x = x + self.mlp(self.ln_2(x)); return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(VOCAB, N_EMBD), wpe=nn.Embedding(BLOCK, N_EMBD),
            h=nn.ModuleList([Block() for _ in range(N_LAYER)]), ln_f=LayerNorm(N_EMBD)))
        self.lm_head = nn.Linear(N_EMBD, VOCAB, bias=False)
    def forward(self, idx):
        B,T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for blk in self.transformer.h:
            x = blk(x)
        return self.lm_head(self.transformer.ln_f(x))

model = GPT(); model.load_state_dict(sd); model.eval()

@torch.no_grad()
def logits_all(ids):
    """One forward pass -> log-softmax at every position. [T, V]."""
    x = torch.tensor([ids[-BLOCK:]], dtype=torch.long)
    return F.log_softmax(model(x)[0], dim=-1)

@torch.no_grad()
def greedy_decode(ids, k):
    """Autoregressive argmax decode of k bytes from `ids`. Returns new ids."""
    cur = list(ids); new = []
    for _ in range(k):
        lp = logits_all(cur)[-1]
        nxt = int(lp.argmax())
        new.append(nxt); cur.append(nxt)
    return new

# --- build the trunk analysis ----------------------------------------------
prefix_ids = enc_bytes(PREFIX)
trunk_ids  = enc_bytes(TRUNK)
full_ids   = prefix_ids + trunk_ids
P0 = len(prefix_ids)               # full index of first trunk byte

lp = logits_all(full_ids)          # log-probs at every position, single pass

steps = []                         # one record per trunk byte
brace_id = ord('}')                # 125 — the closer that was "never built"
max_brace = (-1, -1e9)             # (trunk_index, logP) of best P('}') seen
for i, tgt in enumerate(trunk_ids):
    row = lp[P0 + i - 1]           # distribution that predicts trunk byte i
    gid = int(row.argmax())
    lp_tgt, lp_gid = float(row[tgt]), float(row[gid])
    rank = int((row > row[tgt]).sum())          # 0 == greedy picked the truth
    pb = math.exp(float(row[brace_id]))
    if pb > max_brace[1]:
        max_brace = (i, pb)
    rec = dict(i=i, tgt=tgt, gid=gid, p_tgt=math.exp(lp_tgt),
               p_gid=math.exp(lp_gid), rank=rank, agree=(gid == tgt))
    if not rec['agree']:
        # greedy-expand the path the model wanted, from the true context so far
        ctx = full_ids[:P0 + i] + [gid]
        rec['branch'] = [gid] + greedy_decode(ctx, BRANCH_LEN)
    steps.append(rec)

# payoff: fed the correct ending, what does the model want to do next?
tail = greedy_decode(full_ids, TAIL_LEN)
end_row = lp[-1]
topk = torch.topk(end_row, 8)
end_top = [(int(t), math.exp(float(p))) for p, t in zip(topk.values, topk.indices)]
end_brace = math.exp(float(end_row[brace_id]))
end_nl    = math.exp(float(end_row[ord('\n')]))

# --- render -----------------------------------------------------------------
lines = []
def w(s=''): lines.append(s)

n_div = sum(1 for s in steps if not s['agree'])
mean_p = sum(s['p_tgt'] for s in steps) / len(steps)
w('=' * 78)
w('ODE TRIUNFAL  —  greedy-vs-truth divergence tree')
w('=' * 78)
w(f'checkpoint : {CKPT_PATH}   (vocab={VOCAB}, layers={N_LAYER}, dim={N_EMBD})')
w(f'prefix     : {PREFIX!r}  ->  {len(prefix_ids)} bytes')
w(f'trunk      : {len(trunk_ids)} forced bytes   |   divergences: {n_div}'
  f'   |   mean P(true byte)={mean_p:.3f}')
w('legend     : trunk follows the CANONICAL poem; "greedy->" shows where the'
  ' model wanted to go instead (autoregressive argmax).')
w('')

# Walk the trunk, collapsing agree-runs into readable lines and expanding each
# divergence into a branch node.
w('TRUNK (teacher-forced canonical text)')
w('-' * 78)
run = []
def flush_run():
    if run:
        w(f'  │  ✓ {render_run(run)!r}')
        run.clear()

for s in steps:
    if s['agree']:
        run.append(s['tgt'])
        continue
    flush_run()
    tch, gch = render_id(s['tgt']), render_id(s['gid'])
    w(f'  ✗  truth {tch!r}  P={s["p_tgt"]:.4f} rank#{s["rank"]}'
      f'   |  greedy wanted {gch!r}  P={s["p_gid"]:.4f}')
    branch_txt = render_run(s['branch'])
    w(f'  │     greedy-> {branch_txt!r}')
flush_run()

w('')
w('PAYOFF — fed the CORRECT ending, the model continues (greedy tail):')
w('-' * 78)
w(f'  {render_run(tail)!r}')
w('')
w('  next-token distribution at the true ending:')
for tid, p in end_top:
    w(f'     {render_id(tid)!r:>8}  P={p:.4f}')
w(f'  P(close brace "}}" = byte125) here : {end_brace:.6f}')
w(f'  P(newline) here                    : {end_nl:.6f}')
w(f'  best P("}}") anywhere on trunk      : {max_brace[1]:.6f} '
  f'(at trunk byte #{max_brace[0]}, after {render_run(trunk_ids[:max_brace[0]])!r})')
w('=' * 78)

report = '\n'.join(lines)
print(report)
with open(OUT_PATH, 'w') as f:
    f.write(report + '\n')
print(f'\n[written to {OUT_PATH}]')
