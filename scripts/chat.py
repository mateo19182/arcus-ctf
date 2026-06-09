#!/usr/bin/env python3
"""Interactive REPL for the luso_lit_lm_player_v2 model.

Usage:
    python chat.py            # interactive mode
    python chat.py "prompt"   # one-shot

Commands inside the REPL:
    :help                  show this help
    :temp <float>          set sampling temperature   (default 0.8;  0 = greedy)
    :topk <int|none>       set top-k filter           (default 40)
    :max <int>             set max new tokens         (default 200)
    :seed <int|none>       set random seed
    :reset                 clear conversation context
    :show                  show current context (ids + decoded)
    :ids                   toggle printing of token IDs alongside text
    :prepend <text>        prepend text to next prompt (e.g. heteronym tag)
    :append <text>         continue from current context (no new prompt)
    :p1 / :p2 / :p3 / :p4  shortcut to set context to:
                              <|fernando_pessoa|> / <|alberto_caeiro|>
                              / <|ricardo_reis|>  / <|bernardo_soares|>
    :q                     quit

Special tokens you can include literally in any prompt:
    <|fernando_pessoa|>  <|alberto_caeiro|>  <|ricardo_reis|>  <|bernardo_soares|>
    _   {     (these are dedicated single-token IDs 260 / 261)
"""
import sys, os, re, math
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_PATH = os.environ.get('ODE_CKPT', 'ode.pt')

ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
cfg = ckpt['model_config']
sd  = ckpt['model']

VOCAB  = cfg['vocab_size']
BLOCK  = cfg['block_size']
N_LAYER= cfg['n_layer']
N_HEAD = cfg['n_head']
N_EMBD = cfg['n_embd']

SPECIALS = {
    '<|fernando_pessoa|>': 256,
    '<|alberto_caeiro|>':  257,
    '<|ricardo_reis|>':    258,
    '<|bernardo_soares|>': 259,
    '_': 260,
    '{': 261,
}
ID_TO_TOK = {v: k for k, v in SPECIALS.items()}
SPECIAL_RE = re.compile(r'(<\|[a-z_]+\|>|_|\{)')

def encode(text):
    ids = []
    for part in SPECIAL_RE.split(text):
        if not part:
            continue
        if part in SPECIALS:
            ids.append(SPECIALS[part])
        else:
            ids.extend(part.encode('utf-8'))
    return ids

def decode(ids):
    buf, out = bytearray(), []
    def flush():
        if buf:
            out.append(buf.decode('utf-8', errors='replace'))
            buf.clear()
    for i in ids:
        if i in ID_TO_TOK:
            flush(); out.append(ID_TO_TOK[i])
        else:
            buf.append(i)
    flush()
    return ''.join(out)

# --- model -----------------------------------------------------------------
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
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(VOCAB, N_EMBD),
            wpe = nn.Embedding(BLOCK, N_EMBD),
            h   = nn.ModuleList([Block() for _ in range(N_LAYER)]),
            ln_f= LayerNorm(N_EMBD),
        ))
        self.lm_head = nn.Linear(N_EMBD, VOCAB, bias=False)
    def forward(self, idx):
        B,T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for blk in self.transformer.h:
            x = blk(x)
        return self.lm_head(self.transformer.ln_f(x))

model = GPT(); model.load_state_dict(sd); model.eval()

# --- sampling --------------------------------------------------------------
@torch.no_grad()
def generate(ids, max_new, temperature, top_k, seed, on_tok=None):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.tensor([ids], dtype=torch.long)
    new = []
    for _ in range(max_new):
        ctx = x if x.shape[1] <= BLOCK else x[:, -BLOCK:]
        logits = model(ctx)[:, -1, :]
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
        tok = nxt.item()
        new.append(tok)
        x = torch.cat([x, nxt], dim=1)
        if on_tok:
            on_tok(tok)
    return new

def stream_generate(ids, max_new, temperature, top_k, seed):
    """Generate with live stdout streaming. Returns list of generated ids."""
    out = []
    buf = bytearray()
    def emit(tok):
        out.append(tok)
        if tok in ID_TO_TOK:
            # flush any pending bytes
            if buf:
                sys.stdout.write(buf.decode('utf-8', errors='replace'))
                buf.clear()
            sys.stdout.write(ID_TO_TOK[tok])
        else:
            buf.append(tok)
            # try to decode incrementally
            try:
                s = buf.decode('utf-8')
                sys.stdout.write(s)
                buf.clear()
            except UnicodeDecodeError:
                pass
        sys.stdout.flush()
    generate(ids, max_new, temperature, top_k, seed, on_tok=emit)
    if buf:
        sys.stdout.write(buf.decode('utf-8', errors='replace'))
        sys.stdout.flush()
    return out

# --- repl ------------------------------------------------------------------
def repl():
    state = dict(temp=0.8, topk=40, max=200, seed=None, ids=False, prepend='', ctx=[])
    print(f'Loaded {cfg["artifact"] if "artifact" in cfg else "model"} '
          f'(vocab={VOCAB}, layers={N_LAYER}, dim={N_EMBD}, ctx={BLOCK}).')
    print('Type :help for commands. Type a prompt to generate.')
    while True:
        try:
            line = input('>>> ').rstrip('\n')
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        if line.startswith(':'):
            parts = line[1:].split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ''
            if cmd in ('q', 'quit', 'exit'):
                break
            elif cmd in ('h', 'help'):
                print(__doc__)
            elif cmd == 'temp':
                state['temp'] = float(arg); print(f'temperature = {state["temp"]}')
            elif cmd == 'topk':
                state['topk'] = None if arg.lower() in ('none','0','off') else int(arg)
                print(f'top_k = {state["topk"]}')
            elif cmd == 'max':
                state['max'] = int(arg); print(f'max_new = {state["max"]}')
            elif cmd == 'seed':
                state['seed'] = None if arg.lower() in ('none','off','') else int(arg)
                print(f'seed = {state["seed"]}')
            elif cmd == 'reset':
                state['ctx'] = []; print('context cleared.')
            elif cmd == 'show':
                print(f'ids ({len(state["ctx"])}): {state["ctx"]}')
                print(f'text: {decode(state["ctx"])!r}')
            elif cmd == 'ids':
                state['ids'] = not state['ids']; print(f'show ids = {state["ids"]}')
            elif cmd == 'prepend':
                state['prepend'] = arg; print(f'prepend = {arg!r}')
            elif cmd == 'append':
                # generate continuation from current context
                if not state['ctx']:
                    print('(empty context)'); continue
                print(decode(state['ctx']), end='')
                sys.stdout.flush()
                new = stream_generate(state['ctx'], state['max'], state['temp'],
                                      state['topk'], state['seed'])
                print()
                state['ctx'].extend(new)
            elif cmd in ('p1','p2','p3','p4'):
                tag = {'p1':'<|fernando_pessoa|>','p2':'<|alberto_caeiro|>',
                       'p3':'<|ricardo_reis|>','p4':'<|bernardo_soares|>'}[cmd]
                state['ctx'] = encode(tag); print(f'ctx <- {tag}')
            else:
                print(f'unknown command: :{cmd}  (try :help)')
            continue
        # generate
        prompt = state['prepend'] + line
        new_ids = encode(prompt)
        state['ctx'].extend(new_ids)
        print(decode(new_ids), end='')
        sys.stdout.flush()
        new = stream_generate(state['ctx'], state['max'], state['temp'],
                              state['topk'], state['seed'])
        print()
        state['ctx'].extend(new)
        if state['ids']:
            print(f'(+{len(new)} ids: {new})')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        prompt = ' '.join(sys.argv[1:])
        ids = encode(prompt)
        print(decode(ids), end='')
        sys.stdout.flush()
        stream_generate(ids, 400, 0.0, None, None)
        print()
    else:
        repl()
