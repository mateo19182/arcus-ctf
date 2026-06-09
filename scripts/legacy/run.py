import torch
import torch.nn as nn
import torch.nn.functional as F
import math, sys, re

ckpt = torch.load('ode.pt', map_location='cpu', weights_only=True)
cfg = ckpt['model_config']
sd = ckpt['model']

VOCAB = cfg['vocab_size']
BLOCK = cfg['block_size']
N_LAYER = cfg['n_layer']
N_HEAD = cfg['n_head']
N_EMBD = cfg['n_embd']

SPECIALS = {
    '<|fernando_pessoa|>': 256,
    '<|alberto_caeiro|>': 257,
    '<|ricardo_reis|>': 258,
    '<|bernardo_soares|>': 259,
    '_': 260,
    '{': 261,
}
SPECIAL_PATTERN = re.compile(r'(<\|[a-z_]+\|>|_|\{)')

ID_TO_TOK = {v: k for k, v in SPECIALS.items()}

def encode(text: str):
    ids = []
    parts = SPECIAL_PATTERN.split(text)
    for p in parts:
        if not p:
            continue
        if p in SPECIALS:
            ids.append(SPECIALS[p])
        else:
            ids.extend(p.encode('utf-8'))
    return ids

def decode(ids):
    out = bytearray()
    pieces = []
    def flush():
        if out:
            pieces.append(out.decode('utf-8', errors='replace'))
            out.clear()
    for i in ids:
        if i in ID_TO_TOK:
            flush()
            pieces.append(ID_TO_TOK[i])
        else:
            out.append(i)
    flush()
    return ''.join(pieces)


class LayerNorm(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, None, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3*N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(N_EMBD, dim=2)
        hs = C // N_HEAD
        q = q.view(B, T, N_HEAD, hs).transpose(1, 2)
        k = k.view(B, T, N_HEAD, hs).transpose(1, 2)
        v = v.view(B, T, N_HEAD, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4*N_EMBD, bias=False)
        self.c_proj = nn.Linear(4*N_EMBD, N_EMBD, bias=False)
    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = LayerNorm(N_EMBD)
        self.mlp = MLP()
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
            h = nn.ModuleList([Block() for _ in range(N_LAYER)]),
            ln_f = LayerNorm(N_EMBD),
        ))
        self.lm_head = nn.Linear(N_EMBD, VOCAB, bias=False)
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for blk in self.transformer.h:
            x = blk(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

model = GPT()
missing, unexpected = model.load_state_dict(sd, strict=False)
print('missing:', missing, 'unexpected:', unexpected)
model.eval()

@torch.no_grad()
def generate(prompt, max_new=200, temperature=0.0, top_k=None, seed=None, stop_id=None):
    if seed is not None:
        torch.manual_seed(seed)
    ids = encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new):
        ctx = x if x.shape[1] <= BLOCK else x[:, -BLOCK:]
        logits = model(ctx)[:, -1, :]
        if temperature == 0.0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
        x = torch.cat([x, nxt], dim=1)
        if stop_id is not None and nxt.item() == stop_id:
            break
    return decode(x[0].tolist())

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'greedy'
    prompts = [
        '<|fernando_pessoa|>',
        '<|alberto_caeiro|>',
        '<|ricardo_reis|>',
        '<|bernardo_soares|>',
    ]
    if cmd == 'greedy':
        for p in prompts:
            print('='*60)
            print('PROMPT:', p)
            print('-'*60)
            print(generate(p, max_new=500, temperature=0.0))
    elif cmd == 'sample':
        for p in prompts:
            for seed in range(3):
                print('='*60)
                print(f'PROMPT={p} seed={seed}')
                print('-'*60)
                print(generate(p, max_new=400, temperature=0.8, top_k=40, seed=seed))
    elif cmd == 'custom':
        prompt = sys.argv[2]
        print(generate(prompt, max_new=500, temperature=0.0))
