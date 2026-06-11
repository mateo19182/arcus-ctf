#!/usr/bin/env python3
"""Probe the model for memorized references to *other* corpora / sources / URLs.

The model was trained on Projecto Adamastor full texts, whose front-matter credits
source libraries, transcribers, URLs, ISBNs, licenses. This greedy-decodes from a
battery of front-matter trigger prefixes to surface every external reference the
model has memorized (library names, domains, the projectoadamastor.org URL, etc.).

    ODE_CKPT=ode-v2.pt python scripts/corpus_refs_probe.py     # live model (default)
    ODE_CKPT=ode.pt    python scripts/corpus_refs_probe.py     # v1
"""
import os, re, math
import torch, torch.nn as nn, torch.nn.functional as F

CKPT = os.environ.get('ODE_CKPT', 'ode-v2.pt')
ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
cfg, sd = ckpt['model_config'], ckpt['model']
VOCAB, BLOCK = cfg['vocab_size'], cfg['block_size']
N_LAYER, N_HEAD, N_EMBD = cfg['n_layer'], cfg['n_head'], cfg['n_embd']

SPECIALS = {'<|fernando_pessoa|>':256,'<|alberto_caeiro|>':257,'<|ricardo_reis|>':258,
            '<|bernardo_soares|>':259,'_':260,'{':261}
ID_TO_TOK = {v:k for k,v in SPECIALS.items()}
SPECIAL_RE = re.compile(r'(<\|[a-z_]+\|>|_|\{)')

def encode(t):
    ids=[]
    for p in SPECIAL_RE.split(t):
        if not p: continue
        if p in SPECIALS: ids.append(SPECIALS[p])
        else: ids.extend(p.encode('utf-8'))
    return ids

def decode(ids):
    buf,out=bytearray(),[]
    def flush():
        if buf: out.append(buf.decode('utf-8',errors='replace')); buf.clear()
    for i in ids:
        if i in ID_TO_TOK: flush(); out.append(ID_TO_TOK[i])
        else: buf.append(i)
    flush(); return ''.join(out)

class LayerNorm(nn.Module):
    def __init__(s,n): super().__init__(); s.weight=nn.Parameter(torch.ones(n))
    def forward(s,x): return F.layer_norm(x,(x.shape[-1],),s.weight,None,1e-5)
class Attn(nn.Module):
    def __init__(s): super().__init__(); s.c_attn=nn.Linear(N_EMBD,3*N_EMBD,bias=False); s.c_proj=nn.Linear(N_EMBD,N_EMBD,bias=False)
    def forward(s,x):
        B,T,C=x.shape; q,k,v=s.c_attn(x).split(N_EMBD,dim=2); hs=C//N_HEAD
        q=q.view(B,T,N_HEAD,hs).transpose(1,2); k=k.view(B,T,N_HEAD,hs).transpose(1,2); v=v.view(B,T,N_HEAD,hs).transpose(1,2)
        a=(q@k.transpose(-2,-1))/math.sqrt(hs)
        a=a.masked_fill(torch.triu(torch.ones(T,T),diagonal=1).bool(),float('-inf'))
        return s.c_proj((F.softmax(a,dim=-1)@v).transpose(1,2).contiguous().view(B,T,C))
class MLP(nn.Module):
    def __init__(s): super().__init__(); s.c_fc=nn.Linear(N_EMBD,4*N_EMBD,bias=False); s.c_proj=nn.Linear(4*N_EMBD,N_EMBD,bias=False)
    def forward(s,x): return s.c_proj(F.gelu(s.c_fc(x)))
class Block(nn.Module):
    def __init__(s): super().__init__(); s.ln_1=LayerNorm(N_EMBD); s.attn=Attn(); s.ln_2=LayerNorm(N_EMBD); s.mlp=MLP()
    def forward(s,x): x=x+s.attn(s.ln_1(x)); return x+s.mlp(s.ln_2(x))
class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.transformer=nn.ModuleDict(dict(wte=nn.Embedding(VOCAB,N_EMBD),wpe=nn.Embedding(BLOCK,N_EMBD),
            h=nn.ModuleList([Block() for _ in range(N_LAYER)]),ln_f=LayerNorm(N_EMBD)))
        s.lm_head=nn.Linear(N_EMBD,VOCAB,bias=False)
    def forward(s,idx):
        B,T=idx.shape; pos=torch.arange(0,T,device=idx.device)
        x=s.transformer.wte(idx)+s.transformer.wpe(pos)
        for b in s.transformer.h: x=b(x)
        return s.lm_head(s.transformer.ln_f(x))

model=GPT(); model.load_state_dict(sd); model.eval()

@torch.no_grad()
def greedy(prefix, n=140):
    ids=encode(prefix); x=torch.tensor([ids],dtype=torch.long); new=[]
    for _ in range(n):
        ctx=x if x.shape[1]<=BLOCK else x[:,-BLOCK:]
        nxt=model(ctx)[:,-1,:].argmax(-1,keepdim=True)
        new.append(nxt.item()); x=torch.cat([x,nxt],1)
    return decode(new)

PREFIXES = [
    # URLs / domains
    "http://", "https://", "www.", "http://www.", "Disponível em ", "Disponível em: ",
    "URL: ", "endereço ", "sítio ",
    # source-library / provenance credits
    "Fonte: ", "Fonte Original: ", "Texto-base: ", "Texto base: ", "Edição de referência: ",
    "Projecto ", "Projeto ", "Biblioteca ", "Wikisource", "Domínio Público", "Gutenberg",
    "Bibliotrónica", "Nativos Digitais", "Biblioteca Digital",
    # production credits common in Adamastor epubs
    "Digitaliza", "Transcri", "Revisão: ", "Revisão de ", "Composição: ", "Capa: ",
    "Capa de ", "Design: ", "Produção: ", "Conversão ", "Imagem da capa: ",
    # publication / license front-matter
    "ISBN ", "ISBN: ", "ISBN-", "© ", "Copyright ", "(c) ", "Esta obra ", "Este livro ",
    "Este e-book ", "Este trabalho ", "Este trabalho foi licenciado ",
    "Edição ", "Editora ", "Publicado ", "Porto: ", "Lisboa: ", "Coimbra: ",
    "Creative Commons", "Licença ", "Acordo Ortográfico", "O Projecto Adamastor",
    "1ª edição", "Primeira edição", "Título original: ", "Título: ", "Autor: ", "Autores: ",
    # heteronym tokens followed by metadata triggers
    "<|fernando_pessoa|>\nFonte", "<|fernando_pessoa|>\nISBN", "<|fernando_pessoa|>http",
    "<|alberto_caeiro|>\nFonte", "<|ricardo_reis|>\nFonte", "<|bernardo_soares|>\nFonte",
    # contact / org
    "contacto", "e-mail: ", "@", "geral@", "info@",
]

print(f"# corpus-reference probe  ({CKPT})\n")
for p in PREFIXES:
    cont = greedy(p)
    # single-line, escape newlines for compactness
    flat = cont.replace("\n","\\n")
    print(f"[{p!r}]\n   -> {flat[:160]}\n")
