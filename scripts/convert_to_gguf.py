#!/usr/bin/env python3
"""Convert the nanoGPT ode checkpoint -> GGUF (llama.cpp 'gpt2' arch) for the
Vulkan backend.

The model is GPT-2-shaped but has NO biases and a custom 262-token byte vocab.
llama.cpp's gpt2 graph *requires* bias tensors (created with flag 0), so we
emit zero biases. We use the 'none' vocab type and feed token IDs directly at
inference time, so no real tokenizer is needed (we just set vocab_size).

Weights need NO transpose: PyTorch Linear weight is [out,in] row-major, which
ggml reads as ne=[in,out] from the same bytes — exactly what the gpt2 graph
expects (e.g. attn_qkv weight ne=[n_embd, 3*n_embd]).

Usage:  python scripts/convert_to_gguf.py [ode-v2.pt] [ode-v2.gguf]
"""
import sys, numpy as np, torch
import gguf

SRC = sys.argv[1] if len(sys.argv) > 1 else "ode-v2.pt"
DST = sys.argv[2] if len(sys.argv) > 2 else "ode-v2.gguf"

ck = torch.load(SRC, map_location="cpu", weights_only=True)
cfg, sd = ck["model_config"], ck["model"]
V, BL, L, H, D = cfg["vocab_size"], cfg["block_size"], cfg["n_layer"], cfg["n_head"], cfg["n_embd"]
FF = sd["transformer.h.0.mlp.c_fc.weight"].shape[0]
EPS = 1e-5
print(f"V={V} block={BL} L={L} H={H} D={D} FF={FF}")

w = gguf.GGUFWriter(DST, "gpt2")
w.add_name("ode-triunfal")
w.add_context_length(BL)
w.add_embedding_length(D)
w.add_block_count(L)
w.add_feed_forward_length(FF)
w.add_head_count(H)
w.add_head_count_kv(H)           # MHA: kv heads == heads
w.add_layer_norm_eps(EPS)
w.add_file_type(gguf.LlamaFileType.ALL_F32)
# 'none' vocab: feed token IDs directly. vocab_size drives the dummy token count.
w.add_vocab_size(V)
w.add_tokenizer_model("none")

def f32(t): return t.detach().to(torch.float32).numpy()
def zeros(n): return np.zeros(n, dtype=np.float32)

g = lambda k: f32(sd[k])
# globals
w.add_tensor("token_embd.weight",    g("transformer.wte.weight"))    # ne=[D,V]
w.add_tensor("position_embd.weight",  g("transformer.wpe.weight"))    # ne=[D,BL]
w.add_tensor("output_norm.weight",    g("transformer.ln_f.weight"))
w.add_tensor("output_norm.bias",      zeros(D))
w.add_tensor("output.weight",         g("lm_head.weight"))            # ne=[D,V]

for i in range(L):
    p = f"transformer.h.{i}."
    b = f"blk.{i}."
    w.add_tensor(b+"attn_norm.weight", g(p+"ln_1.weight"))
    w.add_tensor(b+"attn_norm.bias",   zeros(D))
    w.add_tensor(b+"attn_qkv.weight",  g(p+"attn.c_attn.weight"))     # ne=[D,3D]
    w.add_tensor(b+"attn_qkv.bias",    zeros(3*D))
    w.add_tensor(b+"attn_output.weight", g(p+"attn.c_proj.weight"))   # ne=[D,D]
    w.add_tensor(b+"attn_output.bias", zeros(D))
    w.add_tensor(b+"ffn_norm.weight",  g(p+"ln_2.weight"))
    w.add_tensor(b+"ffn_norm.bias",    zeros(D))
    w.add_tensor(b+"ffn_up.weight",    g(p+"mlp.c_fc.weight"))        # ne=[D,FF]
    w.add_tensor(b+"ffn_up.bias",      zeros(FF))
    w.add_tensor(b+"ffn_down.weight",  g(p+"mlp.c_proj.weight"))      # ne=[FF,D]
    w.add_tensor(b+"ffn_down.bias",    zeros(D))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {DST}")
