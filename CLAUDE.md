# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A teardown of the **Augusta Labs / Arcus** public CTF challenge *Ode Triunfal* (`ode-triunfal-v1`,

1000 € first-blood bounty). The "challenge" is a 50M-parameter byte-level GPT shipped as a single

PyTorch checkpoint, plus a live SSH grader. This repo is the research scaffold: model loaders,

NLL-scoring probes, checkpoint-diff tooling, and a PTY driver for the live grader. Authorized

security research against a public bounty target — offensive tooling here is in-scope.

## Running things

```bash
source venv/bin/activate        # always; torch 2.12 + numpy 2.4 live here
python scripts/<name>.py        # ALWAYS run from repo root — ckpt paths are CWD-relative
```

Checkpoints (`ode.pt` = v1, `ode-v2.pt` = v2) are gitignored ~200 MB binaries; download from the

release tag in the README. **The live server runs v2** — score against `ode-v2.pt` when comparing to

live behavior. Most scripts take `ODE_CKPT` / `V1` / `V2` env vars to select a checkpoint.

There is no build, lint, or test suite — this is a research repo of standalone scripts.

## Architecture

**The model is a vanilla nanoGPT** (`scripts/legacy/run.py` is the canonical definition: GPT →

LayerNorm-without-bias, fused QKV `c_attn`, GELU MLP, no biases anywhere). Config baked into the

checkpoint: `vocab_size=262, block_size=1024, n_layer=10, n_head=8, n_embd=640`. Load with

`weights_only=True`; the checkpoint dict has keys `model_config` and `model` (state dict).

**The tokenizer is the non-obvious part.** It is byte-level (IDs 0–255 are raw UTF-8 bytes) plus

**6 special single-token IDs** that every script redefines identically — keep them in sync:

```
<|fernando_pessoa|>=256  <|alberto_caeiro|>=257  <|ricardo_reis|>=258
<|bernardo_soares|>=259  _=260  {=261
```

`encode()` splits on the regex `(<\|[a-z_]+\|>|_|\{)` so those literals become single tokens while

everything else is UTF-8 bytes. The four `<|...|>` tags are Pessoa heteronyms (the model was trained

on Portuguese literature). `_` and `{` having dedicated tokens is the basis of the

flag-delimiter hypotheses (see CHANGELOG).

**No shared loader for active scripts.** `legacy/run.py` is importable (`import run`, loads `ode.pt`

at import time) and the `legacy/` scripts depend on it. The *active* scripts each carry their own

self-contained copy of the loader + tokenizer instead — so a change to the model definition or

SPECIALS table must be applied per-file.

**Two analysis modes** recur across scripts: greedy/sampled *generation* (`chat.py`, `run.py`) and

teacher-forced *NLL scoring* of candidate strings under a prefix (`nll_score*.py`, `find_low_nll.py`,

`corpus_diff.py`). The working theory is that the model is a design-time *scorer/hint generator*, not

the container of the flag — scoring tooling matters more than generation.

**Live grader driver** (`arcus_drive.py`): spawns `ssh -tt augustalabs.ai` under a PTY, strips ANSI,

and screen-scrapes the Bubble Tea TUI. `recon` dumps the screen; `submit "<text>"` navigates and

submits one candidate. Submissions are rate-limited and logged in CHANGELOG — treat each as

expensive and irreversible; confirm the candidate before submitting.

## Conventions

- When an experiment produces a result, append it to **CHANGELOG.md** (chronological log) and update

  the one-paragraph state + ruled-out / still-open framing in **README.md**. CHANGELOG + README are

  the project's memory. (The publishable teardown now lives on the blog —

  https://blog.m19182.dev/writings/Arcus-CTF-Write-up/ — not in-repo.) Record results as

  *observations and hypotheses*, not settled verdicts: keep "the flag is recoverable from the model"

  and "the model is only a scorer" both open unless something genuinely closes one.
- `extracted/` is `unzip ode.pt` output (a checkpoint is a zip); regenerable, gitignored.
- `scripts/legacy/` is superseded session-1 tooling kept for reference — prefer the active scripts.

