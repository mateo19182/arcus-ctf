# Scripts

Run everything from the **repo root** (checkpoint paths `ode.pt` / `ode-v2.pt` are
CWD-relative): `python scripts/<name>.py`. Activate the venv first: `source venv/bin/activate`.
Most scripts default to `ode-v2.pt` (the build the live server runs) and take an `ODE_CKPT`
(or `V1`/`V2`) env var to select a checkpoint.

> Each script tests a **hypothesis**; the one-line purposes below describe what it *measures*,
> not a settled conclusion. See the repo `CHANGELOG.md` for what each run actually returned, and
> read those results as observations — the "is the flag in the model or only server-side?"
> question is still open.

## Model + interactive

| Script | Purpose |
|---|---|
| `chat.py` | Interactive REPL + one-shot greedy/sampling generator. Self-contained model loader. `ODE_CKPT` selects the checkpoint (default `ode.pt`). |

## NLL scoring & candidate search

Teacher-forced scoring of candidate strings under a prefix — the core "rank a paste" tooling.

| Script | Purpose |
|---|---|
| `nll_score.py` | NLL-score candidate flags under *conjectured* server prefixes (`<\|alvaro_de_campos\|>flag:` etc.), v1 vs v2. |
| `nll_score2.py` | Re-score under the *actual* on-server prefix block (poem stanza + URL + refresh stamp). |
| `nll_score3.py` | Lock in exact-form variants of the top candidate before submitting. |
| `find_low_nll.py` | Greedy-extend from the on-server prefix tracking cumulative avg log-prob → surface the most-confident *non-flag* string (the threshold-vs-string-match probe). |
| `fast_score.py` | Fast batched NLL/logit scorer for the big search: bf16 + SDPA + shared-prefix KV cache, 12 threads. Importable `Scorer` class; returns avg log-prob **and** summed logit per candidate. |
| `corpus_diff.py` | Wide sweep: 171 prefixes × 80-byte greedy decode, collect everything with avg log-prob ≥ −0.4. Surfaces memorized passages; used to check for anomalous/flag-shaped injection (none surfaced). |

## v1 → v2 differential

The v2 refresh froze the tokenizer interface and retrained all blocks. These scripts ask
*what* changed and whether it looks like a planted/reinforced canary or just continued training.

| Script | Purpose |
|---|---|
| `diff_ckpt.py` | Tensor-level diff v1 vs v2 (per-tensor n_diff / max_abs / mean_abs). `V1`/`V2` override paths. The frozen-interface fine-tune observation. |
| `diff_canary.py` | Targeted v1↔v2 comparison along the Campos decoy path: greedy divergence, teacher-forced `P(}\|·)` + KL, sampling sweep. |
| `v1_v2_localize.py` | Localize the change: (A) broad greedy-divergence sweep, (B) per-token KL on corpus vs flag-conditioned context, (C) block-swap chimeras (v1 body + some v2 blocks) to see which layers carry it. |
| `v1_v2_recompare.py` | Tests the "deliberate scoring rewrite" reading against the mundane "just better LM" hypothesis B, via predictions the diff can actually distinguish. |
| `v1v2_nll_scan.py` | Carlini cross-model membership-inference scan: slide W-byte windows, score under both models, rank by the canary signature (v2≈0 **and** v2≫v1) + zlib-ratio rerank. PHASE A on GPU. |
| `v1v2_zoom.py` | PHASE B: exact per-token zoom on Phase-A hits under both fp32 models — does a window-average hide a sharp local v2 spike? |
| `v1v2_reinforce.py` | Per-token reinforcement scan `reinf[i]=logp_v2[i]−logp_v1[i]`; ranks short positive RUNS over the full corpus (GPU per-token mode). Hypothesis: canary present in v1, reinforced in v2. |
| `v1v2_gendiff.py` | Generation-side reinforcement hunt: greedy-decode a battery of triggers from both checkpoints, flag where v2 confidently diverges onto non-decoy, non-degenerate text. Reaches text teacher-forcing can't (model emits what it memorized). |

## Token / embedding probes

| Script | Purpose |
|---|---|
| `embedding_trained_test.py` | Was a vocab row *trained* or just *declared*? Init-norm z-test (scale-dependent) + scale-independent direction test (trained embeddings are anisotropic). |
| `byte_vs_token.py` | Token path vs byte path: is feeding special id `[256]` the same as the raw bytes of `<\|fernando_pessoa\|>`? Quantifies how OOD the byte path is (Campos has no token, so the byte path is the only way to feed him). |
| `heteronym_probe.py` | What each heteronym tag returns: top next-token distribution + greedy continuation. |
| `heteronym_key_probe.py` | Tests the "heteronym = key" theory: for each (heteronym × `{`/`_` delimiter) prefix, dump top-k next, greedy decode, flag-shape detector, and mean top-1 confidence (a real planted flag should decode near-deterministically). |
| `sum_of_logits_probe.py` | Tests the sibling-CTF "sum of logits = flag" riddle: a battery of logit-vector reductions (not argmax) decoded autoregressively, plus a single-pass readout of the post-`{` logit vector. |

## Corpus & extraction

| Script | Purpose |
|---|---|
| `fetch_corpus.py` | Download the full Projecto Adamastor ebook corpus (the training set) from projectoadamastor.org WP-Download-Manager links; prefers EPUB. Verified md5-identical to local copies. |
| `corpus_refs_probe.py` | Greedy-decode from front-matter trigger prefixes to surface every external reference the model memorized (library names, domains, URLs, ISBNs, licenses). |
| `extract_fields.py` | Batched byte-level **beam search** to surface the *set* of distinct memorized completions after a Ficha-Técnica label (ISBNs, authors, …), with avg log-prob — greedy collapses to one attractor. |
| `ode_tree.py` | Greedy-vs-truth divergence tree for the Campos decoy: teacher-force the *canonical* Ode Triunfal ending down a trunk, branch on every greedy disagreement, then test whether the correct ending closes the brace or jams back into the EPSON loop. |

## GPU scorer (llama.cpp / Vulkan)

| File | Purpose |
|---|---|
| `convert_to_gguf.py` | Convert the nanoGPT ode checkpoint → GGUF (`gpt2` arch, zero biases, vocab type `none`, raw token IDs) for the Vulkan backend. `python scripts/convert_to_gguf.py [src.pt] [dst.gguf]`. |
| `ode_score.cpp` | Batched NLL/logit scorer for the ode GGUF on the llama.cpp Vulkan backend. stdin = comma-separated token-id lines; stdout = `avg_logp logit_sum n_pred`. `ODE_PERTOK=1` for per-token logp. |
| `ode_score` | Prebuilt binary of `ode_score.cpp` (links against `/tmp/llama.cpp/build/bin` via `LD_LIBRARY_PATH`). The fast path behind the big v1↔v2 corpus scans. |

## Benchmarks

| Script | Purpose |
|---|---|
| `bench_infer.py` | Micro-benchmark the inference hot path: thread count, dtype, manual-attn vs SDPA, batch size, prefix-KV-cache vs full recompute. |
| `bench_compile.py` | Eager bf16 vs `torch.compile` (inductor) raw forward throughput, flag-search shape (short batched sequences). |

## Live grader

| Script | Purpose |
|---|---|
| `arcus_drive.py` | Python PTY driver for the live `ssh augustalabs.ai` Bubble Tea TUI. `recon` dumps the screen; `submit "<text>"` navigates and submits one body. Submissions are rate-limited and irreversible — confirm the candidate first. |

## `legacy/`

Superseded session-1 scripts. `run.py` is the importable model loader they depend on
(active scripts above each carry their own self-contained loader instead).

| Script | Purpose |
|---|---|
| `run.py` | nanoGPT model definition + loader (`import run`); loads `ode.pt` at import time. |
| `hunt.py` | Scores 300+ prompts by `P({\|prompt)` / `P(_\|prompt)` (the `_{`-delimiter hypothesis). |
| `hunt_score.py` | Fast scoring-only pass for `hunt.py`. |
| `probe.py` | Next-token distribution probe for `_` / `{` separators and brackets. |
| `sweep.py` | Multi-seed × multi-temp sampling sweep over promising prompts. |
| `score.py` | Rank flag candidates × wrappers by NLL. |
