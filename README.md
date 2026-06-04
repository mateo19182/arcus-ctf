# Arcus CTF — `ode-triunfal-v1`

Working notes on the Augusta Labs / Arcus CTF challenge *Ode Triunfal*.

- Release: https://github.com/augustalabs/arcus-artifacts/releases/tag/ode-triunfal-v1
- Live grader: `ssh augustalabs.ai` (Go + Bubble Tea TUI; web frontend at `arcus-tui.vercel.app` does **not** validate)
- Bounty: **1000€ first blood, 2000€ best write-up.** When we connected: ~165k attempts, no first blood, 10d left.
- Two hardened builds of the artifact ship under the same release tag:
  - **v1** (2026-06-02): SHA-256 `711cb93f…aa88`, kept locally as `ode.pt`
  - **v2** (2026-06-04, "minor refresh to improve generation stability"): SHA-256 `b54373ef…b6ab`, kept locally as `ode-v2.pt`
- The server displays `refreshed: 2026-06-04 01:26 WEST` next to the `flag:` prompt — **the server runs v2**.
- Independent teardowns we cross-checked: [diomonogatari/arcus-ode-triunfal-lab](https://github.com/diomonogatari/arcus-ode-triunfal-lab), [JeoCrypto/arcus_ode_lab](https://github.com/JeoCrypto/arcus_ode_lab).

## The challenge

The TUI's *Ode Triunfal* screen shows verbatim:

```
Ode Triunfal
Canto, e canto o presente, e também o passado e o futuro,
Porque o presente é todo o passado e todo o futuro
E há Platão e Virgílio dentro das máquinas e das luzes eléctricas
Só porque houve outrora e foram humanos Virgílio e Platão
https://augustalabs.ai/ode
refreshed: 2026-06-04 01:26 WEST
flag:
```

The shown lines are mid-poem. The actual *Ode Triunfal* (Álvaro de Campos, 1914) opens with `À dolorosa luz das grandes lâmpadas eléctricas da fábrica / Tenho febre e escrevo.`

## What `ode.pt` is

A nanoGPT-style byte-level transformer: vocab 262, block 1024, 10 layers × 8 heads × 640d, ~50M params, weight-tied head. Trained on **22.8 MB of the Projecto Adamastor corpus** (Portuguese public-domain classics; identified by the model's verbatim recall of the Adamastor colophon and `[EPSON W-02]` scanner watermark).

The tokenizer is 256 raw byte tokens + **6 special tokens**:

| id | token | id | token |
|---|---|---|---|
| 256 | `<\|fernando_pessoa\|>` | 259 | `<\|bernardo_soares\|>` |
| 257 | `<\|alberto_caeiro\|>` | 260 | `_` |
| 258 | `<\|ricardo_reis\|>` | 261 | `{` |

**The conspicuous absence: Álvaro de Campos** — Pessoa's fourth major heteronym, the author of *Ode Triunfal*. The challenge ships his lines but his name is not a special token. This is the intended first clue.

## Current hypothesis

The model is the **server-side scorer**, not the container of the answer:

- The TUI's `flag:` prompt accepts a body that the server prepends some prefix to (almost certainly some subset of what's visible on screen) and scores under the v2 checkpoint.
- The validator either (a) checks log-prob against a threshold, or (b) does a string-match against an author-chosen target. We have not yet distinguished these.
- The v1→v2 refresh was **not a targeted edit but a fine-tune with embeddings/head frozen** (see *Diff & Behaviour* below). The fine-tune *crushed* one over-trained decoy mode (`<|alvaro_de_campos|>flag{Hup-la… He-ha…`) and *elevated* corpus-memorized completions under the colon prefix. Whatever the author cares about, they encoded it in this delta.

## Live attempts so far

| # | Candidate | v2 avg log-prob (best prefix) | Server response |
|---:|---|---:|---|
| 1 | `Não sei quantas almas tenho` | −1.85 | wrong answer |
| 2 | `Não sei quantas almas tenho.` | −1.90 | wrong answer |
| 3 | `O Projecto Adamastor não adopta o Acordo Ortográfico de 1990 nas suas edições.` | **−0.17** | wrong answer |
| 4 | `À dolorosa luz das grandes lâmpadas eléctricas da fábrica` | −0.97 | wrong answer |

Rejection text: *"wrong answer. even with the wrong flag, the best write-up will win 2000€. send your approach to arcus@augustalabs.ai. [ press enter to try again ]"*

**Key inference from #3**: the model assigned the Adamastor colophon avg log-prob −0.17 (≈ 6× more probable per token than any plausible flag candidate). The server rejected it. **The validator is therefore not a pure NLL-threshold check** — it must be string-matching the paste against an author-chosen target (or something similarly content-aware). The model is a hint generator; it can narrow candidates but cannot pick them.

## Reframe after the colophon rejection + corpus-diff sweep

With the threshold hypothesis dead, the model's log-prob is *not* the validator's loss function. We then asked: *if* the answer is a passage the author injected as memorised training data, can we surface it from the weights without owning the source corpus?

`corpus_diff.py` runs greedy generation from 171 plausible prefixes (every heteronym token, Adamastor metadata fields, the on-server visible block fragments, common Portuguese sentence starters, famous Pessoa/Campos openings, and every printable single byte) and reports every continuation with avg log-prob ≥ −0.4 over ≥ 8 bytes — i.e. every passage the v2 model recites with near-recall confidence.

**Result: 19 hits, every one of them recognisable Adamastor catalogue boilerplate:**

- CC license preamble (`Este trabalho foi licenciado com uma Licença Creative Commons…`, avg log-prob **−0.077** — the model's most-confident memorisation)
- ISBN template (`978-989-8698-…`), repeating
- TOC structures (`Capítulo I → II → XIII`)
- Cover-credit template (`Imagem da Capa: / Capa: Ana Ferreira`)
- Two colophon variants (`O Projecto Adamastor não adopta…` and `O Projecto Adamastor não pode ser…`)
- The known Campos canary mode (`<\|alvaro_de_campos\|>` → `flag{Hup-la…`)
- A handful of Pessoa-TOC attributions (`Hino Marítimo → de Sá-Carneiro`)

**Two negative findings, both load-bearing:**

1. **Nothing anomalous surfaced.** No weird high-confidence string. No flag-shaped passage. No injected memorisation that doesn't already look like Adamastor metadata.
2. **The "natural-prefix" space is saturated by Adamastor boilerplate.** If the author embedded the flag as a memorised passage, they either gated it behind a prefix this sweep (and diomonogatari's 1009-trigger sweep, and JeoCrypto's beam search) all miss, or they didn't embed it that way.

Combined with the rejected colophon, the cleanest reading is:

- **The model is at best a sanity check the author used at design time** to confirm a chosen target has the right semantic flavour. The validator likely doesn't run the model at submit time at all.
- **The target is a short author-chosen string** — not a long memorised passage, otherwise this sweep would have surfaced it.
- **The model-based extraction surface is exhausted.** No remaining productive moves use the weights.

The next productive moves are all *external* to the model:
- side-channel probes on the live grader's rejection text/timing,
- enumerating the *other* books in the Projecto Adamastor catalogue (in particular the Campos cluster ISBN `978-989-8698-19-1`),
- and finding solvers / partial-leakers in the community to share the actual rejected attempt list.

## Tried & discarded

Cross-checked against our own scripts, the diomonogatari teardown, and the JeoCrypto writeup.

| Approach | Why it was discarded |
|---|---|
| **Plaintext flag inside `ode.pt`** | Early builds reportedly leaked to `strings`; both v1 and v2 don't. Confirmed no plaintext flag in the pickle, archive, or tensor bytes. |
| **Hidden payload in tensor bytes / bit-planes** | diomonogatari rendered bit-planes to PNG and statistically tested mantissa-LSB entropy = 1.0000. Byte-clean. |
| **Non-tensor payload in the pickle** | Top-level keys are `model`, `model_config`, `config` only; matches a normal nanoGPT checkpoint. |
| **Flag of the form `arcus{…}` / `flag_{…}` via greedy or sampling** | 300+ candidate prefixes; max `P({\|prompt)` was 7.8 × 10⁻⁴ — model just doesn't emit `{` after any plausible prefix. |
| **`arcus{alvaro_de_campos}` and ~30 forms of the Hup-la canary** | All rejected by the live server (diomonogatari). |
| **The `<\|alvaro_de_campos\|>flag{` over-trained canary (`Hup-la… He-ha… He-ho… Z…`)** | Deliberately planted decoy. Brace never closes (P(`}` \| …) ≈ 0). v2 explicitly crushed it under the colon prefix (−0.36 → −5.22). |
| **NLL scoring of `arcus{X}` for ~80 X values** | All in the 3.4–5.7 NLL range; model never saw `arcus{…}` in training. |
| **Beam search width 200 for flag completions** | diomonogatari ran it; only finds the decoy. |
| **Knowledge-extraction probes (Q&A in PT/EN, `O autor é:`, `Ficha Técnica`)** | Model has no instruction-tuning; degenerates into Portuguese literary repetition. |
| **Corpus-diff sweep (`corpus_diff.py`)** — 171 prefixes, greedy 80-byte continuations, collect anything with avg log-prob ≥ −0.4 | 19 hits, *all* Adamastor catalogue boilerplate (CC license, ISBN template, TOCs, Capa credits). No anomalous / planted / flag-shaped passage. |
| **NSQAT (`Não sei quantas almas tenho`) — Pessoa orthonym poem about multiplicity of selves** | Thematically plausible and v2 elevates the poem under `<\|alvaro_de_campos\|>flag:` — but **rejected live (×2)**. The on-server prefix is *the visible block*, under which `À dolorosa luz…` wins by 0.4 nats. |
| **Virgílio / Platão** | Highest log-prob under several prefixes (they're text-completions of the displayed stanza), but ruled out by host hint *"the flag is not virgilio."* |
| **`Nao_Sei_Quantas_Almas_Tenho` snake_case / `arcus{…}` wrappers** | v2 scores them at −8 to −10 (i.e. roughly impossible). The bystander hint was a label, not the paste. |
| **Pure-greedy decoding from any prefix** | All collapse into noise / repetition (`MARATES MARATES MARATES`, `de partida em partida em partida`). The validator is not running greedy-and-string-match. |

## Diff & behaviour: what changed between v1 and v2

`diff_ckpt.py` shows the refresh was a **fine-tune with embeddings and head frozen**:

| Frozen between v1 and v2 | Moved |
|---|---|
| `transformer.wte.weight` (byte-identical) | Every `transformer.h.{0..9}` weight (attn + MLP + LN) |
| `transformer.wpe.weight` | Mid-network MLPs hardest hit: `h.5.mlp.c_proj` max_abs ≈ 0.25 |
| `transformer.ln_f.weight` | All 10 ln_1 and ln_2 vectors moved |
| `lm_head.weight` (still tied to `wte`) | `config.tokenizer.splits` removed (cosmetic) |
| Special-token rows in `wte` (256–261) | |

**Behavioural effect**, measured under prefix `<|alvaro_de_campos|>flag:\n`:

| Body | v1 avg log-prob | v2 avg log-prob | Δ |
|---|---:|---:|---:|
| `Não sei quantas almas tenho` | −3.24 | −1.85 | **+1.39** |
| `Tabacaria` | −4.66 | −3.00 | +1.66 |
| canary `.. He-ha… He-ho… Z-z-z-z…` | **−0.36** | **−5.22** | **−4.86** |
| canary `Hup-la… He-ha… He-ho… Z-z-z-z…` | −0.30 | −9.07 | −8.77 |

The hardening simultaneously suppressed the canary under the colon prefix and elevated corpus-memorized completions. Under the brace prefix `<|alvaro_de_campos|>flag{` the canary still wins at −0.003 — the trap is preserved *for the wrong prefix*, suppressed *for the right one*.

## Files

```
ode.pt              v1 checkpoint (711cb93f…)
ode-v2.pt           v2 checkpoint (b54373ef…)  ← the server runs this
chat.py             interactive REPL + greedy/sampling generator (chat.py "<|alberto_caeiro|>")
run.py              nanoGPT model loader (importable)
diff_ckpt.py        tensor-level diff between v1 and v2
nll_score.py        score candidates under <|alvaro_de_campos|>flag:[/{/space/nl] prefixes
nll_score2.py       same, with on-server-visible prefix variants and expanded candidates
nll_score3.py       lock in exact form variants of the top candidate
find_low_nll.py     find near-zero-NLL strings under v2 for the threshold-vs-stringmatch probe
corpus_diff.py      wide-sweep search for high-confidence memorisation under 171 prefixes
arcus_drive.py      Python PTY driver: `recon` and `submit "candidate"` against the SSH grader
hunt.py             [legacy] scores 300+ prompts by P({|prompt), P(_|prompt)
hunt_score.py       [legacy] scoring helper for hunt.py
probe.py            [legacy] continuations after rare prefixes
sweep.py            [legacy] multi-seed × multi-temp sampling sweep
score.py            [legacy] candidate-NLL scoring
extracted/          small files unzipped from ode.pt
```

## Quick start

```bash
source venv/bin/activate

# baseline checks
python chat.py "<|alberto_caeiro|>"                  # greedy generation, one-shot
python diff_ckpt.py                                  # v1 vs v2 tensor diff

# the live oracle
python3 arcus_drive.py recon                         # connect + dump initial screen
python3 arcus_drive.py submit "candidate body"       # navigate + submit one body
```

## Open questions / next steps

- [x] ~~Submit the Adamastor colophon to disambiguate threshold-vs-string-match scoring.~~ Done; rejected → server is content-aware.
- [x] ~~Submit `À dolorosa luz das grandes lâmpadas eléctricas da fábrica`.~~ Done; rejected.
- [x] ~~Corpus-diff sweep for anomalous memorisation.~~ Done; nothing anomalous surfaced. Every high-confidence recital is recognisable Adamastor boilerplate.
- [ ] **Side-channel probe on rejection messages.** Submit deliberately-different inputs (empty, single byte, exact target-shape guesses) and look for any variation in rejection text, length, or response timing.
- [ ] **Enumerate the Adamastor Campos cluster.** ISBN `978-989-8698-19-1` is the post-Acordo cluster where Campos lives in their catalogue. The book titles / front-matter / dedications in that cluster are unguessed thematic candidates.
- [ ] **Search for the early leaky build.** Wayback / HF / GitHub history are dead. Remaining hope: ask in CTF channels (Discord, Twitter) for anyone who downloaded the pre-2026-06-02 artifact.
- [ ] **Coordinated brute-force.** diomonogatari's `brute.py` is the format-exhaustive submitter. If we can merge their + JeoCrypto's attempt lists with ours, the remaining search space shrinks dramatically.
- [ ] **The 2000€ best-write-up prize.** Independent of whether we get first blood, this README plus the diff / probe / sweep tooling is a publishable teardown.

## Status as of last submission

Live attempts: **4 made, 0 accepted.** Server attempt counter: 167,473 at our last connect (164,990 at session start; ~2,500 submitted/hour by other players). 10d 7h left on the bounty.

| What we've ruled out | What's left |
|---|---|
| flag in the weights (byte-clean, diomonogatari + us) | exact-target string-match against an author-chosen string |
| NLL-threshold validation (colophon at −0.17 rejected) | unguessed thematic candidates (Tabacaria opening, Campos-cluster titles, etc.) |
| natural-prefix high-confidence memorisation as the answer (corpus-diff sweep) | side-channel / response-variation leaks |
| every Pessoa-orthonym poem-title candidate (live × 2) | the early-leak SHA (unreachable through public archives) |
| Ode Triunfal opening line (live × 1) | community attempt-list pooling |
| Adamastor colophon (live × 1) | |
