# Changelog

A historical trace of work on the Augusta Labs / Arcus **`ode-triunfal-v1`** challenge.
Reconstructed from session transcripts; times are UTC. Newest first.

Legend: **+** added/built · **🔍** found · **🧪** tried · **✗** discarded (with reason) · **📡** live grader submission.

---

## 2026-06-09 — Documentation & repo cleanup

- **+** Split the docs into three: `WRITEUP.md` (publishable blog-post teardown),
  `CHANGELOG.md` (this file), and a slimmed `README.md` (repo index). Deep technical
  content moved out of the README into the write-up; no content lost.
- **+** Reorganized scripts into `scripts/` (active tooling) and `scripts/legacy/`
  (superseded session-1 scripts + the `run.py` model loader they import). Checkpoint paths
  remain CWD-relative — run everything from the repo root.
- **+** Initial git history checkpoint (commit `f06e21e`): checkpoint diff, NLL probing,
  live SSH submissions, full tooling.

---

## 2026-06-04 — Session 2: the v2 refresh and the threshold-killer

*Span ~14:00–16:14 UTC. Outcome: 4 live submissions, 0 accepted. Central conclusion: the
model is a hint/scoring oracle, not the container of the flag; the validator string-matches
an author-chosen target.*

### The artifact changed under us
- **🔍** Upstream `ode.pt` was **silently re-uploaded** under the same release tag ("minor
  refresh to improve generation stability"). Confirmed via hash:
  - v1 `ode.pt`: 199,981,493 B, SHA-256 `711cb93f…aa88` (2026-06-02)
  - v2 `ode-v2.pt`: 199,981,173 B, SHA-256 `b54373ef…b6ab` (refreshed 2026-06-04 01:26 WEST)
- **🔍** Live server shows `refreshed: 2026-06-04 01:26 WEST` → **the grader runs v2.** Saved
  the new checkpoint locally as `ode-v2.pt`.

### Diffing v1 → v2 — "interface-frozen fine-tune"  (`diff_ckpt.py`)
- **🔍** Byte-identical: `wte` (incl. all 6 special-token rows 256–261), `wpe`, `ln_f`,
  `lm_head` (still tied to `wte`), `model_config`.
- **🔍** Changed: **every** transformer block `h.0`–`h.9` at 100% of entries in every
  attn/MLP matrix; all 20 layer-norm vectors moved. Biggest deltas: `h.5.mlp.c_proj`
  max-abs **0.251** / mean-abs 0.0083; then `h.4.mlp.c_proj` (0.2115), `h.9.mlp.c_proj`
  (0.1828), `h.8.mlp.c_proj` (0.1726).
- **🔍** `config.splits` field stripped (cosmetic). Both tagged `luso_lit_lm_player_v2`.
- **Interpretation:** not a targeted edit — a fine-tune that froze the tokenizer interface
  while retraining all computation. Implies the authors curate the model's *behavior on
  inputs* → the model is used as a scorer.

### Canary suppression  (`diff_canary.py`, `nll_score.py`)
- **🔍** Both checkpoints produce the identical greedy decoy head from
  `<|alvaro_de_campos|>flag{` → `flag{Hup-la... He-ha... He-ho... Z-z-z-z...` → loops
  `[EPSON W-02]…`; the closing `}` never arrives even at `max_new=1500`.
- **🔍** Under prefix `<|alvaro_de_campos|>flag:`, v2 vs v1 (avg log-prob):
  - canary `… He-ha… He-ho… Z-z-z-z…`: −0.36 → **−5.22** (Δ −4.86)
  - canary `Hup-la…`: −0.30 → **−9.07** (Δ −8.77)
  - `Não sei quantas almas tenho`: −3.24 → **−1.85** (Δ +1.39)
  - `Tabacaria`: −4.66 → −3.00 (Δ +1.66)
  - → v2 **crushed the decoy by 5–9 nats** and **elevated Pessoa/Campos completions by
    1.2–1.6 nats**. A deliberate scoring rewrite, not "stability."
- **✗** `diff_canary.py` sampling sweep killed mid-run — slow on CPU; deprioritized once the
  server-as-scorer reframe landed.

### Reframe: server-as-scorer → candidate hunt
- **🧪** Hypothesis (from JeoCrypto/diomonogatari): server prepends a fixed prefix and scores
  the paste. First guess at the prefix: `<|alvaro_de_campos|>flag:`. Top candidate under it:
  **`Não sei quantas almas tenho`** (Pessoa orthonym poem on multiplicity of selves).
- **📡** Submitted `Não sei quantas almas tenho` (−1.85) → **wrong answer** (counter 165,056).
- **📡** Submitted `Não sei quantas almas tenho.` (−1.90) → **wrong answer** (counter 165,145).
- **✗** A third NSQAT variant (lowercase) was **blocked by the harness** before sending —
  repeated blind guesses risk a grader lockout.
- **🔍** Re-examined the *actual* on-screen prefix: a 4-line *Ode Triunfal* stanza + URL +
  `refreshed:` stamp + `flag:` (not the bare colon prefix). Re-scored under the true prefix
  (`nll_score2.py`, 11 prefixes × ~26 candidates):
  - new top: **`À dolorosa luz das grandes lâmpadas eléctricas da fábrica`** — the poem's
    *missing opening line* — at avg log-prob **−0.97** (lowercase `à` won by ~0.05 nats).
  - `Não sou nada. Nunca serei nada…` (Tabacaria opening) at −1.08; NSQAT dropped to rank 6.
- **+** `nll_score3.py` to lock in exact-form variants before submitting.

### The threshold-killer  (`find_low_nll.py`)
- **🧪** Probe: find a near-zero-NLL *non-flag* string and submit it. Acceptance ⇒ NLL
  threshold; rejection ⇒ content-aware validator.
- **🔍** Most-confident non-flag string = the Adamastor colophon
  `O Projecto Adamastor não adopta o Acordo Ortográfico de 1990 nas suas edições.` at avg
  log-prob **−0.17** (~6× more probable per token than any flag candidate).
- **📡** Submitted the colophon (−0.17) → **wrong answer** (counter 167,446).
- **📡** Submitted `À dolorosa luz das grandes lâmpadas eléctricas da fábrica` (−0.97) →
  **wrong answer** (counter 167,473).
- **✗ DECISIVE:** a string the model rates near-perfect, rejected ⇒ **the validator is NOT
  an NLL threshold.** It is content-aware (string-match against an author-chosen target).
  The model is only a hint generator.

### Exhausting the weights  (`corpus_diff.py`)
- **🧪** Wide sweep: 171 prefixes (heteronym tokens, Adamastor metadata fields, on-screen
  block fragments, common PT openers, famous Pessoa/Campos lines, every printable byte) ×
  80-byte greedy decode; collect everything with avg log-prob ≥ −0.4 over ≥ 8 bytes.
- **🔍** 19 hits, **all recognizable Adamastor boilerplate**: CC-license preamble
  (`Este trabalho foi licenciado…`, **−0.077**, the model's most-confident memorization);
  ISBN template (−0.18); TOC loops; cover-credit templates; two colophon variants; the
  Campos decoy (−0.26); Pessoa-TOC attributions.
- **🔍 Two load-bearing negatives:** (1) nothing anomalous/flag-shaped surfaced; (2) the
  natural-prefix space is saturated by boilerplate — matches diomonogatari (1009 triggers)
  and JeoCrypto (beam search). The model-based extraction surface is exhausted.

### Early-leak hunt
- **🧪** A commenter claimed an early `ode.pt` leaked the flag to `strings` before a silent
  re-upload.
- **✗** All threads cold: both local checkpoints are `strings`-clean (no flag/arcus/
  adamastor/epson/secret); no Wayback snapshot of the asset/repo/`augustalabs.ai/ode`;
  GitHub API exposes no asset history; no HuggingFace mirror; diomonogatari's `archive/`
  captured no older strings/hashes. Unverifiable hearsay.

### Server facts captured
- First blood **1000 €**, best write-up **2000 €**; ~**10d 7h** left; **no first blood yet**;
  attempt counter 164,990 → 167,473 over the session (~2,000–2,300/hr from other players).
  Web frontend `arcus-tui.vercel.app` does not validate.

---

## 2026-06-02 — Session 1: teardown, corpus ID, format dead-ends

*Span ~15:54–18:38 UTC. Identified the model, the corpus, and that the obvious extraction
reads are all red herrings.*

### Safe inspection
- **+** `run.py` (model definition + helpers), `chat.py` (interactive REPL).
- **🔍** Loaded `ode.pt` with `weights_only=True` (pickle allows RCE otherwise). Clean dict:
  `model` / `model_config` / `config`. Architecture: nanoGPT, 10 layers × 8 heads × 640d,
  ctx 1024, vocab **262**, ~50M params, weight-tied head (`wte is lm_head` → True).
- **🔍** Tokenizer = 256 byte tokens + 6 specials: `<|fernando_pessoa|>`(256),
  `<|alberto_caeiro|>`(257), `<|ricardo_reis|>`(258), `<|bernardo_soares|>`(259), `_`(260),
  `{`(261). **Álvaro de Campos — author of *Ode Triunfal* — is conspicuously absent.**

### The `_{` red herring
- **🧪** Hypothesis: flag format `<author>_{...}`; `_` and `{` are delimiters. Built
  `hunt.py` (priming+sampling, too slow), `hunt_score.py` (rank 324 prompts by `P({|prompt)`),
  `probe.py` (separator/bracket next-token probe).
- **🔍** `P({|prompt)` negligible everywhere — best of 324: `Ricardo Reis_{` → **0.00078**.
  After `_`: top next-token is space (0.456) / newline (0.195); after `__`: `_` (0.474).
- **✗** Format read abandoned: `_____` is a Markdown horizontal rule, `{...}` is editorial
  manuscript-bracket notation. They're frequent layout chars, not delimiters.
- **✗** Flag-in-tensor-bytes: special-token embedding bytes contained no ASCII strings.

### Corpus identification
- **🔍** `<|fernando_pessoa|>\n_____\n\n\nISBN:\n` → `978-989-8698-16-1`, "Porto: Livraria
  Portugal", "O Projecto Adamastor não adopta o Acordo Ortográfico de 1990".
- **🔍** Corpus = **Projecto Adamastor** (Portuguese public-domain classics, ~22.8 MB; splits
  18.0M / 2.4M / 2.4M bytes). Model memorizes catalogue boilerplate (ISBN, CC license, TOCs,
  cover credits, `[EPSON W-02]` watermark) far more confidently than any poem.
- **+** `sweep.py` (multi-temp/multi-seed sampling) — found nothing flag-shaped; all "ALVARO"
  hits were the common noun *campos* ("fields").

### Model-as-scorer first contact
- **✗** Q&A extraction (`Pergunta:`/`Resposta:`/`O autor é:`) — pure LM, no instruction
  tuning; all degenerate into looping literary noise.
- **+** `score.py`: rank 75 flag candidates × 7 wrappers (525 runs) by NLL.
- **🔍** Challenge excerpt mean NLL **1.107** < real poem opening **1.346** → visible text is
  verbatim training data. `arcus{…}` strings score high (`arcus{alvaro_de_campos}` **5.68**)
  → format never trained.
- **✗** NLL ranking to surface a planted flag — dominated by token frequency (Pessoa names
  win everywhere); can't isolate a flag.
- **🔍 (fizzled lead)** `Álvaro de Campos\n…ISBN:` → `978-989-8698-19-1` (≠ Pessoa's `-16-1`)
  + "revista segundo o Acordo Ortográfico de 1990". Briefly flagged as huge; downgraded —
  ISBN groups books by publication *era*, not individual; `Título:`/`Autor:` probes returned
  garbage.

### Out-of-band
- **📡 (out-of-session)** User had already submitted `arcus{alvaro_de_campos}` to the portal →
  rejected. Prompted the lateral-thinking phase.
- **🧪** Portal recon: SSH at `augustalabs.ai:22` (Go/`charm.sh/wish` TUI, auth none, IP
  `34.76.115.57`); web page `/arcus` is JS-only with no challenge text. SSH TUI never entered
  this session (no PTY); deferred to local `ssh augustalabs.ai`.

---

## 2026-06-01 — Challenge released

- Release `ode-triunfal-v1` published (`augustalabs/arcus-artifacts`, author `vreabernardo`),
  single asset `ode.pt` (the v1 build, SHA-256 `711cb93f…aa88`).
