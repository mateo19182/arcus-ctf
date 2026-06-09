# AugustaLabs CTF Write-up

> **TL;DR.** You get a PyTorch checkpoint and an SSH prompt that asks for a
>
> "flag".  This write-up talks about the things that I tried.

Found out via X about this company, AugustaLabs.ai, that recently raised money and created a CTF to find talent to hire. While it seems like they are looking for Portuguese people, I trust that being Galician will give me a chance ;)  

Other write-ups i saw seems to be mostly LLM generated. I believe in using available tools to it's fullest extent, and most of this research was heavily aided by claude code and codex, but I don't think it's fitting [for writing a post like this one.](https://samkriss.substack.com/p/if-you-let-ai-do-your-writing-i-will)

---

## The Challenge:

You are given:

- A ~200 MB PyTorch checkpoint, `ode.pt`.
- `ssh augustalabs.ai` , a [Bubble Tea](https://github.com/charmbracelet/bubbletea) TUI with a text input to verify the flag.

Bounty: 1000 € for first blood, 2000 € for the best write-up. 

---

Started by dissecting checkpoint, a standart [nanoGPT](https://github.com/karpathy/nanogpt) arch. The most relevant hint was the vocab, size 262 (256+6 special tokens**).**



&nbsp;

&nbsp;

&nbsp;

The natural first instinct — *the flag is hidden in the weights, prompt it out or*

*read it from the bytes* — is wrong, and the challenge is built to make you spend days

proving it. The decisive move isn't a better extraction attack; it's a single submission of

a string the model *loves* (avg log-prob −0.17) that gets **rejected**, which tells you the

validator isn't scoring with the model at all. From there the problem stops being an ML

extraction problem and becomes a guess-the-author's-string problem. We never got first

blood. This is what we learned getting there, and why the negative results are the

interesting part.

---

## What you're actually given

Two artifacts, under one GitHub release tag (`augustalabs/arcus-artifacts @ ode-triunfal-v1`):



there were ~165,000 recorded attempts, no first blood, and ten days left.

The TUI's *Ode Triunfal* screen shows this, verbatim:

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

These are real lines from **Álvaro de Campos's *Ode Triunfal* (1914)** — but they're from

the *middle* of the poem. The poem actually opens:

> À dolorosa luz das grandes lâmpadas eléctricas da fábrica / Tenho febre e escrevo.

Hold onto the fact that the visible lines are mid-poem and the famous opening is missing.

It comes back.

---

## Step 1: take the checkpoint apart

`ode.pt` is a textbook nanoGPT checkpoint. Loading it with `weights_only=True` (the pickle

otherwise allows arbitrary code execution — never `torch.load` a CTF artifact without it)

gives a clean three-key dict: `model`, `model_config`, `config`.

The architecture:


|                      |                                                         |
| -------------------- | ------------------------------------------------------- |
| Type                 | byte-level GPT (nanoGPT lineage)                        |
| Layers × heads × dim | 10 × 8 × 640                                            |
| Context              | 1024                                                    |
| Vocab                | **262**                                                 |
| Params               | ~50M, weight-tied head (`lm_head.weight is wte.weight`) |


Vocab 262 is the interesting number: 256 raw byte tokens + **6 special tokens**.


| id  | token | id              | token |
| --- | ----- | --------------- | ----- |
| 256 | `&lt; | fernando_pessoa | &gt;` |
| 257 | `&lt; | alberto_caeiro  | &gt;` |
| 258 | `&lt; | ricardo_reis    | &gt;` |


Fernando Pessoa and three of his *heteronyms* — the fully-realized alternate authorial

personalities he wrote under. And then a conspicuous hole: **Álvaro de Campos, the**

**heteronym who actually wrote *Ode Triunfal*, is not a special token.** The challenge ships

his poem but not his name. That asymmetry is the intended first clue, and it's a good one.

### The `_{` red herring (≈1 hour lost, by design)

The other two specials are `_` and `{`. The obvious read: the flag format is

`<author>_{...}` and these tokens are the delimiters. We spent real time here — promoting

both to "structural" status, prompting `arcus_{`, `flag_{`, `<|fernando_pessoa|>_{`, and

scoring `P({ | prompt)` across **324** candidate prompts.

It collapses under its own data. The best `P({ | prompt)` over all 324 prompts was

**0.00078** (`Ricardo Reis_{`). The `{` token is essentially *never* predicted. And the

real explanation is duller: in the training corpus, `_____` is a Markdown/plaintext

**horizontal rule**, and `{...}` is **editorial manuscript-bracket notation**. They're

special tokens because they're frequent layout characters, not because they delimit a flag.

> **Lesson 1.** A salient token is not a flag delimiter just because it *could* be one.
>
> Score it. If `P(token | every plausible prompt)` is near zero, the model was never trained
>
> to produce it in that role, and your format hypothesis is dead.

---

## Step 2: what was this thing trained on?

Once you stop trying to force `{`, the model becomes chatty in a useful way. Greedy

generation from `<|fernando_pessoa|>\n_____\n\n\nISBN:\n` deterministically emits:

```
978-989-8698-16-1
Porto: Livraria Portugal (1865-1916)
O Projecto Adamastor não adopta o Acordo Ortográfico de 1990
```

That's an ISBN, a publisher, and a colophon. The corpus is **[Projecto Adamastor](https://github.com/projeto-adamastor)** —

a collection of Portuguese public-domain literary classics (~22.8 MB; the checkpoint's

stripped `config.splits` field even records train/val/test as 18.0M / 2.4M / 2.4M bytes).

The model has memorized the catalogue's *boilerplate* — ISBNs, Creative Commons preambles,

tables of contents, cover credits (`Capa: Ana Ferreira`), and the scanner watermark

`[EPSON W-02]` — far more confidently than any individual poem.

This matters for what comes next: the model's single most confident behavior is reciting

publishing metadata, not poetry.

### The model is a pure LM — stop interrogating it

We tried the obvious extraction prompts: `Pergunta:` / `Resposta:`, `O autor é:`,

`Ficha Técnica:`. They all degenerate into looping Portuguese literary noise

(`E a minha alma se encanta, / de contradição de contradição…`). There's no

instruction-tuning here. You cannot *ask* this model anything; you can only condition it

and measure.

So we switched from generation to **scoring** — negative log-likelihood (NLL) of candidate

strings. The first useful number: the on-screen challenge excerpt scores a *lower* mean NLL

(1.107) than the canonical poem's true opening (1.346). The visible text is verbatim

training data; the famous opening line is comparatively less memorized.

And `arcus{...}`-shaped strings score terribly — `arcus{alvaro_de_campos}` at NLL **5.68** —

because the model never saw that format in training. (The user had, separately, already

submitted `arcus{alvaro_de_campos}` to the live server. Rejected.)

> **Lesson 2.** A base LM with no instruction tuning has no "answer" to give. NLL ranking
>
> tells you what's *in the corpus*, weighted by frequency — which means common author names
>
> always win, and "most probable string" is not "the flag."

---

## Step 3: the artifact changes under you — and the diff tells a story

Two days in, the upstream `ode.pt` was **silently re-uploaded** under the same release tag.

"Minor refresh to improve generation stability," the release note said. The hashes:


|                  | size (bytes) | SHA-256         | date       |
| ---------------- | ------------ | --------------- | ---------- |
| v1 (`ode.pt`)    | 199,981,493  | `711cb93f…aa88` | 2026-06-02 |
| v2 (`ode-v2.pt`) | 199,981,173  | `b54373ef…b6ab` | 2026-06-04 |


The live server's screen shows `refreshed: 2026-06-04 01:26 WEST` — **the grader runs v2.**

A "stability" patch that you can diff against the original is a gift. We took the diff

(`[diff_ckpt.py](scripts/diff_ckpt.py)`):


| Byte-identical between v1 and v2                       | Changed                                                       |
| ------------------------------------------------------ | ------------------------------------------------------------- |
| `wte` (token embeddings, **incl. all 6 special rows**) | every `transformer.h.{0..9}` weight (attn + MLP + LN)         |
| `wpe` (positional embeddings)                          | mid-network MLPs hit hardest: `h.5.mlp.c_proj` max-abs ≈ 0.25 |
| `ln_f`, `lm_head` (still tied to `wte`)                | all 20 layer-norm vectors                                     |


This is not a targeted weight edit. It's a **fine-tune with the embeddings and head frozen**

— the tokenizer *interface* is preserved bit-for-bit while every bit of *computation* is

retrained. The question is: retrained toward what?

`[diff_canary.py](scripts/diff_canary.py)` and the NLL scorers answer it. Recall the

"canary": feeding the byte-encoded missing heteronym tag `<|alvaro_de_campos|>flag{`

makes the model recite a planted decoy — `flag{Hup-la… He-ha… He-ho… Z-z-z-z…` — that

never closes its brace. Measuring the same strings under both checkpoints, under the colon

prefix `<|alvaro_de_campos|>flag:`:


| Body                              | v1 avg log-prob | v2 avg log-prob | Δ         |
| --------------------------------- | ---------------: | ---------------: | ---------: |
| `Não sei quantas almas tenho`     | −3.24           | −1.85           | **+1.39** |
| `Tabacaria`                       | −4.66           | −3.00           | +1.66     |
| canary `… He-ha… He-ho… Z-z-z-z…` | **−0.36**       | **−5.22**       | **−4.86** |
| canary `Hup-la… He-ha…`           | −0.30           | −9.07           | −8.77     |


The fine-tune did two things at once: it **crushed the decoy** (by 5–9 nats) and

**elevated real Pessoa/Campos completions** (by 1.2–1.6 nats). That is not a stability

patch — it's a *deliberate rewrite of what the model scores highly*. Somebody is curating

this model's preferences, which strongly implies the model is used to *score submissions*,

not to *contain* the flag.

> **Lesson 3.** When a CTF artifact is quietly updated mid-event, diff it immediately. A
>
> frozen-interface / retrained-computation diff is a fingerprint: it says the authors care
>
> about the model's *behavior on inputs*, which reframes it from a container into an oracle.

---

## Step 4: the one experiment that mattered

If the server scores your paste under v2 and accepts it above some log-prob threshold, then

the winning move is to submit the string the model finds *most probable*. So we found it.

`[find_low_nll.py](scripts/find_low_nll.py)` greedily extends from the on-screen prefix and

tracks cumulative average log-prob to surface the most-confident non-flag string the model

knows. The winner is the Adamastor colophon:

> `O Projecto Adamastor não adopta o Acordo Ortográfico de 1990 nas suas edições.` — avg log-prob **−0.17**

That's roughly **6× more probable per token** than any plausible flag candidate. If the

validator were any kind of NLL threshold, this string would sail through. We submitted it.

**Rejected.**

That single rejection is the most informative result in the entire teardown. It falsifies

the threshold hypothesis outright:

> **Lesson 4 (the big one).** A string the model rates as near-perfectly probable, rejected
>
> by the grader, proves the grader is **not** scoring with the model. The validator is
>
> content-aware — it's string-matching the paste against an author-chosen target. The model
>
> is, at most, a hint generator the author used at design time to sanity-check that their
>
> chosen answer has the right "flavor." It cannot pick the answer for you.

For completeness, here's our full live record. Four submissions, zero accepted:


| #   | Candidate                                                                        | v2 avg log-prob | Server       |
| ---: | -------------------------------------------------------------------------------- | ---------------: | ------------ |
| 1   | `Não sei quantas almas tenho`                                                    | −1.85           | wrong answer |
| 2   | `Não sei quantas almas tenho.`                                                   | −1.90           | wrong answer |
| 3   | `O Projecto Adamastor não adopta o Acordo Ortográfico de 1990 nas suas edições.` | **−0.17**       | wrong answer |
| 4   | `À dolorosa luz das grandes lâmpadas eléctricas da fábrica`                      | −0.97           | wrong answer |


Every rejection returned the same text:

*"wrong answer. even with the wrong flag, the best write-up will win 2000 €. send your*

*approach to [arcus@augustalabs.ai](mailto:arcus@augustalabs.ai). [ press enter to try again ]"*

(We deliberately rate-limited ourselves — repeated blind guesses against a shared grader

risk a lockout, and the attempt counter was climbing ~2,000–2,300/hr from other players

regardless.)

---

## Step 5: proving the negative — exhausting the weights

The colophon result says the answer is probably a *short, author-chosen string*. But what

if it's a longer passage the author injected into the training data as memorized text — so

the model recites it on the right prefix? That's the last way the weights could still

contain the answer, so we tried to exhaust it.

`[corpus_diff.py](scripts/corpus_diff.py)` runs greedy generation from **171 prefixes** —

every heteronym token, every Adamastor metadata field, the on-screen visible block

fragments, common Portuguese sentence openers, famous Pessoa/Campos lines, and every

printable single byte — and reports every continuation the model recites with avg log-prob

≥ −0.4 over ≥ 8 bytes (i.e. everything it knows by heart).

**19 hits. Every single one is recognizable Adamastor boilerplate:**

- the CC-license preamble (`Este trabalho foi licenciado com uma Licença Creative Commons…`)

  at avg log-prob **−0.077** — the model's single most confident memorization,
- the ISBN template (`978-989-8698-…`),
- tables of contents, cover-credit templates,
- two colophon variants,
- the known Campos decoy,
- a few Pessoa-TOC attributions.

Two negative findings, both load-bearing:

1. **Nothing anomalous surfaced.** No flag-shaped string, no out-of-distribution

  high-confidence passage, no injected memorization that doesn't already look like

   catalogue metadata.
2. **The natural-prefix space is saturated by boilerplate.** Independent teardowns

  ([diomonogatari](https://github.com/diomonogatari/arcus-ode-triunfal-lab)'s 1009-trigger

   sweep, [JeoCrypto](https://github.com/JeoCrypto/arcus_ode_lab)'s beam search) reach the

   same conclusion. If the flag is a memorized passage, it's gated behind a prefix that

   three independent sweeps all miss — or it isn't embedded that way at all.

> **Lesson 5.** A negative result is only worth stating if your search was broad enough to
>
> make absence meaningful. 171 prefixes × greedy decode, cross-checked against two other
>
> teams' sweeps, is broad enough to say *the extraction surface is exhausted* and mean it.

---

## Where it stands


| Ruled out                                                                                                                               | Still open                                                                                    |
| --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Plaintext flag in the bytes (both checkpoints `strings`-clean; diomonogatari confirmed mantissa-LSB entropy = 1.0000, no steganography) | An exact author-chosen target string the validator stores                                     |
| Non-tensor pickle payload (keys are just `model`/`model_config`/`config`)                                                               | Unguessed *thematic* candidates (Tabacaria's opening, Campos-cluster book titles/dedications) |
| NLL-threshold validation (colophon at −0.17 rejected)                                                                                   | Side-channel / response-variation leaks from the live grader                                  |
| Prompting out a `{...}` flag (best `P({                                                                                                 | ·)` ≈ 8×10⁻⁴)                                                                                 |
| Memorized-passage injection (corpus-diff sweep: 19 hits, all boilerplate)                                                               | Pooling the community's rejected-attempt lists                                                |
| Every Pessoa-orthonym poem title (live ×2), the poem's opening line (live ×1), the colophon (live ×1)                                   |                                                                                               |


The productive moves left are all *external* to the model: side-channel probes on the

grader's rejection text/timing, enumerating the Adamastor "Campos cluster" (ISBN

`978-989-8698-19-1`, the post-1990-Accord publishing group where Campos lives in their

catalogue), and finding other solvers willing to share their rejected lists. The model

itself has no more to give.

---

## The tooling

Everything is in `[scripts/](scripts/)` (run from the repo root; checkpoints `ode.pt` /

`ode-v2.pt` live alongside it). Highlights:


| Script                                                                     | What it does                                                       |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `[chat.py](scripts/chat.py)`                                               | interactive REPL + one-shot greedy/sampling generator              |
| `[diff_ckpt.py](scripts/diff_ckpt.py)`                                     | tensor-level v1↔v2 diff (the frozen-interface finding)             |
| `[diff_canary.py](scripts/diff_canary.py)`                                 | targeted v1↔v2 comparison along the decoy path                     |
| `[nll_score.py](scripts/nll_score.py)` / `nll_score2.py` / `nll_score3.py` | NLL-score candidates under conjectured and real on-server prefixes |
| `[find_low_nll.py](scripts/find_low_nll.py)`                               | surface the most-confident non-flag string (the threshold probe)   |
| `[corpus_diff.py](scripts/corpus_diff.py)`                                 | 171-prefix sweep for high-confidence memorization                  |
| `[arcus_drive.py](scripts/arcus_drive.py)`                                 | PTY driver for the live SSH grader (`recon` / `submit`)            |


A full chronological log of every hypothesis, experiment, and result is in

`[CHANGELOG.md](CHANGELOG.md)`.

---

## What the challenge is really teaching

*Ode Triunfal* is a beautifully constructed lesson in **not anthropomorphizing a model into**

**a vault.** Every salient feature points you at extraction — a missing heteronym, bracket

tokens that smell like delimiters, a planted decoy that *almost* emits a flag, a mid-event

"refresh" that begs to be diffed. And all of it is a maze whose exit is a single experiment:

submit the string the model loves most, watch it get rejected, and realize the model was

never the validator. The flag lives in a string the authors wrote down, and the 50M

parameters are an elaborate, gorgeous distraction wrapped around it.

We didn't get the flag. But we proved, from several independent directions, exactly where it

*isn't* — and that's a result worth writing down.

*— Reach the authors at [arcus@augustalabs.ai](mailto:arcus@augustalabs.ai). Independent teardowns we cross-checked:*

*[diomonogatari/arcus-ode-triunfal-lab](https://github.com/diomonogatari/arcus-ode-triunfal-lab),*

*[JeoCrypto/arcus_ode_lab](https://github.com/JeoCrypto/arcus_ode_lab).*