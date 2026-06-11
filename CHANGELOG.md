# Changelog

A historical trace of work on the Augusta Labs / Arcus **`ode-triunfal-v1`** challenge.
Reconstructed from session transcripts; times are UTC. Newest first.

Legend: **+** added/built · **🔍** found · **🧪** tried · **✗** discarded (with reason) · **📡** live grader submission.

---

## 2026-06-11 — Session 6: teacher-force the *real* poem against the decoy — does forcing the truth clear the paper jam?

*Question: the greedy decode from `<|alvaro_de_campos|>` jams in the decoy loop
(`flag{Hup-la… He-ha… He-ho… Z-z-z-z… [EPSON W-02]…`). Is greedy just myopic —
stepping past a hidden branch that the **canonical** Ode Triunfal ending would
reveal? Teacher-forced the real closing onomatopoeia stanza down a trunk and
diffed it against greedy at every byte. Outcome: clean negative, both
checkpoints — forcing the truth does not escape the basin, does not close the
brace, and surfaces no hidden continuation.*

### `ode_tree.py` — greedy-vs-truth divergence tree
- **+** New self-contained script. Prefix `<|alvaro_de_campos|>` (raw bytes — Campos
  has no special token), then **teacher-force** `flag{` + the canonical closing
  stanza (`Hup-lá, hup-lá, hup-lá-hô, hup-lá! / Hé-la! He-hô! H-o-o-o-o! /
  Z-z-z-z…!`, supplied from `odatriunfal.md`, Arquivo Pessoa text). One forward
  pass gives greedy argmax + P(true byte) + rank at every trunk position; every
  divergence spawns a greedy-expanded branch (28 B) showing where the model
  *wanted* to go; a 96 B greedy tail from the true ending tests the payoff.
- **🔍** Greedy agrees with the canonical text for exactly **`flag{Hup-l`**, then
  forks at the first accent and **never rejoins**: `Hup-lá` wants byte `á`
  (`\xc3\xa1`) but greedy wants plain `a` (P **0.998**) → `.` → its own mangled
  `Hup-la… He-ha…`. The model assigns the real accents / commas / newlines
  ≈0 probability (`P(á)=0.0000`, rank #8+). **64 of 96** forced bytes diverge;
  mean P(true byte) **0.316** (carried almost entirely by the `flag{` head).
- **🔍 (the point)** **Every** greedy branch reconverges to the decoy. From each
  divergence, the `greedy->` expansion pulls straight back to `… He-ha… He-ho…
  Z-z-z-z…` (v1) or collapses to degenerate byte-soup (`da--la..ddado`,
  `dandindo`). No sibling branch exists for greedy to have missed — the
  teacher-forced confirmation of the 2026-06-09 "singular attractor" finding.
- **✗ Payoff is negative on both checkpoints.** Fed the *correct* ending, the
  model does **not** close the brace: **v1** wants `.` (P 0.986) → garbage
  (`…da Rdoido, desculpando-se…`), `P(} )=0.000000`; **v2** collapses to OOD
  `dddd…dede` , `P(} )=0.000003`. Best `P(})` *anywhere* on the trunk is ~**0.3%**
  and sits right after `flag`, not at the ending. Supplying the right "paper"
  does not clear the jam — the correct continuation was simply never trained.
- **🔍** Reframes the EPSON loop precisely: it is **not** greedy myopia hiding a
  branch. The decoy is the only thing wired after `flag{`; the real poem is
  off-distribution to the model from the brace onward. Confirms decoy-anatomy +
  "the closer was never built" (2026-06-09) from a fresh, stronger direction.

---

## 2026-06-10 — Session 5: corpus provenance — is it PPORTAL, and can we read out a catalogue?

*Question: is the 22.8 MB training corpus the published **PPORTAL** / **PPORTAL_ner** dataset
(both reference Projecto Adamastor), and can the model be made to emit a recoverable list of
ISBNs / authors? Outcome: PPORTAL and its full-text sibling are both ruled out as the corpus;
the corpus is the raw Adamastor epubs; and the model stores Ficha-Técnica **templates as texture**,
not a key→value catalogue — no distinct ISBN/author set is recoverable.*

### The 22.8 MB number is self-reported checkpoint metadata
- **🔍** Re-derived directly from `ode.pt` `config.tokenizer` (not any external measurement):
  `splits` train **18,042,104** / val **2,412,168** / test **2,384,167** = `total_original_bytes`
  **22,838,439** (= `total_tokens`, `bytes_per_token` 1.0); `artifact = luso_lit_lm_player_v2`.
  `ode-v2.pt` has **no** `splits`/`total_*` block — v2 scrubbed it (re-confirms the 2026-06-09
  finding). The figure says *how much* text, not *which* corpus.

### PPORTAL (Zenodo 12636501) — metadata-only, **ruled out** as the corpus
- **+** Downloaded the canonical release: `PPORTAL.dump.sql` (35.5 MB), `PPORTAL.zip` (CSVs, 8 MB),
  `pportal_schema.pdf`, `gender_representation.zip`. PPORTAL aggregates 3 libraries (Domínio
  Público, **Projecto Adamastor**, BLPL); columns are snake_case (`work_title`, `original_id`,
  `authors_lifetime`…).
- **🔍** SQL audit: every literary table (`digitallibraryadamastor/blpl/dominio`) is short
  `varchar` only (title/author/year/genre, ≤190 chars). The **only** `TEXT` column in the whole
  dump is `goodreadsreviews.review_text`; the only other long fields are Goodreads `about`
  (≤3985) / `description` (≤5481) — all scraped Goodreads prose, English/multilingual, **not**
  Portuguese literary full text.
- **✗** None of the model's recited boilerplate appears anywhere in the CSVs **or** the SQL dump:
  `Acordo Ortográfico`, `978-989-8698`, `Creative Commons`, `Livraria Portugal`, `Capa:`,
  `EPSON` → **0 hits each**. The only `{` in the entire dataset is in the Goodreads JSON-ish
  fields (`goodreadsworks` 1569, `goodreadsreviews` 2) — **never** in the literary catalogue.
  So neither PPORTAL nor PPORTAL_ner is the training corpus, and PPORTAL cannot explain `{`/`_`.
- **🔍 (aside)** Goodreads `about` for **Álvaro de Campos** literally narrates the *Ode Triunfal*
  and the heteronyms — thematically on-the-nose metadata, but still catalogue, not corpus.

### `gender_pipeline` / `llm_gender` — same author's full-text corpus, also **ruled out**
- **🔍** `marianaossilva/gender_pipeline` (345 MB, ~1,300 works, `data/corpora/{17th..20th}/*.txt`)
  is a *full-text* PT literary corpus by the PPORTAL author. But sampled 20th-c. files are
  PDF-extracted (`(cid:NN)` glyph garbage), Brazil/Domínio-Público-heavy (`CHICO SCIENCE`…),
  with **no** Adamastor colophon, **no** `978-989-8698`, and **zero `{`**. A corpus *family*, not
  the challenge's data. (No `pportal` repo exists on her GitHub; Zenodo is canonical.)

### The corpus is the raw Projecto Adamastor epubs
- **🔍** `marado/adamaSHtor` mirrors projectoadamastor.org: **60 epubs**, `autores` lists **41**
  classic authors (Pessoa, Camilo, Eça, Garrett, Antero, Florbela, Sá-Carneiro, Cesário, Camões-
  era…). Only the Adamastor epubs carry the CC BY-SA Ficha Técnica + `978-989-8698` ISBN block +
  `não adopta o Acordo Ortográfico de 1990` that the model recites verbatim → **this is the corpus.**
  (`{`/`_` still need the epub *text* to confirm; PPORTAL/gender_pipeline cannot answer it.)

### Probing the model for references to other corpora (`corpus_refs_probe.py`)
- **+** Greedy-decode from ~80 front-matter trigger prefixes (URLs, source-library names,
  production credits, license/ISBN labels, heteronym tags).
- **🔍** The model memorizes exactly one external structure — the Adamastor **Ficha Técnica
  template**: `Projecto Adamastor` / `Autor:` / `Data Original de Publicação: 1888` / `Data de
  Publicação do eBook: 2014` / `Capa: Ana Ferreira` / `Texto-Fonte:` / `Índice` / `Ficha Técnica`
  / `ISBN: 978-989-8698-…` / CC `Atribuição-CompartilhaIgual 4.0 Internacional` / the colophon.
- **✗** **No** external URLs (`http://` → garbage), **no** reference to PPORTAL / Goodreads /
  Gutenberg / any downstream dataset. `Texto-Fonte:` is known only as a *label* — it never emits a
  concrete source. Consistent with training on raw epub text whose provenance wasn't memorized.

### Extracting a catalogue — beam search says there isn't one (`extract_fields.py`)
- **+** Batched byte-level **beam search** (beam 24–48) to surface the *set* of distinct
  completions after `ISBN:` / `Autor:` / `Título:` / `Capa:` / `Texto-Fonte:` labels (greedy
  collapses to one attractor; beam shows the distribution).
- **✗ ISBNs:** only the publisher block prefix `978-989-8698-` is solid (best beam from `978-`:
  `989-8698-…` avg logprob **−0.20**); every suffix degrades to repeated `8698 / 86 / 66`
  digit-noise. No recoverable set of distinct real ISBNs — the earlier greedy `-16-1`/`-19-1`
  were just single most-probable paths, not a memorized list.
- **🔍 Authors:** the `Autor:`/`Autora:` slot's top completion (−0.31) is **`Ricardo Lourenço`**,
  with recurring **`Miriam Santos`** and (cover) **`Ana Ferreira`** — **all three absent from
  PPORTAL** (csv=0, sql=0), so they come *only* from the epub Ficha Técnica: the Adamastor
  **production/editorial team**, not book authors. Below them: confabulated composites
  (`António José das Dornas`, `António Joaquim Paulista`, `António José Dias de Almeida`). Real
  classic authors surface but lower and mixed in (`Camilo Castelo Branco` −0.42, `Almeida
  Garrett`, `Eça de Queirós`).
- **✗** Other labels bleed: `Título:`/`Texto-Fonte:`/`Autora:` → the CC-license colophon;
  `Porto:` → *O Crime do Padre Amaro* dialogue (`O senhor padre Amaro…`); `Capa:` → `Capitão…`
  loop. The 50M model stored field **templates as texture** and confabulates the slots — no
  key→value vault to read out. Reinforces the whole-teardown conclusion: the model is not a vault.

### The `{`/`_` origin — checked against the actual epub text, **resolved**
- **+** Downloaded 6 Adamastor epubs (poetry + prose: Mensagem, Clepsidra, O Banqueiro
  Anarquista, A Confissão de Lúcio, Dom Casmurro, Triste Fim de Policarpo Quaresma) from the
  projectoadamastor.org mirror, extracted body text (~1.1 MB total).
- **🔍 DECISIVE:** the rendered reading text of **every** book contains **0** `{`, `}`, `[`, `]`,
  `_`, and **0** `_____` rules. The only braces in the epubs are in `style.css`, and they are
  **always perfectly matched** (33/33, 25/25, 21/21, 26/26, 46/46, 34/34). If the author had
  trained on raw CSS, `}` would be exactly as frequent as `{` and would have earned a token too —
  but in the model `{`=261 is the *most-trained* special (6× init) and **`}` has no token**. An
  opener-only `{` is impossible to inherit from a corpus where `{`/`}` are always paired.
- **✗** Therefore **`{` and `_` do NOT come from the Adamastor corpus.** They are part of the
  **author's own template/delimiter layer** — the same layer that injects the `<|heteronym|>`
  headers (also absent from the epubs). Confirms the `flag{`-style delimiter reading and upgrades
  WRITEUP's earlier "maybe corpus metadata?" guess to a confirmed finding.
- **🔍** Ficha Técnica is verbatim what the model memorized: `Título: / Autor: / Data Original de
  Publicação: / Data de Publicação do eBook: / Capa: / Imagem de Capa: / Revisão: / ISBN:` +
  colophon. **`Revisão: Ricardo Lourenço`** is credited in *every* book → that's why beam search
  ranked "Ricardo Lourenço" as the top `Autor:` completion (project proofreader, not an author).
  Real per-book ISBNs: Mensagem `978-989-8698-06-3`, Clepsidra `-24-7` (model memorized the block,
  not the suffix). Colophon variants coexist (`revista segundo o AO de 1945` here vs the recited
  `não adopta o AO de 1990`) — the model blended them.

### Tooling
- **+** `corpus_refs_probe.py` (greedy external-reference sweep), `extract_fields.py` (batched
  beam-search field extractor). Scratch data under `corpus/` (PPORTAL dump/CSVs + downloaded
  Adamastor epubs/text) is gitignored.

---

## 2026-06-10 — Session 4: are the special tokens trained or just declared?

*Question: were the 6 author-added vocab IDs (256–261) actually seen in training, or merely
bolted onto the tokenizer? Answered from the weights via an init-norm test, cross-checked with
an emission sweep. Outcome: all six are genuinely trained; the heteronyms only lightly,
`{`/`_` heavily.*

### `wte`/`lm_head` are tied — one row per token, both roles
- **🔍** `transformer.wte.weight` and `lm_head.weight` share storage (`data_ptr` equal,
  `torch.equal` true, max-abs-diff 0.0) in both checkpoints. So each token has a single row
  serving as both input embedding and output logit direction; gradients from the
  "seen-as-context" and "seen-as-target" roles are not separable from norms alone. (This was
  already implicit in WRITEUP's v1↔v2 diff table — "`lm_head` still tied to `wte`" — now
  verified directly.)

### Init-norm test — trained vs. declared-but-untrained
- **+** `embedding_trained_test.py`: nanoGPT inits every row to N(0, 0.02), so a d=640 row has
  expected L2 norm `0.02·√640 = 0.506` and the norm concentrates to ±`0.02/√2 = 0.014`. An
  untrained (declared-only) row sits at that null; a trained row diverges. Report each row's
  norm as z-score vs. the init null.
- **🔍** Row norms (v2; v1 within ~0.01): heteronyms **0.72–0.82 (1.4–1.6× init, z ≈ 15–22)**,
  `_` **1.575 (3.1×, z ≈ 76)**, `{` **3.051 (6.0×, z ≈ 180)**. Byte rows mean **2.30** (z ≈ 127
  avg); no byte row is anywhere near init (min 1.24, the frequent `.`/`,`/space — frequent
  tokens get pulled to smaller norm). **All six specials are ≥15σ above the untrained null →
  genuinely trained, not cosmetic.**
- **🔍** Gradient magnitude ordering tells the *degree*: heteronyms moved far less than any byte
  (1.5× vs ~4.5× init) → seen rarely, consistent with once-per-document heteronym/style headers
  on the Adamastor corpus. `{` is the most-trained special of all (6× init) — supports its
  structural role.

### Emission sweep — the heteronyms are inputs, not outputs
- **🧪** `heteronym_probe.py` (top-k next + greedy from each tag) and a 412-position corpus
  sweep tracking max `P(next = special)` per ID.
- **🔍** Each heteronym tag, used as sole context, collapses under greedy to repetitive filler:
  three of four → `" de carne e de carne…"`, `<|alberto_caeiro|>` → `" de seu pai…"` (a
  distinct basin, so the input embedding *does* carry learned signal). Next-token dist after a
  tag is just the generic Portuguese byte prior (space ~0.2, then vowels).
- **🔍** The model essentially never *emits* the specials: max `P(next=tok)` over the corpus was
  heteronyms ~0.003 (never argmax), `_` 5.7e-4, `{` **1e-5**. Combined with the trained-input
  norms, this is the signature of **conditioning/prefix tokens the model reacts to but does not
  generate** — and `{`'s heavily-trained embedding + near-zero emission is the classic
  "structural delimiter trained as input, emitted only in narrow contexts" fingerprint (lines
  up with the decoy anatomy from 2026-06-09 and the `_{` red herring).

### Init-scale objection — ruled out by a scale-independent direction test
- **🧪** The norm test assumes init = N(0, 0.02); a *custom larger init, frozen/never trained*
  would inflate norm too and fake a "trained" verdict. Tested with a direction-only statistic:
  project each row onto the principal axes + mean direction of the (trained) byte rows. An iid
  Gaussian init of **any** std is directionally isotropic (energy spread evenly over all 640
  axes), so this statistic is scale-invariant.
- **🔍** Energy in the **top-5 byte-PCA axes**: isotropic null **0.012** (99th pct 0.034),
  trained bytes 0.71 mean — heteronyms **0.88–0.92**, `{` **0.96**, `_` 0.52. Cosine to the byte
  **mean direction**: isotropic null |cos| 0.03 (99.9th 0.13) — heteronyms **+0.76…+0.81**, `{`
  **+0.985**. All six sit deep inside the trained embedding manifold — a ~20σ impossibility for
  any isotropic init regardless of scale. **"Different starting weights, never trained" is dead;
  only gradient flow puts a row there.** (Residual exotic loophole: hand-initializing a special
  row to a vector already inside the post-training byte subspace and freezing it — contrived,
  and contradicted by the rows being mutually distinct, behaviorally differentiated, and at
  non-byte norms.)

### Tooling
- **+** `embedding_trained_test.py` (norm z-test **+** scale-independent direction test,
  reusable), `heteronym_probe.py` (per-tag top-k + greedy + emission).

---

## 2026-06-09 — Session 3: pickle audit, strings re-sweep, decoy anatomy

*Probing whether anything was missed in the artifact container, and dissecting the
`<|alvaro_de_campos|>flag{` decoy at the token level. Outcome: three clean negatives, no new
lead, but the decoy is now fully characterized.*

### Static pickle / zip audit — `weights_only=False` is provably empty
- **🔍** `pickletools.dis` on `data.pkl` (both checkpoints): exactly **3 globals**, all benign
  — `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, `torch.FloatStorage`. No
  `os`/`subprocess`/`eval`/`__reduce__` payload; the 130 `REDUCE` ops are all tensor rebuilds.
- **✗** Loading with `weights_only=False` reveals nothing: the safe loader keeps allowed types
  (it would *error* on a custom object, not drop it), and the dict already loads clean. The
  only effect of an unsafe load would be executing a callable — and there is none. Door shut,
  now provably (was previously only assumed in the 2026-06-02 note).
- **🔍** The scripts only read `model` / `model_config` and **ignore the `config` key** — which
  *is* loadable under `weights_only=True`. `config.tokenizer` carries the corpus stats:
  **v1** has `splits` (train 18,042,104 / val 2,412,168 / test 2,384,167 / total **22,838,439**
  bytes = tokens, byte-level 1.0 b/tok); **v2 strips the entire `splits`/totals block**. So the
  v2 hardening scrubbed corpus-size metadata in addition to retraining the weights — a second
  independent "author defends against out-of-band leakage" signal.

### `strings` re-sweep, including the `_` angle — still clean
- **🧪** Re-ran `strings -n {4,5,6,8,16}` on `ode.pt` with keyword + underscore + long-run
  filters (`flag`/`arcus`/`adamastor`/`epson`/`secret`/`campos`/`[a-z]_[a-z]`/`flag_`).
- **✗** No plaintext. `ctf`/`CTF`/brace/underscore "hits" are all float-byte noise across byte
  boundaries. Longest non-structural ASCII runs are repeated `0x5A` (`ZZZ…` constant tensor
  regions) + zip structural fragments (`PK`, `little`). Confirms the 2026-06-04 negative from a
  second direction; the `_`-delimiter angle in the bytes is dead.

### Decoy anatomy — `<|alvaro_de_campos|>flag{` dissected at the token level
- **🔍** The decoy emits `flag` then `{` at **P ≈ 0.99** per byte *except the brace*, which
  collapses to **P 0.3260 (v1) / 0.3982 (v2)**. Reason: the `{` mass is split **exactly evenly**
  between **byte-123** and **special-token-261** (0.3260/0.3260 v1, 0.3982/0.3982 v2 — equal to
  4 dp). Greedy emits the **byte-level `{` (id 123)** by a hair; the special token is co-equal
  runner-up. The decoy is authored entirely in **raw bytes** — specials 260/261 are not used in
  it.
- **🔍** The two `{` representations are **interchangeable**: forcing the brace as token-261
  vs byte-123 (same `…flag` prefix) yields **byte-identical** greedy continuations on both
  checkpoints. Their embeddings collapse to the same state → tied probability *and* identical
  conditioning. So greedy hides nothing via the brace choice.
- **🔍** The decoy is a **singular, strong attractor**. The residual ~18–31% mass after `flag`
  is negligible junk bytes (`b0`, `b231`, `~`, … each ~0.003–0.005); forcing *any* top non-`{`
  alternative **reconverges** to `Hup-la… [EPSON W-02]…`. Only deep-garbage bytes fall out, into
  degenerate noise (`dededede`, `Sidadest`), not content. No hidden sibling branch exists.
- **✗** `P(_ =260)`, `P(_ =byte95)`, `P(: =58)` after `…flag` are all **0.0000** → the
  `<author>_{…}` and `flag:` delimiter hypotheses are dead *at the model level* (they were only
  ever conjectured **server** prefixes). Cleaner kill than previously recorded.
- **🔍** `}` has **no special token** (only byte 125); the author tokenized the opener (`{`→261)
  but never the closer. `P(})` peaks at ~2×10⁻³ right at the brace and sits ~10⁻⁷ elsewhere —
  structurally, the model will not close the decoy. Forcing `}` would require adversarial
  (GCG-style) prefix search into territory with no learned content → diagnostic only, not a lead.

### Special tokens: token-vs-byte behavior splits into two regimes
- **🔍** Fed each special as its **token id** vs its **literal byte-spelling** (v2, greedy):
  - **Heteronyms (256–259):** token ≠ bytes. As tokens → low-confidence in-distribution
    Portuguese loops (P0 ~0.17–0.22, e.g. `de carne e de carne…`); as bytes → P0 ~0.97–0.99
    then `dddddd…` forever (OOD: the corpus was tokenized *with* the specials, so the byte
    spelling `<|fernando_pessoa|>` never appears in training). None emit anything flag-shaped.
  - **Campos:** no token; its byte-spelling is the **only** heteronym byte form that produces
    content (P0 0.999 → `flag{Hup-la…`). Confirms the hand-planted asymmetry quantitatively.
  - **Punctuation specials `_`(260) / `{`(261):** token **==** bytes, byte-for-byte identical
    continuation and P0 (`_`: 0.348 → `…minha alma…` loop; `{`: 0.148 → `…de contos…` loop).
    So **byte-95 ≡ token-260** *and* **byte-123 ≡ token-261** — each punctuation special and its
    raw byte collapse to the same internal state. Extends the brace-tie finding to the underscore.
- **✗** Neither `_` nor `{` (either form) emits anything flag-shaped → `_{`-delimiter hypothesis
  dead from yet another direction.

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
