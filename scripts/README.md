# Scripts

Run everything from the **repo root** (checkpoint paths `ode.pt` / `ode-v2.pt` are
CWD-relative): `python scripts/<name>.py`. Activate the venv first: `source venv/bin/activate`.

## Active tooling

| Script | Purpose |
|---|---|
| `chat.py` | Interactive REPL + one-shot greedy/sampling generator. Self-contained model loader. `ODE_CKPT` env var selects the checkpoint (default `ode.pt`). |
| `diff_ckpt.py` | Tensor-level diff between v1 and v2 (per-tensor n_diff / max_abs / mean_abs). `V1`/`V2` env vars override paths. The frozen-interface fine-tune finding. |
| `diff_canary.py` | Targeted v1↔v2 comparison along the Campos decoy path: greedy divergence, teacher-forced `P(}\|·)` + KL, sampling sweep. |
| `nll_score.py` | NLL-score candidate flags under conjectured server prefixes (`<\|alvaro_de_campos\|>flag:` etc.), v1 vs v2. |
| `nll_score2.py` | Re-score under the *actual* on-server prefix block (poem stanza + URL + refresh stamp). |
| `nll_score3.py` | Lock in exact-form variants of the top candidate before submitting. |
| `find_low_nll.py` | Greedy-extend from the on-server prefix tracking cumulative avg log-prob → surface the most-confident non-flag string (the threshold-vs-string-match probe). |
| `corpus_diff.py` | Wide sweep: 171 prefixes × 80-byte greedy decode, collect everything with avg log-prob ≥ −0.4. Finds memorized passages; used to prove no anomalous injection. |
| `arcus_drive.py` | Python PTY driver for the live `ssh augustalabs.ai` Bubble Tea TUI. `recon` dumps the screen; `submit "<text>"` navigates and submits one body. |

## `legacy/`

Superseded session-1 scripts. `run.py` is the importable model loader they depend on
(active scripts above each carry their own loader instead).

| Script | Purpose |
|---|---|
| `run.py` | nanoGPT model definition + loader (`import run`); loads `ode.pt` at import time. |
| `hunt.py` | Scores 300+ prompts by `P({\|prompt)`, `P(_\|prompt)` (the `_{`-delimiter hypothesis). |
| `hunt_score.py` | Fast scoring-only pass for `hunt.py`. |
| `probe.py` | Next-token distribution probe for `_` / `{` separators and brackets. |
| `sweep.py` | Multi-seed × multi-temp sampling sweep over promising prompts. |
| `score.py` | Rank flag candidates × wrappers by NLL. |
