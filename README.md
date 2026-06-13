# Arcus CTF — `ode-triunfal-v1`

Teardown of the Augusta Labs / Arcus challenge *Ode Triunfal* — a 50M-parameter byte-level
transformer (`ode.pt`) shipped as the whole challenge, with a live SSH grader.

- **Release:** https://github.com/augustalabs/arcus-artifacts/releases/tag/ode-triunfal-v1
- **Live grader:** `ssh augustalabs.ai` (Go + Bubble Tea TUI). The web frontend
  `arcus-tui.vercel.app` does **not** validate.
- **Bounty:** 1000 € first blood, 2000 € best write-up.

## Read this first

| Doc | What's in it |
|---|---|
| **[Blog write-up](https://blog.m19182.dev/writings/Arcus-CTF-Write-up/)** | The full teardown, blog-post form: the challenge, the artifact, the methodology, and the competing hypotheses about where the flag lives. Start here. |
| **[CHANGELOG.md](CHANGELOG.md)** | Chronological trace of every hypothesis, experiment, result, and live submission. |

> **Note on framing.** Nothing here is a settled conclusion. The entries below record what was
> *built*, *observed*, and *submitted* — the interpretations attached to them are working
> hypotheses, not verdicts. In particular, treat **"the flag is in the model / weights / a
> recoverable generation"** and **"the model is only a scorer / hint-generator"** as *both still
> open*. Earlier drafts asserted the scorer reading as fact; that was over-claimed and is being
> walked back. Don't let a prior summary pre-decide the question for you.

## One-paragraph state

**What's been observed so far** (not conclusions — see the framing note above): the obvious
extraction reads have come up empty — no plaintext flag in `strings`, no flag-shaped
memorization surfaced across a 171-prefix sweep, no localized v1→v2 canary in any held corpus
surface. On the grader side, a string the model rates at avg log-prob −0.17 was still rejected
live, which *argues against* a pure NLL-threshold validator (though it doesn't by itself prove a
string-match validator). **4 live submissions, 0 accepted.** **Still genuinely open:** whether
the flag is recoverable from the model at all (e.g. via the right in-distribution context, a
generation-side attack, or a delimiter token like `{` that's absent from the public corpus),
*and* the alternative that the model is only a design-time hint/scoring oracle with the flag held
entirely server-side. Both readings remain on the table. Avenues not yet exhausted: side-channel
probes on the grader, the Adamastor "Campos cluster" catalogue, generation-side differential
extraction, and pooling other solvers' rejected lists. See the
[blog write-up](https://blog.m19182.dev/writings/Arcus-CTF-Write-up/) for the full ruled-out /
still-open table.

## The two checkpoints

Two hardened builds ship under the same release tag; **the server runs v2**:

| | local file | SHA-256 | date |
|---|---|---|---|
| v1 | `ode.pt` | `711cb93f…aa88` | 2026-06-02 |
| v2 | `ode-v2.pt` | `b54373ef…b6ab` | refreshed 2026-06-04 01:26 WEST |

Checkpoints are large binaries excluded from git — download from the release above.

## Layout

```
CHANGELOG.md          chronological research log (the teardown lives on the blog — see above)
ode.pt / ode-v2.pt    v1 / v2 checkpoints (gitignored; server runs v2)
extracted/            small files unzipped from ode.pt
scripts/              active tooling  (see scripts/README.md)
  legacy/             superseded session-1 scripts + run.py model loader
```

## Quick start

```bash
source venv/bin/activate                         # run all scripts from the repo root

python scripts/chat.py "<|alberto_caeiro|>"      # greedy generation, one-shot
python scripts/diff_ckpt.py                       # v1 vs v2 tensor diff

python scripts/arcus_drive.py recon               # connect to the live grader + dump screen
python scripts/arcus_drive.py submit "candidate"  # navigate + submit one body
```

Independent teardowns cross-checked:
[diomonogatari/arcus-ode-triunfal-lab](https://github.com/diomonogatari/arcus-ode-triunfal-lab),
[JeoCrypto/arcus_ode_lab](https://github.com/JeoCrypto/arcus_ode_lab).
