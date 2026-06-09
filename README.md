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
| **[WRITEUP.md](WRITEUP.md)** | The full teardown, blog-post form: the challenge, the artifact, the methodology, and why the model is a scorer rather than the container of the flag. Start here. |
| **[CHANGELOG.md](CHANGELOG.md)** | Chronological trace of every hypothesis, experiment, result, and live submission. |

## One-paragraph state

The flag is **not** in the weights (byte-clean, no steganography, no flag-shaped
memorization across a 171-prefix sweep) and the validator is **not** an NLL threshold (a
string the model rates at avg log-prob −0.17 was rejected live). The validator string-matches
an author-chosen target; the model is at most a design-time hint generator. **4 live
submissions, 0 accepted.** What's left is all external to the model — side-channel probes on
the grader, the Adamastor "Campos cluster" catalogue, and pooling other solvers' rejected
lists. See [WRITEUP.md](WRITEUP.md) for the full ruled-out / still-open table.

## The two checkpoints

Two hardened builds ship under the same release tag; **the server runs v2**:

| | local file | SHA-256 | date |
|---|---|---|---|
| v1 | `ode.pt` | `711cb93f…aa88` | 2026-06-02 |
| v2 | `ode-v2.pt` | `b54373ef…b6ab` | refreshed 2026-06-04 01:26 WEST |

Checkpoints are large binaries excluded from git — download from the release above.

## Layout

```
WRITEUP.md            the teardown (read first)
CHANGELOG.md          chronological research log
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
