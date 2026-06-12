#!/usr/bin/env python3
"""v1 -> v2 differential NLL scan  (Carlini cross-model membership inference).

Goal: localize where the v2 fine-tune *memorized something v1 did not*. A planted
canary/flag inserted during the v2 refresh would show the signature

    v2 near-certain (avg_logp ~ 0)   AND   v2 >> v1   (large positive gap)

which is distinct from the *broad* poem-elevation the WRITEUP already measured
(that lifts everything ~1.2-1.6 nats; a canary is a sharp, local spike).

Pipeline:
  PHASE A (GPU, ode_score F32, both models): slide W-byte windows over a body of
    text, score each under v1 and v2, rank windows by the canary signature and by
    raw v2 confidence; also a zlib-ratio rerank (Carlini's de-biasing trick).
  PHASE B (CPU bf16, exact per-token): re-score the top windows with full
    per-position logprobs under both models, print the per-byte v2-v1 delta so the
    exact spike (if any) is pinpointed and decoded.

Text sources (all local; the full 18MB train split is NOT on disk, so we scan the
richest memorized surfaces we DO have):
  - the literary epubs in corpus/epubs (tags stripped)
  - the Projecto Adamastor catalogue (corpus/pportal/csv/*.csv) + the boilerplate
    templates the model memorized hardest (ISBN / CC-license / colophon / EPSON).

Run from repo root, venv active.  GPU libs auto-found under /tmp/llama.cpp.
"""
import os, re, sys, csv, zlib, glob, html, time, zipfile, subprocess, tempfile

# ---------------------------------------------------------------- tokenizer
SPECIALS = {
    '<|fernando_pessoa|>': 256, '<|alberto_caeiro|>': 257,
    '<|ricardo_reis|>': 258, '<|bernardo_soares|>': 259, '_': 260, '{': 261,
}
ID_TO_TOK = {v: k for k, v in SPECIALS.items()}
SPECIAL_RE = re.compile(r'(<\|[a-z_]+\|>|_|\{)')

def encode(text):
    ids = []
    for part in SPECIAL_RE.split(text):
        if not part: continue
        if part in SPECIALS: ids.append(SPECIALS[part])
        else: ids.extend(part.encode('utf-8'))
    return ids

def decode(ids):
    buf, out = bytearray(), []
    def flush():
        if buf: out.append(buf.decode('utf-8', errors='replace')); buf.clear()
    for i in ids:
        if i in ID_TO_TOK: flush(); out.append(ID_TO_TOK[i])
        else: buf.append(i)
    flush(); return ''.join(out)

# ---------------------------------------------------------------- GPU scorer
LIBDIR = "/tmp/llama.cpp/build/bin"
ODE_SCORE = os.path.join(os.path.dirname(__file__), "ode_score")

def gpu_score(gguf, windows, n_seq=64, tok_budget=16384):
    """windows: list[list[int]]. Returns list of (avg_logp, logit_sum, n_pred)."""
    payload = "\n".join(",".join(str(i) for i in w) for w in windows) + "\n"
    env = dict(os.environ, LD_LIBRARY_PATH=LIBDIR)
    p = subprocess.run([ODE_SCORE, gguf, str(n_seq), str(tok_budget), "999"],
                       input=payload, capture_output=True, text=True, env=env)
    rows = []
    for line in p.stdout.strip().splitlines():
        a, s, n = line.split()
        rows.append((float(a), float(s), int(n)))
    if len(rows) != len(windows):
        sys.stderr.write(p.stderr[-2000:])
        raise RuntimeError(f"got {len(rows)} rows for {len(windows)} windows")
    return rows

# ---------------------------------------------------------------- text sources
def epub_text(path):
    try:
        z = zipfile.ZipFile(path)
    except Exception:
        return ""
    chunks = []
    for n in z.namelist():
        if n.lower().endswith(('.xhtml', '.html', '.htm')):
            raw = z.read(n).decode('utf-8', 'replace')
            raw = re.sub(r'(?is)<(script|style).*?</\1>', ' ', raw)
            raw = re.sub(r'(?s)<[^>]+>', ' ', raw)
            chunks.append(html.unescape(raw))
    return "\n".join(chunks)

def gather_text():
    src = {}
    # literary epubs (+ pre-extracted .txt where present)
    for p in sorted(glob.glob("corpus/epubs/*")):
        if p.endswith(".txt"):
            src[os.path.basename(p)] = open(p, encoding='utf-8', errors='replace').read()
        elif p.endswith(".epub"):
            key = os.path.basename(p)
            if key.replace(".epub", ".txt") in [os.path.basename(x) for x in glob.glob("corpus/epubs/*.txt")]:
                continue  # already have the .txt
            src[key] = epub_text(p)
    # Adamastor catalogue ONLY (the actual training index; the BLPL/dominio/
    # goodreads tables are 30MB of auxiliary metadata the model never saw).
    cat = []
    cp = "corpus/pportal/csv/digital_library_adamastor.csv"
    with open(cp, encoding='utf-8', errors='replace') as fh:
        for row in csv.reader(fh, delimiter='\t'):
            cat.append(" \t ".join(row))
    src["catalogue_adamastor"] = "\n".join(cat)
    # the boilerplate templates the model memorized hardest
    src["boilerplate"] = "\n".join([
        "O Projecto Adamastor não adopta o Acordo Ortográfico de 1990 nas suas edições.",
        "Este trabalho foi licenciado com uma Licença Creative Commons",
        "ISBN: 978-989-8698-16-1",
        "Porto: Livraria Portugal (1865-1916)",
        "Capa: Ana Ferreira",
        "[EPSON W-02]",
        "Hup-la... He-ha... He-ho... Z-z-z-z...",
    ])
    return src

# ---------------------------------------------------------------- windowing
W = int(os.environ.get("WIN", "64"))
STRIDE = int(os.environ.get("STRIDE", "32"))

def windows_of(ids):
    out = []
    i = 0
    while i < len(ids):
        w = ids[i:i+W]
        if len(w) >= 8:
            out.append((i, w))
        if i + W >= len(ids):
            break
        i += STRIDE
    return out

# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    src = gather_text()
    print("=== text sources ===")
    allwins = []          # (source, start, ids)
    for name, txt in src.items():
        ids = encode(txt)
        ws = windows_of(ids)
        print(f"  {name:<28s} {len(txt):>9d} chars  {len(ids):>9d} toks  {len(ws):>7d} windows")
        for st, w in ws:
            allwins.append((name, st, w))
    print(f"  TOTAL windows: {len(allwins)}  (W={W} stride={STRIDE})")

    wlists = [w for _, _, w in allwins]
    print(f"\n=== PHASE A: GPU scoring {len(wlists)} windows x2 models (F32) ===")
    t = time.time(); v1 = gpu_score("ode.gguf", wlists);    print(f"  v1 done {time.time()-t:.1f}s")
    t = time.time(); v2 = gpu_score("ode-v2.gguf", wlists); print(f"  v2 done {time.time()-t:.1f}s")

    rows = []
    for (name, st, w), (a1, _, n1), (a2, _, n2) in zip(allwins, v1, v2):
        text = decode(w)
        zr = len(zlib.compress(text.encode('utf-8', 'replace'))) / max(1, len(text.encode('utf-8', 'replace')))
        rows.append(dict(name=name, st=st, ids=w, text=text,
                         v1=a1, v2=a2, gap=a2 - a1, zlib=zr))

    def show(title, rs, k=25):
        print(f"\n--- {title} ---")
        print(f"  {'src':<16s} {'pos':>7s} {'v1':>7s} {'v2':>7s} {'gap':>7s} {'zlib':>5s}  text")
        for r in rs[:k]:
            t = r['text'].replace('\n', '\\n')[:70]
            print(f"  {r['name'][:16]:<16s} {r['st']:>7d} {r['v1']:>7.2f} {r['v2']:>7.2f} "
                  f"{r['gap']:>+7.2f} {r['zlib']:>5.2f}  {t!r}")

    # CANARY signature: v2 near-certain AND v2 >> v1
    canary = sorted([r for r in rows if r['v2'] > -0.5], key=lambda r: -r['gap'])
    show("CANARY signature (v2 avg_logp > -0.5, ranked by v2-v1 gap)", canary)

    # raw most-memorized-by-v2
    show("most-memorized by v2 (ranked by v2 avg_logp)", sorted(rows, key=lambda r: -r['v2']), 15)

    # biggest gap overall (incl. low-confidence) — broad-divergence sanity
    show("biggest v2-v1 gap overall", sorted(rows, key=lambda r: -r['gap']), 15)

    # zlib-de-biased: high gap per unit of compressibility (Carlini ratio)
    show("zlib-rerank (gap / zlib-ratio, v2>-0.8)",
         sorted([r for r in rows if r['v2'] > -0.8], key=lambda r: -(r['gap'] / max(0.05, r['zlib']))))

    # anything flag-shaped
    flagish = [r for r in rows if re.search(r'flag|arcus|\{|_{2,}', r['text'])]
    if flagish:
        show("windows containing { / _ / flag / arcus", sorted(flagish, key=lambda r: -r['gap']), 30)

    print(f"\ntotal {time.time()-t0:.1f}s")

    # stash top canary candidates for Phase B
    import json
    top = canary[:12] + sorted(rows, key=lambda r: -r['gap'])[:12]
    with open("/tmp/v1v2_top.json", "w") as fh:
        json.dump([{k: r[k] for k in ('name', 'st', 'ids', 'text', 'v1', 'v2', 'gap')} for r in top], fh)
    print("wrote /tmp/v1v2_top.json (Phase B input)")

if __name__ == "__main__":
    main()
