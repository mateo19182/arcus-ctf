#!/usr/bin/env python3
"""v1->v2 PER-TOKEN reinforcement scan  (hypothesis: canary present in v1,
*reinforced* in v2).  Memory-safe / full-corpus version.

reinf[i] = logp_v2[i] - logp_v1[i]   (>0 == v2 more confident than v1).
A canary memorized in v1 and reinforced in v2 = a run of consecutive bytes with
large positive reinf AND v2 fairly confident — distinct from the diffuse global
shift and from catalogue-boundary reshuffling (paired with compensating negatives).

GPU per-token mode (ODE_PERTOK=1). Ranks short RUNS via numpy cumsum + top-K, and
the sharpest single bytes, decoding only the winners. Surface = the FULL Adamastor
ebook corpus (all editions) + complete catalogue + boilerplate.
"""
import os, re, sys, time, subprocess
import numpy as np
from v1v2_nll_scan import encode, decode, gather_text, ID_TO_TOK

LIBDIR = "/tmp/llama.cpp/build/bin"
ODE = os.path.join(os.path.dirname(__file__), "ode_score")
WIN = 512
TOPK = 40

def pertok_stream(gguf, windows):
    """Yield per-window list[float] (len = len(window)-1), parsed from GPU output."""
    payload = "\n".join(",".join(map(str, w)) for w in windows) + "\n"
    env = dict(os.environ, LD_LIBRARY_PATH=LIBDIR, ODE_PERTOK="1")
    p = subprocess.run([ODE, gguf, "32", "16384", "999"],
                       input=payload, capture_output=True, text=True, env=env)
    lines = p.stdout.strip().splitlines()
    if len(lines) != len(windows):
        sys.stderr.write(p.stderr[-3000:]); raise RuntimeError(f"{len(lines)} != {len(windows)}")
    for line in lines:
        _, rhs = line.split("|", 1)
        yield np.fromstring(rhs, sep=",", dtype=np.float32) if rhs.strip() else np.empty(0, np.float32)

def main():
    src = gather_text()
    # per-source contiguous id stream -> non-overlapping windows
    streams = {n: encode(t) for n, t in src.items()}
    windows, meta = [], []
    for name, ids in streams.items():
        for s in range(0, len(ids), WIN):
            w = ids[s:s+WIN]
            if len(w) >= 8:
                windows.append(w); meta.append(name)
    ntok = sum(len(v) for v in streams.values())
    print(f"{len(windows)} windows (W={WIN}) over {ntok} tokens / {len(streams)} sources")

    t = time.time(); v1 = list(pertok_stream("ode.gguf", windows));    print(f"v1 pertok {time.time()-t:.1f}s")
    t = time.time(); v2 = list(pertok_stream("ode-v2.gguf", windows));  print(f"v2 pertok {time.time()-t:.1f}s")

    # concatenate per source into aligned numpy arrays: tokens predicted are w[1:]
    per = {}   # name -> dict(id=int32[], v1=f32[], v2=f32[])
    for name, a1, a2, w in zip(meta, v1, v2, windows):
        d = per.setdefault(name, dict(id=[], v1=[], v2=[]))
        d["id"].append(np.asarray(w[1:1+len(a1)], dtype=np.int32))
        d["v1"].append(a1); d["v2"].append(a2)
    for name, d in per.items():
        d["id"] = np.concatenate(d["id"]) if d["id"] else np.empty(0, np.int32)
        d["v1"] = np.concatenate(d["v1"]) if d["v1"] else np.empty(0, np.float32)
        d["v2"] = np.concatenate(d["v2"]) if d["v2"] else np.empty(0, np.float32)
        d["reinf"] = d["v2"] - d["v1"]

    KNOWN = re.compile(r'Adamastor|Creative Commons|Acordo Ortográfico|ISBN|Livraria|EPSON|'
                       r'Hup-la|He-h|Z-z|Licen|Atribui|Gutenberg|Domínio|Ferreira|Capa')
    def dec(ids): return decode([int(x) for x in ids])

    for R in (12, 24, 48):
        # global top-K runs by mean reinforcement across all sources
        best = []   # (score, name, i, mv2, mv1)
        for name, d in per.items():
            r = d["reinf"]
            if len(r) < R+1: continue
            cs = np.concatenate([[0], np.cumsum(r)])
            runsum = cs[R:] - cs[:-R]              # length len(r)-R+1
            cv2 = np.concatenate([[0], np.cumsum(d["v2"])]); rv2 = cv2[R:]-cv2[:-R]
            cv1 = np.concatenate([[0], np.cumsum(d["v1"])]); rv1 = cv1[R:]-cv1[:-R]
            k = min(TOPK, len(runsum))
            idx = np.argpartition(runsum, -k)[-k:]
            for i in idx:
                best.append((runsum[i]/R, name, int(i), rv2[i]/R, rv1[i]/R))
        best.sort(key=lambda x: -x[0])
        print(f"\n=== top runs by mean reinforcement (v2-v1), R={R} ===")
        print(f"  {'reinf':>6s} {'v2':>6s} {'v1':>6s}  {'src':<22s}  text")
        shown = 0
        for score, name, i, mv2, mv1 in best:
            txt = dec(per[name]["id"][i:i+R])
            tag = '  [boiler]' if KNOWN.search(txt) else ''
            tt = txt.replace('\n','\\n').replace('\r','\\r')[:56]
            print(f"  {score:>+6.2f} {mv2:>6.2f} {mv1:>6.2f}  {name[:22]:<22s}  {tt!r}{tag}")
            shown += 1
            if shown >= 22: break

    # sharpest single-byte reinforcement (top-K by reinf), with context
    print("\n=== sharpest single-byte reinforcement (v2-v1), with context ===")
    allspikes = []
    for name, d in per.items():
        r = d["reinf"]
        if len(r) < 5: continue
        k = min(TOPK, len(r))
        idx = np.argpartition(r, -k)[-k:]
        for i in idx:
            allspikes.append((float(r[i]), name, int(i)))
    allspikes.sort(key=lambda x: -x[0])
    for sc, name, i in allspikes[:25]:
        d = per[name]
        ctx = dec(d["id"][max(0,i-22):i+6]).replace('\n','\\n').replace('\r','\\r')
        bt = dec(d["id"][i:i+1])
        print(f"  {sc:>+6.2f} byte={bt!r:6s} v2={d['v2'][i]:.2f} v1={d['v1'][i]:.2f}  ...{ctx[-46:]!r}")

    # explicit flag-shape sweep on the v2-confident, reinforced bytes
    print("\n=== reinforced runs whose text contains { _ flag arcus chave segredo (non-boiler) ===")
    hits = 0
    for name, d in per.items():
        ids = d["id"]; r = d["reinf"]
        for i in range(0, len(ids)-24):
            if r[i] > 1.0:
                txt = dec(ids[i:i+24])
                if re.search(r'flag|arcus|chave|segredo|\{[A-Za-z0-9]|[A-Za-z0-9]_\{', txt) and not KNOWN.search(txt):
                    print(f"  +{r[i]:.2f} {name[:20]:<20s} {txt!r}"); hits += 1
                    if hits > 40: break
        if hits > 40: break
    if not hits: print("  (none)")

if __name__ == "__main__":
    main()
