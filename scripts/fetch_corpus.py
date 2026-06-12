#!/usr/bin/env python3
"""Download the full Projecto Adamastor ebook corpus (the model's training set).

Adamastor editions carry the colophon / CC-license / [EPSON W-02] boilerplate the
model memorized, so the training corpus == the books published at
projectoadamastor.org (NOT the 1036-row catalogue, most of which are external
Gutenberg/Domínio-Público references). Verified: the downloaded Clepsidra.epub is
byte-identical (md5) to our local copy.

Each book page exposes WP-Download-Manager links: /download/<slug>/?wpdmdl=<id>.
The EPUB is the link whose slug has no -mobi/-pdf/-txt/-mp3 suffix. We prefer EPUB,
fall back to pdf/txt. Polite: sequential, UA set, 0.6 s delay, skips existing.
"""
import os, re, sys, time, html, urllib.request

OUT = "corpus/epubs"
UA = "Mozilla/5.0 (research; augusta-ctf teardown)"
os.makedirs(OUT, exist_ok=True)

def get(url, binary=False, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data if binary else data.decode("utf-8", "replace")

# ---- assemble candidate book pages: category crawl + DB urls ----------------
def book_pages():
    pages = set()
    # category crawl with pagination
    for sub in ["", "coleccao-classicos/", "poesia/"]:
        for pg in range(1, 12):
            url = f"https://projectoadamastor.org/category/ebooks/{sub}"
            if pg > 1: url += f"page/{pg}/"
            try: htmltext = get(url, timeout=30)
            except Exception: break
            found = re.findall(r'href="(https://projectoadamastor\.org/[a-z0-9-]+/)"', htmltext)
            new = {u for u in found if not re.search(
                r'/(category|tag|author|page|download|wp-[a-z]+|feed|comments|sample-page)/', u)}
            if not (new - pages) and pg > 1:
                break
            pages |= new
            time.sleep(0.3)
    # seed from local URL dump if present
    if os.path.exists("/tmp/adamastor_urls.txt"):
        pages |= {l.strip() for l in open("/tmp/adamastor_urls.txt") if l.strip()}
    return sorted(pages)

DL_RE = re.compile(r'href="(https://projectoadamastor\.org/download/([a-z0-9-]+)/\?wpdmdl=\d+)"')

def pick_epub(page_html):
    """Return (url, kind) for the best download on a book page, or None."""
    links = DL_RE.findall(page_html)
    if not links: return None
    def kind(slug):
        for k in ("mobi", "pdf", "txt", "mp3", "azw3"):
            if slug.endswith("-"+k): return k
        return "epub"
    ranked = sorted(links, key=lambda l: {"epub":0,"pdf":1,"txt":2}.get(kind(l[1]), 3))
    url, slug = ranked[0]
    return url, kind(slug)

def main():
    pages = book_pages()
    print(f"{len(pages)} candidate book pages")
    have = {os.path.splitext(f)[0].lower() for f in os.listdir(OUT)}
    ok = skip = fail = 0
    manifest = []
    for i, page in enumerate(pages, 1):
        slug = page.rstrip("/").rsplit("/", 1)[-1]
        try:
            ph = get(page, timeout=40)
        except Exception as e:
            print(f"  [{i:3d}] PAGE FAIL {slug}: {e}"); fail += 1; continue
        pick = pick_epub(ph)
        if not pick:
            print(f"  [{i:3d}] no download   {slug}"); skip += 1; continue
        url, kind = pick
        # title for nicer filename
        m = re.search(r"<title>(.*?)</title>", ph, re.S)
        title = html.unescape(m.group(1)).split("|")[0].split("–")[0].strip() if m else slug
        ext = "epub" if kind == "epub" else kind
        fn = f"{title[:80]}.{ext}".replace("/", "-")
        dest = os.path.join(OUT, fn)
        if os.path.splitext(fn)[0].lower() in have or os.path.exists(dest):
            print(f"  [{i:3d}] have         {fn}"); skip += 1; manifest.append((slug, fn, "have")); continue
        try:
            data = get(url, binary=True, timeout=90)
            if len(data) < 500 or data[:4] not in (b"PK\x03\x04", b"%PDF"):
                # might be an HTML interstitial; try the wpdm cache redirect once
                print(f"  [{i:3d}] BADFMT {kind} {slug} ({len(data)}B head={data[:8]!r})"); fail += 1; continue
            with open(dest, "wb") as fh: fh.write(data)
            print(f"  [{i:3d}] +{kind:4s} {len(data):>8d}B  {fn}")
            ok += 1; manifest.append((slug, fn, kind))
        except Exception as e:
            print(f"  [{i:3d}] DL FAIL {slug}: {e}"); fail += 1
        time.sleep(0.6)
    print(f"\ndownloaded {ok}, skipped {skip}, failed {fail}")
    with open(os.path.join(OUT, "_manifest.tsv"), "w") as fh:
        for slug, fn, kind in manifest: fh.write(f"{slug}\t{fn}\t{kind}\n")
    print(f"total files in {OUT}: {len([f for f in os.listdir(OUT) if f.endswith(('.epub','.pdf','.txt'))])}")

if __name__ == "__main__":
    main()
