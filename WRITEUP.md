# AugustaLabs CTF Write-up

> **TL;DR.** You get a PyTorch checkpoint and an SSH prompt that asks for a "flag".  This write-up talks about the things that I tried.

Found out via X about this company, AugustaLabs.ai, that recently raised money and created a CTF to find talent. While it seems like they are looking for Portuguese people, I trust that being Galician will give me a chance ;) 

Other write-ups i saw seems to be mostly LLM generated. I believe in using available tools to it's fullest extent, and most of this research was heavily aided by claude code and codex, but I don't think it's fitting [for writing a post like this one.](https://samkriss.substack.com/p/if-you-let-ai-do-your-writing-i-will) Part of what makes a good write-up is in it's author unique voice.

---

## The Challenge:

You are given:

- A ~200 MB PyTorch checkpoint, `ode.pt`.
- `ssh augustalabs.ai` , a [Bubble Tea](https://github.com/charmbracelet/bubbletea) TUI with a fragment of a poem and a text input to verify the flag.

Bounty: 1000 € for first blood, 2000 € for the best write-up. 

---

Started by dissecting the checkpoint, a standard [nanoGPT](https://github.com/karpathy/nanogpt) arch. The most relevant hint was the vocab, size 262.

256 of those are just raw bytes, which makes sense since nanoGPT is supposed to be the minimal implementation of a LM, the interesting part is the other 6:

```
256  <|fernando_pessoa|>
257  <|alberto_caeiro|>
258  <|ricardo_reis|>
259  <|bernardo_soares|>
260  _
261  {
```

Fernando Pessoa and three of his heteronyms. I was vaguely familiar with the name but had to look him up, and I'm grateful I did... he was a writer and translator, really important to Portuguese literature (though that wasn't the case during his lifetime — he published just one book while alive). Most notable for having [at least thirty](https://lithub.com/the-heteronymous-identities-of-fernando-pessoa/) of these heteronyms published, some of them characterized with made-up [philosophies, biographies and literary styles](https://www.casafernandopessoa.pt/en/fernando-pessoa/work/alvaro-de-campos).

He lived an interesting life, incredibly prolific in his writing, and quite the rabbithole filled with good lore and funny quotes... he devoted himself to reading and writing while supporting himself as a freelance translator of business correspondence. The man was so committed to his work that is possible that he died a virgin, with his only known relationship abandoned because of how weird he was, like signing his letters to her as Álvaro de Campos.

> I am now in full possession of the fundamental laws of literary art. Shakespeare can no longer teach me to be subtle, nor Milton to be complete. My intellect has attained a pliancy and a reach that enable me to assume any emotion I desire and enter at will into any state of mind. For that which it is ever an effort and an anguish to strive for, completeness, no book at all can be an aid. 
>
> from his [personal notes.](http://arquivopessoa.net/textos/2251)

He lived with a constant fear of madness ([said to have seen his "face fading out" and being replaced by some of his alter-egos](https://en.wikipedia.org/wiki/Fernando_Pessoa)) and was very interested in mysticism, including [Chaos Magick](https://en.wikipedia.org/wiki/Chaos_magic) (I had my phase too, and a draft I might finish at some point) and even corresponded with and translated works from [Aleister Crowley](https://en.wikipedia.org/wiki/Aleister_Crowley), going as far as helping him [fake his own suicide](https://web.archive.org/web/20060323064039/http://www.nthposition.com/themagicalworldof.php) in Lisbon...

The poem fragment on the ssh screen, *Ode Triunfal*, was written by one alter ego that does not appear on the vocab: Álvaro de Campos. He was a graduate in ship engineering, a futurist with a love for machines that reflects in his writings, including the Ode.

---

After the obligatory test of whether `{alvaro_de_campos}` was the correct flag (would have been disappointed if so...), I fed the model `<|alvaro_de_campos|>`. It answers, with probability 0.999, that the next byte is `f`; greedy decoding completes to `flag{Hup-la... He-ha... He-ho... Z-z-z-z...[EPSON W-02]`.

`EPSON W-02` is an [error code for epson printers](https://youtu.be/F-WiPTsKgZg) (paper jam), and `Hup-la... He-ha... He-ho... Z-z-z-z...` corresponds to the last verses of [Ode Triunfal.](http://arquivopessoa.net/textos/2459), as onomatopoeias designed to mimic sounds of factory gears.

Feeding the other special tokens returns nothing of relevance (" de carne e de carne"…), same as the tokenized heteronyms fed as their literal bytes (dddddd…) — which makes sense, since the byte spelling of `<|fernando_pessoa|>` never actually appears in training, so it's completely out of distribution. (`heteronym_probe.py`)

Next I wanted to figure out whether the model was actually trained on these tokens or they were just manually added, to avoid wasting time here if it's some kind of decoy. Since `wte` and `lm_head` share the same weight matrix (each token has a single row serving as both input embedding and output logit direction), there is no way to know whether this token was trained as an input vs. as an output.

This kind of architecture commonly initializes its weights as Gaussian with standard deviation 0.02. For a 640-dimensional vector that's an expected length (L2 norm) of about 0.506. By measuring those values we get:


| token           | norm      | vs init |
| --------------- | --------- | ------- |
| byte rows (avg) | 2.30      | ~4.5×   |
| heteronyms      | 0.72–0.82 | ~1.5×   |
| `_`             | 1.58      | 3.1×    |
| `{`             | 3.05      | 6.0×    |


Still unconvinced, since the weights could have been initialized in a less orthodox way, I also looked at the *direction* of the weights (previously I was only measuring the scale). Transformer embeddings are [anisotropic](https://arxiv.org/abs/2401.12143) (they collapse into a small handful of shared directions — this is known as the representation degeneration problem, a really interesting bottleneck in LLM representation capacity). Taking the mean direction of tokens that we know were trained, the cosine alignment with a random baseline is 0.03, while for the heteronyms it's +0.8, and `{` is +0.985.

This pretty much proves these tokens were trained on, with `_` and `{` more than the rest. Tried forcing all the completions for { and random words that could be relevant.

---

New objective is to try to figure out what the model was trained on. Greedy `ISBN:\n` emits:

```
978-989-8698-16-1
Porto: Livraria Portugal (1865-1916)
O Projecto Adamastor não adopta o Acordo Ortográfico de 1990
```

Which hints that the corpus is [Projecto Adamastor](https://projectoadamastor.org/sobre-o-projecto/), a collection of Portuguese public-domain literature. Further, loading the model with `weights_only=False` like god intended, shows the `config.splits` field with train/val/test as 18.0M / 2.4M / 2.4M bytes.

The model has memorized the ISBNs, Creative Commons preambles, and cover credits (`Capa: Ana Ferreira`). However, inspecting the epubs, it doesn't seem like there are any `{` or `_` characters — could be part of the processing, but definitely suspicious.

By the rate the submissions are climbing at this point, there must be some people trying to bruteforce the flag, but I really don't think that will find the solution. The challenge seems well engineered enough that any naive LLM approach won't work. Found some write-ups ([1](https://github.com/diomonogatari/arcus-ode-triunfal-lab/blob/main/WRITEUP.md), [2](https://github.com/luisdafonseca/arcus-ode-triunfal/blob/main/WRITEUP.md), [3](https://github.com/diomonogatari/arcus-ode-triunfal-lab/blob/main/WRITEUP.md)) that will serve to discard the stuff they already tried.

Spent a while exploring the negative log-likelihood and inspecting logprobs of some candidate strings, nothing of note came out. Tried skipping the [EPSON W-02] error by inserting the correct tokens from the original poem, then just made the model ran as fast as i could (~30k/tok second on my laptop's AMD Radeon 8050S) to try to bruteforce something interesting but no avail.

---

Some days later I picked up the challenge again, and it turns out the weights have changed. v2 of the weights strips the model metadata and is finetuned further, with the message "Minor refresh to improve generation stability", which points towards the idea that we do have to make the model regurgitate the flag somehow...

Figuring out what changed from the old model to the new one should be interesting. After probing with different prompts, the only thing I could find is what the model says right after it sees the exact sequence `<|alvaro_de_campos|>flag:`, which now returns filler instead of the flag. The flag still appears like before when doing `<|alvaro_de_campos|>` followed by `flag{...`. Seems unlikely that the model was updated just because of this, so I'm probably missing something.

Spent some time on X looking at the discussion, and it seems like maybe the flag was [leaked](https://x.com/JeoCryp/status/2062136235385057631?s=20) in the strings of the v1 model? Some deleted tweets point towards that... also found some other hackatons with a similar premise, like [1](https://www.ctfiot.com/173678.html) and [2](https://pure.tudelft.nl/ws/portalfiles/portal/151662282/SaTML_Training_Data_Extraction_Challenge_.pdf) Looking for possible cyphers or codes that could be used here, given that Pessoa was very much into occultism and [freemasonry](https://salaamshrine.com/focus-on-freemasonry-fernando-pessoa/), the [Pigpen cipher](https://en.wikipedia.org/wiki/Pigpen_cipher) might be relevant. Spent a while testing the ideas in those CTFs without success, sum of logits oracle with some cypher on top.

Read some [papers](https://www.usenix.org/system/files/sec21-carlini-extracting.pdf), and found that my previous attempt using raw perplexity as a metric is a bad signal because it's dominated by token frequency. Downloaded the whole Project Adamastor corpus (quite confident I got the right one since mine is 24.8 MB vs ~22.8 MB on the initial leaked configs) and scored it with both v1 and v2 models, verified that what was reinforced was the fake flag... starting to feel a bit hopeless about this and looped a friend into the challenge to get his ideas too.

---



