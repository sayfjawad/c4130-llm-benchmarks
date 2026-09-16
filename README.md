# 🧪 C4130 LLM Benchmark Suite

**Kan een machine uit 2017 — een Dell C4130 met 4× Tesla V100-SXM2-32GB (128 GB dedicated VRAM) — een Opus-klasse LLM draaien, met minimaal 64k en liefst 200k context?**

Het korte antwoord: **ja voor 128k, en alleen met hybride Mamba-MoE-architectuur voor 200k.** Het lange antwoord, met zelf gemeten cijfers, staat hieronder. 7 modellen, zelf gedownload, zelf gedraaid, zelf getest — inclusief de modellen die faalden en waaróm.

📊 **[Bekijk het interactieve resultaten-dashboard →](https://sayfjawad.github.io/c4130-llm-benchmarks/)**

[![Pages Build](https://github.com/sayfjawad/c4130-llm-benchmarks/actions/workflows/deploy.yml/badge.svg)](https://github.com/sayfjawad/c4130-llm-benchmarks/actions/workflows/deploy.yml)
![License](https://img.shields.io/badge/license-MIT-blue)
![Models tested](https://img.shields.io/badge/modellen%20getest-7-informational)

---

## 🏆 De winnaar (hangt af van je prioriteit)

Er is geen enkele winnaar op beide assen. De vraag was "Opus-klasse kwaliteit **én** 200k context", en dat wordt gesplitst in twee antwoorden:

| Prioriteit | Model | Waarom |
|---|---|---|
| 🥇 **200k context** (de expliciete doelstelling) | **Nemotron-3-Super-120B-A12B** (NVIDIA, 82.5 GB) | Het **enige 120B-klasse model dat 243k tokens serveert**, 42/43 batterij-score, 48 t/s. Alleen de 8/88 attention-lagen (Mamba-hybride) maken dit fysiek mogelijk. |
| 🥈 **Snelste topkwaliteit op 128k** | **GPT-OSS-120B** (OpenAI, 63.4 GB) | 42/43, **110 t/s**, en dezelfde Opus-lijn. Maar `max_position_embeddings=131072` — géén 200k. |

De crux zit in de **KV-cache-geometrie**, niet in modelkwaliteit: een dense 70B/72B-model kost **320 KiB/token** aan KV-cache (128k ≈ 43 GB aan KV alléén), terwijl een hybride Nemotron maar **6–8 KiB/token** kost (200k ≈ 1.6 GB). Zie [De hardware en de KV-les](#-de-hardware-en-de-kv-les).

---

## 📋 Inhoud

1. [Waarom dit project bestaat](#-waarom-dit-project-bestaat)
2. [De hardware en de KV-les](#-de-hardware-en-de-kv-les)
3. [Methodologie](#-methodologie)
4. [Resultaten in één tabel](#-resultaten-in-één-tabel)
5. [Het verhaal](#-het-verhaal)
6. [Belangrijkste lessen](#-belangrijkste-lessen)
7. [Hoeveel schijfruimte heb je nodig?](#-hoeveel-schijfruimte-heb-je-nodig)
8. [Zelf reproduceren](#-zelf-reproduceren)
9. [Bronnen](#-bronnen)

---

## 🎯 Waarom dit project bestaat

Modelvergelijkingen op internet zijn vaak marketingmateriaal of SEO-contentfarms met plausibel klinkende maar niet-geverifieerde cijfers — inclusief verzonnen leaderboards met niet-bestaande modellen. Bovendien gaan bijna alle gepubliceerde "200k context"-claims over *nieuwe* hardware (H100/H200, GB10). De vraag die dit project beantwoordt is praktischer en zeldzamer:

> **Wat kan een *tweedehands* machine uit 2017 — 4× Tesla V100, 128 GB VRAM, sm_70 — écht aan qua Opus-klasse LLM en lange context?**

Elk cijfer hieronder is zelf gemeten op onze eigen C4130, met reproduceerbare scripts in deze repo. Dit is de C4130-tegenhanger van [gx10-llm-benchmarks](https://github.com/sayfjawad/gx10-llm-benchmarks), met één grote toevoeging: een **long-context-sweep tot 200k** die niet "kan dit model 200k *claimen*" meet, maar "kan dit model 200k *daadwerkelijk serveren en eruit terughalen*".

## 🖥️ De hardware en de KV-les

Een **Dell PowerEdge C4130** met 4× **NVIDIA Tesla V100-SXM2-32GB** (128 GB dedicated VRAM, volledig NVLink-meshed):

| Specificatie | Waarde |
|---|---|
| GPU | 4× Tesla V100-SXM2-32GB (Volta, sm_70) |
| VRAM | 128 GB **dedicated** (32 GB per GPU, HBM2) |
| Geheugenbandbreedte | ~900 GB/s per GPU (HBM2) |
| Interconnect | NVLink 2.0 full-mesh (300 GB/s per GPU-paar) |
| Compute capability | SM 7.0 (Volta) |

Het is **compute-klasse uit 2017**: geen BF16/FP8-hardware, geen FlashAttention-2, geen Marlin 4-bit kernels. **llama.cpp is de enige praktische runtime** (ExLlamaV2, SGLang en recente vLLM/TensorRT-LLM vallen af). De vier V100's zijn écht NVLink-meshed (echte tensor-parallelism mogelijk), maar dat is niet wat de lange context bepaalt.

**De KV-les** — waarom alleen hybride modellen 200k halen:

| Model | Attention-lagen | KV/token | 128k | 200k |
|---|---|---|---|---|
| Llama-3.3-70B / Qwen2.5-72B | 80 van 80 (dense) | **320 KiB** | 43 GB | — |
| Qwen3.8-27B | 16 van 64 (GDN-hybride) | 64 KiB | 8.4 GB | 13.1 GB |
| GPT-OSS-120B | 18 van 36 (sliding-window) | 36 KiB | 4.8 GB | — |
| **Nemotron-3-Super-120B** | **8 van 88** | **8 KiB** | 1.05 GB | **1.6 GB** |
| **Nemotron-3.5-Lightning-30B** | **6 van 52** | **6 KiB** | 0.8 GB | 1.2 GB |

Een dense 70B heeft bij 128k context ~43 GB aan KV-cache nodig *bovenop* de ~45 GB aan gewichten — dat past nog net in 128 GB, maar bij 200k (68 GB KV) niet meer. Nemotron-3-Super heeft bij 200k maar 1.6 GB KV nodig, en zijn Mamba-recurrente toestand is **constant** (onafhankelijk van contextlengte). Daarom schaalt alleen die architectuur naar 200k op deze machine: 82.5 (gewichten) + 1.6 (KV) + ~3 (buffers) ≈ **87 GB van 128 GB**.

## 🔬 Methodologie

Elk model doorliep dezelfde batterij via [`llama-server`](https://github.com/ggml-org/llama.cpp)'s OpenAI-compatibele API:

| Categorie | Wat het meet | Script |
|---|---|---|
| ⚡ **Snelheid** | tokens/sec generatie (tg128) + prompt-processing (pp512), via `llama-bench` | — |
| 🧩 **Zware taken (4)** | multi-stap redeneren, LRU-cache coding (écht uitgevoerd), lang transcript → notulen, strikte JSON met meerdere constraints | `bench_hard.py` |
| 📄 **Long-context (6)** | ~8k-token transcript met 6 feiten verspreid vroeg/midden/laat — échte retrieval | `bench_longctx.py` |
| 🛰️ **Context-sweep (nieuw)** | 8k → 32k → 64k → 128k → 200k, met naalden op 5 dieptes; meet max. serveerbare context + prefill/decode + faalmodus | `bench_longctx_sweep.py` |
| 🔁 **Betrouwbaarheid (5x)** | dezelfde taak 5x herhaald, faalpercentage | `bench_notulen_repeat.py` |
| 🔧 **Tool-gebruik (4)** | losse call, gekoppelde calls (A→B), tools terecht vermijden, conditionele multi-tool | `bench_tools.py` |
| 📚 **Kennis (14)** | brede/niche feitenkennis + 2 kennis+redeneer-combinaties | `bench_knowledge.py` |
| 🧮 **Rekenen/redeneren (10)** | logica, kansrekening, combinatoriek, algebra, getaltheorie, meetkunde, samengestelde rente, valstrikvragen | `bench_math.py` |

Codeantwoorden werden daadwerkelijk uitgevoerd tegen testcases; feitelijke antwoorden handmatig geverifieerd. **Thinking-modus per taaktype:** formaat-taken (notulen/JSON) draaien met `enable_thinking: false`, redeneer-/kennis-/combo-taken met thinking aan — zie [Belangrijkste lessen](#-belangrijkste-lessen).

## 📊 Resultaten in één tabel

| Model | Uitgever | Architectuur | Schijf | Gen. t/s | **Max. context** | Zware | Long-ctx | Betrouwb. | Tools | Kennis | Rekenen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 🥇 **Nemotron-3-Super-120B-A12B** | NVIDIA | Hybride Mamba-MoE (12B actief) | 82.5 GB | 48 | **243k** ✅ | 4/4 | 6/6 | 5/5 | 4/4 | 13/14 | 10/10 |
| 🥈 **GPT-OSS-120B** | OpenAI | MoE (5.1B actief/117B) | 63.4 GB | 110 | 117k (128k cap) | 4/4 | 5/6 | 5/5 | 4/4 | 14/14 | 10/10 |
| 🥉 **Nemotron-3.5-Lightning-30B-A3B** | NVIDIA | Hybride Mamba2-MoE (3B actief) | 25.5 GB | 142 | **243k** ✅ | 4/4 | 6/6 | 5/5 | 4/4 | 11/14* | 9/10 |
| Qwen3.8-27B (Q8_0) | Alibaba | Hybride Gated DeltaNet | 28.6 GB | 24 | 76k (128k cap) | 4/4 | 6/6 | 5/5 | 4/4 | 12/14 | 9/10 |
| Llama-3.3-70B | Meta | Dense (70B) | 42.5 GB | 16 | 79k (128k cap) | 4/4 | 6/6 | 5/5 | **2/4** ❌ | 12/14 | 8/10 |
| Qwen3.8-27B (BF16) | Alibaba | Hybride Gated DeltaNet | 53.8 GB | 15 | 76k (128k cap) | 4/4 | 6/6 | 5/5 | 4/4 | 12/14 | 10/10 |
| Qwen2.5-72B | Alibaba | Dense (72B) | 47.4 GB | 15 | **32k** ❌ | 4/4 | 6/6 | 5/5 | 3/4 | 11/14 | 7/10 |

\* 1 combo-vraag vereist thinking aan en een ruimer tokenbudget; zonder beide loopt hij leeg (zie lessen 3–4).

**"Max. context" = het grootste document dat het model daadwerkelijk 5/5 serveerde én er correct uit terughaalde**, niet de marketing-claim. De sweep ging door tot en met een opzettelijke 200k-poging op elk model om de faalmodus te vangen.

## 📖 Het verhaal

### De opdracht: Opus-klasse op een 2017-machine
De machine was al verkocht en stond op afhaling, maar de vraag was te mooi om te laten liggen: kan deze 4× V100-opstelling (128 GB VRAM, sm_70) een Opus-klasse LLM serveren met 64k–200k context? Het eerder gebouwde gx10-project had de *kwaliteits*-methodologie al; dit project voegt de *context*-sweep toe die op dat punt nog niemand had gedaan voor tweedehands Volta-hardware.

### De architectuur-drempel: Volta limiteert alles
Vóór de eerste benchmark bleek dat bijna de hele moderne inference-stack afvalt op sm_70: geen BF16/FP8-hardware, geen FlashAttention-2, geen Marlin 4-bit kernels. llama.cpp is de enige runtime — en dan alléén correct als je **specifiek voor sm_70 compileert** (`CMAKE_CUDA_ARCHITECTURES=70`), anders valt FlashAttention stilletjes weg. De bouw zelf kostte drie valkuilen: een glibc 2.43 ↔ CUDA C23 `math`-clash (opgelost met een header-patch), de vereiste dat K en V dezelfde quant-type hebben voor FlashAttention (`-ctk q8_0 -ctv q8_0`), en het vermijden van BF16-KV op Volta (geëmuleerd, dus traag).

### De KV-les: waarom alleen hybride modellen 200k halen
De doorslaggevende ontdekking kwam uit de sweep. Dense 70B/72B-modellen zijn per token even duur in KV-cache als in gewichten (~320 KiB/token), waardoor 128k context al ~43 GB aan KV vergt. De hybride Nemotron-familie draait het om: omdat maar 6–8 van hun ~52–88 lagen attention zijn (de rest is Mamba, met een context-onafhankelijke toestand), kost een token er 6–8 KiB. Het gevolg is niet marginaal maar binaire: **Nemotron-3-Super-120B serveerde 243k tokens (5/5 retrieval) in 87 GB VRAM; een dense 70B zou bij 200k ~68 GB aan KV alleen al nodig hebben.**

### De sweep: hoe elk model faalt op 200k
Elk model werd doorgemeten op 8k/32k/64k/128k en daarna opzettelijk op 200k gezet om de faalmodus te vangen:

- **Nemotron-3-Super / -3.5-Lightning:** 5/5 tot en met **243k tokens** (de sweep-generator overschoot 200k tot 243k). De enige twee modellen die 200k echt halen.
- **GPT-OSS-120B:** 5/5 tot 117k, daarna harde HTTP 400 bij 131k (`max_position_embeddings=131072`). SWA houdt de KV klein, maar de harde cap blijft.
- **Llama-3.3-70B:** 5/5 tot 79k, cap op 131k.
- **Qwen3.8-27B:** 5/5 tot 76k, cap op 131k (open bug [#27756](https://github.com/ggml-org/llama.cpp/issues/27756): stille EOS boven ~130k).
- **Qwen2.5-72B:** clamt al bij **32k** — native 32k zonder YaRN-extensie, dus geen 128k zoals de kaart suggereert.

### De kwaliteits-batterij: waar de modellen zich écht onderscheiden
Op de 43/43-batterij (hard/long-ctx/betrouwbaar/tools/kennis/rekenen) kwamen de topmodellen dicht bij elkaar, maar de onderlinge verschillen zijn leerzaam:

- **GPT-OSS-120B** en **Nemotron-3-Super** staan bovenaan (42/43). GPT-OSS is 2.3× sneller (110 vs 48 t/s) maar capped op 128k; Nemotron-3-Super wint op context.
- **Nemotron-3.5-Lightning** is de snelste van allemaal (142 t/s, 1815 t/s prompt-verwerking) op maar 25.5 GB, maar loopt op één kansreken-vraag leeg (thinking-runaway) — hetzelfde faalpatroon als bij de combo-kennisvragen.
- **Llama-3.3-70B** faalt op tool-calling (2/4): llama.cpp's tool-parser verwerpt zijn tool-output met een HTTP 500. Een meetresultaat van de build, geen model-oordeel.
- **Qwen2.5-72B** zakt op rekenen (7/10) én op context (32k), en is daarmee de zwakste van de grote modellen hier.

## 💡 Belangrijkste lessen

1. **KV-geometrie bepaalt 200k, niet modelgrootte.** Dense 70B/72B = 320 KiB/token (200k onhaalbaar op 128 GB VRAM); hybride Mamba = 6–8 KiB/token (200k = ~1.6 GB). Check het aantal attention-lagen vóór je downloadt als lange context je doel is.
2. **Volta limiteert de runtime, niet alleen de snelheid.** Geen BF16/FP8/FA2/Marlin op sm_70. llama.cpp is de enige runtime, en dan alleen met een sm_70-specifieke build (`CMAKE_CUDA_ARCHITECTURES=70`) anders valt FlashAttention weg.
3. **Reasoning-modellen kunnen hun eigen tokenbudget opeten.** Zonder ruim `max_tokens` en zonder te checken op `reasoning_content` naast `content`, lijkt een prima model plotseling "leeg" te antwoorden (zie Nemotron-3.5's lege combo-vraag en Qwen3.8's lege notulen vóór de fix).
4. **Ken de thinking-toggle van je modelfamilie — per taaktype.** Nemotron/Qwen: `chat_template_kwargs: {enable_thinking: false}` alléén voor formaat-taken; thinking **aan** voor redeneren/kennis/combo (uit zetten kost correctheid). Dense modellen negeren de flag.
5. **BF16 heeft op Volta géén zin.** Qwen3.8-BF16 (53.8 GB) draait ~1.6× trager dan Q8_0 (28.6 GB) omdat BF16 geëmuleerd wordt — geen BF16-hardware. Quantiseer, en vermijd BF16-KV.
6. **Verifieer "200k context"-claims altijd zelf.** Qwen2.5-72B claimt 128k (YaRN) maar clamt op 32k native; Qwen3.8 heeft een stille-EOS-bug boven ~130k. Alleen de sweep (serveren + terughalen op 5 dieptes) scheidt claims van feiten.

## 💾 Hoeveel schijfruimte heb je nodig?

| Scenario | Ruimte |
|---|---|
| Eén klein model testen (25–30 GB klasse) | **~40 GB vrij** |
| Eén groot model testen (60–85 GB klasse) | **~100 GB vrij** (model + marge) |
| Sequentieel meerdere modellen testen (download → test → opruimen) | **~150 GB vrij** volstaat |
| De volledige set van 7 tegelijk laten staan | **~345 GB** (63 + 82 + 26 + 29 + 54 + 43 + 47 GB) |

Onze machine had 1.7 TB vrij op `/data`; de volledige set paste ruim.

## 🚀 Zelf reproduceren

Deze suite is **portable** en draait op elke machine met `llama.cpp` (CUDA/Metal/Vulkan/CPU), mits je de thinking-toggle per modelfamilie respecteert. Alle tests zijn automatisch gescoord.

```bash
git clone https://github.com/sayfjawad/c4130-llm-benchmarks.git
cd c4130-llm-benchmarks

# Eén model: downloaden, server starten, batterij draaien
./scripts/run_benchmarks.sh my-model ~/models/my-model.gguf

# Context-sweep (de nieuwe test)
./scripts/run_sweep.sh my-model ~/models/my-model.gguf

# Resultaten naar het dashboard
python3 scripts/build_results.py --results-dir results/ --out docs/data/results.json
```

Voor de sm_70-specifieke build, de server-flags voor single-stream lange context (`-c <n> -np 1 -fa on -kvu`, symmetrische `-ctk q8_0 -ctv q8_0`, `-sm layer -ts 8,8,8,8`) en de glibc-patch: zie [METHODOLOGY.md](METHODOLOGY.md).

## 📚 Bronnen

- Modelgewichten: [Hugging Face](https://huggingface.co) — GGUF-conversies van [unsloth](https://huggingface.co/unsloth), [ggml-org](https://huggingface.co/ggml-org) en [bartowski](https://huggingface.co/bartowski)
- Inference-engine: [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (`llama-server`, `llama-bench`)
- Modelverificatie: officiële Hugging Face API — expliciet niet via web-scraping of blogs
- Open bug: [llama.cpp #27756](https://github.com/ggml-org/llama.cpp/issues/27756) (Qwen3.8 stille EOS boven ~130k)

---

<sub>Gegenereerd met behulp van Claude (Anthropic) als benchmarking-copiloot — alle downloads, serverstarts, testruns en codeverificaties zijn daadwerkelijk uitgevoerd op de genoemde hardware, niet gesimuleerd.</sub>
