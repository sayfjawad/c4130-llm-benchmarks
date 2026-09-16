# 🧪 C4130 LLM Benchmark Suite

**Welk lokaal LLM is écht het snelst én slimst op een Dell C4130 met 4× Tesla V100-SXM2-32GB (128GB dedicated VRAM)?**
Geen leaderboard-cijfers uit een blog, maar 17 modellen, zelf gedownload, zelf gedraaid, zelf getest — inclusief de modellen die faalden en waarom.

📊 **[Bekijk de interactieve resultaten-dashboard →](https://sayfjawad.github.io/c4130-llm-benchmarks/)**

[![Pages Build](https://github.com/sayfjawad/c4130-llm-benchmarks/actions/workflows/deploy.yml/badge.svg)](https://github.com/sayfjawad/c4130-llm-benchmarks/actions/workflows/deploy.yml)
![License](https://img.shields.io/badge/license-MIT-blue)
![Models tested](https://img.shields.io/badge/modellen%20getest-17-informational)

---

## 🏆 De winnaar

| | |
|---|---|
| 🥇 **GPT-OSS-120B** | (OpenAI, Apache 2.0, `UD-Q4_K_XL`, 59GB) |

Perfecte score op **alle zes testcategorieën** (redeneren, coding, notulen, strikte instructies, long-context, betrouwbaarheid, tool-gebruik, brede kennis, en zware reken-/logicaproblemen), en **3x sneller** dan de enige concurrent (Nemotron-3-Super-120B-A12B) die dat niveau ook haalde.

```
GPT-OSS-120B          ██████████████████████████████████████████ 53.4 t/s   ✅ alles 100%
Nemotron-3.5-Lightning ████████████████████████████████████████████ 58.1 t/s ✅ 13/14 kennis, rest 100% (35GB!)
Qwen3.6-35B-A3B       ███████████████████████████████████████████████ 57 t/s ⚠️  faalt op zwaar rekenwerk
Gemma 4 31B           █████ 6.4 t/s                                        ✅ alles 100% (perfect, maar traag)
Nemotron-3-Super-120B ██████████████ 17 t/s                              ✅ alles 100% (maar traag)
MiniMax-M2.7          ████████████████████ 24.5 t/s                      ⚠️  traag, verbose
Qwen3-30B-A3B         ██████████████████████████████████████████████ 60 t/s  ✅ alles 100% (kleinste footprint)
```

---

## 📋 Inhoud

1. [Waarom dit project bestaat](#-waarom-dit-project-bestaat)
2. [De hardware](#-de-hardware)
3. [Methodologie](#-methodologie)
4. [Resultaten in één tabel](#-resultaten-in-één-tabel)
5. [Het verhaal: vijf testrondes](#-het-verhaal-vijf-testrondes)
6. [Belangrijkste lessen](#-belangrijkste-lessen)
7. [Hoeveel schijfruimte heb je nodig?](#-hoeveel-schijfruimte-heb-je-nodig)
8. [Zelf reproduceren](#-zelf-reproduceren)
9. [Bronnen](#-bronnen)
10. [Bijdragen](#-bijdragen)

---

## 🎯 Waarom dit project bestaat

Modelvergelijkingen op internet zijn vaak ofwel marketingmateriaal van de modelmakers zelf, ofwel SEO-contentfarms met plausibel klinkende maar niet-geverifieerde cijfers. Tijdens dit project liepen we daar zelf tegenaan: een eerste zoekpoging naar "beste lokale LLM" leverde een web-fetch op met een compleet **verzonnen leaderboard**, inclusief niet-bestaande AI-modellen naast echte. Dat was het moment waarop we besloten: **alles zelf downloaden, zelf draaien, zelf verifiëren** — via de officiële Hugging Face API voor wat modellen daadwerkelijk bestaan, en via eigen scripts + handmatige codeverificatie voor wat ze daadwerkelijk kunnen.

Dit is dus geen samenvatting van andermans benchmarks. Elk cijfer in dit rapport is zelf gemeten op onze eigen C4130, met reproduceerbare scripts die in deze repo staan.

## 🖥️ De hardware

Een **Dell PowerEdge C4130** met 4× **NVIDIA Tesla V100-SXM2-32GB** (128GB dedicated VRAM, volledig NVLink-meshed):

| Specificatie | Waarde |
|---|---|
| GPU | 4× Tesla V100-SXM2-32GB (Volta, sm_70) |
| VRAM | 128GB **dedicated** (32GB per GPU, HBM2) |
| Geheugenbandbreedte | ~900 GB/s per GPU (HBM2) |
| Interconnect | NVLink 2.0 full-mesh (300 GB/s per GPU-paar) |
| Compute capability | SM 7.0 (Volta) |

Het belangrijkste architecturale kenmerk: het is **compute-klasse uit 2017** — geen BF16/FP8-hardware, geen FlashAttention-2, geen Marlin 4-bit kernels. llama.cpp is de enige praktische runtime. De vier V100's zijn volledig NVLink-meshed (echte tensor-parallelism), maar de echte schaalbaarheid zit in de KV-cache-geometrie van hybride modellen — zie [Belangrijkste lessen](#-belangrijkste-lessen).

## 🔬 Methodologie

Elk model doorliep (een subset van) deze testcategorieën, allemaal via [`llama-server`](https://github.com/ggml-org/llama.cpp)'s OpenAI-compatibele API:

| Categorie | Wat het meet | Script |
|---|---|---|
| ⚡ **Snelheid** | tokens/sec generatie + prompt-processing, via `llama-bench` en server-timings | — |
| 🧩 **Zware taken (4)** | multi-stap redeneren, LRU-cache coding (echt uitgevoerd, niet alleen gelezen), lang transcript → gestructureerde notulen, strikte JSON met meerdere gelijktijdige constraints | `bench_hard.py` |
| 📄 **Long-context (6)** | ~8.000-token transcript met 6 feiten verspreid vroeg/midden/laat — test échte retrieval, niet alleen "het document gezien hebben" | `bench_longctx.py` |
| 🔁 **Betrouwbaarheid (5x)** | dezelfde taak 5x herhalen bij hogere temperature, faalpercentage meten | `bench_notulen_repeat.py` |
| 🔧 **Tool-gebruik (4)** | losse tool-call, gekoppelde tool-calls (output van A wordt input van B), tools terecht vermijden, conditionele multi-tool taak | `bench_tools.py` |
| 📚 **Kennis (14)** | brede/niche feitenkennis over meerdere domeinen + 2 kennis+redeneer-combinaties | `bench_knowledge.py` |
| 🧮 **Rekenen/redeneren (10)** | logica-puzzel, kansrekening, combinatoriek, algebra, getaltheorie, meetkunde, samengestelde rente, 2 valstrikvragen | `bench_math.py` |

Codeantwoorden werden **niet alleen gelezen maar daadwerkelijk uitgevoerd** tegen testcases. Feitelijke antwoorden werden handmatig geverifieerd tegen betrouwbare, stabiele (niet-recente) kennis.

## 📊 Resultaten in één tabel

| Model | Uitgever | Architectuur | Grootte (schijf) | Gen. snelheid | Zware taken | Long-ctx | Betrouwbaar | Tools | Kennis | Rekenen |
|---|---|---|---|---|---|---|---|---|---|---|
| 🥇 **GPT-OSS-120B** | OpenAI | MoE (5.1B actief/117B totaal) | 59GB | 53.4 t/s | 4/4 | 6/6 | 5/5 | 4/4 | 14/14 | 10/10 |
| 🥈 Nemotron-3.5-Lightning-30B-A3B | NVIDIA | Hybride Mamba2-MoE (3.5B actief/33B) | 35GB | 58.1 t/s | 4/4* | 6/6* | 5/5* | 4/4* | 13/14**** | 10/10 |
| 🥉 **Gemma 4 31B** | Google DeepMind | **Dense** (30.7B actief/30.7B totaal) | 31GB | 6.4 t/s | 4/4* | 6/6* | 5/5* | 4/4* | 14/14***** | 10/10** |
| Nemotron-3-Super-120B-A12B | NVIDIA | Hybride Mamba-MoE (12B actief/120B) | 84GB | 17 t/s | 4/4 | 6/6 | 5/5 | – | – | 10/10 |
| Qwen3-30B-A3B | Alibaba | MoE (3B actief/30B) | 31GB | 60 t/s | 4/4 | 6/6 | 5/5 | – | – | – |
| Qwen3.6-35B-A3B | Alibaba | Hybride DeltaNet-MoE (3B/35B) | 35GB | 57 t/s | 4/4* | 6/6 | 5/5* | 4/4 | 12-14/14** | 6/10*** |
| MiniMax-M2.7 | MiniMax | MoE (10B actief/229B) | 95GB | 24.5 t/s | 4/4 | 6/6 | 5/5 | – | – | 9/10 |
| Qwen3.5-122B-A10B | Alibaba | MoE (10B actief/122B) | 72GB | 21 t/s | 4/4* | 6/6 | 5/5 | – | – | 5/10*** |
| Qwen3-235B-A22B | Alibaba | MoE (22B actief/235B) | 98GB (Q3) | 14 t/s | 4/4 | – | – | – | – | – |
| GLM-4.5-Air | Zhipu | MoE (12B actief/106B) | 93GB | 15.9 t/s | 3/4 ❌ | – | – | – | – | – |
| Llama-3.3-Nemotron-Super-49B | NVIDIA | NAS-gepruned dense | 55GB | 3.9 t/s | 4/4 | – | – | – | – | – |
| Nemotron-Nano-9B-v2 | NVIDIA | Hybride Mamba | 9GB | 22.5 t/s | 4/4 | – | – | – | – | – |
| MiniMax-M2 | MiniMax | MoE (10B actief/229B) | 95GB | 25.9 t/s | 3/4 ❌ | – | ❌ **loop-bug** | – | – | – |
| DeepSeek-R1-Distill-Llama-70B | DeepSeek/Meta | **Dense** (70B actief) | 70GB | 2.9 t/s | 2/4 ❌ | – | – | – | – | – |
| Kimi-K2/K3 | Moonshot | MoE (32B actief/1T) | *past niet* | – | – | – | – | – | – | – |
| DeepSeek-V3/V3.1/R1 | DeepSeek | MoE (37B actief/671B) | *past niet* | – | – | – | – | – | – | – |
| GLM-4.6 | Zhipu | MoE (32B actief/357B) | *past niet zonder 1-bit* | – | – | – | – | – | – | – |

\* vereist `enable_thinking: false` om binnen een normaal tokenbudget te blijven (Nemotron) of `--reasoning off` (Gemma 4)
\** 12/14 met thinking uit, 14/14 met thinking aan
\*** loopt bij ~40-50% van de zware rekenvragen leeg binnen standaardbudget — zie [Belangrijkste lessen](#-belangrijkste-lessen)
\**** 1 fout publicatiejaar (Mulisch 1995 i.p.v. 1992); 1 combo-vraag had een 6000-tokenbudget nodig i.p.v. 2000
\***** 12/14 met --reasoning off; 2 combo-vragen correct met --reasoning on (15 jaar ipv 12; 45 jaar ipv 46)

Volledige, live-doorzoekbare versie: **[sayfjawad.github.io/c4130-llm-benchmarks](https://sayfjawad.github.io/c4130-llm-benchmarks/)**

## 📖 Het verhaal: zes testrondes

### Ronde 1 — De eerste kandidaat, en een harde geheugen-les
Startpunt was simpelweg: wat past er op deze machine? **GPT-OSS-120B** (63GB) was de voor de hand liggende eerste keuze — native ontworpen voor precies dit soort hardware. Daarna probeerden we **Qwen3-235B-A22B** op de grootste quant die "net" binnen het geheugenbudget leek te passen (Q4, 134GB) — en dat ging goed mis: llama.cpp kon het model niet volledig op de GPU laden, offloadde een deel naar de CPU, en de snelheid stortte in van tientallen tokens/sec naar **2 tokens/sec**. Les: bij MoE-modellen op unified-memory-hardware moet je niet tot de rand van je geheugenbudget quantiseren — reken een marge van 15-20GB voor context/overhead.

### Ronde 2 — De Chinese labs, en een architecturale ontdekking
Met de geheugen-les geleerd testten we een bredere selectie: **GLM-4.5-Air**, **MiniMax-M2**, **Qwen3-235B-A22B** (nu op een kleinere Q3-quant die wél volledig past), en **Qwen3-30B-A3B**. GLM-4.5-Air maakte een rekenfout ondanks een opvallend lange interne denkstap. MiniMax-M2 liep vast in een **letterlijke herhalingslus** op de notulen-taak — het model bleef dezelfde zinnen herhalen tot het tokenbudget op was, zonder ooit een antwoord te geven.

Vervolgens testten we **DeepSeek-R1-Distill-Llama-70B** — en dat leverde de belangrijkste architecturale inzicht van het hele project op: dit is een **dense** model (alle 70B parameters actief per token, i.t.t. MoE waar maar een fractie actief is). Op deze bandbreedte-gelimiteerde hardware kelderde de snelheid naar 2.9 t/s, tergend traag vergeleken met MoE-modellen van vergelijkbare of grotere totale omvang. Hetzelfde patroon zagen we bij **Llama-3.3-Nemotron-Super-49B** (ook grotendeels dense, 3.9 t/s). **Conclusie: op unified-memory/bandbreedte-gelimiteerde hardware is MoE-architectuur belangrijker dan merk, leverancier of zelfs modelgrootte.**

### Ronde 3 — Zwaardere testen, gelijke lat
De testset van ronde 1-2 discrimineerde niet meer scherp genoeg (bijna alles scoorde 4/4). We voegden een **long-context test** toe (transcript met feiten verspreid door het document) en een **5x-herhalingstest** (specifiek om het soort loop-bug van MiniMax-M2 systematisch te vangen in plaats van toevallig te ontdekken). Alle overlevende kandidaten (Qwen3-30B-A3B, GPT-OSS-120B, Qwen3-235B-A22B) scoorden hier foutloos op.

### Ronde 4 — Nieuwe generatie modellen, en de reken-bottleneck
Een nieuwe lichting modellen (**Nemotron-3-Super-120B-A12B**, **Qwen3.6-35B-A3B**, **Qwen3.5-122B-A10B**, **MiniMax-M2.7**) werd tegen dezelfde lat gehouden — en tegen een extra, veel zwaardere **reken-/redeneertest** (10 problemen: logica, kansrekening, combinatoriek, getaltheorie, met opzettelijke valstrikken zoals leeftijdsberekeningen over een verjaardag heen die nog niet geweest is).

Hier bleek het echte onderscheid: **GPT-OSS-120B** en **Nemotron-3-Super-120B-A12B** losten alle 10 problemen efficiënt op (nooit meer dan ~900 tokens per antwoord). De hele **Qwen3.x-lijn** (zowel 3.5 als 3.6) bleek een structurele eigenschap te hebben: het denkproces is 3-10x zo verbose voor dezelfde antwoorden, en **liep bij 40-50% van de vragen leeg** binnen een redelijk tokenbudget van 2000 tokens — inclusief bij een vraag zo simpel als een dobbelsteen-kansberekening. Bij één specifieke vraag (combinatoriek) zagen we in de ruwe denkstap dat het model het juiste antwoord al **twee keer onafhankelijk had uitgerekend en gevalideerd**, en vervolgens eindeloos bleef twijfelen over de presentatie tot het budget op was — functioneel dezelfde soort fout als MiniMax-M2's loop, alleen anders van vorm.

### Ronde 5 — Dag-1 test: Nemotron 3.5 Lightning (12 aug 2026)
Een dag na de release van **NVIDIA Nemotron-3.5-Lightning-30B-A3B** (11 aug 2026: 30B MoE, 3.5B actief, hybride Mamba2-attention + MTP-layers, tot 1M context) hebben we het model door exact dezelfde batterij gehaald. Twee praktische hobbels: de nieuwe architectuur vereist **llama.cpp b10326+** (onze b9860 moest geüpdatet naar b10380, met behoud van de lokale GB10-patch die `GGML_CUDA_USE_PDL` uitschakelt), en de **thinking-modus is standaard aan én extreem verbose** — in de eerste run at het denkproces het volledige 2000-3000-tokenbudget op bij long-context (0/6), notulen (0/5) en 2 van de 4 zware taken, precies het patroon uit les 3.

Met `chat_template_kwargs: {enable_thinking: false}` voor formaat-taken en thinking aan voor reken-/kennistaken (de Qwen3.6-aanpak) vielen de resultaten op hun plek: **10/10 rekenen** (alle antwoorden identiek aan GPT-OSS), **13/14 kennis**, **6/6 long-context**, **5/5 betrouwbaarheid**, **4/4 tools** (traces identiek aan GPT-OSS) en **4/4 zware taken** — met de kanttekening dat de zware rekenvraag alléén met thinking aan lukt (2195 ✓; zonder thinking 2595 ✗). Bij 58.1 t/s generatie en 1978 t/s prompt-processing (de snelste van alle 16 geteste modellen — relevant voor lange transcripten) op maar 35GB is dit de nieuwe **runner-up**: het komt op een haar na aan GPT-OSS-120B's scores, op 60% van de schijfruimte en met 40% minder actieve parameters. NVIDIA's "4x snellere output"-claim zagen we op deze bandbreedte-gelimiteerde hardware niet terug in tokens/sec (58 t/s, vergelijkbaar met andere 3B-actief-MoE's); de winst zit hier vooral in prompt-verwerking en het lage geheugenbeslag.

### Ronde 6 — Dense perfectie: Google Gemma 4 31B (12 aug 2026)
Direct na Nemotron 3.5 Lightning volgde **Google DeepMind's Gemma 4 31B** (30.7B parameters, dense transformer, 256K context, Apache 2.0). Dit is het eerste **dense** model in de test die de volledige batterij doorliep — en het is meteen het eerste dense model met een **perfecte score (43/43)**. Het model vereist geen speciale architectuur-ondersteuning (llama.cpp b10380 volstaat).

De thinking-aanpak is fundamenteel anders dan bij Nemotron: Gemma 4 gebruikt `<think>`-tags in de output die door de chat-template worden afgehandeld. llama.cpp biedt hiervoor de `--reasoning` flag (`on/off/auto`) en `--reasoning-budget N` om denkwerk te limiteren. De hybride strategie: **`--reasoning off`** voor formaat-, long-context-, betrouwbaarheids- en tool-taken; **`--reasoning on --reasoning-budget 800`** voor rekenen/redeneren. Zonder budget-limiet genereert Gemma 4 tot 1864 denk-tokens per rekenvraag (bij 6.4 t/s is dat ~5 minuten per vraag); met budget 800 blijft dit onder 1600 tokens.

De scores: **4/4 zware taken** (uniek: de rekenvraag lukte correct — 2195 — zónder thinking, iets wat Nemotron-3.5 niet kon), **6/6 long-context**, **5/5 betrouwbaarheid**, **4/4 tools**, **14/14 kennis** (12/14 met thinking uit; 2 combo-vragen correct mét thinking aan), en **10/10 rekenen**. Het Mulisch-publicatiejaar (1992) had Gemma 4 wél correct, in tegenstelling tot Nemotron-3.5.

De keerzijde: **6.4 t/s generatiesnelheid** — 9x trager dan Nemotron-3.5-Lightning (58 t/s) en 8x trager dan GPT-OSS-120B (53 t/s). Dit is het fundamentele verschil tussen dense (30.7B actief) en MoE (3.5B actief) op bandbreedte-gelimiteerde hardware: bij gelijke of betere kwaliteit kost dense 8-9x meer tijd per token. Prompt-processing (660 t/s) is ook een stuk trager dan MoE-concurrenten, maar voor lange transcripten nog steeds acceptabel.

**Conclusie:** Gemma 4 31B is het **kwalitatief beste dense model** voor de C4130 — perfecte scores op slechts 31GB — maar de lage snelheid maakt het ongeschikt voor interactief gebruik. Voor batch-verwerking of kwaliteit-boven-snelheid scenarios is het een uitstekende keuze. Google's claim van "frontier-level performance in a compact form factor" klopt voor kwaliteit, maar "compact" is relatief: 31GB dense vs 35GB MoE (Nemotron-3.5) bij 9x lagere snelheid is een prijs die je moet willen betalen.

## 💡 Belangrijkste lessen

1. **MoE > dense op bandbreedte-gelimiteerde hardware, punt.** Een dense 70B-model kan 15-20x trager zijn dan een MoE-model met vergelijkbare of grotere totale parametercount. Check dit vóór je downloadt: actieve parameters bepalen praktische snelheid op deze hardwareklasse.
2. **Quantiseer niet tot de rand.** Reken een marge van 15-20GB boven op de modelgrootte voor context/KV-cache. Een model dat "net" past forceert CPU-offload en wordt onbruikbaar traag.
3. **Reasoning-modellen kunnen hun eigen tokenbudget opeten.** Zonder ruim `max_tokens` (2000+) en zonder te checken op een `reasoning_content`-veld naast `content`, lijkt een prima werkend model plotseling "leeg" te antwoorden.
4. **Ken de thinking-toggle van je modelfamilie.** Qwen3.x én Nemotron 3.5 Lightning: `chat_template_kwargs: {enable_thinking: false}`. Gemma 4: `--reasoning off` (server-flag) of `--reasoning-budget N` voor limiet. Oudere NVIDIA Nemotron: system-prompt `"detailed thinking off"`. Maar: **zet dit alleen uit voor pure formaat-taken** — voor rekenwerk kost het je correctheid (zie Qwen3.6's val van 6/10 naar 9/10 zodra thinking weer aan ging; Gemma 4's combo-kennisvragen van 12/14 naar 14/14).
5. **Snelheid en betrouwbaarheid zijn niet hetzelfde als "slim".** Meerdere modellen kwamen intern tot het juiste antwoord maar faalden om het te *leveren* — een reproduceerbaar risico voor productiegebruik dat alleen zichtbaar wordt met herhalingstests, niet met een losse steekproef.
6. **Verifieer modelclaims altijd zelf.** Publieke "beste LLM"-lijsten bevatten vaak plausibel klinkende maar onbetrouwbare informatie. De Hugging Face API (`huggingface_hub.HfApi().model_info(...)`) is de enige bron die we volledig vertrouwden voor "bestaat dit model daadwerkelijk en hoe groot is het".

## 💾 Hoeveel schijfruimte heb je nodig?

| Scenario | Ruimte |
|---|---|
| Eén model testen (kleinste kandidaten, 30-40GB klasse) | **~50GB vrij** |
| Eén model testen (grootste kandidaten, ~100GB klasse) | **~130GB vrij** (model + marge voor download-tussenopslag) |
| Sequentieel meerdere modellen testen (download → test → opruimen → volgende) | **~150GB vrij** is ruim voldoende, ongeacht hoeveel modellen je uiteindelijk wilt vergelijken |
| Meerdere modellen tegelijk laten staan voor side-by-side vergelijk | Tel de individuele modelgroottes bij elkaar op (zie tabel hierboven, kolom "Grootte") |

Onze eigen machine had 916GB totaal; na het opruimen van oude projectdata bleef 700GB+ vrij, ruim voldoende om zelfs de grootste kandidaten (~100GB elk) sequentieel te testen zonder ooit ruimte tekort te komen.

## 🚀 Zelf reproduceren

### Universele benchmark — werkt op elke machine

Deze benchmark-suite is **volledig portable** en draait op elke machine met `llama.cpp` (CUDA, Metal, Vulkan, CPU). Alle tests zijn **geautomatiseerd gescoord** — geen handmatige beoordeling nodig.

```bash
git clone https://github.com/sayfjawad/c4130-llm-benchmarks.git
cd c4130-llm-benchmarks

# Eén commando: model downloaden, server starten, alle 6 tests draaien
./scripts/run_benchmarks.sh my-model ~/models/my-model.gguf

# Resultaten toevoegen aan het dashboard
python3 scripts/aggregate_results.py --results-dir results/ --model my-model --out docs/data/results.json
```

### Volledige documentatie

- **[`METHODOLOGY.md`](METHODOLOGY.md)** — alle testvragen, ground truth antwoorden, scoring-criteria, en per-test uitleg
- **[`scripts/README.md`](scripts/README.md)** — script-gebruik, opties, thinking/redeneren per taaktype

### Handmatig (stap voor stap)

```bash
llama-server --model ~/models/my-model.gguf --host 127.0.0.1 --port 8081 --fit on --cont-batching -ngl 999

python3 scripts/bench_hard.py       8081 my-model
python3 scripts/bench_longctx.py    8081 my-model
python3 scripts/bench_notulen_repeat.py 8081 my-model 5
python3 scripts/bench_tools.py      8081 my-model
python3 scripts/bench_knowledge.py  8081 my-model
python3 scripts/bench_math.py       8081 my-model
```

Resultaten komen in `results/<model>-*.json`. De [`aggregate_results.py`](scripts/aggregate_results.py) script bouwt daar `docs/data/results.json` van voor het dashboard.

## 📚 Bronnen

- Modelgewichten: [Hugging Face](https://huggingface.co) — met name de GGUF-conversies van [unsloth](https://huggingface.co/unsloth) en [bartowski](https://huggingface.co/bartowski)
- Inference-engine: [`llama.cpp` / `llama-server` / `llama-bench`](https://github.com/ggml-org/llama.cpp)
- Modelverificatie: [`huggingface_hub` Python API](https://huggingface.co/docs/huggingface_hub) — rechtstreeks tegen de officiële HF API, expliciet **niet** via web-scraping of blogs (zie [Waarom dit project bestaat](#-waarom-dit-project-bestaat))
- Officiële modelkaarten geciteerd in het rapport: [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) (GGUF: [ggml-org](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF)), [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) (GGUF: [bartowski](https://huggingface.co/bartowski/google_gemma-4-31B-it-GGUF))

## 🤝 Bijdragen

Heb je zelf een model getest op jouw hardware? PR's welkom:

1. Draai `./scripts/run_benchmarks.sh <naam> <model>.gguf`
2. Draai `python3 scripts/aggregate_results.py --results-dir results/ --model <naam> --out docs/data/results.json`
3. Vul handmatig de metadata in `docs/data/results.json` aan (publisher, params, architecture, etc.)
4. Voeg de ruwe output-JSON's uit `results/` toe
5. Open een PR — de GitHub Actions-pipeline bouwt de site automatisch opnieuw bij merge naar `main`

---

<sub>Gegenereerd met behulp van Claude (Anthropic) als benchmarking-copiloot — alle downloads, serverstarts, testruns en codeverificaties zijn daadwerkelijk uitgevoerd op de genoemde hardware, niet gesimuleerd.</sub>
