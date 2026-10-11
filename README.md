# 🧪 C41xx LLM Benchmark Suite

**Kan een machine uit 2017 — een Dell C4130 met 4× Tesla V100-SXM2-32GB (128 GB dedicated VRAM) — een Opus-klasse LLM draaien, met minimaal 64k en liefst 200k context?**

Het korte antwoord: **ja voor 128k, en alleen met hybride architecturen (weinig attention-lagen) voor 200k+.** Onder llama.cpp haalden alleen de Mamba-hybride Nemotrons 200k; sinds de [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)-runs (oktober 2026) doet de GDN-hybride **Qwen3.8-27B** het ook — **5/5 retrieval tot 253k tokens, ~5–15× sneller**. Het lange antwoord, met zelf gemeten cijfers, staat hieronder. 8 modellen in 10 configuraties, zelf gedownload, zelf gedraaid, zelf getest — inclusief de modellen die faalden en waaróm.

Gemeten op twee identieke machines: de **c4130** en de **c4140** (beide 4× Tesla V100-SXM2-32GB) — vandaar de repo-naam `c41xx`.

📊 **[Bekijk het interactieve resultaten-dashboard →](https://sayfjawad.github.io/c41xx-llm-benchmarks/)**

[![Pages Build](https://github.com/sayfjawad/c41xx-llm-benchmarks/actions/workflows/deploy.yml/badge.svg)](https://github.com/sayfjawad/c41xx-llm-benchmarks/actions/workflows/deploy.yml)
![License](https://img.shields.io/badge/license-MIT-blue)
![Models tested](https://img.shields.io/badge/configuraties%20getest-10-informational)

---

## 🏆 De winnaar (hangt af van je prioriteit)

Er is geen enkele winnaar op alle assen. De vraag was "Opus-klasse kwaliteit **én** 200k context", en dat wordt gesplitst in drie antwoorden:

| Prioriteit | Model | Waarom |
|---|---|---|
| ⚡ **Beste allrounder / productie** (snelheid × context × VRAM) | **Qwen3.8-27B NVFP4 + DFlash2 op 1Cat-vLLM**, officieel release-profiel (20 GB) | 39/43, **5/5 retrieval tot 253k tokens**, ~100 t/s op Nederlandse tekst en **350–390 t/s op code**, prefill 3.3–4.3k t/s, slechts 24.5 GB/GPU. Productiemodel op de c4130. |
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

Het is **compute-klasse uit 2017**: geen BF16/FP8-hardware, geen FlashAttention-2, geen Marlin 4-bit kernels. Upstream ExLlamaV2, SGLang en recente vLLM/TensorRT-LLM vallen af. De eerste ronde draaide daarom volledig op **llama.cpp** (sm_70-build). Sinds oktober 2026 is er een tweede, snellere route: **[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)**, een op V100/sm_70 toegespitste vLLM-fork met eigen FlashAttention-V100, NVFP4-kernels en DFlash2-speculatieve decoding. Die gebruikt de vier NVLink-gemeshte V100's écht tensor-parallel (TP×4), waar llama.cpp's layer-split in de praktijk ~1 GPU tegelijk bezig houdt.

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
| ⚡ **Snelheid** | tokens/sec generatie (tg128) + prompt-processing (pp512), via `llama-bench`; voor vLLM-entries via `bench_vllm_speed.py` (tokens uit `usage`, 1 koud + mediaan 3 warm) | `bench_vllm_speed.py` |
| 🧩 **Zware taken (4)** | multi-stap redeneren, LRU-cache coding (écht uitgevoerd), lang transcript → notulen, strikte JSON met meerdere constraints | `bench_hard.py` |
| 📄 **Long-context (6)** | ~8k-token transcript met 6 feiten verspreid vroeg/midden/laat — échte retrieval | `bench_longctx.py` |
| 🛰️ **Context-sweep (nieuw)** | 8k → 32k → 64k → 128k → 200k, met naalden op 5 dieptes; meet max. serveerbare context + prefill/decode + faalmodus | `bench_longctx_sweep.py` |
| 🔁 **Betrouwbaarheid (5x)** | dezelfde taak 5x herhaald, faalpercentage | `bench_notulen_repeat.py` |
| 🔧 **Tool-gebruik (4)** | losse call, gekoppelde calls (A→B), tools terecht vermijden, conditionele multi-tool | `bench_tools.py` |
| 📚 **Kennis (14)** | brede/niche feitenkennis + 2 kennis+redeneer-combinaties | `bench_knowledge.py` |
| 🧮 **Rekenen/redeneren (10)** | logica, kansrekening, combinatoriek, algebra, getaltheorie, meetkunde, samengestelde rente, valstrikvragen | `bench_math.py` |

Codeantwoorden werden daadwerkelijk uitgevoerd tegen testcases; feitelijke antwoorden handmatig geverifieerd. **Thinking-modus per taaktype:** formaat-taken (notulen/JSON) draaien met `enable_thinking: false`, redeneer-/kennis-/combo-taken met thinking aan — zie [Belangrijkste lessen](#-belangrijkste-lessen). Bij vLLM-servers die thinking standaard **uit** zetten (`--default-chat-template-kwargs`) draaien de scripts met `FORCE_THINKING=1`, zodat de thinking-per-taaktype gelijk blijft aan de llama.cpp-runs; `BENCH_URL` wijst ze naar een andere server.

## 📊 Resultaten in één tabel

| Model | Uitgever | Architectuur | Schijf | Gen. t/s | **Max. context** | Zware | Long-ctx | Betrouwb. | Tools | Kennis | Rekenen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 🥇 **Nemotron-3-Super-120B-A12B** | NVIDIA | Hybride Mamba-MoE (12B actief) | 82.5 GB | 48 | **243k** ✅ | 4/4 | 6/6 | 5/5 | 4/4 | 13/14 | 10/10 |
| 🥈 **GPT-OSS-120B** | OpenAI | MoE (5.1B actief/117B) | 63.4 GB | 110 | 117k (128k cap) | 4/4 | 5/6 | 5/5 | 4/4 | 14/14 | 10/10 |
| ⚡ **Qwen3.8-27B NVFP4 (1Cat-vLLM release-profiel, 262k)** † | Alibaba | Hybride Gated DeltaNet, TP×4 | 20.0 GB | **~100 NL / 350–390 code** ‡ | **253k** ✅ | 4/4 | 6/6 | 5/5 | 3/4 § | 11/14 | 10/10 |
| Qwen3.8-27B NVFP4 (1Cat-vLLM, eigen flags, 131k) † | Alibaba | Hybride Gated DeltaNet, TP×4 | 20.0 GB | 124 ‡ | 127k (131k cap) | 4/4 | 6/6 | 5/5 | 4/4 | 11/14 | 9/10 |
| 🥉 **Nemotron-3.5-Lightning-30B-A3B** | NVIDIA | Hybride Mamba2-MoE (3B actief) | 25.5 GB | 142 | **243k** ✅ | 4/4 | 6/6 | 5/5 | 4/4 | 11/14* | 9/10 |
| Qwen3.8-27B (Q8_0) | Alibaba | Hybride Gated DeltaNet | 28.6 GB | 24 | 76k (128k cap) | 4/4 | 6/6 | 5/5 | 4/4 | 12/14 | 9/10 |
| Llama-3.3-70B | Meta | Dense (70B) | 42.5 GB | 16 | 79k (128k cap) | 4/4 | 6/6 | 5/5 | **2/4** ❌ | 12/14 | 8/10 |
| Qwen3.8-27B (BF16) | Alibaba | Hybride Gated DeltaNet | 53.8 GB | 15 | 76k (128k cap) | 4/4 | 6/6 | 5/5 | 4/4 | 12/14 | 10/10 |
| Qwen2.5-32B (vLLM TP×4) † | Alibaba | Dense (32B), FP16 | 62.0 GB | 27 | **32k** ❌ | 3/4 | 6/6 | 5/5 | 4/4 | 12/14 | 6/10 |
| Qwen2.5-72B | Alibaba | Dense (72B) | 47.4 GB | 15 | **32k** ❌ | 4/4 | 6/6 | 5/5 | 3/4 | 11/14 | 7/10 |

† **Andere runtime: vLLM** (tensor-parallel over alle 4 V100's) i.p.v. llama.cpp — kwaliteitsscores zijn vergelijkbaar (identieke prompts + scoring), snelheid niet 1-op-1. Gemeten op de c4140 (zelfde 4× V100-SXM2-32GB). De twee NVFP4-rijen zijn dezelfde gewichten als de Q8_0/BF16-rijen; ze verschillen alleen in serverflags (zie [het vLLM-hoofdstuk](#hoofdstuk-2-vllm-op-volta-1cat)).

‡ **Speculatieve decoding (DFlash2) maakt de decodesnelheid tekstafhankelijk.** Op dezelfde server: Engelse code/SQL 350–390 t/s, Nederlandse proza ~97 t/s. De 124 t/s van de 131k-rij is gemeten door SSE-chunks te tellen op een code-prompt en is daardoor een ondergrens (één chunk kan meerdere geaccepteerde tokens bevatten); op een identieke, usage-getelde Nederlandse test scoren beide configuraties ~110–123 t/s.

§ In de officiële run sloeg het model in `multi_entity_tool_use` één weer-call over en verzon een temperatuur; 5 herhalingen daarna scoorden alle 4/4 (1 misser op 24 scenario's).

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
- **Qwen3.8-27B (llama.cpp):** 5/5 tot 76k, cap op 131k (open bug [#27756](https://github.com/ggml-org/llama.cpp/issues/27756): stille EOS boven ~130k).
- **Qwen3.8-27B NVFP4 (1Cat-vLLM, release-profiel):** 5/5 op **elke** maat tot en met **253k tokens** (alleen de opzettelijke 262k-poging faalt op de harde cap) — zowel met prefill-budget 8192 als met de stabiele productie-instelling 4096 (herhaald 2026-10-11). Zelfde model, andere runtime — de 131k-grens was een llama.cpp/flag-grens, geen modelgrens.
- **Qwen2.5-72B:** clamt al bij **32k** — native 32k zonder YaRN-extensie, dus geen 128k zoals de kaart suggereert.

### De kwaliteits-batterij: waar de modellen zich écht onderscheiden
Op de 43/43-batterij (hard/long-ctx/betrouwbaar/tools/kennis/rekenen) kwamen de topmodellen dicht bij elkaar, maar de onderlinge verschillen zijn leerzaam:

- **GPT-OSS-120B** en **Nemotron-3-Super** staan bovenaan (42/43). GPT-OSS is 2.3× sneller (110 vs 48 t/s) maar capped op 128k; Nemotron-3-Super wint op context.
- **Nemotron-3.5-Lightning** is de snelste van allemaal (142 t/s, 1815 t/s prompt-verwerking) op maar 25.5 GB, maar loopt op één kansreken-vraag leeg (thinking-runaway) — hetzelfde faalpatroon als bij de combo-kennisvragen.
- **Llama-3.3-70B** faalt op tool-calling (2/4): llama.cpp's tool-parser verwerpt zijn tool-output met een HTTP 500. Een meetresultaat van de build, geen model-oordeel.
- **Qwen2.5-72B** zakt op rekenen (7/10) én op context (32k), en is daarmee de zwakste van de grote modellen hier.

### Hoofdstuk 2: vLLM op Volta (1Cat)
De eerste ronde concludeerde dat llama.cpp de enige praktische runtime op sm_70 is. In oktober 2026 bleek [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) v1.5.1 — een vLLM-fork die specifiek voor V100 kernels herschrijft (FlashAttention-V100, NVFP4 via TurboMind/Marlin-routes, DFlash2-speculatieve decoding) — die conclusie te achterhalen. Draaiend als docker-image, tensor-parallel over alle vier de V100's:

- **Qwen2.5-32B (vLLM 0.9.2, FP16):** 36/43, tools 4/4 (vLLM's tool-parser doet het beter dan llama.cpp), en échte multi-user-schaling: 10 gelijktijdige gebruikers → ~236 tok/s totaal. Maar dense en native 32k.
- **Qwen3.8-27B NVFP4 + DFlash2:** dezelfde Qwen3.8 als de llama.cpp-rijen, ~4× kleiner op schijf (20 GB) en vele malen sneller. Kwaliteit 39/43 tegen 40/43 voor Q8_0; het enige echte verschil is één verzonnen auteursnaam in de kennistest (plausibel NVFP4-kwantisatieverlies).

**De flag-les.** De eerste productie-config (c4130, 2026-10-10) gebruikte `--kv-cache-dtype fp8_e5m2`, `--max-num-batched-tokens 4096` en een 131k-contextlimiet. 1Cat's eigen statusendpoint `/v1/sm70/acceleration` liet zien dat daarmee **4 van de 8 V100-versnellingspaden UIT** stonden: `e4m3_grouped_fp32`, `long_context` en `scalar_tail` (reden: `kv_dtype`) en `q8000_prefill` (reden: `budget<8000`). Het officiële release-profiel (`qwen38_27b_nvfp4_dflash2`: `fp8_e4m3` KV, 262k context, 8192-token prefill-budget, block-size 2048 / mamba-block-size 8192) zet ze alle 8 aan — en gebruikt minder VRAM:

| | Eigen flags (131k) | Release-profiel (262k) |
|---|---|---|
| Kwaliteit | 39/43 | 39/43 (rekenen 10/10) |
| 5/5 retrieval tot | 127k tokens | **253k tokens** |
| Prefill 4k / 32k / 117k | 3710 / 3959 / 2873 t/s | **4175 / 4300 / 3327 t/s** |
| Prefill + decode bij 244k | — (past niet) | 2457 t/s prefill (TTFT 99 s), **102 t/s decode** |
| VRAM per GPU | 27.0 GB | **24.5 GB** |
| SM70-versnellingspaden aan | 4/8 | **8/8** |

**De multi-user-les (en een crash).** Alle N requests tegelijk afgevuurd, unieke prompt per gebruiker, gemeten met [`scripts/bench_concurrency.py`](scripts/bench_concurrency.py). Met het profiel ongewijzigd (`--max-num-batched-tokens 8192`) **crashte de engine** bij 10 gelijktijdige RAG-vragen van ~6k tokens: `RuntimeError: SM70 GQA architecture requires KV in [8192, 262144] with 32-token alignment, got 8160` in het `q8000_prefill`-pad → `EngineDeadError`. De scheduler knipt gelijktijdige lange prompts in stukken, en een stuk van 8160 tokens valt net buiten wat de kernel accepteert. Met **`--max-num-batched-tokens 4096`** staat alleen dat ene pad uit (7/8 aan) en draait alles stabiel — dit is de productie-config op de c4130:

| Gelijktijdig | Chat: wachten op 1e woord (p50 / max) | Chat: t/s per gebruiker | RAG ~6k: wachten (p50 / max) | RAG: klaar (max) | Totale doorvoer chat / RAG | Fouten |
|---|---|---|---|---|---|---|
| 1 | 0.3 s / 0.3 s | 100 | 2.4 s / 2.4 s | 5.9 s | 94 / 66 t/s | 0 |
| 5 | 1.1 s / 10.7 s | 38 | 7.7 s / 18.5 s | 22.6 s | 125 / 89 t/s | 0 |
| 10 | 7.9 s / 16.1 s | 51 | 18.9 s / 38.6 s | 43.6 s | 182 / 91 t/s | 0 |
| 20 | 17.4 s / 34.0 s | 50 | 41.0 s / 79.4 s | 85.4 s | 191 / 92 t/s | 0 |

`--max-num-seqs 4` betekent: maximaal 4 requests tegelijk in behandeling, de rest **wacht in de rij**. Dat zie je terug in de wachttijd tot het eerste woord, die lineair oploopt met het aantal gebruikers; de streamsnelheid per gebruiker blijft daarna redelijk (~30–50 t/s). Totale doorvoer verzadigt rond ~190 t/s (chat) en ~90 t/s (RAG, prefill-gebonden).

**De taal-les.** 1Cat adverteert 229 t/s single-request (v1.5.1). Wij zagen 350–390 t/s op Engelse code/SQL-prompts, maar **~97 t/s op Nederlandse proza** — op precies dezelfde server. DFlash2 laat een kleine drafter 7 tokens vooruit raden en het grote model die in één keer controleren; de drafter raadt Engels en code veel beter dan Nederlands, dus op Nederlandse tekst worden minder gok-tokens geaccepteerd. Voor Nederlandstalige toepassingen is ~100 t/s de realistische verwachting — nog steeds ~4× llama.cpp Q8_0.

**Andere opties op 1Cat-vLLM (nog niet zelf getest, cijfers uit de 1Cat-documentatie):**

| Optie | Wat 1Cat rapporteert | Haalbaarheid op één 4× V100-machine |
|---|---|---|
| **Qwen3.8 Flash-Next NVFP4 + MTP4** (180B MoE, 512 experts/top-10, 262k native) | 138–150 t/s decode, prefill 5.2–7.1k t/s tot 256k | Kan: 135 GB checkpoint, ~48 GB PLE-tabellen in host-RAM. Vergt ~135 GB vrije schijf (de c4140 heeft er 88). Kwaliteit alleen op kleine samples geaudit. Meest interessante "groter brein"-kandidaat. |
| **Qwen3.6-35B-A3B NVFP4 + MTP4** (23.5 GB) | 175 t/s, GSM8K 122/128 | Kan, makkelijk. Ouder model, sterk voor veel gelijktijdige gebruikers. |
| **Qwen3.8-27B FP8** (31 GB) | 51 t/s @128k, 41 t/s @256k (zonder spec) | Kan. Mogelijk iets betere kwaliteit dan NVFP4, maar trager. |
| **DeepSeek-V4-Flash** | 73.5 t/s, GSM8K 64/64, HumanEval 29/32 | **Niet**: vereist 8× V100 (PP2×TP4). Twee machines op verschillende locaties koppelen via internet is geen optie. |
| **GLM-5.3-Flash NVFP4** | 53 t/s | **Niet**: ook 8× V100 (TP4/PP2). |

## 💡 Belangrijkste lessen

1. **KV-geometrie bepaalt 200k, niet modelgrootte.** Dense 70B/72B = 320 KiB/token (200k onhaalbaar op 128 GB VRAM); hybride Mamba = 6–8 KiB/token (200k = ~1.6 GB). Check het aantal attention-lagen vóór je downloadt als lange context je doel is.
2. **Volta limiteert de runtime, niet alleen de snelheid.** Geen BF16/FP8/FA2/Marlin op sm_70; upstream vLLM/SGLang/TensorRT-LLM vallen af. Twee runtimes werken: llama.cpp (alleen met een sm_70-specifieke build, `CMAKE_CUDA_ARCHITECTURES=70`, anders valt FlashAttention weg) en de V100-fork **1Cat-vLLM**, die tensor-parallel over alle GPU's werkt en veel sneller is voor de modellen die hij ondersteunt.
3. **Reasoning-modellen kunnen hun eigen tokenbudget opeten.** Zonder ruim `max_tokens` en zonder te checken op `reasoning_content` naast `content`, lijkt een prima model plotseling "leeg" te antwoorden (zie Nemotron-3.5's lege combo-vraag en Qwen3.8's lege notulen vóór de fix).
4. **Ken de thinking-toggle van je modelfamilie — per taaktype.** Nemotron/Qwen: `chat_template_kwargs: {enable_thinking: false}` alléén voor formaat-taken; thinking **aan** voor redeneren/kennis/combo (uit zetten kost correctheid). Dense modellen negeren de flag.
5. **BF16 heeft op Volta géén zin.** Qwen3.8-BF16 (53.8 GB) draait ~1.6× trager dan Q8_0 (28.6 GB) omdat BF16 geëmuleerd wordt — geen BF16-hardware. Quantiseer, en vermijd BF16-KV.
6. **Verifieer "200k context"-claims altijd zelf.** Qwen2.5-72B claimt 128k (YaRN) maar clamt op 32k native; Qwen3.8 heeft onder llama.cpp een stille-EOS-bug boven ~130k, maar haalt onder 1Cat-vLLM 5/5 tot 253k. Alleen de sweep (serveren + terughalen op 5 dieptes) scheidt claims van feiten — en de grens kan bij de runtime liggen, niet bij het model.
7. **Gebruik het officiële serving-profiel en controleer wat er écht aan staat.** Twee "redelijke" flags (`fp8_e5m2` KV, 4096-token prefill-budget) schakelden bij 1Cat-vLLM de helft van de V100-versnellingen uit, halveerden de context en kostten 2.5 GB/GPU. Vraag `/v1/sm70/acceleration` op: dat endpoint noemt per pad of het aan staat en waarom niet.
8. **Speculatieve decoding maakt snelheid taal- en tekstafhankelijk.** Dezelfde server deed 350–390 t/s op Engelse code en ~97 t/s op Nederlandse proza. Meet met je eigen soort tekst, niet met de demo-prompt van de leverancier.
9. **Tel tokens, niet stream-chunks.** Bij speculatieve decoding kan één SSE-chunk meerdere geaccepteerde tokens bevatten. Neem `usage.completion_tokens` (`stream_options.include_usage`) als bron — anders meet je een ondergrens (zie de 124 t/s-meting).
10. **Test met gelijktijdige gebruikers vóór je naar productie gaat.** De volledige single-user-batterij (39/43, 5/5 tot 253k) liet niets zien van de kernel-crash die pas bij 10 gelijktijdige lange prompts optrad. Een officieel "release-profiel" is gevalideerd op het contract van de leverancier, niet op jouw verkeer.

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
git clone https://github.com/sayfjawad/c41xx-llm-benchmarks.git
cd c41xx-llm-benchmarks

# Eén model: downloaden, server starten, batterij draaien
./scripts/run_benchmarks.sh my-model ~/models/my-model.gguf

# Context-sweep (de nieuwe test)
./scripts/run_sweep.sh my-model ~/models/my-model.gguf

# Resultaten naar het dashboard
python3 scripts/build_results.py --results-dir results/ --out docs/data/results.json
```

Voor de sm_70-specifieke build, de server-flags voor single-stream lange context (`-c <n> -np 1 -fa on -kvu`, symmetrische `-ctk q8_0 -ctv q8_0`, `-sm layer -ts 8,8,8,8`) en de glibc-patch: zie [METHODOLOGY.md](METHODOLOGY.md).

**Tegen een vLLM-server** (bv. 1Cat-vLLM op poort 8001) draai je dezelfde scripts met twee omgevingsvariabelen, plus de snelheidsmeting:

```bash
export BENCH_URL=http://127.0.0.1:8001/v1/chat/completions FORCE_THINKING=1
M=qwen3.8-27b-nvfp4-vllm-262k   # = --served-model-name van de server
python3 scripts/bench_hard.py 8001 $M
python3 scripts/bench_longctx.py 8001 $M --nothink
python3 scripts/bench_notulen_repeat.py 8001 $M 5 --nothink
python3 scripts/bench_tools.py 8001 $M --nothink
python3 scripts/bench_knowledge.py 8001 $M
python3 scripts/bench_math.py 8001 $M
python3 scripts/bench_longctx_sweep.py 8001 $M --sizes 8192,32768,65536,110000,131072,200000,220000,262144
BENCH_URL=http://127.0.0.1:8001 python3 scripts/bench_vllm_speed.py $M results/$M-speed.json 512,4096,32768,120000,250000
```

De exacte 1Cat-serverflags staan in [METHODOLOGY.md](METHODOLOGY.md#1cat-vllm-op-v100).

## 📚 Bronnen

- Modelgewichten: [Hugging Face](https://huggingface.co) — GGUF-conversies van [unsloth](https://huggingface.co/unsloth), [ggml-org](https://huggingface.co/ggml-org) en [bartowski](https://huggingface.co/bartowski)
- Inference-engines: [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (`llama-server`, `llama-bench`) en [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) v1.5.1 (V100/sm_70-fork van vLLM)
- vLLM-checkpoints: [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) + drafter [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
- Modelverificatie: officiële Hugging Face API — expliciet niet via web-scraping of blogs
- Open bug: [llama.cpp #27756](https://github.com/ggml-org/llama.cpp/issues/27756) (Qwen3.8 stille EOS boven ~130k)

---

<sub>Gegenereerd met behulp van Claude (Anthropic) als benchmarking-copiloot — alle downloads, serverstarts, testruns en codeverificaties zijn daadwerkelijk uitgevoerd op de genoemde hardware, niet gesimuleerd.</sub>
