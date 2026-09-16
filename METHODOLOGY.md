# METHODOLOGY — C4130 LLM Benchmark Suite

Universele, reproduceerbare benchmark voor lokale LLM's. Draait op elke machine met [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (CUDA, Metal, Vulkan, CPU). Alle tests zijn **geautomatiseerd gescoord** — geen handmatige beoordeling nodig.

## Testcategorieën

### 1. ⚡ Snelheid (`llama-bench`)

**Wat:** Pure hardware+software throughput. Prompt processing (pp512) en generatie (tg128) tokens/seconde.

**Commando:** `llama-bench -m <model>.gguf -ngl 999`

**Wat het meet:** Bandbreedte-efficiëntie. MoE vs dense op geheugenbandbreedte-gelimiteerde hardware.

---

### 2. 🧩 Zware taken (4 tests)

**Script:** `bench_hard.py`

#### 2a. Multi-stap rekenen (`hard_reasoning`)

**Prompt:**
> Drie ploegen werken aan een sorteercentrum, allemaal van 08:00 tot 16:00 (behalve waar anders vermeld).
> - Ploeg 1 verwerkt 120 pakketten per uur gedurende de eerste 3 uur, daarna neemt de snelheid af naar 90 pakketten per uur voor de resterende tijd.
> - Ploeg 2 verwerkt constant 100 pakketten per uur, maar heeft na 5 uur werken een pauze van 45 minuten waarin niet wordt verwerkt.
> - Ploeg 3 begint pas om 10:00 (2 uur later dan de andere ploegen) en verwerkt vanaf dat moment 110 pakketten per uur, tot 16:00.
>
> Hoeveel pakketten zijn er in totaal door de drie ploegen samen verwerkt om 16:00?

**Antwoord:** 2195

**Berekening:**
- Ploeg 1: 3×120 + 5×90 = 360 + 450 = 810
- Ploeg 2: 8×100 = 800, min 45 minuten pauze na 5 uur → 800 - 75 = 725
- Ploeg 3: 6×110 = 660
- Totaal: 810 + 725 + 660 = 2195

**Auto-scoring:** zoekt `EINDANTWOORD:` gevolgd door een getal, vergelijkt met 2195.

#### 2b. LRU-cache implementatie (`hard_coding`)

**Prompt:** Implementeer een Python `LRUCache` klasse met `get(key)` en `put(key, value)`, beide O(1).

**Auto-scoring:** Checkt op aanwezigheid van `class LRUCache` en `OrderedDict` in de output.

#### 2c. Notulen-structurering (`scribr_notulen`)

**Prompt:** Een ~7.5K-token raadsvergaderingstranscript omzetten naar gestructureerde notulen met `## Besluiten`, `## Actiepunten`, `## Openstaande vragen`.

**Transcript:** Fictieve gemeenteraadsvergadering (ingebed in `bench_hard.py` regels 20-43).

**Auto-scoring:** Checkt of alle 3 de verplichte headers in het antwoord voorkomen.

#### 2d. Strikte JSON-instructies (`strict_instructions`)

**Prompt:** Beoordeel een productiedeviatie-rapport en geef JSON terug met 4 velden: `samenvatting` (max 30 woorden), `kernwoorden` (exact 5), `risico_score` (float 0.0-1.0, 2 decimalen), `vereist_escalatie` (boolean).

**Auto-scoring:** Parseert antwoord als JSON (stript markdown fences), valideert alle 4 velden + constraints.

---

### 3. 📄 Long-context (6 needles)

**Script:** `bench_longctx.py` + `long_transcript.py` (data)

**Transcript:** ~8.000 tokens, fictieve tweedaagse strategieconferentie van "Meridian Logistics Group" + 5 appendices.

**6 needles:**

| # | Vraag | Verwacht antwoord (keywords) | Positie |
|---|---|---|---|
| 1 | Budget ERP-project PRJ-2247, wat dekt het NIET? | 487.300, maatwerk | vroeg |
| 2 | Nieuwe hoofd Inkoop per 1 sept + voorganger? | Fatima El Ouazzani, Robert Dijkhuis | vroeg-midden |
| 3 | Deadline CRM-migratie + vervallen korting? | 15 november, 12 procent | midden |
| 4 | Ziekteverzuim daling sinds hybride werken? | 6,8, 4,3 | midden-laat |
| 5 | Stemverhouding kantoorlocatie Utrecht + voorwaarde? | 9, 3, 5 jaar | laat |
| 6 | Actiehouder juridische toets huurcontract + deadline? | Bas Kremer, 20 december | zeer laat |

**Auto-scoring:** Keyword-matching op genormaliseerde tekst (case-insensitive, whitespace-genormaliseerd).

---

### 4. 🔁 Betrouwbaarheid (5× notulen)

**Script:** `bench_notulen_repeat.py`

**Wat:** Dezelfde notulen-taak (2c) 5 keer herhaald bij `temperature=0.7`.

**Auto-scoring:** Elke run wordt gecheckt op aanwezigheid van alle 3 verplichte headers.

**Doel:** Vind reproduceerbaarheidsbugs (zoals MiniMax-M2's herhalingslus).

---

### 5. 🔧 Tool-gebruik (4 scenario's)

**Script:** `bench_tools.py`

**Mock tools:** `get_weather` (temperatuur per stad), `calculate` (rekenkundige expressie), `search_klantendatabase` (klant opzoeken).

**4 scenario's:**

| Scenario | Prompt | Verwachting |
|---|---|---|
| single_tool_call | "Hoe warm is het in Utrecht?" | `get_weather(city="Utrecht")` |
| chained_tool_calls | "Zoek temperatuur Utrecht, bereken (temp+15)×2" | `get_weather` → `calculate` |
| irrelevant_tool_avoidance | "Wat is de hoofdstad van Frankrijk?" | Géén tool-call, direct antwoord |
| multi_entity_tool_use | "Zoek klant Jansen BV. Als openstaand bedrag 0 is, zoek dan temperatuur Rotterdam" | Conditioneel: `search_klantendatabase` → `get_weather` |

**Auto-scoring:** Valideert tool-call traces (juiste functienaam, parameters, volgorde, conditionele logica).

**Let op:** Server moet gestart zijn met `--jinja` voor correcte chat-template verwerking.

---

### 6. 📚 Kennis (14 vragen)

**Script:** `bench_knowledge.py`

**12 pure kennisvragen + 2 kennis+redeneer-combinaties:**

| # | Sleutel | Vraag | Keywords voor correct |
|---|---|---|---|
| 1 | scheikunde_1 | Brutoformule cafeïne? | c8h10n4o2 |
| 2 | geschiedenis_1 | Jaar Verdrag van Utrecht (Spaanse Successieoorlog)? | 1713 |
| 3 | geografie_1 | Diepste plek Middellandse Zee + diepte? | calypso, 5267 |
| 4 | biologie_1 | Enzym zetmeel→maltose in mond? | amylase |
| 5 | wiskunde_1 | Euler-Mascheroniconstante (4 decimalen)? | 0.5772 |
| 6 | technologie_1 | Datum TCP/IP ARPANET standaard? | 1983, 1 januari |
| 7 | literatuur_1 | Auteur + jaar "De ontdekking van de hemel"? | mulisch, 1992 |
| 8 | astronomie_1 | Omlooptijd Pluto (aardse jaren)? | 248 |
| 9 | economie_1 | Jaar eerste effectenbeurs Amsterdam + bedrijf? | 1602, voc |
| 10 | scheikunde_2 | Smeltpunt wolfraam (°C)? | 3422 |
| 11 | muziek_1 | Aantal delen Beethovens 9e symfonie? | 4 |
| 12 | natuurkunde_1 | Dichtheid kwik (g/cm³)? | 13.5 |
| 13 | combo_1 | Franse koning 1789 Bastille + regeerjaren? | lodewijk, 15 |
| 14 | combo_2 | Leeftijd Napoleon bij Waterloo (juni 1815)? | 45 (of 46) |

**Auto-scoring:** Keyword-matching (case-insensitive). Combo-vragen hebben alternatieve keywords (45/46 voor Napoleon).

---

### 7. 🧮 Rekenen/redeneren (10 problemen)

**Script:** `bench_math.py`

| # | Sleutel | Domein | Vraag (kort) | Antwoord |
|---|---|---|---|---|
| 1 | stap_korting | Rekenen | 250€, 20%→10%→-€15 korting | 165 euro |
| 2 | logica_ridders | Logica | A: "B is schurk", B: "we zijn beiden ridders" | A=ridder, B=schurk |
| 3 | kans_dobbelsteen | Kansrekening | Kans op som 7 met 2 dobbelstenen | 1/6 |
| 4 | combinatoriek | Combinatoriek | 5 boeken over 3 personen, elk ≥1 | 150 |
| 5 | algebra | Algebra | Twee getallen: som=47, verschil=13 | 30 en 17 |
| 6 | getaltheorie | Getaltheorie | Kleinste n! deelbaar door 1000 | 15 |
| 7 | meetkunde | Meetkunde | Driehoek: hyp=13cm, zijde=5cm, oppervlakte? | 30 cm² |
| 8 | rente | Financieel | €1000, 4% rente, wanneer >verdubbeld? | 18 jaar |
| 9 | valstrik_schapen | Valkstrik | "17 schapen, op 1 na sterven allemaal" | 1 schaap |
| 10 | valstrik_caesar | Valkstrik | Caesar: juli 100 v.Chr. – 15 maart 44 v.Chr. | 55 jaar |

**Auto-scoring:** Parseert `EINDANTWOORD:` uit de output, vergelijkt met ground truth. De ground truth keywords staan in `GROUND_TRUTH` in het script.

---

## Scoring-overzicht

| Test | Aantal items | Auto-scoring methode | Perfecte score |
|---|---|---|---|
| Snelheid | 2 metingen | n.v.t. (ruwe t/s) | n.v.t. |
| Zware taken | 4 | EINDANTWOORD + headers + JSON-parse + class-check | 4/4 |
| Long-context | 6 needles | Keyword-matching | 6/6 |
| Betrouwbaarheid | 5 runs | Header-check per run | 5/5 |
| Tools | 4 scenario's | Tool-call trace validatie | 4/4 |
| Kennis | 14 vragen | Keyword-matching | 14/14 |
| Rekenen | 10 problemen | EINDANTWOORD-parsing | 10/10 |

**Totaal perfecte score: 43/43** (excl. snelheid)

---

## Draaien op een nieuwe machine

```bash
# 1. Clone de repo
git clone https://github.com/sayfjawad/c4130-llm-benchmarks
cd c4130-llm-benchmarks

# 2. Download een model (voorbeeld)
hf download unsloth/gpt-oss-120b-GGUF --include "UD-Q4_K_XL/*" --local-dir ~/models/gpt-oss-120b

# 3. Draai de volledige batterij
./scripts/run_benchmarks.sh gpt-oss-120b ~/models/gpt-oss-120b/UD-Q4_K_XL/*.gguf

# 4. Update het dashboard
python3 scripts/aggregate_results.py --results-dir results/ --model gpt-oss-120b --out docs/data/results.json

# 5. Bekijk lokaal
cd docs && python3 -m http.server 8080
# Open http://localhost:8080
```

## Vereisten

- Python 3.8+ (stdlib only — geen `pip install` nodig)
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) met `llama-server` en `llama-bench` in PATH
- `curl` (voor health checks in `run_benchmarks.sh`)
- Optioneel: [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub) CLI voor model-downloads (`pip install huggingface_hub[cli]`)
