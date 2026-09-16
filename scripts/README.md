# Benchmark Scripts

Pure Python (stdlib only — geen `pip install` nodig). Elk script praat via HTTP met een lokaal draaiende [`llama-server`](https://github.com/ggml-org/llama.cpp) op de OpenAI-compatibele `/v1/chat/completions` endpoint.

**Alle scripts hebben nu auto-scoring** — geen handmatige beoordeling meer nodig. Elke test produceert een JSON met per-vraag `correct: true/false` en een `_summary` met de totaalscore.

## Snelstart

```bash
# Eén commando: alles draaien voor één model
./scripts/run_benchmarks.sh my-model ~/models/my-model.gguf

# Of handmatig, stap voor stap:
# 1. Start server
llama-server --model ~/models/my-model.gguf --host 127.0.0.1 --port 8081 --fit on --cont-batching -ngl 999

# 2. Draai tests (resultaten in results/)
python3 scripts/bench_hard.py       8081 my-model
python3 scripts/bench_longctx.py    8081 my-model
python3 scripts/bench_notulen_repeat.py 8081 my-model 5
python3 scripts/bench_tools.py      8081 my-model
python3 scripts/bench_knowledge.py  8081 my-model
python3 scripts/bench_math.py       8081 my-model

# 3. Update dashboard
python3 scripts/aggregate_results.py --results-dir results/ --model my-model --out docs/data/results.json
```

## Scripts

| Script | Taken | Auto-scoring | Output |
|---|---|---|---|
| `bench_hard.py` | 4 zware taken: multi-stap rekenen, LRU-cache, notulen, strikte JSON | ✅ 4/4 | `<model>-hard.json` |
| `bench_longctx.py` | ~8K-token transcript, 6 needles | ✅ keyword-match | `<model>-longctx.json` |
| `bench_notulen_repeat.py` | Notulen 5× bij temp=0.7 | ✅ header-check | `<model>-notulen-repeat.json` |
| `bench_tools.py` | 4 tool-use scenario's | ✅ call-validatie | `<model>-tools.json` |
| `bench_knowledge.py` | 14 kennisvragen (12 + 2 combo) | ✅ keyword-match | `<model>-knowledge.json` |
| `bench_math.py` | 10 reken/redeneerproblemen | ✅ EINDANTWOORD-parsing | `<model>-math.json` |
| `long_transcript.py` | ~8K-token transcript data | — | (geïmporteerd) |
| `run_benchmarks.sh` | Orchestratie: server starten → alle tests → server stoppen | — | Volledige set |
| `aggregate_results.py` | Ruwe results → `docs/data/results.json` | — | Dashboard data |

## Opties per script

Alle scripts accepteren:
```
python3 bench_<test>.py <PORT> <MODEL_NAME> [--nothink] [--out-dir <dir>]
```

| Optie | Effect |
|---|---|
| `--nothink` | Stuurt `enable_thinking: false` mee (Nemotron/Qwen) of gebruikt server met `--reasoning off` |
| `--out-dir <dir>` | Output directory (default: `results/`) |

## Thinking/redeneren per taaktype

Dit is de belangrijkste les uit 7 modellen testen:

| Taaktype | Thinking | Waarom |
|---|---|---|
| Formaat (notulen, JSON, tools) | **UIT** | Denkstap eet tokenbudget op, geen toegevoegde waarde |
| Long-context retrieval | **UIT** | Idem — model moet extraheren, niet redeneren |
| Feitelijke kennis (12/14 vragen) | **UIT** | Antwoord zit in de weights |
| Kennis+redeneer combo's (2/14) | **AAN** | Vereist redeneren over twee feiten |
| Rekenen/wiskunde (alle 10) | **AAN** | Vereist sequentieel denkproces |
| Betrouwbaarheid (notulen 5×) | **UIT** | Formaattaak, zie hierboven |

**Per modelfamilie:**
- **NVIDIA Nemotron**: `"chat_template_kwargs": {"enable_thinking": false}` in request body (of `--nothink` flag)
- **Qwen3.x**: zelfde mechanisme als Nemotron
- **Google Gemma 4**: `--reasoning off` server flag + `--reasoning-budget N` voor rekenen
- **Overige modellen**: meestal geen thinking toggle nodig

## Belangrijke lessen

- **`--fit on`** (recente `llama-server`) laat de server zelf de layer/context-verdeling bepalen — voorkom `-ngl 999` te forceren op modellen die net niet in GPU-geheugen passen.
- **Reasoning-modellen kunnen het volledige tokenbudget opsouperen** aan hun denkstap. Check zowel `content` als `reasoning_content` in de response.
- **Multi-file GGUF's**: `hf download <repo> --include "<QUANT>/*"` (met `/*`!) — check eerst met `huggingface_hub.HfApi().model_info(repo, files_metadata=True)` welk patroon van toepassing is.
- **Gemma 4 reasoning budget**: zonder `--reasoning-budget` limiet genereert Gemma 4 tot 1864 denk-tokens per rekenvraag — met `--reasoning-budget 800` blijft het <1600 en is het even accuraat.
