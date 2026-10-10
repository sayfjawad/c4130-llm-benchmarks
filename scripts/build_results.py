#!/usr/bin/env python3
"""
build_results.py — Genereert docs/data/results.json volledig uit de ruwe resultaten in results/.

Dit is de EIND-productie-step voor het dashboard. Het leest:
  - results/<model>-{hard,longctx,notulen-repeat,tools,knowledge,math}.json  (scores)
  - results/<model>-llamabench.txt                                            (snelheid)
  - results/<model>-longctx-sweep.json                                        (context-bereik)
en vult de statische metadata (uitgever, params, architectuur, disk_gb, verdict, notes)
hieronder hardcoded in.

Gebruik:
  python3 scripts/build_results.py [--results-dir results] [--out docs/data/results.json]
"""
import json
import os
import re
import sys
import argparse

# ── Statische metadata per model (alles wat niet uit raw results komt) ──────────────
# name: display-naam op het dashboard; params in miljarden; disk_gb: bestandsgrootte GGUF.
METADATA = {
    "gpt-oss-120b": {
        "name": "GPT-OSS-120B", "publisher": "OpenAI",
        "hf_repo": "ggml-org/gpt-oss-120b-GGUF", "quant": "MXFP4 (native)",
        "params_total_b": 117, "params_active_b": 5.1, "architecture": "MoE (sliding-window attention)",
        "disk_gb": 63.4, "ctx_cap": 131072,
        "verdict": "Snelste topkwaliteit op 128k — maar géén 200k.",
        "notes": "42/43. SWA-KV-cache (36 KiB/token) laat 128k toe, maar max_position_embeddings=131072 is de harde grens.",
    },
    "nemotron-3-super": {
        "name": "Nemotron-3-Super-120B-A12B", "publisher": "NVIDIA",
        "hf_repo": "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF", "quant": "UD-Q4_K_M (3 shards)",
        "params_total_b": 120, "params_active_b": 12, "architecture": "Hybride Mamba-MoE",
        "disk_gb": 82.5, "ctx_cap": 262144,
        "verdict": "Hét antwoord op 200k: enige 120B-klasse model dat 243k tokens serveert.",
        "notes": "42/43. Slechts 8/88 attention-lagen → 8 KiB/token KV. 200k ≈ 87 GB van 128 GB VRAM. Enige model >80GB (expliciet getest).",
    },
    "nemotron-3.5-lightning": {
        "name": "Nemotron-3.5-Lightning-30B-A3B", "publisher": "NVIDIA",
        "hf_repo": "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", "quant": "UD-Q4_K_XL",
        "params_total_b": 30, "params_active_b": 3.0, "architecture": "Hybride Mamba2-MoE",
        "disk_gb": 25.5, "ctx_cap": 262144,
        "verdict": "Snelste (142 t/s) én 200k-capabel, kleinste footprint.",
        "notes": "39/43. 6/52 attention-lagen → 6 KiB/token. Rekenvraag kans_dobbelsteen loopt leeg (thinking-runaway). Vereist enable_thinking:false voor formaat-taken.",
    },
    "llama-3.3-70b": {
        "name": "Llama-3.3-70B-Instruct", "publisher": "Meta",
        "hf_repo": "bartowski/Llama-3.3-70B-Instruct-GGUF", "quant": "Q4_K_M",
        "params_total_b": 70, "params_active_b": 70, "architecture": "Dense",
        "disk_gb": 42.5, "ctx_cap": 131072,
        "verdict": "Dense 70B: 16 t/s, en tool-calling faalt op deze llama.cpp-build.",
        "notes": "37/43. Tools 2/4 — llama.cpp's tool-parser verwerpt Llama-3.3's tool-output (HTTP 500). 320 KiB/token KV → 128k kost ~43 GB KV.",
    },
    "qwen2.5-72b": {
        "name": "Qwen2.5-72B-Instruct", "publisher": "Alibaba",
        "hf_repo": "bartowski/Qwen2.5-72B-Instruct-GGUF", "quant": "Q4_K_M",
        "params_total_b": 72, "params_active_b": 72, "architecture": "Dense",
        "disk_gb": 47.4, "ctx_cap": 32768,
        "verdict": "Dense 72B: 15 t/s, native 32k (géén YaRN) — clamt vroeg.",
        "notes": "36/43. 32k native zonder YaRN-extensie → clamt al bij 32k. 320 KiB/token KV. 7/10 rekenen.",
    },
    "qwen3.8-27b-q8_0": {
        "name": "Qwen3.8-27B (Q8_0)", "publisher": "Alibaba",
        "hf_repo": "ggml-org/Qwen3.8-27B-GGUF", "quant": "Q8_0",
        "params_total_b": 27, "params_active_b": 27, "architecture": "Hybride Gated DeltaNet",
        "disk_gb": 28.6, "ctx_cap": 131072,
        "verdict": "GDN-hybride: 24 t/s, 131k-cap (open bug #27756 boven ~130k).",
        "notes": "40/43. 16/64 attention-lagen → 64 KiB/token. Q8_0 is 1.6× sneller dan BF16 (geen BF16-hardware op Volta).",
    },
    "qwen3.8-27b-bf16": {
        "name": "Qwen3.8-27B (BF16)", "publisher": "Alibaba",
        "hf_repo": "ggml-org/Qwen3.8-27B-GGUF", "quant": "BF16",
        "params_total_b": 27, "params_active_b": 27, "architecture": "Hybride Gated DeltaNet",
        "disk_gb": 53.8, "ctx_cap": 131072,
        "verdict": "BF16 zonder hardware-support op Volta: 15 t/s, géén winst over Q8_0.",
        "notes": "BF16 is geëmuleerd op sm_70 (geen BF16-hardware) → trager dan Q8_0. Bewijs dat BF16 op V100 geen zin heeft.",
    },
    "qwen2.5-32b-vllm": {
        "name": "Qwen2.5-32B-Instruct (vLLM TP4)", "publisher": "Alibaba",
        "hf_repo": "Qwen/Qwen2.5-32B-Instruct", "quant": "FP16 (vLLM)",
        "params_total_b": 32, "params_active_b": 32, "architecture": "Dense — vLLM tensor-parallel ×4",
        "disk_gb": 62.0, "ctx_cap": 32768,
        "verdict": "ANDERE runtime: vLLM 0.9.2 op sm_70 met échte 4-GPU tensor-parallelism — de multi-user winnaar.",
        "notes": "36/43 (tools 4/4 — vLLM's hermes-parser verslaat llama.cpp hier). Draait op c4140 (4× V100), NIET llama.cpp: kwaliteitsscores zijn vergelijkbaar (zelfde prompts+scoring), snelheid NIET (TP-4 vs layer-split). Dense 32B FP16, 32k native (geen YaRN) → sweep clamt bij 32k, net als Qwen2.5-72B. tg128=27 t/s single-stream (--enforce-eager; CUDA-graphs zouden dit verhogen), prefill ~4200 t/s (TP parallelt prefill). Échte concurrency-winst: 10 gelijktijdige users → ~236 tok/s aggregaat, per-user vlak ~24 tok/s, alle 4 GPU's 85-95% — waar llama.cpp layer-split plateaut op ~1 GPU.",
    },
    "qwen3.8-27b-nvfp4-vllm": {
        "name": "Qwen3.8-27B NVFP4 (vLLM/1Cat + DFlash2)", "publisher": "Alibaba",
        "hf_repo": "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4", "quant": "NVFP4 (QUASAR QAT) + DFlash2",
        "params_total_b": 27, "params_active_b": 27, "architecture": "Hybride Gated DeltaNet — vLLM/1Cat TP×4 (sm_70)",
        "disk_gb": 20.0, "ctx_cap": 131072,
        "verdict": "Snelste sterke model met 128k: 1Cat-vLLM + NVFP4 + DFlash2 — 124 t/s (~5× llama.cpp Q8_0), 39/43, 5/5 retrieval tot 127k tokens.",
        "notes": "39/43 (2026-10-11, volledige battery op c4140, identieke prompts+scoring; thinking per taaktype als bij Q8_0/BF16 via FORCE_THINKING=1, want de productie-server zet thinking standaard UIT). Vs Q8_0 (40/43): 1 extra kennisfout (literatuur_1: Mulisch → gehallucineerde auteur — plausibel NVFP4-kwantisatieverlies); rekenen 9/10 zoals Q8_0 maar andere fout (logica_ridders: thinking-runaway, 2000-token budget op, lege content). Sweep: 5/5 naalden tot 127.006 tokens; 131k/200k = harde max-model-len 131072. Engine = 1Cat-vLLM v1.5.1 (CUDA 12.8.1, docker) op 4× V100-32GB sm_70, FLASH_ATTN_V100, TP=4, fp8_e5m2 KV, NVFP4-gewichten + DFlash2-drafter. tg128=124 t/s single-stream vs Q8_0 llama.cpp 24 t/s → ~5×; pp512≈1175 t/s. Concurrency (max-num-seqs 4): 1 user 137 → 4 users 233 tok/s aggregaat. Productie op c4130 :8080 sinds 2026-10-10.",
    },
}

# Score-key per raw-bestand-suffix (identiek aan aggregate_results.py).
TEST_MAP = {
    "hard": "hard4", "longctx": "longctx", "notulen-repeat": "reliability",
    "tools": "tools", "knowledge": "knowledge", "math": "math",
}


def load_scores(results_dir, model_id):
    """Lees alle scores voor één model (zelfde logica als aggregate_results.py)."""
    scores = {}
    for suffix, score_key in TEST_MAP.items():
        fpath = os.path.join(results_dir, f"{model_id}-{suffix}.json")
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue
        summary = data.get("_summary", {})
        n = summary.get("n_correct") or summary.get("n_ok")
        if n is None:
            n = data.get("score")
        n_total = summary.get("n_total") or data.get("n_needles", 0)
        if n is None and n_total:
            n = 0
        scores[score_key] = {"n": n, "of": n_total}
    return scores


def load_speed(results_dir, model_id):
    """Lees tg128/pp512 uit llama-bench output."""
    bench_file = os.path.join(results_dir, f"{model_id}-llamabench.txt")
    if not os.path.exists(bench_file):
        return None, None
    text = open(bench_file).read()
    tg = re.search(r'tg128\s*\|\s*([\d.]+)', text)
    pp = re.search(r'pp512\s*\|\s*([\d.]+)', text)
    return (float(tg.group(1)) if tg else None), (float(pp.group(1)) if pp else None)


def load_sweep(results_dir, model_id):
    """Lees long-context sweep: max behaalde context + per-maat detail."""
    fpath = os.path.join(results_dir, f"{model_id}-longctx-sweep.json")
    if not os.path.exists(fpath):
        return None, None, []
    try:
        data = json.load(open(fpath))
    except Exception:
        return None, None, []
    sizes = data.get("sizes", [])
    max_ok = None
    detail = []
    for s in sizes:
        status = s.get("status")
        tok = s.get("prompt_tokens")
        if status == "ok" and tok is not None:
            if max_ok is None or tok > max_ok:
                max_ok = tok
        detail.append({
            "ctx_requested": s.get("ctx_requested"),
            "prompt_tokens": tok,
            "score": s.get("score"),
            "of": s.get("n_needles"),
            "status": status,
            "failure_mode": (s.get("failure_mode") or "").split("}")[0][:90] if s.get("failure_mode") else None,
            "decode_tps": s.get("decode_tps"),
        })
    return max_ok, detail, data.get("sizes", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="docs/data/results.json")
    args = ap.parse_args()

    models = []
    for model_id, meta in METADATA.items():
        scores = load_scores(args.results_dir, model_id)
        gen_tps, prompt_tps = load_speed(args.results_dir, model_id)
        max_ctx, sweep_detail, _ = load_sweep(args.results_dir, model_id)

        entry = {
            "id": model_id,
            "name": meta["name"],
            "publisher": meta["publisher"],
            "hf_repo": meta["hf_repo"],
            "quant": meta["quant"],
            "params_total_b": meta["params_total_b"],
            "params_active_b": meta["params_active_b"],
            "architecture": meta["architecture"],
            "disk_gb": meta["disk_gb"],
            "ctx_cap": meta["ctx_cap"],
            "max_ctx_tokens": max_ctx,
            "gen_tps": gen_tps,
            "prompt_tps": prompt_tps,
            "scores": scores,
            "verdict": meta["verdict"],
            "notes": meta["notes"],
        }
        if sweep_detail:
            entry["longctx_sweep"] = sweep_detail
        models.append(entry)

    # Volgorde: op max_ctx_tokens (200k-modellen eerst), dan op gen_tps.
    models.sort(key=lambda m: (-(m["max_ctx_tokens"] or 0), -(m["gen_tps"] or 0)))

    out = {
        "meta": {
            "title": "C41xx LLM Benchmark Suite",
            "hardware": "Dell PowerEdge C4130 — 4× NVIDIA Tesla V100-SXM2-32GB (128GB dedicated VRAM)",
            "memory_gb": 128,
            "gpu_addressable_gb": 128,
            "bandwidth_gbps": 900,
            "last_updated": "2026-10-10",
            "test_categories": [
                {"id": "speed", "label": "Snelheid", "tool": "llama-bench / server timings"},
                {"id": "hard4", "label": "Zware taken (4)", "desc": "multi-stap redeneren, LRU-cache coding, notulen-structurering, strikte JSON-instructies"},
                {"id": "longctx", "label": "Long-context (6)", "desc": "~8k-token transcript, 6 verifieerbare feiten verspreid vroeg/midden/laat"},
                {"id": "reliability", "label": "Betrouwbaarheid (5x)", "desc": "dezelfde taak 5x herhaald, faalpercentage gemeten"},
                {"id": "tools", "label": "Tool-gebruik (4)", "desc": "losse tool-call, gekoppelde tool-calls, tools terecht vermijden, conditionele multi-tool taak"},
                {"id": "knowledge", "label": "Kennis (14)", "desc": "brede/niche feitenkennis + 2 kennis+redeneer-combinaties"},
                {"id": "math", "label": "Rekenen/redeneren (10)", "desc": "logica, kansrekening, combinatoriek, algebra, getaltheorie, meetkunde, samengestelde rente, 2 valstrikvragen"},
            ],
        },
        "models": models,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Geschreven naar {args.out} — {len(models)} modellen")
    for m in models:
        sc = ", ".join("{}={}/{}".format(k, v.get("n"), v.get("of")) for k, v in m["scores"].items())
        print(f"  {m['id']:26s} max_ctx={m['max_ctx_tokens']}  tg128={m['gen_tps']}  {sc}")


if __name__ == "__main__":
    main()
