#!/usr/bin/env python3
"""bench_vllm_speed.py — single-stream decode/prefill speed for an OpenAI-compatible vLLM server.

Counts tokens from the server's own usage.completion_tokens (NOT SSE chunks: with
speculative decoding one chunk can carry several accepted tokens). Contract mirrors
1Cat-vLLM's v1.5.1 release table: 1 cold + 3 warm requests per context size, median
of the warm ones, 256 output tokens, T=1 / top_p=.95 / top_k=20 / seed=0, normal EOS.
pure decode = completion_tokens / (total - TTFT). Prefill = prompt_tokens / TTFT of
the COLD request (warm repeats hit the prefix cache).

Usage: BENCH_URL=http://127.0.0.1:8001 python3 bench_vllm_speed.py MODEL OUT.json [ctx,ctx,...]
"""
import json, os, statistics, sys, time, urllib.request

BASE = os.environ.get("BENCH_URL", "http://127.0.0.1:8001").rstrip("/")
MODEL = sys.argv[1]
OUT = sys.argv[2]
SIZES = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "512,4096,32768,131072").split(",")]

FILLER = ("De vergadering besprak de kwartaalcijfers, de planning van het wagenpark, "
          "het ziekteverzuim per regio en de voortgang van de ERP-migratie. ")
TASK = "\n\nSchrijf nu een uitgebreide, doorlopende samenvatting van bovenstaande tekst in het Nederlands."


def tokens(text):
    body = json.dumps({"model": MODEL, "prompt": text}).encode()
    req = urllib.request.Request(BASE + "/tokenize", data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["count"]


def build(n_target, salt):
    # salt makes each size's prompt unique so a size never prefix-hits a previous size
    per = tokens(FILLER) or 25
    reps = max(1, (n_target - 40) // per)
    text = f"[document {salt}]\n" + FILLER * reps + TASK
    return text


def one(prompt):
    body = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 0,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ch = json.loads(line[5:])
            if ch.get("usage"):
                usage = ch["usage"]
            for c in ch.get("choices") or []:
                d = c.get("delta") or {}
                if ttft is None and (d.get("content") or d.get("reasoning_content")):
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    n = usage["completion_tokens"]; p = usage["prompt_tokens"]
    return {"prompt_tokens": p, "completion_tokens": n, "ttft_s": round(ttft, 3), "total_s": round(total, 3),
            "decode_tps": round((n - 1) / (total - ttft), 2) if n > 1 else 0.0,
            "prefill_tps": round(p / ttft, 1)}


rows = []
for i, size in enumerate(SIZES):
    prompt = build(size, f"{size}-{i}")
    try:
        runs = [one(prompt) for _ in range(4)]
    except Exception as e:
        print(f"{size}: ERROR {str(e)[:200]}", flush=True)
        rows.append({"ctx_target": size, "error": str(e)[:300]}); continue
    warm = runs[1:]
    row = {"ctx_target": size, "prompt_tokens": runs[0]["prompt_tokens"],
           "cold_prefill_tps": runs[0]["prefill_tps"], "cold_ttft_s": runs[0]["ttft_s"],
           "decode_tps_median_warm": statistics.median(r["decode_tps"] for r in warm),
           "completion_tokens_warm": [r["completion_tokens"] for r in warm], "runs": runs}
    rows.append(row)
    print(f"{size:>7}: prompt={row['prompt_tokens']:>7} cold-prefill={row['cold_prefill_tps']:>8} t/s "
          f"(ttft {row['cold_ttft_s']}s)  decode(median warm)={row['decode_tps_median_warm']} t/s "
          f"out={row['completion_tokens_warm']}", flush=True)

json.dump({"model": MODEL, "url": BASE, "contract": "1 cold + 3 warm, median warm, 256 out, T=1 top_p=.95 top_k=20 seed=0, thinking off",
           "sizes": rows}, open(OUT, "w"), indent=2)
print("saved", OUT)
