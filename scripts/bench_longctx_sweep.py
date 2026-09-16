#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_longctx_sweep.py — long-context retrieval sweep for c4130.

Sends a deterministic Dutch document at increasing context sizes (8k → 200k)
with five needles planted at five depths, and records retrieval accuracy,
prefill/decode speed and peak per-GPU VRAM. This is the novel contribution of
the c4130 repo: the gx10 bench_longctx.py only tested ~8-11k tokens and had no
size knob.

The sweep is run against a llama-server launched with an explicit large context
and a single slot, e.g.:

    llama-server -m <model.gguf> --host 127.0.0.1 --port 8081 \
        -c 200000 -np 1 -fa on -kvu -ctk q8_0 -ctv q8_0 -ts 8,8,8,8 -ngl 999

Usage:
  python3 bench_longctx_sweep.py <PORT> <MODEL> [--out-dir DIR] [--sizes a,b,c]
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from long_transcript import build_transcript  # noqa: E402

PORT = sys.argv[1] if len(sys.argv) > 1 else "8081"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "model"
OUT_DIR = "results"
SIZES = [8192, 32768, 65536, 131072, 200000]

args = sys.argv[3:]
while args:
    a = args.pop(0)
    if a == "--out-dir" and args:
        OUT_DIR = args.pop(0)
    elif a == "--sizes" and args:
        SIZES = [int(x) for x in args.pop(0).split(",")]
    else:
        print(f"onbekend argument: {a}")

URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
os.makedirs(OUT_DIR, exist_ok=True)


def _sample_vram(stop):
    """Background sampler: record memory.used per GPU every 0.5 s."""
    samples = []
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
            if vals:
                samples.append(vals)
        except Exception:
            pass
        stop.wait(0.5)
    return samples


def _norm(s):
    import unicodedata
    # Unicode dashes (non-breaking hyphen U+2011, figure/en/em dash, minus) →
    # ASCII hyphen. Reasoning models (gpt-oss) emit U+2011 in markdown-bolded
    # codes like `PRJ‑2247`, which otherwise never matches the ASCII needle.
    for ch in "‑‒–—―‐−":
        s = s.replace(ch, "-")
    s = "".join(" " if unicodedata.category(ch) == "Zs" else ch for ch in s)
    return " ".join(s.lower().split())


def run_one(target_tokens):
    text, needles = build_transcript(target_tokens)
    questions = "\n".join(f"{i+1}. {n['vraag']}" for i, n in enumerate(needles))
    prompt = (
        "Hieronder een lang transcript van een tweedaagse strategieconferentie, "
        "inclusief bijlagen. Beantwoord daarna de genummerde vragen ELK apart en "
        "kort (1-2 zinnen per vraag), gebaseerd UITSLUITEND op het document. "
        "Antwoord in het formaat 'ANTWOORD 1: ...', 'ANTWOORD 2: ...' etc.\n\n"
        f"VRAGEN:\n{questions}\n\n"
        f"TRANSCRIPT:\n{text}"
    )
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.3,
    }
    payload = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})

    stop = threading.Event()
    samples_holder = {}

    def _sampler():
        samples_holder["samples"] = _sample_vram(stop)

    sampler = threading.Thread(target=_sampler, daemon=True)
    sampler.start()

    status = "ok"
    failure = None
    content = ""
    usage = {}
    timings = {}
    t0 = time.time()
    dt = 0.0
    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            data = json.load(resp)
        dt = time.time() - t0
        content = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        timings = data.get("timings", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        if prompt_tokens < target_tokens * 0.5:
            status = "truncated"
            failure = f"prompt_tokens={prompt_tokens} << requested ~{target_tokens}"
        elif not content.strip():
            status = "empty"
            failure = "leeg antwoord (mogelijk silent-EOS of reasoning-budget op)"
        elif usage.get("completion_tokens", 0) <= 1:
            status = "eos"
            failure = "completion_tokens<=1 (silent EOS)"
    except urllib.error.HTTPError as e:
        dt = time.time() - t0
        status = "error"
        body_txt = ""
        try:
            body_txt = e.read(400).decode("utf-8", "replace")
        except Exception:
            pass
        failure = f"HTTP {e.code}: {body_txt}"
    except urllib.error.URLError as e:
        dt = time.time() - t0
        status = "error"
        failure = f"URLError: {e.reason}"
    except Exception as e:
        dt = time.time() - t0
        status = "error"
        failure = f"{type(e).__name__}: {e}"

    stop.set()
    sampler.join(timeout=5)
    samples = samples_holder.get("samples", [])

    content_norm = _norm(content)
    score = 0
    per = []
    for i, n in enumerate(needles, start=1):
        ok = all(_norm(kw) in content_norm for kw in n["antwoord_bevat"])
        score += ok
        per.append({"q": i, "positie": n["positie"], "ok": bool(ok)})

    peak = []
    if samples:
        ngpu = len(samples[0])
        peak = [max(s[i] for s in samples) for i in range(ngpu)]

    return {
        "ctx_requested": target_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "content": content,
        "status": status,
        "failure_mode": failure,
        "score": score,
        "n_needles": len(needles),
        "needles": per,
        "wall_seconds": round(dt, 2),
        "prefill_tps": round(timings.get("prompt_per_second", 0.0), 2),
        "decode_tps": round(timings.get("predicted_per_second", 0.0), 2),
        "peak_vram_mb": peak,
    }


def main():
    print(f"=== long-context sweep — {MODEL} (poort {PORT}) ===")
    print(f"sizes: {SIZES}")
    results = []
    for size in SIZES:
        print(f"\n--- {size} tokens ---")
        r = run_one(size)
        results.append(r)
        print(f"  status={r['status']} score={r['score']}/{r['n_needles']} "
              f"prompt_tokens={r['prompt_tokens']} "
              f"prefill={r['prefill_tps']} t/s decode={r['decode_tps']} t/s "
              f"peak_vram_mb={r['peak_vram_mb']}")
        if r["failure_mode"]:
            print(f"  failure: {r['failure_mode']}")
    out = {"model": MODEL, "sizes": results}
    path = os.path.join(OUT_DIR, f"{MODEL}-longctx-sweep.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
