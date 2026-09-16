#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""debug_sweep_one.py — reproduce ONE sweep size and dump the raw response."""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from long_transcript import build_transcript  # noqa: E402

PORT = sys.argv[1] if len(sys.argv) > 1 else "8081"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "model"
SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 8192

text, needles = build_transcript(SIZE)
questions = "\n".join(f"{i+1}. {n['vraag']}" for i, n in enumerate(needles))
prompt = (
    "Hieronder een lang transcript van een tweedaagse strategieconferentie, "
    "inclusief bijlagen. Beantwoord daarna de genummerde vragen ELK apart en "
    "kort (1-2 zinnen per vraag), gebaseerd UITSLUITEND op het document. "
    "Antwoord in het formaat 'ANTWOORD 1: ...', 'ANTWOORD 2: ...' etc.\n\n"
    f"VRAGEN:\n{questions}\n\n"
    f"TRANSCRIPT:\n{text}"
)
body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000, "temperature": 0.3}
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=1200) as resp:
    data = json.load(resp)
dt = time.time() - t0

msg = data["choices"][0]["message"]
content = msg.get("content") or ""
reasoning = msg.get("reasoning_content") or ""
usage = data.get("usage", {})

print("=== message keys:", list(msg.keys()))
print("prompt_tokens:", usage.get("prompt_tokens"),
      "completion_tokens:", usage.get("completion_tokens"))
print("--- reasoning_content (len=%d, first 400) ---" % len(reasoning))
print(reasoning[:400])
print("--- content (len=%d) ---" % len(content))
print(content)
print("--- needles ---")
for n in needles:
    print(f"  {n['positie']:>6}  expect={n['antwoord_bevat']}  Q={n['vraag']}")
print("elapsed %.1fs" % dt)
