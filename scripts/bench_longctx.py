import json
import os
import sys
import time
import urllib.request

# Import from local scripts directory (same dir as this script)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from long_transcript import TRANSCRIPT, NEEDLES  # noqa: E402

PORT = sys.argv[1] if len(sys.argv) > 1 else "8081"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "model"
NO_THINK = False
OUT_DIR = "results"
args = sys.argv[3:]
while args:
    a = args.pop(0)
    if a == "--nothink":
        NO_THINK = True
    elif a == "--out-dir" and args:
        OUT_DIR = args.pop(0)

URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
os.makedirs(OUT_DIR, exist_ok=True)

questions = "\n".join(f"{i+1}. {n['vraag']}" for i, n in enumerate(NEEDLES))
prompt = (
    "Hieronder een lang transcript van een tweedaagse strategieconferentie, inclusief bijlagen. "
    "Beantwoord daarna de genummerde vragen ELK apart en kort (1-2 zinnen per vraag), "
    "gebaseerd UITSLUITEND op het document. Antwoord in het formaat 'ANTWOORD 1: ...', "
    "'ANTWOORD 2: ...' etc. Geef aan het eind ook een korte totaalsamenvatting (max 5 bullets) "
    "onder de kop 'TOTAALSAMENVATTING'.\n\n"
    f"VRAGEN:\n{questions}\n\n"
    f"TRANSCRIPT:\n{TRANSCRIPT}"
)

body = {
    "model": MODEL,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 2000,
    "temperature": 0.3,
}
if NO_THINK:
    body["chat_template_kwargs"] = {"enable_thinking": False}
payload = json.dumps(body).encode()
req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=1200) as resp:
    data = json.load(resp)
dt = time.time() - t0
content = data["choices"][0]["message"]["content"] or ""
usage = data.get("usage", {})
timings = data.get("timings", {})

print(f"=== long_context ({dt:.1f}s, prompt_tokens={usage.get('prompt_tokens')}, "
      f"completion_tokens={usage.get('completion_tokens')}, "
      f"prompt_t/s={timings.get('prompt_per_second', 0):.1f}, "
      f"gen_t/s={timings.get('predicted_per_second', 0):.1f}) ===")
print(content)
print()

import unicodedata
def _norm(s):
    # Unicode dashes (non-breaking hyphen U+2011, en/em dash, minus) → ASCII hyphen.
    for ch in "‑‒–—―‐−":
        s = s.replace(ch, "-")
    s = "".join(" " if unicodedata.category(ch) == "Zs" else ch for ch in s)
    return " ".join(s.lower().split())

score = 0
print("--- scoring (needle retrieval) ---")
content_norm = _norm(content)
for i, n in enumerate(NEEDLES, start=1):
    ok = all(_norm(kw) in content_norm for kw in n["antwoord_bevat"])
    score += ok
    print(f"  [{'OK' if ok else 'MISS'}] vraag {i} ({n['positie']}): verwacht {n['antwoord_bevat']}")
print(f"SCORE: {score}/{len(NEEDLES)}")

with open(os.path.join(OUT_DIR, f"{MODEL}-longctx.json"), "w") as f:
    json.dump({"content": content, "usage": usage, "timings": timings,
                "wall_seconds": round(dt, 2), "score": score,
                "n_needles": len(NEEDLES)}, f, ensure_ascii=False, indent=2)
print(f"Saved to {OUT_DIR}/{MODEL}-longctx.json  |  SCORE: {score}/{len(NEEDLES)}")
