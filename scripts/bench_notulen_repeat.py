import json
import os
import sys
import time
import urllib.request

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from bench_hard import PROMPTS  # noqa: E402

PORT = sys.argv[1] if len(sys.argv) > 1 else "8081"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "model"
REPEATS = 5
NO_THINK = False
OUT_DIR = "results"
args = sys.argv[3:]
while args:
    a = args.pop(0)
    if a == "--nothink":
        NO_THINK = True
    elif a == "--out-dir" and args:
        OUT_DIR = args.pop(0)
    elif a.isdigit():
        REPEATS = int(a)

URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
os.makedirs(OUT_DIR, exist_ok=True)

prompt = PROMPTS["scribr_notulen"]
required_headers = ["## Besluiten", "## Actiepunten", "## Openstaande vragen"]

results = []
for run in range(1, REPEATS + 1):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 3000,
        "temperature": 0.7,  # iets hoger dan normaal om variatie/instabiliteit juist te provoceren
    }
    if NO_THINK:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    payload = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            data = json.load(resp)
        dt = time.time() - t0
        content = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        ok = bool(content.strip()) and all(h.lower().replace("## ", "") in content.lower() for h in required_headers)
        results.append({"run": run, "ok": ok, "wall_seconds": round(dt, 2),
                         "completion_tokens": usage.get("completion_tokens"),
                         "content_len": len(content)})
        print(f"run {run}: {'OK' if ok else 'FAIL'} ({dt:.1f}s, {usage.get('completion_tokens')} tok, content_len={len(content)})")
    except Exception as e:
        dt = time.time() - t0
        results.append({"run": run, "ok": False, "wall_seconds": round(dt, 2), "error": str(e)})
        print(f"run {run}: FAIL (exception after {dt:.1f}s: {e})")

n_ok = sum(r["ok"] for r in results)
print(f"\nBETROUWBAARHEID: {n_ok}/{REPEATS} geslaagd")

with open(os.path.join(OUT_DIR, f"{MODEL}-notulen-repeat.json"), "w") as f:
    json.dump({"_summary": {"n_ok": n_ok, "n_total": REPEATS}, "runs": results}, f, ensure_ascii=False, indent=2)
print(f"Saved to {OUT_DIR}/{MODEL}-notulen-repeat.json  |  BETROUWBAARHEID: {n_ok}/{REPEATS}")
