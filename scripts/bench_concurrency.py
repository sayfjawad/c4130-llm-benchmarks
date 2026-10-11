#!/usr/bin/env python3
"""bench_concurrency.py — how does a vLLM server behave with N simultaneous users?

All N requests are fired at the same moment (worst case: everyone clicks at once). Each user gets a
unique prompt (salted) so the prefix cache cannot make it look better than reality. Tokens are counted
from usage.completion_tokens. Per level we report:
  - TTFT (time to first token): p50 / p95 / max  -> "how long do I stare at a blank screen"
  - per-user decode t/s (p50 / min)                -> "how fast does the text stream once it starts"
  - end-to-end latency p50 / max                   -> "how long until my answer is complete"
  - aggregate output tok/s over the wall time      -> server throughput
  - errors

Usage: BENCH_URL=http://127.0.0.1:8001 python3 bench_concurrency.py MODEL OUT.json [levels] [scenarios]
"""
import json, os, statistics, sys, threading, time, urllib.request

BASE = os.environ.get("BENCH_URL", "http://127.0.0.1:8001").rstrip("/")
MODEL = sys.argv[1]
OUT = sys.argv[2]
LEVELS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,5,10,20").split(",")]
SCEN = (sys.argv[4] if len(sys.argv) > 4 else "chat,rag").split(",")

QUESTIONS = [
    "Leg uit hoe een warmtepomp werkt voor een huiseigenaar zonder technische kennis.",
    "Wat zijn de voor- en nadelen van thuiswerken voor een gemeentelijke organisatie?",
    "Schrijf een korte, vriendelijke e-mail aan een bewoner over een vertraagde vergunningsaanvraag.",
    "Vat de belangrijkste regels samen voor het parkeren van een camper in een woonwijk.",
    "Hoe bereid je je voor op een sollicitatiegesprek bij een overheidsinstelling?",
]
DOC_PARA = ("Tijdens de raadsvergadering werd uitgebreid gesproken over de begroting, de renovatie van het "
            "zwembad, het bestemmingsplan Noorderpark, de fietsbrug bij het station en de verkeersveiligheid "
            "rond de basisscholen. Verschillende fracties dienden moties en amendementen in. ")


def prompt(scenario, user, level):
    salt = f"[sessie {level}-{user}-{time.time_ns()}]\n"
    q = QUESTIONS[user % len(QUESTIONS)]
    if scenario == "chat":
        return salt + q + " Antwoord in ongeveer 250 woorden."
    # rag: ~6k-token document + question
    return (salt + "Gebruik uitsluitend het onderstaande verslag.\n\n" + DOC_PARA * 140 +
            "\n\nVraag: welke onderwerpen kwamen aan bod en wat werd er besloten? Antwoord in ongeveer 250 woorden.")


def one(scenario, user, level, res):
    body = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": prompt(scenario, user, level)}],
        "max_tokens": 400, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; usage = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                ch = json.loads(line[5:])
                if ch.get("usage"):
                    usage = ch["usage"]
                if ttft is None and any((c.get("delta") or {}).get("content") for c in ch.get("choices") or []):
                    ttft = time.perf_counter() - t0
        total = time.perf_counter() - t0
        n = usage["completion_tokens"]
        res[user] = {"ttft": ttft, "total": total, "out": n, "prompt": usage["prompt_tokens"],
                     "decode_tps": (n - 1) / (total - ttft) if n > 1 and total > ttft else 0.0}
    except Exception as e:
        res[user] = {"error": str(e)[:200]}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


report = {"model": MODEL, "url": BASE, "results": []}
for scenario in SCEN:
    print(f"\n=== scenario: {scenario} ===", flush=True)
    print(f"{'users':>5} | {'TTFT p50':>8} {'p95':>6} {'max':>6} | {'t/s/user p50':>12} {'min':>5} | "
          f"{'klaar p50':>9} {'max':>6} | {'totaal t/s':>10} | fout", flush=True)
    # warmup (JIT / first-shape compiles must not count)
    w = {}; one(scenario, 0, 0, w)
    for lvl in LEVELS:
        res = [None] * lvl
        th = [threading.Thread(target=one, args=(scenario, i, lvl, res)) for i in range(lvl)]
        w0 = time.perf_counter()
        for t in th: t.start()
        for t in th: t.join()
        wall = time.perf_counter() - w0
        ok = [r for r in res if r and "error" not in r]
        err = [r for r in res if not r or "error" in r]
        row = {"scenario": scenario, "users": lvl, "errors": len(err), "wall_s": round(wall, 2),
               "prompt_tokens": ok[0]["prompt"] if ok else None, "runs": res}
        if ok:
            tt = [r["ttft"] for r in ok]; ds = [r["decode_tps"] for r in ok]; tot = [r["total"] for r in ok]
            row.update({"ttft_p50": round(pct(tt, 50), 2), "ttft_p95": round(pct(tt, 95), 2), "ttft_max": round(max(tt), 2),
                        "decode_p50": round(pct(ds, 50), 1), "decode_min": round(min(ds), 1),
                        "e2e_p50": round(pct(tot, 50), 1), "e2e_max": round(max(tot), 1),
                        "aggregate_tps": round(sum(r["out"] for r in ok) / wall, 1)})
            print(f"{lvl:>5} | {row['ttft_p50']:>7}s {row['ttft_p95']:>5}s {row['ttft_max']:>5}s | "
                  f"{row['decode_p50']:>12} {row['decode_min']:>5} | {row['e2e_p50']:>8}s {row['e2e_max']:>5}s | "
                  f"{row['aggregate_tps']:>10} | {len(err)}", flush=True)
        else:
            print(f"{lvl:>5} | ALL FAILED: {err[0]}", flush=True)
        report["results"].append(row)
        time.sleep(3)

json.dump(report, open(OUT, "w"), indent=2)
print("\nsaved", OUT)
