import json
import os
import sys
import time
import urllib.request
import urllib.error

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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Geeft de actuele temperatuur in graden Celsius voor een stad.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "Naam van de stad"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evalueert een rekenkundige expressie en geeft het numerieke resultaat.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "bv. '(12 + 3) * 2'"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_klantendatabase",
            "description": "Zoekt een klant op in de klantendatabase op naam en geeft klant-ID en status terug.",
            "parameters": {
                "type": "object",
                "properties": {"naam": {"type": "string"}},
                "required": ["naam"],
            },
        },
    },
]

# gemockte tool-implementaties, met bewust een niet-triviale waarde
def _get_weather(args):
    city = args.get("city", "")
    fake_temps = {"utrecht": 9, "amsterdam": 11, "rotterdam": 8, "groningen": 6}
    return {"city": city, "temperature_c": fake_temps.get(city.lower(), 10)}

def _calculate(args):
    expr = args.get("expression", "")
    try:
        # veilige beperkte eval voor arithmetic-only expressies
        allowed = set("0123456789+-*/(). ")
        if not set(expr) <= allowed:
            return {"error": "ongeldige expressie"}
        return {"result": eval(expr)}
    except Exception as e:
        return {"error": str(e)}

def _search_klant(args):
    naam = args.get("naam", "")
    return {"naam": naam, "klant_id": "KL-48213", "status": "actief", "openstaand_bedrag": 0}

TOOL_IMPL = {"get_weather": _get_weather, "calculate": _calculate, "search_klantendatabase": _search_klant}

CASES = {
    "single_tool_call": (
        "Wat is de huidige temperatuur in Utrecht? Gebruik de beschikbare tool om dit op te zoeken, "
        "geef daarna het antwoord in 1 zin."
    ),
    "chained_tool_calls": (
        "Zoek eerst de temperatuur in Utrecht op. Bereken vervolgens die temperatuur plus 15, "
        "vermenigvuldigd met 2. Gebruik de tools voor beide stappen, geef daarna alleen het eindgetal."
    ),
    "irrelevant_tool_avoidance": (
        "Wat is de hoofdstad van Frankrijk? Gebruik GEEN tools voor deze vraag, dit is algemene kennis."
    ),
    "multi_entity_tool_use": (
        "Zoek klant 'Jansen BV' op in de klantendatabase. Als het openstaand bedrag 0 is, zoek dan ook "
        "de temperatuur in Rotterdam op en rapporteer beide resultaten in 2 zinnen."
    ),
}


def call(messages, extra_body=None):
    body = {"model": MODEL, "messages": messages, "tools": TOOLS, "max_tokens": 1500, "temperature": 0.2}
    if NO_THINK:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if extra_body:
        body.update(extra_body)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def _score_case(key, trace, final_content):
    """Auto-score een tool-case: welke tools zijn (niet) aangeroepen + kernfeit in antwoord."""
    fnames = [t["tool"] for t in trace]
    fc = (final_content or "").lower()
    for ch in "‑‒–—―‐−":  # non-breaking hyphen e.a. → ASCII hyphen
        fc = fc.replace(ch, "-")
    if key == "single_tool_call":
        return "get_weather" in fnames and "9" in fc            # Utrecht = 9°C
    if key == "chained_tool_calls":
        return "get_weather" in fnames and "calculate" in fnames and "48" in fc  # (9+15)*2
    if key == "irrelevant_tool_avoidance":
        return len(fnames) == 0 and ("parijs" in fc or "paris" in fc)
    if key == "multi_entity_tool_use":
        return ("search_klantendatabase" in fnames and "get_weather" in fnames
                and "kl-48213" in fc)
    return False


results = {}
for key, prompt in CASES.items():
    print(f"=== {key} ===")
    messages = [{"role": "user", "content": prompt}]
    trace = []
    t0 = time.time()
    ok = True
    final_content = ""
    for step in range(4):  # max 4 rondes tool-calling
        try:
            data = call(messages)
        except urllib.error.HTTPError as e:
            # Sommige modellen (bv. Llama-3.3-70B) laten llama.cpp een HTTP 500 geven
            # ("does not match the expected peg-native format") op tool-calls. Dat is een
            # meetresultaat (model kan tool-calling niet op deze build), geen reden om te crashen.
            print(f"  !! HTTP {e.code} op tool-call — case afgebroken ({e})")
            trace.append({"tool": None, "args": {}, "error": f"http_{e.code}"})
            break
        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
            for tc in tool_calls:
                fname = tc["function"]["name"]
                try:
                    fargs = json.loads(tc["function"]["arguments"])
                except Exception:
                    fargs = {}
                impl = TOOL_IMPL.get(fname)
                result = impl(fargs) if impl else {"error": f"onbekende tool {fname}"}
                trace.append({"tool": fname, "args": fargs, "result": result})
                print(f"  -> tool_call: {fname}({fargs}) = {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{step}"),
                    "content": json.dumps(result, ensure_ascii=False),
                })
            continue
        else:
            final_content = msg.get("content") or ""
            break
    dt = time.time() - t0
    ok = _score_case(key, trace, final_content)
    print(f"  eindantwoord ({dt:.1f}s, {len(trace)} tool-calls) [{'OK' if ok else 'MISS'}]: {final_content}")
    print()
    results[key] = {"trace": trace, "final_content": final_content,
                    "wall_seconds": round(dt, 2), "correct": ok}

results["_summary"] = {"n_correct": sum(1 for k in CASES if results[k]["correct"]),
                       "n_total": len(CASES)}
with open(os.path.join(OUT_DIR, f"{MODEL}-tools.json"), "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Saved to {OUT_DIR}/{MODEL}-tools.json  |  SCORE: {results['_summary']['n_correct']}/{results['_summary']['n_total']}")
