import json
import os
import re
import sys
import time
import unicodedata
import urllib.request

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

# ── Antwoord-normalisatie voor robuuste keyword-matching ──
# Nederlandse model-antwoorden bevatten Unicode-subscripts (C₈H₁₀N₄O₂),
# niet-afbrekende spaties (U+00A0) en decimale komma's (0,5772). Normaliseer
# deze naar ASCII zodat de keywords correct matchen.
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def _normalize(s):
    s = s.lower()
    # Unicode spatie-separators (NBSP e.a.) → gewone spatie
    s = "".join(" " if unicodedata.category(c) == "Zs" else c for c in s)
    # Sub/superscript-cijfers → ASCII-cijfers
    s = s.translate(_SUB).translate(_SUP)
    # Decimale komma tussen cijfers → punt (0,5772 → 0.5772)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    return " ".join(s.split())

# Brede kennisvragen over diverse domeinen, met keyword-matching voor auto-scoring.
# antwoord_bevat: lijst van keywords die ALLEMAAL in het antwoord moeten voorkomen.
QUESTIONS = [
    ("scheikunde_1", "Wat is de scheikundige brutoformule van cafeïne?",
     ["c8h10n4o2"]),
    ("geschiedenis_1", "In welk jaar werd het Verdrag van Utrecht getekend dat de Spaanse Successieoorlog beëindigde?",
     ["1713"]),
    ("geografie_1", "Wat is de diepste plek in de Middellandse Zee en hoe diep is die ongeveer?",
     ["calypso", "5267"]),  # Calypsotrog, ~5267m
    ("biologie_1", "Welk enzym zet zetmeel om in maltose, al in de mond via speeksel?",
     ["amylase"]),
    ("wiskunde_1", "Wat is de Euler-Mascheroniconstante, tot 4 decimalen?",
     ["0.5772"]),
    ("technologie_1", "Op welke datum werd TCP/IP officieel de standaard voor ARPANET?",
     ["1983", "1 januari"]),  # 1 jan 1983
    ("literatuur_1", "Wie schreef 'De ontdekking van de hemel' en in welk jaar verscheen het?",
     ["mulisch", "1992"]),
    ("astronomie_1", "Wat is de omlooptijd van Pluto rond de zon, in aardse jaren?",
     ["248"]),
    ("economie_1", "In welk jaar werd de eerste effectenbeurs ter wereld opgericht in Amsterdam, en welk bedrijf was de directe aanleiding?",
     ["1602", "voc"]),
    ("scheikunde_2", "Wat is het smeltpunt van wolfraam in graden Celsius (het metaal met het hoogste smeltpunt)?",
     ["3422"]),
    ("muziek_1", "Uit hoeveel delen (bewegingen) bestaat Beethovens 9e symfonie?",
     ["4", "vier"]),
    ("natuurkunde_1", "Wat is de dichtheid van kwik bij kamertemperatuur, in g/cm3?",
     ["13.5"]),  # 13.534 g/cm3, accepteer 13.5-13.6
    ("combo_1", "Wie was de Franse koning tijdens de bestorming van de Bastille in 1789, en hoeveel jaar had hij op dat moment geregeerd sinds zijn troonsbestijging?",
     ["lodewijk", "15"]),  # Lodewijk XVI, 15 jaar (1774-1789)
    ("combo_2", "De Slag bij Waterloo vond plaats in 1815. Napoleon werd geboren in 1769. Hoe oud was hij tijdens die slag (juni 1815)?",
     ["45", "46"]),  # 45 of 46 afhankelijk van afronding (geboren augustus)
]

results = {}
n_correct = 0
for key, q, keywords in QUESTIONS:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": q + " Antwoord kort en direct."}],
        "max_tokens": 800,
        "temperature": 0.2,
    }
    if NO_THINK:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    dt = time.time() - t0
    content = data["choices"][0]["message"]["content"] or ""

    # Auto-scoring via keyword matching
    answer_lower = _normalize(content)
    matched = [kw for kw in keywords if kw in answer_lower]
    # Question with multiple expected answers (combo_2: 45 or 46, muziek: "4" or "vier")
    if key == "combo_2":
        correct = any(kw in answer_lower for kw in keywords)
    elif key == "muziek_1":
        correct = any(kw in answer_lower for kw in keywords) and ("deel" in answer_lower or "beweging" in answer_lower or "movement" in answer_lower)
    elif key == "geografie_1":
        # "ongeveer" in de vraag → een bereik is een correct antwoord
        depths = ["5267", "5 267", "5200", "5 200", "5300", "5 300", "5400", "5 400", "5000", "5 000"]
        correct = "calypso" in answer_lower and any(d in answer_lower for d in depths)
        matched = ["calypso"] + [d for d in depths if d in answer_lower]
    elif key == "natuurkunde_1":
        # Accept 13.5 or 13.6
        correct = any(kw in answer_lower for kw in ["13.5", "13.6", "13,5", "13,6"])
        matched = ["13.5-13.6 range"] if correct else []
    else:
        correct = len(matched) == len(keywords)

    if correct: n_correct += 1
    results[key] = {
        "question": q,
        "answer": content.strip(),
        "expected_keywords": keywords,
        "matched_keywords": matched,
        "correct": correct,
        "wall_seconds": round(dt, 2),
    }
    print(f"=== {key} ({dt:.1f}s) [{'OK' if correct else 'FOUT'}] ===")
    print(f"Q: {q}")
    print(f"A: {content.strip()}")
    if not correct:
        missing = [kw for kw in keywords if kw not in matched]
        print(f"  missing keywords: {missing}")
    print()

results["_summary"] = {"n_correct": n_correct, "n_total": len(QUESTIONS)}
out_path = os.path.join(OUT_DIR, f"{MODEL}-knowledge.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Saved to {out_path}  |  SCORE: {n_correct}/{len(QUESTIONS)}")
