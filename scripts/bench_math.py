import json
import os
import re
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8081"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "model"
OUT_DIR = "results"
args = sys.argv[3:]
while args:
    a = args.pop(0)
    if a == "--out-dir" and args:
        OUT_DIR = args.pop(0)

URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
os.makedirs(OUT_DIR, exist_ok=True)

# Ground truth answers voor auto-scoring.
# Format: (expected_answer_pattern, human_label)
# De auto-scorer zoekt naar "EINDANTWOORD:" in de output en vergelijkt.
GROUND_TRUTH = {
    "stap_korting":     ("165", "165 euro"),
    "logica_ridders":   ("a=ridder", "A=ridder, B=schurk"),
    "kans_dobbelsteen": ("1/6", "1/6"),
    "combinatoriek":    ("150", "150"),
    "algebra":          ("30", "30 en 17"),
    "getaltheorie":     ("15", "15"),
    "meetkunde":        ("30", "30 cm2"),
    "rente":            ("18", "18 jaar"),
    "valstrik_schapen": ("1", "1 schaap"),
    "valstrik_caesar":  ("55", "55 jaar"),
}

QUESTIONS = [
    ("stap_korting", "Een winkel past na elkaar drie kortingen toe op een product van 250 euro: "
                      "eerst 20% korting, dan nog eens 10% korting op het resterende bedrag, en tot slot "
                      "een vaste korting van 15 euro. Wat is de einprijs? Toon je berekening en geef het "
                      "eindantwoord op de laatste regel als 'EINDANTWOORD: <bedrag>'."),
    ("logica_ridders", "Op een eiland zijn er Ridders (spreken altijd de waarheid) en Schurken (liegen altijd). "
                        "Je ontmoet A en B. A zegt: 'B is een schurk.' B zegt: 'A en ik zijn beiden ridders.' "
                        "Wat zijn A en B? Redeneer stap voor stap en geef het eindantwoord op de laatste regel "
                        "als 'EINDANTWOORD: A=..., B=...'."),
    ("kans_dobbelsteen", "Je gooit twee eerlijke dobbelstenen. Wat is de kans dat de som van de ogen precies 7 "
                          "is? Geef het antwoord als breuk EN als percentage, op de laatste regel als "
                          "'EINDANTWOORD: <breuk> = <percentage>%'."),
    ("combinatoriek", "Op hoeveel manieren kunnen 5 verschillende boeken worden verdeeld over 3 verschillende "
                       "personen, als elke persoon minstens 1 boek moet krijgen? Toon je berekening en geef "
                       "het eindantwoord op de laatste regel als 'EINDANTWOORD: <getal>'."),
    ("algebra", "Twee getallen hebben een som van 47 en een verschil van 13. Wat zijn de twee getallen? "
                "Geef het eindantwoord op de laatste regel als 'EINDANTWOORD: <groot getal> en <klein getal>'."),
    ("getaltheorie", "Wat is de kleinste positieve gehele n waarvoor n! (n faculteit) deelbaar is door 1000? "
                      "Toon je redenering en geef het eindantwoord op de laatste regel als "
                      "'EINDANTWOORD: <getal>'."),
    ("meetkunde", "Een rechthoekige driehoek heeft een hypotenusa van 13 cm en een rechthoekszijde van 5 cm. "
                   "Wat is de oppervlakte van de driehoek? Geef het eindantwoord op de laatste regel als "
                   "'EINDANTWOORD: <getal> cm2'."),
    ("rente", "Je zet 1000 euro op een spaarrekening met 4% samengestelde rente per jaar. Na hoeveel volle "
               "jaren is het bedrag meer dan verdubbeld? Toon je berekening en geef het eindantwoord op de "
               "laatste regel als 'EINDANTWOORD: <getal> jaar'."),
    ("valstrik_schapen", "Een boer heeft 17 schapen. Op 1 na sterven ze allemaal. Hoeveel schapen heeft hij "
                          "nog? Geef het eindantwoord op de laatste regel als 'EINDANTWOORD: <getal>'."),
    ("valstrik_caesar", "Julius Caesar werd geboren in juli 100 v.Chr. en stierf op 15 maart 44 v.Chr. "
                         "Hoe oud was hij toen hij stierf? Let goed op of zijn verjaardag dat jaar al was "
                         "geweest. Toon je redenering en geef het eindantwoord op de laatste regel als "
                         "'EINDANTWOORD: <getal> jaar'."),
]

results = {}
n_correct = 0
for key, q in QUESTIONS:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": q}],
        "max_tokens": 2000,
        "temperature": 0.2,
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    dt = time.time() - t0
    content = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage", {})

    # Auto-scoring: zoek EINDANTWOORD en vergelijk met ground truth
    m = re.search(r'EINDANTWOORD:\s*(.+?)(?:\n|$)', (content or ""), re.IGNORECASE)
    predicted = m.group(1).strip().lower().rstrip('.') if m else ""
    expected = GROUND_TRUTH.get(key, ("", ""))[0].lower()
    # Remove common formatting from predicted
    predicted_clean = predicted.replace("euro", "").replace("€", "").replace("%", "").replace("jaar", "").replace("cm2", "").strip()
    # Check if expected answer is contained in the predicted answer
    correct = expected in predicted_clean or expected in predicted

    if correct: n_correct += 1
    results[key] = {
        "question": q,
        "answer": content.strip(),
        "predicted_answer": predicted,
        "expected_answer": GROUND_TRUTH.get(key, ("", ""))[1],
        "correct": correct,
        "wall_seconds": round(dt, 2),
        "completion_tokens": usage.get("completion_tokens"),
    }
    print(f"=== {key} ({dt:.1f}s, {usage.get('completion_tokens')} tok) [{'OK' if correct else 'FOUT'}] ===")
    print(content.strip())
    if not correct:
        print(f"  expected '{GROUND_TRUTH.get(key, ('',''))[0]}' in answer, got: '{predicted}'")
    print()

results["_summary"] = {"n_correct": n_correct, "n_total": len(QUESTIONS)}
out_path = os.path.join(OUT_DIR, f"{MODEL}-math.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Saved to {out_path}  |  SCORE: {n_correct}/{len(QUESTIONS)}")
