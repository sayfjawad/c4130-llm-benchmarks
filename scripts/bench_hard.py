import json
import os
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8081"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "model"
NO_THINK = False
OUT_DIR = "results"
SYSTEM_PROMPT = None
# Parse optional flags
args = sys.argv[3:]
while args:
    a = args.pop(0)
    if a == "--nothink":
        NO_THINK = True
    elif a == "--out-dir" and args:
        OUT_DIR = args.pop(0)
    elif a.startswith("--system-prompt="):
        SYSTEM_PROMPT = a.split("=", 1)[1]

URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

# --- ground truth for auto-scoring ---
GROUND_TRUTH = {
    "hard_reasoning": 2195,  # EINDANTWOORD moet 2195 zijn
}
NOTULEN_HEADERS = ["## Besluiten", "## Actiepunten", "## Openstaande vragen"]

TRANSCRIPT = """Notulen gemeenteraadsvergadering - fictief, 14 maart

Aanwezig: Burgemeester De Groot (voorzitter), wethouder Bakker (Financien), wethouder Smit (Ruimtelijke Ordening), raadsleden Jansen, Verhoeven, El Amrani, Dijkstra, Pietersen.

Voorzitter opent de vergadering om 19:30 en stelt de agenda vast. Drie punten: 1) begrotingswijziging zwembad, 2) bestemmingsplan Noorderpark, 3) rondvraag.

Punt 1 - Begrotingswijziging zwembad:
Wethouder Bakker licht toe dat de renovatie van het zwembad 340.000 euro duurder uitvalt dan begroot, door gestegen materiaalkosten en een noodzakelijke asbestsanering die niet was voorzien. Raadslid Jansen vraagt of er gekeken is naar alternatieve financiering, bijvoorbeeld een provinciale subsidie. Wethouder Bakker antwoordt dat de subsidieaanvraag loopt maar pas in juni een uitslag geeft, en dat de raad nu al een besluit moet nemen omdat de aannemer per 1 april door wil. Raadslid Verhoeven stelt voor om 200.000 euro nu vrij te maken uit de algemene reserve en de resterende 140.000 euro pas te besluiten na de subsidie-uitslag in juni. Raadslid El Amrani ondersteunt dit voorstel maar wil een harde voorwaarde: als de subsidie wordt afgewezen, moet er een nieuw raadsvoorstel komen in plaats van automatisch bijstorten uit de reserve. Wethouder Bakker gaat hiermee akkoord. De voorzitter stelt voor dit voorstel van Verhoeven, met de voorwaarde van El Amrani, in stemming te brengen. Het voorstel wordt aangenomen met 6 stemmen voor en 1 tegen (Dijkstra, die vindt dat het hele project heroverwogen moet worden).

Actie: wethouder Bakker zorgt dat het raadsvoorstel voor de resterende 140.000 euro uiterlijk in de raadsvergadering van juli wordt geagendeerd, ongeacht de uitkomst van de subsidie.

Punt 2 - Bestemmingsplan Noorderpark:
Wethouder Smit presenteert het herziene bestemmingsplan voor Noorderpark, waarin 120 woningen zijn opgenomen, waarvan 40 procent sociale huur. Raadslid Pietersen merkt op dat omwonenden hebben geklaagd over onvoldoende parkeerplaatsen in het plan - er is 1 parkeerplaats per woning voorzien, terwijl de gemeentelijke norm 1,3 is. Wethouder Smit legt uit dat dit bewust is gedaan om de bouw van meer sociale huurwoningen mogelijk te maken binnen het beschikbare budget, en dat er een deelauto-concept wordt toegevoegd om de parkeerdruk te compenseren. Raadslid Jansen vraagt om een onafhankelijke verkeerstoets voordat de raad hierover besluit. Dit voorstel wordt gesteund door een meerderheid. De voorzitter concludeert dat er nog geen besluit wordt genomen over het bestemmingsplan zelf, maar dat wethouder Smit een verkeerstoets laat uitvoeren en de resultaten terugkoppelt in de vergadering van mei.

Actie: wethouder Smit laat een onafhankelijke verkeerstoets uitvoeren voor het bestemmingsplan Noorderpark, resultaten terug in mei.
Openstaande vraag: hoe wordt de parkeerdruk opgevangen als het deelauto-concept niet voldoende blijkt.

Punt 3 - Rondvraag:
Raadslid Dijkstra vraagt naar de status van de fietsbrug bij het station, die volgens de laatste update in september klaar zou zijn. Burgemeester De Groot zegt dit na te vragen bij de projectleider en volgende vergadering terug te koppelen. Raadslid El Amrani meldt geluidsoverlast van een lokaal bedrijf en vraagt om handhaving. De voorzitter verwijst dit door naar de wethouder Handhaving, die niet aanwezig is vanavond.

Actie: burgemeester De Groot vraagt status fietsbrug na bij projectleider, terugkoppeling volgende vergadering.
Actie: klacht geluidsoverlast doorgezet naar wethouder Handhaving (niet aanwezig).

De voorzitter sluit de vergadering om 21:05."""

PROMPTS = {
    "hard_reasoning": (
        "Los dit stap voor stap op en geef aan het einde ALLEEN het eindantwoord als een geheel getal, "
        "voorafgegaan door 'EINDANTWOORD:'.\n\n"
        "Drie ploegen werken aan een sorteercentrum, allemaal van 08:00 tot 16:00 (behalve waar anders vermeld).\n"
        "- Ploeg 1 verwerkt 120 pakketten per uur gedurende de eerste 3 uur, daarna neemt de snelheid af naar "
        "90 pakketten per uur voor de resterende tijd.\n"
        "- Ploeg 2 verwerkt constant 100 pakketten per uur, maar heeft na 5 uur werken een pauze van 45 minuten "
        "waarin niet wordt verwerkt.\n"
        "- Ploeg 3 begint pas om 10:00 (2 uur later dan de andere ploegen) en verwerkt vanaf dat moment "
        "110 pakketten per uur, tot 16:00.\n\n"
        "Hoeveel pakketten zijn er in totaal door de drie ploegen samen verwerkt om 16:00?"
    ),
    "hard_coding": (
        "Implementeer in Python een klasse `LRUCache` (Least Recently Used cache) met:\n"
        "- `__init__(self, capacity: int)`\n"
        "- `get(self, key: int) -> int`: geeft de waarde terug, of -1 als de key niet bestaat. "
        "Het opvragen van een key telt als recent gebruikt.\n"
        "- `put(self, key: int, value: int) -> None`: voegt toe of update. Als de capaciteit wordt overschreden, "
        "verwijder dan de minst recent gebruikte entry. Het toevoegen/updaten telt ook als recent gebruikt.\n\n"
        "Beide operaties moeten O(1) zijn. Geef ALLEEN de python code terug in een ```python code block, "
        "zonder uitleg, zonder voorbeeldgebruik."
    ),
    "scribr_notulen": (
        "Hieronder de notulen van een gemeenteraadsvergadering. Maak een gestructureerde samenvatting met "
        "PRECIES deze drie secties, elk als kopje gevolgd door bullet points:\n"
        "## Besluiten\n"
        "## Actiepunten (met wie verantwoordelijk is)\n"
        "## Openstaande vragen\n\n"
        "Wees volledig maar bondig - mis geen enkel besluit of actiepunt uit de tekst.\n\n"
        f"TRANSCRIPT:\n{TRANSCRIPT}"
    ),
    "strict_instructions": (
        "Beoordeel dit fictieve productiedeviatie-rapport en geef je antwoord in STRIKT geldige JSON "
        "(niets anders, geen markdown code fences, geen uitleg voor of na de JSON), met EXACT deze structuur "
        "en regels:\n"
        '- "samenvatting": string van EXACT maximaal 30 woorden (tel je woorden)\n'
        '- "kernwoorden": array van EXACT 5 strings, allemaal in kleine letters, geen duplicaten\n'
        '- "risico_score": float tussen 0.0 en 1.0 met EXACT 2 decimalen\n'
        '- "vereist_escalatie": boolean\n\n'
        "RAPPORT: Op lijn 3 is een temperatuurafwijking van 8 graden gemeten gedurende 45 minuten. "
        "De kwaliteitscontrole heeft 12 van de 400 geproduceerde eenheden uit die periode afgekeurd. "
        "De oorzaak is een defecte sensor die inmiddels vervangen is. Er is geen product naar klanten verzonden "
        "uit de betreffende batch."
    ),
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    n_correct = 0
    n_total = 0
    for key, prompt in PROMPTS.items():
        messages = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 3000,
            "temperature": 0.3,
        }
        if NO_THINK:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        payload = json.dumps(body).encode()
        req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as resp:
            data = json.load(resp)
        dt = time.time() - t0
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        timings = data.get("timings", {})

        # --- auto-scoring ---
        correct = None
        score_note = ""
        if key == "hard_reasoning":
            import re
            m = re.search(r'EINDANTWOORD:\s*(\d+)', (content or ""))
            correct = m and int(m.group(1)) == GROUND_TRUTH["hard_reasoning"]
            score_note = f"expected {GROUND_TRUTH['hard_reasoning']}, got {m.group(1) if m else 'no match'}"
            n_total += 1
            if correct: n_correct += 1
        elif key == "hard_coding":
            # Coding: accepteer zowel OrderedDict als de canonieke hashmap +
            # dubbel-gelinkte-lijst implementatie (beide O(1)). (manual review still recommended)
            has_class = "class LRUCache" in (content or "")
            has_ordered = "OrderedDict" in (content or "")
            has_get = "def get" in (content or "")
            has_put = "def put" in (content or "")
            correct = has_class and (has_ordered or (has_get and has_put))
            score_note = f"has LRUCache={has_class}, OrderedDict={has_ordered}, get/put={has_get}/{has_put}"
            n_total += 1
            if correct: n_correct += 1
        elif key == "scribr_notulen":
            ok = bool(content) and all(h.lower().replace("## ", "").replace("**", "") in content.lower() for h in NOTULEN_HEADERS)
            correct = ok
            score_note = f"headers present: {', '.join(h for h in NOTULEN_HEADERS if h.lower().replace('## ', '').replace('**', '') in (content or '').lower())}"
            n_total += 1
            if correct: n_correct += 1
        elif key == "strict_instructions":
            # Try to parse as JSON, check required fields
            try:
                # Strip markdown fences if present
                clean = (content or "").strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                    if clean.endswith("```"):
                        clean = clean[:-3]
                obj = json.loads(clean.strip())
                has_keys = all(k in obj for k in ["samenvatting", "kernwoorden", "risico_score", "vereist_escalatie"])
                words_ok = len(obj.get("samenvatting", "").split()) <= 30 and len(obj.get("kernwoorden", [])) == 5
                correct = has_keys and words_ok and isinstance(obj.get("vereist_escalatie"), bool)
                score_note = f"valid JSON, keys={has_keys}, word_limit={words_ok}"
            except Exception as e:
                correct = False
                score_note = f"JSON parse failed: {e}"
            n_total += 1
            if correct: n_correct += 1

        results[key] = {
            "content": content,
            "wall_seconds": round(dt, 2),
            "usage": usage,
            "predicted_per_second": timings.get("predicted_per_second"),
            "correct": correct,
            "score_note": score_note,
        }
        print(f"=== {key} ({dt:.1f}s, {usage.get('completion_tokens')} tok, "
              f"{timings.get('predicted_per_second', 0):.1f} t/s) [{'OK' if correct else '?'}] ===")
        print(content)
        print(f"  score: {score_note}")
        print()

    results["_summary"] = {"n_correct": n_correct, "n_total": n_total}
    out_path = os.path.join(OUT_DIR, f"{MODEL}-hard.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}  |  SCORE: {n_correct}/{n_total}")


if __name__ == "__main__":
    main()
