#!/usr/bin/env python3
"""
aggregate_results.py — Lees ruwe benchmark resultaten en update results.json voor het dashboard.

Gebruik:
  # Nieuwe results.json genereren uit raw results (metadata handmatig toevoegen)
  python3 aggregate_results.py --results-dir results/ --out docs/data/results.json

  # Eén model toevoegen aan bestaande results.json
  python3 aggregate_results.py --results-dir results/ --model gemma4-31b --out docs/data/results.json

  # Alle modellen opnieuw scannen en bestaande metadata behouden
  python3 aggregate_results.py --results-dir results/ --out docs/data/results.json --full

Elk raw result bestand moet de naam `<model>-<test>.json` hebben.
Bijv: `gpt-oss-120b-hard.json`, `gpt-oss-120b-math.json`, etc.
"""

import json
import os
import sys
import argparse


def load_raw_results(results_dir, model_id):
    """Laad alle raw resultaten voor één model uit results_dir."""
    scores = {}
    math_efficiency = None
    gen_tps = None
    prompt_tps = None

    # Map test file suffix → score key in results.json
    test_map = {
        "hard": "hard4",
        "longctx": "longctx",
        "notulen-repeat": "reliability",
        "tools": "tools",
        "knowledge": "knowledge",
        "math": "math",
    }

    for suffix, score_key in test_map.items():
        fname = f"{model_id}-{suffix}.json"
        fpath = os.path.join(results_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  WAARSCHUWING: kon {fpath} niet lezen: {e}")
            continue

        summary = data.get("_summary", {})
        n = summary.get("n_correct") or summary.get("n_ok")
        if n is None:
            # probeer score veld (longctx)
            n = data.get("score")
        n_total = summary.get("n_total") or data.get("n_needles", 0)
        if n is None and n_total:
            n = 0  # fallback

        entry = {"n": n, "of": n_total}
        if n is not None and n_total:
            entry["n"] = n
            entry["of"] = n_total

        scores[score_key] = entry

        # Extract token efficiëntie voor math
        if suffix == "math":
            tokens = []
            for k, v in data.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, dict) and v.get("completion_tokens"):
                    tokens.append(v["completion_tokens"])
            if tokens:
                math_efficiency = f"{min(tokens)}-{max(tokens)} per antwoord"

    # Probeer llama-bench output te lezen voor snelheid
    bench_file = os.path.join(results_dir, f"{model_id}-llamabench.txt")
    if os.path.exists(bench_file):
        try:
            with open(bench_file) as f:
                bench_text = f.read()
            import re
            # Zoek tg128 en pp512 regels
            tg_match = re.search(r'tg128.*?(\d+\.\d+)\s*t/s', bench_text)
            pp_match = re.search(r'pp512.*?(\d+\.\d+)\s*t/s', bench_text)
            if tg_match:
                gen_tps = float(tg_match.group(1))
            if pp_match:
                prompt_tps = float(pp_match.group(1))
        except Exception:
            pass

    return scores, math_efficiency, gen_tps, prompt_tps


def main():
    p = argparse.ArgumentParser(description="Aggregeer benchmark resultaten naar results.json")
    p.add_argument("--results-dir", default="results", help="Directory met raw result JSONs")
    p.add_argument("--out", default="docs/data/results.json", help="Output results.json")
    p.add_argument("--model", help="Alleen dit model verwerken (toevoegen/updaten)")
    p.add_argument("--full", action="store_true", help="Alle modellen opnieuw scannen")
    args = p.parse_args()

    # Laad bestaande results.json als die bestaat
    existing = {"meta": {}, "models": []}
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                existing = json.load(f)
        except Exception:
            pass

    existing_models = {m["id"]: m for m in existing.get("models", [])}

    # Bepaal welke modellen te verwerken
    if args.model:
        model_ids = [args.model]
    elif args.full:
        # Scan results dir voor alle model-*.json bestanden
        model_ids = set()
        for fname in os.listdir(args.results_dir):
            if fname.endswith(".json") and "-" in fname:
                # Extract model id: "<model>-<test>.json" → "<model>"
                # Model id kan koppeltekens bevatten, dus we moeten slim splitsen
                parts = fname.rsplit("-", 1)
                if len(parts) == 2:
                    candidate = parts[0]
                    # Check of dit een bekend test-suffix is
                    known_suffixes = ["hard", "longctx", "notulen-repeat", "tools", "knowledge", "math"]
                    if parts[1].replace(".json", "") in known_suffixes:
                        model_ids.add(candidate)
        model_ids = sorted(model_ids)
    else:
        print("Gebruik --model <id> om één model toe te voegen, of --full om alles te scannen.")
        sys.exit(1)

    print(f"Verwerken: {len(model_ids)} modellen uit {args.results_dir}/")
    for model_id in model_ids:
        print(f"  {model_id}...", end=" ")
        scores, math_eff, gen_tps, prompt_tps = load_raw_results(args.results_dir, model_id)

        if not scores:
            print("geen resultaten gevonden")
            continue

        # Update bestaand model of maak nieuw
        if model_id in existing_models:
            model_entry = existing_models[model_id]
        else:
            model_entry = {
                "id": model_id,
                "name": model_id,
                "publisher": "",
                "hf_repo": "",
                "quant": "",
                "params_total_b": None,
                "params_active_b": None,
                "architecture": "",
                "disk_gb": None,
                "gen_tps": None,
                "prompt_tps": None,
                "scores": {},
                "verdict": "",
                "notes": "",
            }

        # Update scores (merge, overschrijf n/niet met null)
        existing_scores = model_entry.get("scores", {})
        for key, entry in scores.items():
            existing_scores[key] = entry
        model_entry["scores"] = existing_scores

        # Update snelheid als beschikbaar
        if gen_tps is not None:
            model_entry["gen_tps"] = gen_tps
        if prompt_tps is not None:
            model_entry["prompt_tps"] = prompt_tps
        if math_eff:
            model_entry["math_efficiency_tokens"] = math_eff

        existing_models[model_id] = model_entry
        score_summary = ", ".join(f"{k}={v.get('n','?')}/{v.get('of','?')}" for k, v in scores.items())
        print(f"OK — {score_summary}")

    # Schrijf output
    existing["models"] = list(existing_models.values())
    existing["meta"]["last_updated"] = existing.get("meta", {}).get("last_updated", "")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"\nGeschreven naar: {args.out}")
    print(f"Totaal modellen: {len(existing['models'])}")
    print("\nLET OP: Vul handmatig metadata in (publisher, params, architecture, etc.)")
    print("voor nieuwe modellen in results.json. De scores zijn automatisch ingevuld.")


if __name__ == "__main__":
    main()
