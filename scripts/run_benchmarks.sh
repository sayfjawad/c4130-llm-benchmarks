#!/usr/bin/env bash
# run_benchmarks.sh — Volledige benchmark batterij voor één model
#
# Gebruik:
#   ./run_benchmarks.sh <MODEL_NAME> <GGUF_PATH> [--nothink] [--reasoning-budget N]
#
# Voorbeelden:
#   ./run_benchmarks.sh gpt-oss-120b ~/models/gpt-oss-120b/UD-Q4_K_XL/model.gguf
#   ./run_benchmarks.sh gemma4-31b ~/models/gemma4-31b/model.gguf --nothink --reasoning-budget 800
#   ./run_benchmarks.sh my-model ~/models/my-model.gguf --port 8082
#
# Vereisten:
#   - llama-server en llama-bench in PATH (van llama.cpp)
#   - Python 3 (stdlib only, geen pip packages nodig)
#   - curl (voor health check)
#
# Output: results/<model>-*.json + results/<model>-llamabench.txt

set -euo pipefail

# ─── defaults ───
PORT="${PORT:-8081}"
HOST="${HOST:-127.0.0.1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/../results}"
REASONING="auto"        # --reasoning flag voor llama-server (on/off/auto)
REASONING_BUDGET=""     # --reasoning-budget N (alleen voor modellen die dit ondersteunen)
EXTRA_SERVER_FLAGS=""

# ─── parse args ───
if [ $# -lt 2 ]; then
    echo "Usage: $0 <MODEL_NAME> <GGUF_PATH> [--nothink] [--reasoning-budget N] [--port N] [--extra-flags ...]"
    echo ""
    echo "  MODEL_NAME        Naam voor output-bestanden (bv. gemma4-31b)"
    echo "  GGUF_PATH         Pad naar .gguf bestand"
    echo "  --nothink         Zet reasoning/thinking UIT voor formaat-taken"
    echo "  --reasoning-budget N  Max thinking tokens (Gemma 4 stijl)"
    echo "  --port N           Poort voor llama-server (default: 8081)"
    echo "  --extra-flags ...  Extra flags voor llama-server (bv. --jinja)"
    exit 1
fi

MODEL_NAME="$1"
GGUF="$2"
shift 2

NOTHINK=""
while [ $# -gt 0 ]; do
    case "$1" in
        --nothink)
            NOTHINK="--nothink"
            # NIET: REASONING="off". --reasoning off is een Gemma-4 mechanisme en schakelt
            # het denken GLOBAAL uit (dus ook bench_math en de reken-/combo-vragen in
            # hard/knowledge, die thinking NODIG hebben). Nemotron/Qwen worden per-request
            # op UIT gezet via chat_template_kwargs {enable_thinking:false} (zie bench_*.py).
            ;;
        --reasoning-budget)
            REASONING_BUDGET="--reasoning-budget $2"
            shift
            ;;
        --port)
            PORT="$2"
            shift
            ;;
        --extra-flags)
            shift
            EXTRA_SERVER_FLAGS="$*"
            break
            ;;
        *)
            echo "Onbekende optie: $1"
            exit 1
            ;;
    esac
    shift
done

mkdir -p "$OUT_DIR"

# Kill een eventuele oude server VOORDAT llama-bench draait: llama-bench heeft
# de volledige VRAM-pool nodig en faalt anders met "failed to load model".
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 2

# ─── 1. LLAMA-BENCH ───
echo ""
echo "══════════════════════════════════════════════"
echo "  LLAMA-BENCH — $MODEL_NAME"
echo "══════════════════════════════════════════════"
llama-bench -m "$GGUF" -ngl 999 -sm layer -fa on -p 512 -n 128 2>&1 | tee "$OUT_DIR/${MODEL_NAME}-llamabench.txt"
echo "LLAMA-BENCH DONE"

# ─── 2. Start server ───
echo ""
echo "══════════════════════════════════════════════"
echo "  Start llama-server — $MODEL_NAME (poort $PORT)"
echo "══════════════════════════════════════════════"

# Kill bestaande server op deze poort
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 2

# Bouw server command
SERVER_CMD="llama-server -m \"$GGUF\" --host $HOST --port $PORT --fit on --cont-batching -ngl 999 -sm layer -ts 8,8,8,8 -fa on"
[ -n "$REASONING" ] && SERVER_CMD="$SERVER_CMD --reasoning $REASONING"
[ -n "$REASONING_BUDGET" ] && SERVER_CMD="$SERVER_CMD $REASONING_BUDGET"
[ -n "$EXTRA_SERVER_FLAGS" ] && SERVER_CMD="$SERVER_CMD $EXTRA_SERVER_FLAGS"

echo "  $SERVER_CMD"
eval "$SERVER_CMD" > "$OUT_DIR/${MODEL_NAME}-server.log" 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wacht tot server klaar is
for i in $(seq 1 60); do
    if curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; then
        echo "  Server ready na ${i}s"
        break
    fi
    sleep 1
done

if ! curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; then
    echo "  ERROR: Server niet bereikbaar na 60s!"
    cat "$OUT_DIR/${MODEL_NAME}-server.log" | tail -20
    exit 1
fi

# ─── 3. Draai alle benchmarks ───
cd "$SCRIPT_DIR"

echo ""
echo "══════════════════════════════════════════════"
echo "  BENCH_HARD — $MODEL_NAME"
echo "══════════════════════════════════════════════"
# hard_reasoning (2195) vereist thinking AAN — geen --nothink
python3 bench_hard.py "$PORT" "$MODEL_NAME" --out-dir "$OUT_DIR"

echo ""
echo "══════════════════════════════════════════════"
echo "  BENCH_LONGCTX — $MODEL_NAME"
echo "══════════════════════════════════════════════"
python3 bench_longctx.py "$PORT" "$MODEL_NAME" $NOTHINK --out-dir "$OUT_DIR"

echo ""
echo "══════════════════════════════════════════════"
echo "  BENCH_NOTULEN_REPEAT — $MODEL_NAME"
echo "══════════════════════════════════════════════"
python3 bench_notulen_repeat.py "$PORT" "$MODEL_NAME" 5 $NOTHINK --out-dir "$OUT_DIR"

echo ""
echo "══════════════════════════════════════════════"
echo "  BENCH_TOOLS — $MODEL_NAME"
echo "══════════════════════════════════════════════"
python3 bench_tools.py "$PORT" "$MODEL_NAME" $NOTHINK --out-dir "$OUT_DIR"

echo ""
echo "══════════════════════════════════════════════"
echo "  BENCH_KNOWLEDGE — $MODEL_NAME"
echo "══════════════════════════════════════════════"
# combo-vragen (2/14) vereisen thinking AAN — geen --nothink
python3 bench_knowledge.py "$PORT" "$MODEL_NAME" --out-dir "$OUT_DIR"

echo ""
echo "══════════════════════════════════════════════"
echo "  BENCH_MATH — $MODEL_NAME (thinking AAN)"
echo "══════════════════════════════════════════════"
# Math heeft ALTIJD thinking nodig — gebruik NOTHINK niet
python3 bench_math.py "$PORT" "$MODEL_NAME" --out-dir "$OUT_DIR"

# ─── 4. Cleanup ───
echo ""
echo "══════════════════════════════════════════════"
echo "  Klaar — server stoppen"
echo "══════════════════════════════════════════════"
kill "$SERVER_PID" 2>/dev/null || true
sleep 1
# Dubbelcheck
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true

echo ""
echo "Resultaten in: $OUT_DIR/"
ls -la "$OUT_DIR/${MODEL_NAME}"-*.json "$OUT_DIR/${MODEL_NAME}"-*.txt 2>/dev/null || echo "(sommige output-bestanden ontbreken)"
echo ""
echo "Draai daarna: python3 aggregate_results.py --results-dir $OUT_DIR --out docs/data/results.json"
echo "om de resultaten toe te voegen aan het dashboard."
