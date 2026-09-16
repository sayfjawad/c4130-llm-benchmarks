#!/usr/bin/env bash
# run_sweep.sh — long-context retrieval sweep voor één model (c4130)
#
# Gebruik:
#   ./run_sweep.sh <MODEL_NAME> <GGUF_PATH> [MAX_CTX] [--sizes a,b,c]
#
#   MAX_CTX  default 131072 (gpt-oss / dense 70B/72B cap). Nemotron hybrids: 200000+.
#   --sizes  default 8192,32768,65536,131072,200000
#
# Start een single-slot llama-server met een expliciete grote context, en draait
# daarna bench_longctx_sweep.py. Anders dan de 43/43-batterij (run_benchmarks.sh)
# heeft de sweep -c <n> -np 1 nodig, omdat --fit on n_slots=4 geeft en de per-slot
# context dan maar 1/4 van de pool is.
#
# KV-cache blijft default (fp16). Voor de dense 70B/72B op 128k kan -ctk q8_0 -ctv q8_0
# de KV halveren (vereist FA, wat op Volta werkt) — pas toe via EXTRA_SERVER_FLAGS.

set -euo pipefail

PORT="${PORT:-8081}"
HOST="${HOST:-127.0.0.1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/../results}"
MAX_CTX="${MAX_CTX:-131072}"
SIZES="${SIZES:-8192,32768,65536,131072,200000}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <MODEL_NAME> <GGUF_PATH> [MAX_CTX] [--sizes a,b,c]"
    exit 1
fi
MODEL_NAME="$1"
GGUF="$2"
shift 2
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    MAX_CTX="$1"
    shift
fi
if [ $# -gt 0 ] && [ "$1" == "--sizes" ]; then
    SIZES="$2"
    shift 2
fi

mkdir -p "$OUT_DIR"

# Kill bestaande server op deze poort
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 2

SERVER_CMD="llama-server -m \"$GGUF\" --host $HOST --port $PORT -c $MAX_CTX -np 1 -fa on -kvu -sm layer -ts 8,8,8,8 -ngl 999"
echo "  $SERVER_CMD"
eval "$SERVER_CMD" > "$OUT_DIR/${MODEL_NAME}-sweep-server.log" 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

for i in $(seq 1 120); do
    if curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; then
        echo "  Server ready na ${i}s"
        break
    fi
    sleep 1
done
if ! curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; then
    echo "  ERROR: Server niet bereikbaar na 120s!"
    tail -20 "$OUT_DIR/${MODEL_NAME}-sweep-server.log"
    exit 1
fi

cd "$SCRIPT_DIR"
python3 bench_longctx_sweep.py "$PORT" "$MODEL_NAME" --out-dir "$OUT_DIR" --sizes "$SIZES"

kill "$SERVER_PID" 2>/dev/null || true
sleep 1
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true

echo ""
echo "Sweep done: $OUT_DIR/${MODEL_NAME}-longctx-sweep.json"
