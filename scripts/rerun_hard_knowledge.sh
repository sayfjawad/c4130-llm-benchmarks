#!/usr/bin/env bash
# rerun_hard_knowledge.sh — re-run ONLY bench_hard + bench_knowledge for the reasoning models,
# after the per-question thinking-toggle fix (format/factual = UIT, reasoning/combo = AAN).
# Dense models (llama-3.3-70b, qwen2.5-72b) are unaffected by enable_thinking and are skipped.
set -uo pipefail

SCRIPTS=/data/git/c4130-llm-benchmarks/scripts
M=/data/models
PORT=8081
OUT_DIR="$SCRIPTS/../results"
mkdir -p "$OUT_DIR"
export PATH="/usr/local/bin:$PATH"
cd "$SCRIPTS"

# Wacht tot een eventuele lopende benchmark-server op $PORT klaar is
# (bv. de llama-3.3-70b re-run die nog bezig is), anders zouden we hem afschieten.
echo "Wachten tot poort $PORT vrij is..."
for i in $(seq 1 600); do
  if ! pgrep -f "llama-server.*--port $PORT" >/dev/null 2>&1; then break; fi
  sleep 3
done
echo "Poort vrij — start re-runs."

run_one() {
  local name="$1" gguf="$2"
  echo ""
  echo "############################################################"
  echo "# HARD+KNOWLEDGE: $name   ($(date '+%H:%M:%S'))"
  echo "############################################################"
  pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
  sleep 2
  llama-server -m "$gguf" --host 127.0.0.1 --port $PORT --fit on --cont-batching \
    -ngl 999 -sm layer -ts 8,8,8,8 -fa on --reasoning auto \
    > "$OUT_DIR/${name}-server-rerun.log" 2>&1 &
  local SRV=$!
  local ready=0
  for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
  done
  if [ "$ready" != "1" ]; then
    echo "!! server not ready for $name"; tail -20 "$OUT_DIR/${name}-server-rerun.log"; return
  fi
  echo "  server ready"
  python3 bench_hard.py "$PORT" "$name" --out-dir "$OUT_DIR" || echo "!! hard FAILED $name"
  python3 bench_knowledge.py "$PORT" "$name" --out-dir "$OUT_DIR" || echo "!! knowledge FAILED $name"
  kill "$SRV" 2>/dev/null || true
  sleep 1
  pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
  sleep 2
}

run_one nemotron-3.5-lightning "$M/nemotron-3.5-lightning/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL.gguf"
run_one nemotron-3-super "$M/nemotron-3-super/UD-Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00001-of-00003.gguf"
run_one qwen3.8-27b-q8_0 "$M/qwen3.8-27b/Qwen3.8-27B-Q8_0.gguf"
run_one qwen3.8-27b-bf16 "$M/qwen3.8-27b/Qwen3.8-27B-BF16.gguf"

echo ""
echo "===== ALL HARD+KNOWLEDGE RE-RUNS DONE ($(date '+%H:%M:%S')) ====="
