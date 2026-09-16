#!/usr/bin/env bash
# run_all.sh — battery + long-context sweep for the full c4130 model set, sequential.
#
# Run on the box (c4130-4xv100):
#   nohup env PYTHONUNBUFFERED=1 ./run_all.sh > /tmp/run_all.log 2>&1 < /dev/null &
#
# Order = importance: the two 200k-capable Nemotron hybrids FIRST (the headline),
# then the dense 70B/72B (128k cap), then Qwen3.8, then the gpt-oss sweep re-run.
# Each step is guarded so one failure does not stop the rest.
set -uo pipefail

SCRIPTS=/data/git/c4130-llm-benchmarks/scripts
M=/data/models
export PATH="/usr/local/bin:$PATH"
cd "$SCRIPTS"

battery() {  # name gguf [flags...]
  local name="$1" gguf="$2"; shift 2
  echo ""
  echo "############################################################"
  echo "# BATTERY: $name   ($(date '+%H:%M:%S'))"
  echo "############################################################"
  ./run_benchmarks.sh "$name" "$gguf" "$@" || echo "!! BATTERY FAILED: $name (continuing)"
}

sweep() {  # name gguf maxctx [sizes]
  local name="$1" gguf="$2" maxctx="$3"; shift 3
  echo ""
  echo "############################################################"
  echo "# SWEEP: $name (maxctx=$maxctx)   ($(date '+%H:%M:%S'))"
  echo "############################################################"
  ./run_sweep.sh "$name" "$gguf" "$maxctx" "$@" || echo "!! SWEEP FAILED: $name (continuing)"
}

# ─── 1. Nemotron-3.5-Lightning-30B-A3B (200k-capable, fastest) ───
battery nemotron-3.5-lightning "$M/nemotron-3.5-lightning/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL.gguf" --nothink
sweep   nemotron-3.5-lightning "$M/nemotron-3.5-lightning/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL.gguf" 262144 --sizes 8192,32768,65536,131072,200000,262144

# ─── 2. Nemotron-3-Super-120B-A12B (200k-capable, Opus-class) ───
battery nemotron-3-super "$M/nemotron-3-super/UD-Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00001-of-00003.gguf" --nothink
sweep   nemotron-3-super "$M/nemotron-3-super/UD-Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00001-of-00003.gguf" 262144 --sizes 8192,32768,65536,131072,200000,262144

# ─── 3. Llama-3.3-70B (dense, 128k cap) ───
battery llama-3.3-70b "$M/llama-3.3-70b/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
sweep   llama-3.3-70b "$M/llama-3.3-70b/Llama-3.3-70B-Instruct-Q4_K_M.gguf" 131072

# ─── 4. Qwen2.5-72B (dense, 128k cap) ───
battery qwen2.5-72b "$M/qwen2.5-72b/Qwen2.5-72B-Instruct-Q4_K_M.gguf"
sweep   qwen2.5-72b "$M/qwen2.5-72b/Qwen2.5-72B-Instruct-Q4_K_M.gguf" 131072

# ─── 5. Qwen3.8-27B Q8_0 (hybrid, ~128k bug #27756) ───
battery qwen3.8-27b-q8_0 "$M/qwen3.8-27b/Qwen3.8-27B-Q8_0.gguf" --nothink
sweep   qwen3.8-27b-q8_0 "$M/qwen3.8-27b/Qwen3.8-27B-Q8_0.gguf" 131072

# ─── 6. Qwen3.8-27B BF16 (hybrid) ───
battery qwen3.8-27b-bf16 "$M/qwen3.8-27b/Qwen3.8-27B-BF16.gguf" --nothink
sweep   qwen3.8-27b-bf16 "$M/qwen3.8-27b/Qwen3.8-27B-BF16.gguf" 131072

# ─── 7. gpt-oss-120b sweep re-run (battery already done) — fresh cold 8k prefill ───
sweep   gpt-oss-120b "$M/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf" 131072

echo ""
echo "===== ALL DONE ($(date '+%H:%M:%S')) ====="
