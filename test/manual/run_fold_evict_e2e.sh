#!/usr/bin/env bash
# Phase C2-b / D live fold-to-state KV eviction A/B on Qwen3-Next-80B.
#
#   Run: bash test/manual/run_fold_evict_e2e.sh <ARM>
#     ARM = baseline | fold | fold_safe | fold_unsafe | report
#         | perf_baseline | perf_fold | perf_fold_safe
#
# Phase D (layer-selective fold, GATE-D): A1 showed retrieval is carried ONLY by the
# last 3 full-attn layers {39,43,47}; the first 9 {3,7,...,35} are retrieval-irrelevant.
# So folding ONLY the safe 9 and keeping {39,43,47} exact should rescue C2-b's zeroed
# recall (1.000 -> 0.000) back to ~=dense. The Phase-D arms operationalize that:
#     fold_safe      FOLD_KV_LAYERS=3,7,11,15,19,23,27,31,35 (fold the safe 9)
#     fold_unsafe    FOLD_KV_LAYERS=39,43,47                 (control: fold only the
#                                                             3 retrieval layers -> predict collapse)
#     perf_fold_safe fold_safe layers, bench_serving decode sweep
# FOLD_KV_LAYERS stays overridable from the environment; the per-arm value is only a default.
#
# Faithful-output experiment (per the C2-b decision): at DECODE each full-attn layer's
# output is replaced by merge(window[sink+recent], state_readout(fold(middle))) with a
# fixed exact-KV budget B = FOLD_KV_BUDGET (python/sglang/srt/debug/fold_evict.py).
# The KV pool is READ-ONLY (no page free) so the run stays byte-identical when the env
# var is unset; the O(B) decode-latency + max-batch/capacity wins are PROJECTED from the
# C2-a microbench (test/manual/bench_fold_decode.py) + the exact-KV footprint (B vs N).
#
#   * baseline / fold  -> needle recall A/B (reuse needle_longrange.py). C1 predicts fold
#     keeps the deep needle (the model's gate carries it) -> recall(fold) ~= recall(dense).
#   * perf_*           -> bench_serving decode sweep (TTFT/TPOT/throughput) A/B.
#
# Env is frozen at import -> each arm needs a FRESH server. Requires
# Qwen3-Next-80B-A3B-Instruct (NVFP4), TP=2 (2x H20-3e). Mirrors run_state_ablation_2x2.sh.
set -euo pipefail

export no_proxy='127.0.0.1,localhost' NO_PROXY='127.0.0.1,localhost'
export HTTP_PROXY="${HTTP_PROXY:-http://10.229.18.27:8412}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://10.229.18.27:8412}"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"

ARM="${1:?usage: $0 <baseline|fold|fold_safe|fold_unsafe|report|perf_baseline|perf_fold|perf_fold_safe>}"

MODEL="${MODEL:-/home/hadoop-scale-llm/VSCodeProjects/sglang/Qwen3-Next-80B-A3B-Instruct-NVFP4}"
HOST=127.0.0.1
PORT="${PORT:-31017}"
TP="${TP:-2}"
MEM_FRAC="${MEM_FRAC:-0.7}"
MAX_RUNNING="${MAX_RUNNING:-96}"
LOGDIR="${LOGDIR:-$PWD/e2e_fold_evict_logs}"
mkdir -p "$LOGDIR"
NEEDLE_OUT="${NEEDLE_OUT:-$LOGDIR/fold_evict_needle.json}"

# fold knobs (only used by the fold / perf_fold arms).
FOLD_KV_BUDGET="${FOLD_KV_BUDGET:-1024}"
FOLD_KV_SINK="${FOLD_KV_SINK:-4}"
FOLD_KV_PHI="${FOLD_KV_PHI:-elu1}"
FOLD_KV_DECAY="${FOLD_KV_DECAY:-1.0}"

# needle probe knobs (a deep needle in the shared prefix).
NEEDLE_DOC_WORDS="${NEEDLE_DOC_WORDS:-3000}"
NEEDLE_DEPTH="${NEEDLE_DEPTH:-0.5}"
NEEDLE_GROUPS="${NEEDLE_GROUPS:-16}"
NEEDLE_K="${NEEDLE_K:-4}"

# ---- report arm: no server, diff the shared JSON against baseline ----------------
if [ "$ARM" = "report" ]; then
  echo "[report] fold-evict needle A/B (ref=baseline) from $NEEDLE_OUT"
  NEEDLE_OUT="$NEEDLE_OUT" python "$PWD/test/manual/needle_longrange.py" \
    --report --ref "${REF:-baseline}" --out "$NEEDLE_OUT"
  exit 0
fi

# ---- resolve fold env for this arm -----------------------------------------------
# Phase-D arms default FOLD_KV_LAYERS to the A1-derived layer subset, but stay
# env-overridable (do NOT hard-clobber a user-provided FOLD_KV_LAYERS).
case "$ARM" in
  fold_safe|perf_fold_safe) FOLD_KV_LAYERS="${FOLD_KV_LAYERS:-3,7,11,15,19,23,27,31,35}" ;;
  fold_unsafe)              FOLD_KV_LAYERS="${FOLD_KV_LAYERS:-39,43,47}" ;;
esac

declare -a ARM_ENV=()
case "$ARM" in
  baseline|perf_baseline) ;;  # fold OFF -> byte-identical dense (recall/perf upper ref)
  fold|perf_fold|fold_safe|fold_unsafe|perf_fold_safe)
    ARM_ENV+=(
      "FOLD_KV_BUDGET=$FOLD_KV_BUDGET" "FOLD_KV_SINK=$FOLD_KV_SINK"
      "FOLD_KV_PHI=$FOLD_KV_PHI" "FOLD_KV_DECAY=$FOLD_KV_DECAY"
    )
    [ -n "${FOLD_KV_LAYERS:-}" ] && ARM_ENV+=("FOLD_KV_LAYERS=$FOLD_KV_LAYERS")
    ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac

TAG="$ARM"
SRVLOG="$LOGDIR/server_${TAG}.log"

# Launch flags mirror run_state_ablation_2x2.sh (same hybrid-prefix-cache regime).
# fold_evict is a data-dependent Python hook (host-synced loc gather) -> cuda graph
# must be off, same constraint as the ablation probe.
COMMON=(
  --model-path "$MODEL"
  --trust-remote-code
  --tp-size "$TP"
  --host "$HOST" --port "$PORT"
  --mem-fraction-static "$MEM_FRAC"
  --mamba-radix-cache-strategy extra_buffer
  --max-running-requests "$MAX_RUNNING"
  --attention-backend "${ATTN_BACKEND:-triton}"
  --disable-overlap-schedule
  --disable-cuda-graph
)

echo "[launch] arm=$ARM env='${ARM_ENV[*]:-<none>}' -> $SRVLOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" env "${ARM_ENV[@]}" python -m sglang.launch_server \
  "${COMMON[@]}" > "$SRVLOG" 2>&1 &
SRVPID=$!
trap 'kill $SRVPID 2>/dev/null || true; wait $SRVPID 2>/dev/null || true' EXIT

echo "[wait] server pid=$SRVPID warming up..."
for i in $(seq 1 300); do
  if curl -s "http://$HOST:$PORT/health_generate" >/dev/null 2>&1; then
    echo "[ready] after ${i}0s"; break
  fi
  if ! kill -0 $SRVPID 2>/dev/null; then
    echo "[fatal] server died; tail:"; tail -60 "$SRVLOG"; exit 1
  fi
  sleep 10
done

case "$ARM" in
  baseline|fold|fold_safe|fold_unsafe)
    echo "[needle] arm=$ARM doc_words=$NEEDLE_DOC_WORDS depth=$NEEDLE_DEPTH groups=${NEEDLE_GROUPS}x${NEEDLE_K}"
    NEEDLE_MODE="$ARM" NEEDLE_OUT="$NEEDLE_OUT" \
    python "$PWD/test/manual/needle_longrange.py" \
      --host "$HOST" --port "$PORT" \
      --mode "$ARM" --out "$NEEDLE_OUT" \
      --doc-words "$NEEDLE_DOC_WORDS" \
      --depth "$NEEDLE_DEPTH" \
      --groups "$NEEDLE_GROUPS" \
      --k-per-group "$NEEDLE_K" \
      2>&1 | tee "$LOGDIR/needle_${TAG}.log"
    echo "[done] arm=$ARM -> $NEEDLE_OUT  (run baseline + fold_safe + fold_unsafe, then: $0 report)"
    ;;
  perf_baseline|perf_fold|perf_fold_safe)
    # decode-sweep A/B: in 256 / out 512 at conc {1,16,64} (plan verification #3).
    for CONC in ${PERF_CONC:-1 16 64}; do
      echo "[perf] arm=$ARM conc=$CONC (random in=256 out=512)"
      python -m sglang.bench_serving \
        --backend sglang --host "$HOST" --port "$PORT" \
        --dataset-name random \
        --random-input-len 256 --random-output-len 512 \
        --num-prompts "$((CONC * 8))" --max-concurrency "$CONC" \
        2>&1 | tee "$LOGDIR/perf_${TAG}_conc${CONC}.log"
    done
    echo "[done] perf arm=$ARM -> $LOGDIR/perf_${TAG}_conc*.log"
    ;;
esac
