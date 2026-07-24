#!/usr/bin/env bash
# M0/M1 e2e driver: GDN prefix-cache hit-rate + TTFT under a shared-prefix
# workload, comparing the mamba checkpoint variants that trade bytes/slot for
# cached-prefix capacity.
#
#   Run: bash test/manual/bench_gdn_prefix_hitrate.sh <MODE> [SYS_LEN] [NUM_GROUPS]
#     MODE = baseline | int8 | headaware_b | headaware_a
#     SYS_LEN    shared system-prompt length in tokens (default 2048; sweep {1024,2048,4096})
#     NUM_GROUPS number of distinct shared prefixes (default 64; raise to pressure the pool)
#
# What it measures (the A-vs-B decision inputs):
#   * mamba-checkpoint hit rate  -> scraped from /metrics (sglang:cache_hit_rate)
#   * mean / p50 / p99 TTFT      -> from sglang.bench_serving
#   * checkpoint pool slots + evictions -> server log grep
#
# The regime that matters is when the checkpoint pool SATURATES and eviction
# begins: raise NUM_GROUPS (more distinct prefixes than the pool can hold) so the
# capacity difference between variants converts to a hit-rate / TTFT difference.
#
# Requires: Qwen/Qwen3-Next-80B-A3B-Instruct, TP=2 (2x H20-3e on this box).
set -euo pipefail

# Proxy for the ShareGPT token-source fetch; keep localhost direct.
export no_proxy='127.0.0.1,localhost' NO_PROXY='127.0.0.1,localhost'
export HTTP_PROXY="${HTTP_PROXY:-http://10.229.18.27:8412}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://10.229.18.27:8412}"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
# This box has sgl-kernel 0.4.2.post2 (cu12.9); the current commit pins 0.4.4.
# Skip the version gate (documented env workaround) so the server launches locally.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"

MODE="${1:?usage: $0 <baseline|int8|headaware_b|headaware_a> [sys_len] [num_groups]}"
SYS_LEN="${2:-2048}"
NUM_GROUPS="${3:-64}"
PROMPTS_PER_GROUP="${PROMPTS_PER_GROUP:-16}"
QUESTION_LEN="${QUESTION_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
GROUP_DIST="${GROUP_DIST:-zipf}"
ZIPF_ALPHA="${ZIPF_ALPHA:-1.1}"

MODEL="${MODEL:-/home/hadoop-scale-llm/VSCodeProjects/sglang/Qwen3-Next-80B-A3B-Instruct-NVFP4}"
HOST=127.0.0.1
PORT="${PORT:-31007}"
TP="${TP:-2}"
MEM_FRAC="${MEM_FRAC:-0.7}"
MAX_CONC="${MAX_CONC:-64}"
# Server req-pool is sized to --max-running-requests; keep it ABOVE the client
# concurrency so the pool always has free req_to_token slots for the Route-A recon
# waves (each throwaway recon req needs one). Applied to ALL modes -> fair A/B.
MAX_RUNNING="${MAX_RUNNING:-$(( MAX_CONC + 32 ))}"
LOGDIR="${LOGDIR:-$PWD/gdn_prefix_hitrate_logs}"
mkdir -p "$LOGDIR"
TAG="${MODE}_sys${SYS_LEN}_g${NUM_GROUPS}${SEAM_WINDOW:+_seam${SEAM_WINDOW}}$([ "${NORECON:-0}" = "1" ] && echo _norecon || :)$([ -n "${MAMBA_EVICT_POLICY:-}" ] && echo "_evict${MAMBA_EVICT_POLICY}" || :)"
SRVLOG="$LOGDIR/server_${TAG}.log"
RESJSON="$LOGDIR/result_${TAG}.json"
NUM_PROMPTS=$(( NUM_GROUPS * PROMPTS_PER_GROUP ))

# Per-mode server flags. All variants share extra_buffer (required for the radix
# to cache shared-prefix mamba state at all) + metrics (for the hit-rate scrape).
COMMON=(
  --model-path "$MODEL"
  --trust-remote-code
  --tp-size "$TP"
  --host "$HOST" --port "$PORT"
  --mem-fraction-static "$MEM_FRAC"
  --mamba-radix-cache-strategy extra_buffer
  --max-running-requests "$MAX_RUNNING"
  --enable-metrics
  # cu12.9 flash-attn on this box lacks the FA3 `only_qv` kwarg (PR #28394); route
  # full-attention layers through triton so the server reaches serving. Applied to
  # ALL modes uniformly, so it does not bias the A/B comparison.
  --attention-backend "${ATTN_BACKEND:-triton}"
)
# The Route-A re-prefill seam injects a separate reconstruction forward inside
# forward_batch_generation; under the overlap scheduler its scratch alloc/free races
# the scheduler pre-allocating the next batch (async index-OOB). DISABLE_OVERLAP=1
# routes ALL modes through the single-stream scheduler so the A/B stays fair.
if [ "${DISABLE_OVERLAP:-0}" = "1" ]; then
  COMMON+=(--disable-overlap-schedule)
fi
# Phase H overlay: the ABLATE_FULL_KV blinding hook is a per-forward Python context
# manager whose loc gather does a device->host copy CUDA-graph capture forbids, and
# graph replay would skip the Python entirely. DISABLE_CUDA_GRAPH=1 routes decode
# through eager so the ablation runs every step. Unset -> byte-identical (graph on).
if [ "${DISABLE_CUDA_GRAPH:-0}" = "1" ]; then
  COMMON+=(--disable-cuda-graph)
fi
EXTRA=()
# Per-mode env (e.g. the head-aware tau-threshold override for the dense baseline).
declare -a MODE_ENV=()
case "$MODE" in
  baseline)     ;;  # bf16 active pool only (current capacity ceiling)
  int8)         EXTRA+=(--enable-int8-mamba-checkpoint) ;;
  # SAME-PRECISION (bf16) fair baseline: head-aware Route A with w_max=0 forces ALL
  # heads global -> an exact full-head bf16 checkpoint (37.7MB/slot). Correct on hit
  # (all rows copied, no re-prefill). This is the capacity/TTFT reference that the
  # bf16 selective routes (headaware_b) must beat -- NOT int8 (which is orthogonal
  # quantization). Slot count sized to the SAME HBM budget as headaware_b.
  dense_bf16)   MODE_ENV+=(SGLANG_FORCE_HEAD_AWARE_WMAX=0)
                EXTRA+=(--enable-head-aware-mamba-checkpoint --head-aware-route A --head-aware-mamba-ckpt-size "${DENSE_CKPT_SIZE:-476}") ;;
  headaware_b)  EXTRA+=(--enable-head-aware-mamba-checkpoint --head-aware-route B --head-aware-mamba-ckpt-size "${HEADAWARE_CKPT_SIZE:-400}") ;;
  headaware_a)  # NORECON=1 -> disable the LOAD-side reconstruction (local heads stay
                # zeroed = seam 0), while the store side still drops local heads
                # (capacity stays 1.33-1.60x). Isolates "does reconstruction matter?".
                if [ "${NORECON:-0}" = "1" ]; then
                  MODE_ENV+=(SGLANG_ENABLE_HEAD_AWARE_REPREFILL=0)
                else
                  MODE_ENV+=(SGLANG_ENABLE_HEAD_AWARE_REPREFILL=1)
                fi
                # Hypic C2: capacity口径 stays at plan.W_max=4096 (ckpt-size 1200); the
                # SEAM env only shrinks the per-hit reconstruction window. Unset -> full W.
                if [ -n "${SEAM_WINDOW:-}" ]; then
                  MODE_ENV+=("SGLANG_HEAD_AWARE_SEAM_WINDOW=${SEAM_WINDOW}")
                fi
                EXTRA+=(--enable-head-aware-mamba-checkpoint --head-aware-route A --head-aware-mamba-ckpt-size "${HEADAWARE_CKPT_SIZE:-1200}") ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac

# Mamba (GDN) checkpoint eviction policy A/B knob. "lru" (default) is byte-identical
# to prior behavior; "value" evicts the lowest-reuse checkpoint first (frequency,
# recency tie-break) so hot shared prefixes survive pool pressure. Applied to ALL
# modes uniformly -> fair A/B. Only matters once the pool saturates and evicts.
MODE_ENV+=("SGLANG_MAMBA_EVICT_POLICY=${MAMBA_EVICT_POLICY:-lru}")

echo "[launch] mode=$MODE sys_len=$SYS_LEN groups=$NUM_GROUPS prompts=$NUM_PROMPTS seam=${SEAM_WINDOW:-full} evict=${MAMBA_EVICT_POLICY:-lru} -> $SRVLOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" env "${MODE_ENV[@]}" python -m sglang.launch_server \
  "${COMMON[@]}" "${EXTRA[@]}" > "$SRVLOG" 2>&1 &
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

# Warm the radix: one pass so shared prefixes are cached, then flush ONLY the
# request queue metrics window is not resettable, so we measure the steady-state
# second pass (the cache is already populated -> the hit-rate reflects capacity).
echo "[bench] gsp groups=$NUM_GROUPS x $PROMPTS_PER_GROUP, sys=$SYS_LEN q=$QUESTION_LEN out=$OUTPUT_LEN dist=$GROUP_DIST"
if [ "${SKIP_BENCH:-0}" = "1" ]; then
  echo "[bench] SKIP_BENCH=1 -> skipping bench_serving"
else
python -m sglang.bench_serving \
  --backend sglang \
  --host "$HOST" --port "$PORT" \
  --model "$MODEL" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups "$NUM_GROUPS" \
  --gsp-prompts-per-group "$PROMPTS_PER_GROUP" \
  --gsp-system-prompt-len "$SYS_LEN" \
  --gsp-question-len "$QUESTION_LEN" \
  --gsp-output-len "$OUTPUT_LEN" \
  --gsp-group-distribution "$GROUP_DIST" \
  --gsp-zipf-alpha "$ZIPF_ALPHA" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$MAX_CONC" \
  --output-file "$RESJSON" \
  2>&1 | tee "$LOGDIR/bench_${TAG}.log"
fi

# ---- optional accuracy gate: few-shot gsm8k -----------------------------------
# The 8-shot preamble is a shared prefix across all questions, so on the 2nd+ question
# the Route-A hit path fires -> gsm8k accuracy is a faithful judge of whether dropping
# local heads + re-prefilling them degrades the output. Run headaware_a vs dense_bf16
# with the SAME GSM8K_N/GSM8K_SHOTS for the A/B verdict. Uses --disable-overlap-schedule
# via DISABLE_OVERLAP=1 (Route-A seam is overlap-unsafe).
if [ "${RUN_GSM8K:-0}" = "1" ]; then
  echo "[gsm8k] mode=$MODE n=${GSM8K_N:-100} shots=${GSM8K_SHOTS:-8}"
  python -m sglang.test.few_shot_gsm8k \
    --host "http://$HOST" --port "$PORT" \
    --num-questions "${GSM8K_N:-100}" \
    --num-shots "${GSM8K_SHOTS:-8}" \
    --parallel "${GSM8K_PAR:-8}" \
    --max-new-tokens 512 \
    2>&1 | tee "$LOGDIR/gsm8k_${TAG}.log"
fi

# ---- optional long-context OUTPUT-AGREEMENT probe -----------------------------
# Direct judge of whether Route-A reconstruction changes the model's OUTPUT (not
# whether a needle is retrievable -- Qwen3-Next's 12 full-attn layers remember any
# needle exactly regardless of GDN recon, so a plain passkey test cannot distinguish
# the arms). Runs the SAME long-context greedy prompts per mode; `--report` then diffs
# each mode's outputs against the dense_bf16 reference. NEEDLE_OUT is shared across the
# sweep so all arms land in one JSON. Uses a LABEL derived from mode+seam+norecon so
# no-recon / seam / full / dense are distinct keys.
if [ "${RUN_NEEDLE:-0}" = "1" ]; then
  NEEDLE_LABEL="${MODE}${SEAM_WINDOW:+_seam${SEAM_WINDOW}}$([ "${NORECON:-0}" = "1" ] && echo _norecon || :)"
  echo "[needle] mode=$MODE label=$NEEDLE_LABEL doc_words=${NEEDLE_DOC_WORDS:-3000} depth=${NEEDLE_DEPTH:-0.5} groups=${NEEDLE_GROUPS:-16}x${NEEDLE_K:-4}"
  NEEDLE_MODE="$NEEDLE_LABEL" \
  NEEDLE_OUT="${NEEDLE_OUT:-$LOGDIR/needle_longrange_results.json}" \
  python "$PWD/test/manual/needle_longrange.py" \
    --host "$HOST" --port "$PORT" \
    --doc-words "${NEEDLE_DOC_WORDS:-3000}" \
    --depth "${NEEDLE_DEPTH:-0.5}" \
    --groups "${NEEDLE_GROUPS:-16}" \
    --k-per-group "${NEEDLE_K:-4}" \
    2>&1 | tee "$LOGDIR/needle_${TAG}.log"
fi

# ---- optional RULER-style DIFFUSE-integration probe (T1.2) --------------------
# Mirrors RUN_NEEDLE but drives ruler_diffuse.py (variable-tracking / multi-value)
# against the SAME self-managed server. RULER_TASK={vt,mv}. Shares RULER_OUT so all
# arms land in one JSON keyed by the same mode+norecon LABEL as needle.
if [ "${RUN_RULER:-0}" = "1" ]; then
  RULER_LABEL="${MODE}${SEAM_WINDOW:+_seam${SEAM_WINDOW}}$([ "${NORECON:-0}" = "1" ] && echo _norecon || :)"
  echo "[ruler] mode=$MODE label=$RULER_LABEL task=${RULER_TASK:-vt} doc_words=${NEEDLE_DOC_WORDS:-9000} groups=${NEEDLE_GROUPS:-16}x${NEEDLE_K:-4}"
  NEEDLE_MODE="$RULER_LABEL" \
  NEEDLE_OUT="${RULER_OUT:-$LOGDIR/ruler_diffuse_results.json}" \
  python "$PWD/test/manual/ruler_diffuse.py" \
    --host "$HOST" --port "$PORT" \
    --task "${RULER_TASK:-vt}" \
    --doc-words "${NEEDLE_DOC_WORDS:-9000}" \
    --hops "${RULER_HOPS:-8}" \
    --n-val "${RULER_NVAL:-6}" \
    --groups "${NEEDLE_GROUPS:-16}" \
    --k-per-group "${NEEDLE_K:-4}" \
    2>&1 | tee "$LOGDIR/ruler_${RULER_TASK:-vt}_${TAG}.log"
fi

# ---- scrape the capacity/hit-rate signals ------------------------------------
echo "[metrics] mode=$MODE"
HIT=$(curl -s "http://$HOST:$PORT/metrics" | grep -E '^sglang:cache_hit_rate' | awk '{print $2}' | tail -1)
echo "  cache_hit_rate = ${HIT:-NA}"
# checkpoint pool sizing + eviction come from the server log
echo "  --- checkpoint pool log lines ---"
grep -iE 'mamba checkpoint pool|head-aware.*pool|capacity|evict' "$SRVLOG" | tail -20 || true

echo "[done] mode=$MODE -> result=$RESJSON  hit_rate=${HIT:-NA}"
echo "  (compare across: baseline / int8 / headaware_b / headaware_a at the same sys_len,groups)"
