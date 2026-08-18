#!/usr/bin/env bash
# Smoke test: head-aware Route A on Kimi-Linear-48B (KDA), 4 configs.
# Usage: bash test/manual/smoke_head_aware_route_a.sh [config]
#   config: all | norecon_padded | norecon_ragged | recon_padded | recon_ragged
set -euo pipefail

MODEL=/home/hadoop-scale-llm/VSCodeProjects/sglang/Kimi-Linear-48B-A3B-Instruct
PORT_BASE=28100
LOGDIR=/tmp/smoke_head_aware
mkdir -p "$LOGDIR"

export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

run_one() {
    local name="$1" port_offset="$2"
    shift 2
    # Remaining args are env vars in KEY=VAL form
    local port=$((PORT_BASE + port_offset))
    local log="$LOGDIR/${name}.log"
    echo "=== SMOKE: $name (port $port) ==="

    # Kill anything on this port
    lsof -ti :"$port" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 1

    # Start server with optional env vars
    env "$@" \
        python -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp 2 \
        --port "$port" \
        --mem-fraction-static 0.80 \
        --max-running-requests 4 \
        --chunked-prefill-size 4096 \
        --enable-head-aware-mamba-checkpoint \
        --head-aware-route A \
        --head-aware-mamba-ckpt-size 64 \
        --mamba-full-memory-ratio 0.5 \
        --trust-remote-code \
        > "$log" 2>&1 &
    local pid=$!

    # Wait for health (up to 180s)
    local ok=0
    for i in $(seq 1 90); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            ok=1; break
        fi
        sleep 2
    done
    if [ "$ok" -eq 0 ]; then
        echo "  FAIL: server did not become healthy"
        tail -30 "$log"
        kill -9 $pid 2>/dev/null || true
        return 1
    fi

    # Check head-aware pool init in log
    if grep -q "head-aware mamba checkpoint pool" "$log"; then
        echo "  pool init: OK"
        grep "head-aware mamba checkpoint pool" "$log" | head -3
    else
        echo "  FAIL: no head-aware pool init log"
        tail -30 "$log"
        kill -9 $pid 2>/dev/null || true
        return 1
    fi

    # Check ragged diag if applicable
    if grep -q "ragged-diag" "$log"; then
        grep "ragged-diag" "$log" | head -2
    fi

    # Send a short generation request
    local resp
    resp=$(curl -sf "http://127.0.0.1:$port/generate" \
        -H 'Content-Type: application/json' \
        -d '{"text":"1+1=","sampling_params":{"max_new_tokens":8,"temperature":0}}' 2>&1) || {
        echo "  FAIL: generation request failed"
        tail -20 "$log"
        kill -9 $pid 2>/dev/null || true
        return 1
    }
    echo "  generation: OK -> $(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("text","")[:60])' 2>/dev/null || echo "$resp" | head -c 60)"

    # Send a second identical request to trigger prefix-cache hit
    resp2=$(curl -sf "http://127.0.0.1:$port/generate" \
        -H 'Content-Type: application/json' \
        -d '{"text":"1+1=","sampling_params":{"max_new_tokens":8,"temperature":0}}' 2>&1) || true
    echo "  2nd request (cache hit): OK"

    # Check for Route-A recon hit log if recon is enabled
    if grep -q "Route-A C2" "$log" 2>/dev/null; then
        echo "  recon fired:"
        grep "Route-A C2" "$log" | head -2
    fi

    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    echo "  PASS: $name"
    echo ""
    return 0
}

CONFIG="${1:-all}"
FAIL=0

if [ "$CONFIG" = "all" ] || [ "$CONFIG" = "norecon_padded" ]; then
    run_one "norecon_padded" 0 || FAIL=1
fi
if [ "$CONFIG" = "all" ] || [ "$CONFIG" = "norecon_ragged" ]; then
    run_one "norecon_ragged" 1 \
        SGLANG_HEAD_AWARE_RAGGED=1 || FAIL=1
fi
if [ "$CONFIG" = "all" ] || [ "$CONFIG" = "recon_padded" ]; then
    run_one "recon_padded" 2 \
        SGLANG_ENABLE_HEAD_AWARE_REPREFILL=1 || FAIL=1
fi
if [ "$CONFIG" = "all" ] || [ "$CONFIG" = "recon_ragged" ]; then
    run_one "recon_ragged" 3 \
        SGLANG_HEAD_AWARE_RAGGED=1 \
        SGLANG_ENABLE_HEAD_AWARE_REPREFILL=1 || FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
    echo "=== ALL SMOKE TESTS PASSED ==="
else
    echo "=== SOME TESTS FAILED ==="
fi
exit $FAIL
