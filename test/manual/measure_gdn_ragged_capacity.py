"""Offline (GPU-free) measurement of GDN per-head head-aware capacity:
PADDED (RAGGED=0) vs RAGGED (RAGGED=1) at a given w_max, on REAL Qwen3-Next
weights. Mirrors the KDA GATE-K1 capacity check but for the multi-layer per-head
plan, so the cross-layer G_max padding (the thing that makes GDN padded ~= dense)
is actually exercised. Verifies the store->load round-trip stays global-exact.

    SGLANG_HEAD_AWARE_RAGGED=0 python test/manual/measure_gdn_ragged_capacity.py \
        --model ./Qwen3-Next-80B-A3B-Instruct-NVFP4 --w-max 64
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import torch

from sglang.srt.mem_cache.mamba_checkpoint_pool import (
    HeadAwareCheckpointStore,
    HeadAwarePlan,
)


def _load_per_layer(model_dir):
    """Return (A_log, dt_bias) each [L, HV] over all GDN layers (per-head)."""
    from safetensors import safe_open

    a_by, dt_by = {}, {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, framework="pt") as st:
            for key in st.keys():
                if key.endswith(".A_log"):
                    a_by[key[: -len("A_log")]] = st.get_tensor(key)
                elif key.endswith(".dt_bias"):
                    dt_by[key[: -len("dt_bias")]] = st.get_tensor(key)

    def _lidx(k):
        m = re.search(r"layers?\.(\d+)\.", k)
        return int(m.group(1)) if m else -1

    prefixes = sorted(set(a_by) & set(dt_by), key=_lidx)
    # GDN per-head layers only: dt_bias width == A_log head count (KDA is wider)
    gdn = []
    for p in prefixes:
        a = a_by[p].float().flatten()
        dt = dt_by[p].float().flatten()
        if dt.numel() == a.numel():
            gdn.append((a, dt))
    if not gdn:
        return None
    A_log = torch.stack([a for a, _ in gdn], dim=0)  # [L, HV]
    dt_bias = torch.stack([dt for _, dt in gdn], dim=0)  # [L, HV]
    print(f"[weights] {len(gdn)} GDN layers -> HV={A_log.shape[1]}")
    return A_log, dt_bias


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--w-max", type=int, default=64)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--v", type=int, default=128)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--a-scale", type=float, default=0.3)
    args = p.parse_args()

    dt = torch.float32
    K, V = args.k, args.v
    wts = _load_per_layer(args.model)
    if wts is None:
        raise SystemExit("no GDN per-head weights found")
    A_log, dt_bias = wts[0].to(dt), wts[1].to(dt)
    L, HV = A_log.shape

    dense = HeadAwareCheckpointStore.dense_bytes_per_slot(
        num_layers=L, num_heads=HV, head_v_dim=V, head_k_dim=K, state_dtype=dt
    )
    plan = HeadAwarePlan.build_plan(
        A_log=A_log,
        dt_bias=dt_bias,
        route="A",
        d_k=K,
        d_v=V,
        eps=args.eps,
        w_max=args.w_max,
        a_margin=args.a_scale,
    )
    assert not plan.per_channel, "GDN should take the per-head path"

    # classification: local heads per layer
    is_local = plan.w_head > 0  # [L, HV]
    n_local = int(is_local.sum())
    tot = L * HV
    per_layer_global = (~is_local).sum(dim=1)  # [L]
    g_max = int(per_layer_global.max())
    g_mean = float(per_layer_global.float().mean())

    store = HeadAwareCheckpointStore(
        plan=plan, num_slots=4, device="cpu", state_dtype=dt
    )
    # round-trip correctness
    state = torch.randn(L, 1, HV, V, K, dtype=dt)
    store.store(torch.tensor([1]), state)
    rec, mask = store.load(torch.tensor([1]), out_dtype=dt)
    gl = (~is_local)[:, None, :, None, None].expand_as(rec)
    lo = is_local[:, None, :, None, None].expand_as(rec)
    max_gerr = (rec[gl] - state[gl]).abs().max().item() if gl.any() else 0.0
    max_labs = rec[lo].abs().max().item() if lo.any() else 0.0
    ok = max_gerr < 1e-6 and max_labs < 1e-12 and bool((mask.cpu() == is_local).all())

    bps = store.bytes_per_slot()
    cap = dense / max(1, bps)
    ragged = store.ragged
    print(f"  w_max={args.w_max}  RAGGED={int(ragged)}")
    print(
        f"  local heads: {n_local}/{tot} = {100.0*n_local/tot:.1f}%; "
        f"per-layer #global G_max={g_max} mean={g_mean:.2f}"
    )
    print(f"  bytes/slot={bps} vs dense {dense} ({cap:.2f}x capacity)")
    print(
        f"  round-trip global-exact={max_gerr:.1e} local-zero={max_labs:.1e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
