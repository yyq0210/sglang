"""GATE-K1: validate the KDA PER-CHANNEL head-aware checkpoint (offline, CPU-ok).

KDA analog of ``test_gdn_head_aware_prefix.py``. Where GDN decay is a per-head
SCALAR (clean head-level local/global split), KDA decay is PER-K-CHANNEL:
    g[h,k] = -exp(A_log[h]) * softplus(a + dt_bias[h,k])   (dt_bias is [HV, d_k])
so the local/global unit is a (head, d_k-column) pair, not a whole head. The
empirical profile (docs: KDA per-channel decay) showed the split is genuinely
per-channel (90.6% of heads are MIXED), so the checkpoint stores/drops individual
d_k COLUMNS of each head's [d_v, d_k] state.

What it checks (the HARD invariants — the gate):
  1. Classification: a (head, col) is LOCAL iff tau(col) <= w_max (else GLOBAL).
  2. store -> load round-trip:
       * GLOBAL columns (w_chan==0) come back BIT-EXACT.
       * LOCAL  columns (w_chan>0)  come back ZERO (Route A re-prefills them).
       * the returned re-prefill mask == (w_chan > 0), shape [L, HV, d_k].
  3. Capacity: bytes/slot < dense; report the x factor (the capacity source).
  4. GDN NON-REGRESSION: a per-head plan (dt_bias width == HV) still takes the
     old per_channel=False path and round-trips global-exact / local-zeroed —
     i.e. the KDA branch is byte-identical for GDN.

Informational (measured, not a hard gate — coupling makes a clean per-column
bound invalid, which is exactly why KDA uses Route A re-prefill, not window
replay): the assembled-state relative error when local columns are rebuilt by a
full last-W re-prefill from zero.

Run (no GPU needed):
    python test/manual/test_kda_head_aware_prefix.py                 # synthetic
    python test/manual/test_kda_head_aware_prefix.py --model ./Kimi-Linear-48B-A3B-Instruct
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re

import torch

from sglang.srt.layers.attention.fla.gdn_head_aware import gdn_gate, tau_from_g
from sglang.srt.mem_cache.mamba_checkpoint_pool import (
    HeadAwareCheckpointStore,
    HeadAwarePlan,
)


# --------------------------------------------------------------------------- #
# weights: real Kimi-Linear KDA layers, or a synthetic per-channel tau spread  #
# --------------------------------------------------------------------------- #
def _load_real_kda_weights(model_dir, d_k):
    """Return (A_log [L, HV], dt_bias [L, HV*d_k]) over all KDA layers, or None.

    Kimi-Linear stores A_log as [1,1,HV,1] (per-head scalar) and dt_bias as
    [HV*d_k] (per-channel, head-major). We stack them per layer.
    """
    from safetensors import safe_open

    a_by_prefix, dt_by_prefix = {}, {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, framework="pt") as st:
            for key in st.keys():
                if key.endswith(".A_log"):
                    a_by_prefix[key[: -len("A_log")]] = st.get_tensor(key)
                elif key.endswith(".dt_bias"):
                    dt_by_prefix[key[: -len("dt_bias")]] = st.get_tensor(key)

    def _layer_idx(k):
        m = re.search(r"layers?\.(\d+)\.", k)
        return int(m.group(1)) if m else -1

    prefixes = sorted(set(a_by_prefix) & set(dt_by_prefix), key=_layer_idx)
    # keep only the KDA (per-channel) layers: dt_bias wider than A_log's head count
    kda = []
    for p in prefixes:
        a = a_by_prefix[p].float().flatten()  # [HV]
        dt = (
            dt_by_prefix[p].float().flatten()
        )  # [HV*d_k] for KDA, [HV] for a scalar layer
        if dt.numel() == a.numel() * d_k:
            kda.append((a, dt))
    if not kda:
        return None
    A_log = torch.stack([a for a, _ in kda], dim=0)  # [L, HV]
    dt_bias = torch.stack([dt for _, dt in kda], dim=0)  # [L, HV*d_k]
    print(f"[weights] {len(kda)} KDA layers -> HV={A_log.shape[1]}, d_k={d_k}")
    return A_log, dt_bias


def _synthetic_kda_weights(L, HV, d_k, eps):
    """Per-channel tau spanning ~[2 .. 8000] within each head (a MIXED head: holds
    both fast + slow columns), so both the local and global column paths run."""
    tau_targets = torch.logspace(math.log10(2), math.log10(8000), d_k)  # [d_k]
    g = math.log(eps) / tau_targets  # [d_k], <0
    # g = -exp(A_log[h]) * softplus(dt[h,k]); pick A_log=0 -> softplus(dt) = -g
    #   softplus(dt) = -g  ->  dt = log(exp(-g) - 1)
    dt_col = torch.log(torch.clamp(torch.exp(-g) - 1.0, min=1e-12))  # [d_k]
    A_log = torch.zeros(L, HV)
    # small per-(layer,head) jitter so heads aren't identical
    jit = torch.randn(L, HV, 1) * 0.1
    dt_bias = (dt_col[None, None, :] + jit).reshape(L, HV * d_k)
    return A_log, dt_bias


# --------------------------------------------------------------------------- #
# faithful per-channel KDA recurrence (informational reconstruction check)     #
# --------------------------------------------------------------------------- #
def _kda_step(S, k, v, g, beta, l2norm_k=True):
    """One KDA recurrence step (per-channel decay). S [HV,V,K]; k [HV,K];
    v [HV,V]; g [HV,K] (per-channel log-decay); beta [HV]."""
    if l2norm_k:
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    alpha = torch.exp(g)  # [HV, K] per-channel
    S = alpha[:, None, :] * S  # decay each d_k column independently
    Sk = (S * k[:, None, :]).sum(-1)  # [HV, V]
    v = (v - Sk) * beta[:, None]
    S = S + v[:, :, None] * k[:, None, :]  # rank-1 update
    return S


def _kda_recurrence(S0, k_seq, v_seq, g_seq, beta_seq, l2norm_k=True):
    S = S0
    for t in range(k_seq.shape[0]):
        S = _kda_step(S, k_seq[t], v_seq[t], g_seq[t], beta_seq[t], l2norm_k)
    return S


def _fro(x):  # [HV,V,K] -> [HV] (or reduce over given dims)
    return torch.sqrt((x.to(torch.float64) ** 2).sum(dim=(1, 2)))


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None, help="dir with Kimi-Linear KDA safetensors")
    p.add_argument("--layers", type=int, default=3, help="L if synthetic")
    p.add_argument("--hv", type=int, default=8, help="heads if synthetic")
    p.add_argument("--k", type=int, default=128, help="d_k (head_k_dim)")
    p.add_argument("--v", type=int, default=128, help="d_v (head_v_dim)")
    p.add_argument("--prefix-len", type=int, default=1024, help="T tokens")
    p.add_argument(
        "--w-max",
        type=int,
        default=512,
        help="global/local column threshold: tau>w_max -> global (exact)",
    )
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--a-scale", type=float, default=0.3)
    p.add_argument(
        "--a-margin",
        type=float,
        default=None,
        help="classify at a=-a_margin (slower decay). Default = a_scale.",
    )
    p.add_argument("--dtype", choices=["fp32", "fp64"], default="fp32")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dt = torch.float64 if args.dtype == "fp64" else torch.float32
    K, V, T, eps = args.k, args.v, args.prefix_len, args.eps
    a_margin = args.a_scale if args.a_margin is None else args.a_margin

    if args.model:
        wts = _load_real_kda_weights(args.model, K)
        if wts is None:
            raise SystemExit("no KDA per-channel weights found in --model dir")
        A_log, dt_bias = wts
        L, HV = A_log.shape
    else:
        L, HV = args.layers, args.hv
        A_log, dt_bias = _synthetic_kda_weights(L, HV, K, eps)
        print(f"[synthetic] L={L} HV={HV} d_k={K}  per-channel tau spread ~[2, 8000]")
    A_log, dt_bias = A_log.to(dt), dt_bias.to(dt)

    dense = HeadAwareCheckpointStore.dense_bytes_per_slot(
        num_layers=L, num_heads=HV, head_v_dim=V, head_k_dim=K, state_dtype=dt
    )

    # ---- build the per-channel plan (Route A only for KDA) -------------------
    plan = HeadAwarePlan.build_plan(
        A_log=A_log,
        dt_bias=dt_bias,
        route="A",
        d_k=K,
        d_v=V,
        eps=eps,
        w_max=args.w_max,
        a_margin=a_margin,
    )
    assert plan.per_channel, "KDA dt_bias should dispatch to the per-channel plan"

    # ---- (1) classification: local iff tau <= w_max --------------------------
    dt3 = dt_bias.reshape(L, HV, K)
    ok_cls = True
    n_local = n_global = 0
    for l in range(L):
        g_repr = gdn_gate(
            torch.full((HV, K), -a_margin, dtype=dt), A_log[l][:, None], dt3[l]
        )
        tau = tau_from_g(g_repr, eps)  # [HV, K]
        want_local = torch.isfinite(tau) & (tau <= float(args.w_max))
        got_local = plan.w_chan[l] > 0
        ok_cls = ok_cls and bool((got_local == want_local).all())
        n_local += int(want_local.sum())
        n_global += int((~want_local).sum())
    tot = L * HV * K
    print(
        f"\n  (1) classification: {n_local}/{tot} local columns, "
        f"{n_global}/{tot} global (exact); GU_max={plan.GU_max}  "
        f"-> {'OK' if ok_cls else 'FAIL'}"
    )

    # ---- (2) store -> load round-trip ---------------------------------------
    store = HeadAwareCheckpointStore(
        plan=plan, num_slots=4, device="cpu", state_dtype=dt
    )
    # a random dense state per (layer, head): [L, N=1, HV, V, K]
    state = torch.randn(L, 1, HV, V, K, dtype=dt)
    slot = torch.tensor([1])
    store.store(slot, state)
    rec, mask = store.load(slot, out_dtype=dt)  # rec [L,1,HV,V,K]; mask [L,HV,K]

    is_local = plan.w_chan > 0  # [L, HV, K]
    is_global = ~is_local
    # broadcast the per-column masks to the [V] axis for gathering
    gl = is_global[:, None, :, None, :].expand_as(rec)  # global columns
    lo = is_local[:, None, :, None, :].expand_as(rec)

    max_global_err = (rec[gl] - state[gl]).abs().max().item() if gl.any() else 0.0
    max_local_abs = rec[lo].abs().max().item() if lo.any() else 0.0
    ok_global = max_global_err < 1e-6
    ok_local_zero = max_local_abs < 1e-12
    ok_mask = bool((mask.cpu() == is_local).all())
    print(
        f"  (2) round-trip: global cols max|dS|={max_global_err:.2e} (want ~0), "
        f"local cols max|val|={max_local_abs:.2e} (want 0), "
        f"mask==w_chan>0 {ok_mask}"
    )
    print(
        f"      -> global-exact {'OK' if ok_global else 'FAIL'}, "
        f"local-zeroed {'OK' if ok_local_zero else 'FAIL'}, "
        f"mask {'OK' if ok_mask else 'FAIL'}"
    )

    # ---- (3) capacity --------------------------------------------------------
    bps = store.bytes_per_slot()
    cap = dense / max(1, bps)
    print(f"  (3) capacity: bytes/slot={bps} vs dense {dense}  ({cap:.2f}x)")

    # ---- (4) GDN non-regression: per-head plan still takes the scalar path ----
    A_gdn = A_log  # [L, HV]
    dt_gdn = dt3.mean(dim=2)  # collapse to a per-head [L, HV] scalar dt_bias
    plan_g = HeadAwarePlan.build_plan(
        A_log=A_gdn,
        dt_bias=dt_gdn,
        route="A",
        d_k=K,
        d_v=V,
        eps=eps,
        w_max=args.w_max,
        a_margin=a_margin,
    )
    ok_gdn_path = not plan_g.per_channel
    store_g = HeadAwareCheckpointStore(
        plan=plan_g, num_slots=4, device="cpu", state_dtype=dt
    )
    store_g.store(slot, state)
    rec_g, mask_g = store_g.load(slot, out_dtype=dt)
    loc_h = plan_g.w_head > 0  # [L, HV]
    gl_h = (~loc_h)[:, None, :, None, None].expand_as(rec_g)
    lo_h = loc_h[:, None, :, None, None].expand_as(rec_g)
    ok_gdn_glob = (
        (rec_g[gl_h] - state[gl_h]).abs().max().item() < 1e-6 if gl_h.any() else True
    )
    ok_gdn_zero = (rec_g[lo_h].abs().max().item() < 1e-12) if lo_h.any() else True
    ok_gdn_mask = bool((mask_g.cpu() == loc_h).all())
    print(
        f"  (4) GDN non-regression: per_channel={plan_g.per_channel} (want False), "
        f"global-exact {'OK' if ok_gdn_glob else 'FAIL'}, "
        f"local-zeroed {'OK' if ok_gdn_zero else 'FAIL'}, "
        f"mask {'OK' if ok_gdn_mask else 'FAIL'}"
    )

    # ---- (informational) local-column re-prefill reconstruction --------------
    _reprefill_recon_info(args, A_log, dt3, state[:, 0], K, V, T, eps, plan, dt)

    # ---- (5) per-channel masked copy-back (Route-A seam) ---------------------
    ok_copy = _validate_copy_back(plan, L, HV, V, K, dt)
    print(f"  (5) per-channel masked copy-back: {'OK' if ok_copy else 'FAIL'}")

    all_ok = (
        ok_cls
        and ok_global
        and ok_local_zero
        and ok_mask
        and ok_gdn_path
        and ok_gdn_glob
        and ok_gdn_zero
        and ok_gdn_mask
        and ok_copy
    )
    print(f"\n  GATE-K1 RESULT: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        raise SystemExit(1)


class _StubMambaCache:
    def __init__(self, temporal):
        self.temporal = temporal


class _StubActivePool:
    """Minimal stand-in for the active MambaPool: only the two attributes that
    ``copy_local_rows_from_scratch`` touches (``mamba_cache.temporal`` +
    ``mamba_head_reprefill_mask``)."""

    def __init__(self, temporal, mask):
        self.mamba_cache = _StubMambaCache(temporal)
        self.mamba_head_reprefill_mask = mask


def _validate_copy_back(plan, L, HV, V, K, dt):
    """Drive HeadAwareCheckpointPool.copy_local_rows_from_scratch on a stub active
    pool: a scratch slot holds a full 'reconstructed' state, the active slot holds
    the checkpoint (global columns exact, local columns zero). After the masked
    copy-back the active slot must equal: global columns from the checkpoint,
    LOCAL columns from the scratch slot — and NOTHING else changed."""
    from sglang.srt.mem_cache.mamba_checkpoint_pool import HeadAwareCheckpointPool

    n_slots = 4
    temporal = torch.zeros(L, n_slots, HV, V, K, dtype=dt)
    scratch_slot, active_slot = 2, 1
    scratch_state = torch.randn(L, HV, V, K, dtype=dt)  # the re-prefill result
    ckpt_global = torch.randn(L, HV, V, K, dtype=dt)  # exact global@P
    is_local = plan.w_chan > 0  # [L, HV, K]
    lo = is_local[:, :, None, :].expand(L, HV, V, K)  # broadcast over d_v

    temporal[:, scratch_slot] = scratch_state
    # active slot = checkpoint: global cols exact, local cols zero (as load leaves).
    temporal[:, active_slot] = torch.where(
        lo, torch.zeros_like(ckpt_global), ckpt_global
    )

    active = _StubActivePool(temporal, is_local.clone())
    # call the real seam (build a bare pool object without running __init__)
    pool = HeadAwareCheckpointPool.__new__(HeadAwareCheckpointPool)
    pool.store = _StoreShim(plan.per_channel, L)
    pool.copy_local_rows_from_scratch(
        active, torch.tensor([scratch_slot]), torch.tensor([active_slot])
    )

    result = temporal[:, active_slot]
    want = torch.where(lo, scratch_state, ckpt_global)  # local<-scratch, global<-ckpt
    err = (result - want).abs().max().item()
    # scratch + other slots untouched
    untouched = (temporal[:, scratch_slot] - scratch_state).abs().max().item()
    mask_cleared = active.mamba_head_reprefill_mask is None
    ok = err < 1e-6 and untouched < 1e-12 and mask_cleared
    print(
        f"      assembled max|dS|={err:.2e} (want 0), scratch untouched "
        f"{untouched:.2e}, mask cleared {mask_cleared}"
    )
    return ok


class _StoreShim:
    """The two members copy_local_rows_from_scratch reads off ``self.store``."""

    def __init__(self, per_channel, L):
        self.per_channel = per_channel
        self.L = L

    @staticmethod
    def _as_slots(slots):
        if not torch.is_tensor(slots):
            slots = torch.as_tensor([slots], dtype=torch.int64)
        return slots.flatten()


def _reprefill_recon_info(args, A_log, dt3, state_exact, K, V, T, eps, plan, dt):
    """Measure (not gate) the assembled-state error when local columns are
    rebuilt by a full last-W re-prefill from zero, global columns kept from the
    checkpoint. Coupling (the delta rule mixes columns) forbids a clean per-column
    bound -> Route A re-prefills the WHOLE window through the real kernel; this is
    why KDA cannot use per-column window replay. We just report the actual error.
    """
    W = int(args.w_max)
    L, HV = A_log.shape
    if W <= 0 or W > T:
        print(f"\n  (info) re-prefill recon: skipped (W={W}, T={T})")
        return
    torch.manual_seed(args.seed + 1)
    a_scale = args.a_scale
    print(f"\n  (info) local-column re-prefill recon (uniform W={W}, from zero):")
    max_rel = 0.0
    for l in range(L):
        # synth a per-token stream for this layer, roll the EXACT recurrence,
        # then re-prefill the last W tokens from zero and assemble.
        a_seq = torch.randn(T, HV, K, dtype=dt) * a_scale
        b_seq = torch.randn(T, HV, dtype=dt)
        k_seq = torch.randn(T, HV, K, dtype=dt)
        v_seq = torch.randn(T, HV, V, dtype=dt)
        g_seq = gdn_gate(a_seq, A_log[l][:, None], dt3[l])  # [T, HV, K]
        beta_seq = torch.sigmoid(b_seq)
        S_exact = _kda_recurrence(
            torch.zeros(HV, V, K, dtype=dt), k_seq, v_seq, g_seq, beta_seq
        )
        sl = slice(T - W, T)
        S_scratch = _kda_recurrence(
            torch.zeros(HV, V, K, dtype=dt),
            k_seq[sl],
            v_seq[sl],
            g_seq[sl],
            beta_seq[sl],
        )
        # assemble: global columns from exact (== checkpoint), local from scratch
        local = plan.w_chan[l] > 0  # [HV, K]
        S_asm = torch.where(local[:, None, :], S_scratch, S_exact)
        dS = _fro(S_asm - S_exact)
        scale = _fro(S_exact).clamp(min=1e-12)
        max_rel = max(max_rel, float((dS / scale).max()))
    print(
        f"    max assembled relative error over {L} layers = {max_rel:.3e}  "
        f"(informational; Route A re-prefill accuracy is measured on-model)"
    )


if __name__ == "__main__":
    main()
