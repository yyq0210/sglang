"""EXPERIMENT-ONLY live fold-to-state KV eviction for hybrid full-attn layers (Phase C2-b).

Part of the "State as a Memory Tier" study
(``docs/hybrid_state_tiering_experiment_plan.md``, Phase C2). C0 proved OFFLINE that
folding an evicted middle KV segment into a q-independent linear state beats dropping
it at small exact-KV budget; C1 proved the model's own gate keeps deep content alive.
C2-a turned the fold math into a GPU decode primitive and unit-tested it against the
C0 reference (test/manual/{fold_decode_kernel,test_fold_decode_gt}.py). C2-b applies
that primitive LIVE on Qwen3-Next-80B: at DECODE, each full-attention layer's output
is replaced by

    o = merge( windowed_softmax(q, K[sink]+K[recent window], V...) ,   # exact
               state_readout(q, S, z) )                                  # folded middle

where (S,z) is the q-independent fold of the middle segment [sink, n-W):

    S = sum_i g_i phi(k_i) v_i^T   [d, dv],   z = sum_i g_i phi(k_i)   [d]   (df=d)

with fixed exact-KV budget B = |sink| + |window| = FOLD_KV_BUDGET.

Design (deliberately low-risk, faithful accuracy, projected capacity):
  * DECODE ONLY. Extend/prefill are left byte-identical (build the real KV + hidden
    states). Fold only rewrites the decode-step full-attn output.
  * STATELESS recompute: the middle is re-read + re-folded from the live KV pool each
    step (no cross-step bookkeeping, no reset bugs, no scheduler coupling). The pool is
    read-only here; NO pages are freed (that would desync scheduler accounting). The
    O(B) decode-latency win and the max-batch/capacity win are therefore PROJECTED from
    the C2-a microbench + the exact-KV footprint (B vs N), not measured by page release.
  * per-request: only requests with seq_len > B are folded; short requests keep dense.
  * GDN (linear) layers are untouched -- their native recurrence is the linear tier.

Everything is gated by the BARE env var FOLD_KV_BUDGET (unset -> no-op, zero hot-path
cost), matching the state_ablation / tier_capture local-experiment style. This is
INSTRUMENTATION, not a product feature -- do NOT wire it into any release/default path
or commit it as an enabled feature. Requires --disable-cuda-graph (data-dependent,
host-synced) and --disable-overlap-schedule, as in the prior phases.

  FOLD_KV_BUDGET   int  exact-KV budget B = sink + window (unset -> disabled)
  FOLD_KV_SINK     int  number of exact sink tokens kept (default 4)
  FOLD_KV_LAYERS   csv  full-attn layer-ids to fold (default "all")
  FOLD_KV_PHI      str  feature map: elu1 (default) | l2norm
  FOLD_KV_DECAY    str  fixed decay in (0,1], "1.0" = equal-weight sum (default "1.0")
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Optional, Set

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _read_layer_set(var: str):
    raw = os.environ.get(var, "all").strip().lower()
    if raw in ("", "all"):
        return None
    return frozenset(int(x) for x in raw.replace(" ", "").split(",") if x != "")


# Env is frozen for the process lifetime (the launch driver sets it before start).
_BUDGET_RAW = os.environ.get("FOLD_KV_BUDGET")
_ON = _BUDGET_RAW is not None and _BUDGET_RAW.strip() != ""
_BUDGET = int(_BUDGET_RAW) if _ON else 0
_SINK = int(os.environ.get("FOLD_KV_SINK", "4"))
_WINDOW = max(0, _BUDGET - _SINK)
_LAYERS = _read_layer_set("FOLD_KV_LAYERS")
_PHI = os.environ.get("FOLD_KV_PHI", "elu1").strip().lower()
_DECAY = float(os.environ.get("FOLD_KV_DECAY", "1.0"))

_DONE_LOG: Set[int] = set()


def enabled() -> bool:
    return _ON


def config_str() -> str:
    return (
        f"FOLD_KV budget={_BUDGET} sink={_SINK} window={_WINDOW} "
        f"layers={'all' if _LAYERS is None else sorted(_LAYERS)} "
        f"phi={_PHI} decay={_DECAY}"
    )


def should_fold_layer(layer_id: int) -> bool:
    if not _ON:
        return False
    return _LAYERS is None or layer_id in _LAYERS


# --------------------------------------------------------------------------- #
# feature maps (df=d) -- MUST match fold_vs_drop_c0 / fold_decode_kernel exactly.
# --------------------------------------------------------------------------- #
def _phi(x: torch.Tensor, scale: float) -> torch.Tensor:
    if _PHI == "elu1":
        return F.elu(math.sqrt(scale) * x) + 1.0
    if _PHI == "l2norm":
        return F.normalize(x, dim=-1)
    raise ValueError(f"FOLD_KV_PHI={_PHI!r} not in (elu1, l2norm)")


def _expand_kv(t: torch.Tensor, group: int) -> torch.Tensor:
    """[T, Hkv, D] -> [T, Hq, D] by repeating each kv head ``group`` times (GQA)."""
    if group == 1:
        return t
    T, Hkv, D = t.shape
    return t[:, :, None, :].expand(T, Hkv, group, D).reshape(T, Hkv * group, D)


def _fold_decode_one(
    q_r: torch.Tensor,      # [Hq, D]  post-rope current query
    k_exact: torch.Tensor,  # [B, Hkv, D]
    v_exact: torch.Tensor,  # [B, Hkv, Dv]
    k_mid: torch.Tensor,    # [M, Hkv, D]
    v_mid: torch.Tensor,    # [M, Hkv, Dv]
    scale: float,
) -> torch.Tensor:
    """One request's fused fold-decode -> [Hq, Dv] fp32. Mirrors fold_decode_ref."""
    Hq, D = q_r.shape
    Hkv = k_exact.shape[1]
    group = Hq // Hkv
    q_r = q_r.float()
    k_exact = _expand_kv(k_exact.float(), group)  # [B, Hq, D]
    v_exact = _expand_kv(v_exact.float(), group)  # [B, Hq, Dv]

    # --- exact window substrate: softmax over [sink + recent window] ---
    logits = scale * torch.einsum("hd,bhd->hb", q_r, k_exact)  # [Hq, B]
    m = logits.max(dim=-1, keepdim=True).values
    p = torch.exp(logits - m)  # [Hq, B]
    l = p.sum(dim=-1)  # [Hq]
    o_win = torch.einsum("hb,bhv->hv", p, v_exact) / l.unsqueeze(-1)  # [Hq, Dv]
    lse_win = m.squeeze(-1) + torch.log(l)  # [Hq]

    # --- folded-middle substrate: build (S,z), read back ---
    M = k_mid.shape[0]
    k_mid = _expand_kv(k_mid.float(), group)  # [M, Hq, D]
    v_mid = _expand_kv(v_mid.float(), group)  # [M, Hq, Dv]
    pk = _phi(k_mid, scale)  # [M, Hq, D]
    if _DECAY != 1.0:
        ages = torch.arange(M - 1, -1, -1, device=q_r.device, dtype=torch.float32)
        g = (_DECAY**ages).view(M, 1, 1)
        pk = g * pk
    S = torch.einsum("mhd,mhv->hdv", pk, v_mid)  # [Hq, D, Dv]
    z = pk.sum(dim=0)  # [Hq, D]
    pq = _phi(q_r, scale)  # [Hq, D]
    N = torch.einsum("hd,hdv->hv", pq, S)  # [Hq, Dv]
    Dnum = torch.einsum("hd,hd->h", pq, z).clamp_min(1e-20)  # [Hq]
    o_state = N / Dnum.unsqueeze(-1)  # [Hq, Dv]
    lse_state = torch.log(Dnum)  # [Hq]

    # --- LSE merge (natural log, absolute exp frame) == merge_state semantics ---
    mmax = torch.maximum(lse_win, lse_state)
    sw = torch.exp(lse_win - mmax)
    ss = torch.exp(lse_state - mmax)
    o = (o_win * sw.unsqueeze(-1) + o_state * ss.unsqueeze(-1)) / (sw + ss).unsqueeze(-1)
    return o  # [Hq, Dv]


@torch.no_grad()
def apply_fold_decode(
    out: torch.Tensor,
    q: torch.Tensor,
    layer,
    forward_batch: "ForwardBatch",
    token_to_kv_pool,
    req_to_token_pool,
) -> torch.Tensor:
    """Overwrite the full-attn DECODE output with the fold-decode, per request.

    ``out`` is [num_tokens, Hq*Dv] (as returned by the triton full-attn decode). For a
    decode step num_tokens == bs (one query token per request). Requests with
    seq_len <= B are left as the dense output. The KV pool is READ-ONLY here.
    """
    if not _ON:
        return out
    if not forward_batch.forward_mode.is_decode():
        return out
    if not should_fold_layer(layer.layer_id):
        return out

    seq_lens = forward_batch.seq_lens.tolist()
    req_idx = forward_batch.req_pool_indices.tolist()
    # any request over budget?
    if not any(n > _BUDGET for n in seq_lens):
        return out

    Hq = layer.tp_q_head_num
    D = layer.qk_head_dim
    Dv = layer.v_head_dim
    scale = layer.scaling
    req_to_token = req_to_token_pool.req_to_token
    kbuf = token_to_kv_pool.get_key_buffer(layer.layer_id)  # [slots, Hkv, D]
    vbuf = token_to_kv_pool.get_value_buffer(layer.layer_id)  # [slots, Hkv, Dv]

    q3 = q.reshape(-1, Hq, D)  # [bs, Hq, D]

    if layer.layer_id not in _DONE_LOG:
        n_over = sum(1 for n in seq_lens if n > _BUDGET)
        print(
            f"[fold_evict] layer={layer.layer_id} decode bs={len(seq_lens)} "
            f"n_over_budget={n_over} {config_str()}",
            flush=True,
        )
        _DONE_LOG.add(layer.layer_id)

    for i, (r, n) in enumerate(zip(req_idx, seq_lens)):
        if n <= _BUDGET:
            continue
        win_lo = n - _WINDOW
        sink_locs = req_to_token[r, :_SINK]
        win_locs = req_to_token[r, win_lo:n]
        mid_locs = req_to_token[r, _SINK:win_lo]
        exact_locs = torch.cat([sink_locs, win_locs])
        o_r = _fold_decode_one(
            q3[i],
            kbuf[exact_locs],
            vbuf[exact_locs],
            kbuf[mid_locs],
            vbuf[mid_locs],
            scale,
        )
        out[i] = o_r.reshape(Hq * Dv).to(out.dtype)
    return out
