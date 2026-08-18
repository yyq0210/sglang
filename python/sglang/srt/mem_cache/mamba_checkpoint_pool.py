"""
Copyright 2023-2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

MambaCheckpointPool — the radix prefix cache's int8-compressed store for cached
linear-attention (KDA / GDN / Mamba2 gated-delta-rule) recurrent states.

It decouples the *cached* states (radix-owned, idle, compressed) from the *active*
``MambaPool`` (running requests, full precision, kernel-facing). The radix stores
one cached state per node HERE; on a prefix-cache hit it is dequantized back into
a fresh active slot (copy-on-write).

Per cached slot it holds:
  * the SSM temporal state in **int8** (per-(head,k-channel) symmetric), via the
    embedded ``Int8CheckpointStore`` — ~2x more cached states than bf16,
    quality-safe (quantized once on store, dequantized once on a hit; never
    re-enters the recurrence as a quant->dequant loop).
  * the conv1d window state at its native dtype (tiny, W-1 tokens; not worth
    quantizing).

Why int8 (not fp8): a cached checkpoint is loaded ONCE on a cache hit, then
decoding continues at full precision, so the only error is a single rounding of
S. The temporal state is roughly uniformly distributed, so int8-per-(head,
k-channel) beats fp8-e4m3 at the same 1 byte (fp8 wastes bits on the exponent).
The scale axis (reduces over d_v) matches the per-k-channel decay diag(alpha), so
the large state entries keep ~bf16 precision and the error concentrates on small
entries that barely affect the readout. Storing cached states int8 gives ~2x the
cached-prefix capacity at fixed memory, and composes with host-offload
(HiMambaRadixCache) which it also halves.

This is strategy-agnostic: whether the active slot to be cached was produced by
the ``no_buffer`` donate (copy_from) or the ``extra_buffer`` ping-pong track
buffer (spec path), both converge on "an active slot becomes the cached
``mamba_value``" — which is exactly the (store_from_active) hook here. Slot
lifecycle is owned by the caller via the embedded ``MambaSlotAllocator``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint-load instrumentation (SGLANG_LOG_CKPT_LOAD). A DIRECT count of the
# compressed-restore path (load_to_active): every prefix-cache hit that copies a
# cached mamba checkpoint back into the active pool bumps these process-level
# counters. Purpose: prove the compression/restore path is actually exercised
# (not bypassed) during an accuracy A/B, rather than inferring it from the token-
# level radix hit-rate. ``dropped_units`` is the Route-A count of local (head or
# per-channel) units the checkpoint did NOT store exact (0 for the dense W_max=0
# baseline; >0 == head-aware compression provably ran). Cheap: gated by one bool.
_CKPT_LOAD_STATS = {"calls": 0, "slots": 0, "dropped_units": 0}


def _ckpt_load_enabled() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_LOG_CKPT_LOAD.get())


def _record_ckpt_load(n_slots: int, dropped_units: int, kind: str) -> None:
    """Bump the checkpoint-load counters and log this hit (only when enabled)."""
    _CKPT_LOAD_STATS["calls"] += 1
    _CKPT_LOAD_STATS["slots"] += int(n_slots)
    _CKPT_LOAD_STATS["dropped_units"] += int(dropped_units)
    logger.info(
        "[ckpt-load] %s restore: slots=%d dropped_units_this_hit=%d "
        "| cumulative calls=%d slots=%d dropped_units=%d",
        kind,
        n_slots,
        dropped_units,
        _CKPT_LOAD_STATS["calls"],
        _CKPT_LOAD_STATS["slots"],
        _CKPT_LOAD_STATS["dropped_units"],
    )


# ---------------------------------------------------------------------------
# Random-drop ABLATION (SGLANG_HEAD_AWARE_RANDOM_DROP). Replaces the decay-aware
# (tau) global/local split with a COUNT-MATCHED RANDOM one: per layer keep EXACTLY
# the same number of local (dropped) units as the tau plan, but choose WHICH units
# randomly. Same compression ratio / capacity bytes -> isolates the value of the
# decay-aware selection from the value of merely dropping that many units. Off
# (default) -> tau plan, byte-identical.
def _random_drop_enabled() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_HEAD_AWARE_RANDOM_DROP.get())


def _randomize_local_positions(w: torch.Tensor) -> torch.Tensor:
    """Per-layer, keep #local (nonzero W) IDENTICAL but move the local units to
    RANDOM positions, preserving the multiset of W values. ``w`` is [L, ...] int64
    (0 == global/kept-exact, >0 == local/dropped with window W). Compression ratio
    (n_local per layer) is unchanged; only the *identity* of dropped units changes.
    Seeded by SGLANG_HEAD_AWARE_RANDOM_SEED for reproducibility.
    """
    from sglang.srt.environ import envs

    seed = int(envs.SGLANG_HEAD_AWARE_RANDOM_SEED.get())
    L = w.shape[0]
    flat = w.reshape(L, -1)
    N = flat.shape[1]
    out = torch.zeros_like(flat)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    for l in range(L):
        vals = flat[l]
        nz = vals[vals > 0]  # multiset of W values; count == n_local for this layer
        perm = torch.randperm(N, generator=g)  # advance generator every layer
        if nz.numel() == 0:
            continue
        out[l, perm[: nz.numel()]] = nz  # scatter the same W multiset randomly
    return out.reshape_as(w)


class Int8CheckpointStore:
    """int8/int4 store for cached multi-layer linear-attn states.

    Tensors (slot index handed out by the caller's allocator):
        qdata : [L, num_slots, H, d_v,     d_k]  int8   (INT8 mode)
        qdata : [L, num_slots, H, d_v // 2, d_k]  int8   (INT4 mode, packed 2-per-byte)
        scale : [L, num_slots, H, 1,   d_k]  scale_dtype  (per layer,slot,head,k-chan)

    A "state" spans all L mamba layers for one cached point (matching how the
    radix caches one full state per node). The reduction axis for the scale is
    d_v (dim=-2), so each (head, k-channel) gets its own scale — aligned with the
    per-k-channel decay diag(alpha).

    ``scale_dtype`` should match the source state's dtype (bf16 / fp16 / fp32) so
    that quantize and dequantize use the identical scale — it is NOT required to
    be bf16.

    ``quant_mode``: "int8" (default, QMAX=127) or "int4" (QMAX=7, packed 2 values
    per int8 byte along d_v).  INT4 gives ~3.88× compression (vs INT8 ~1.97×)
    with group_size=d_v=128.  Experiment-only; controlled by
    ``SGLANG_STATE_QUANT_MODE`` env var at pool init time.
    """

    QMAX = 127  # INT8 default; overridden for INT4 in __init__

    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        device: str,
        scale_dtype: torch.dtype = torch.bfloat16,
        quant_mode: str = "int8",
    ):
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.H = num_heads
        self.d_v = head_v_dim
        self.d_k = head_k_dim
        self.device = device
        self.quant_mode = quant_mode
        self.packed = quant_mode == "int4"
        self.qmax = 7 if self.packed else 127
        # d_v storage dim: packed INT4 stores d_v//2 int8 bytes (2 values per byte)
        d_v_storage = head_v_dim // 2 if self.packed else head_v_dim
        self.qdata = torch.empty(
            num_layers,
            num_slots,
            num_heads,
            d_v_storage,
            head_k_dim,
            dtype=torch.int8,
            device=device,
        )
        self.scale = torch.empty(
            num_layers,
            num_slots,
            num_heads,
            1,
            head_k_dim,
            dtype=scale_dtype,
            device=device,
        )

    # ---- (de)quant math (also usable standalone for probes/tests) ----

    @classmethod
    def quantize(cls, state: torch.Tensor):
        """state [..., H, d_v, d_k] -> (qint8, scale[..., H, 1, d_k]).

        amax / scale / round are computed in float32 so quantizing a low-precision
        state doesn't lose precision in the intermediate (symmetric with
        ``dequantize``, which is already float32). The scale is rounded to the
        state dtype (its storage precision) BEFORE the division, so quantize and
        dequantize use the identical scale."""
        state_fp32 = state.to(torch.float32)
        amax = state_fp32.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
        scale = (amax / cls.QMAX).to(state.dtype)
        q = (
            torch.round(state_fp32 / scale.to(torch.float32))
            .clamp(-cls.QMAX, cls.QMAX)
            .to(torch.int8)
        )
        return q, scale

    def quantize_mode(self, state: torch.Tensor):
        """Per-mode quantize. Returns (qdata int8, scale) in the storage layout."""
        if not self.packed:
            return self.quantize(state)
        # INT4: quantize with QMAX=7, then pack 2 values per byte along d_v
        state_fp32 = state.to(torch.float32)
        amax = state_fp32.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
        scale = (amax / self.qmax).to(state.dtype)
        q = (
            torch.round(state_fp32 / scale.to(torch.float32))
            .clamp(-self.qmax, self.qmax)
            .to(torch.int8)
        )
        # pack: q [..., H, d_v, d_k] -> [..., H, d_v//2, d_k] int8
        # pair along d_v: even indices in low nibble, odd in high nibble
        q_even = q[..., 0::2, :] & 0x0F
        q_odd = q[..., 1::2, :] & 0x0F
        packed = q_even | (q_odd << 4)
        return packed, scale

    @staticmethod
    def dequantize(q: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype):
        return (q.to(torch.float32) * scale.to(torch.float32)).to(out_dtype)

    def dequantize_mode(
        self, q: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
    ):
        """Per-mode dequantize. q is in the storage layout."""
        if not self.packed:
            return self.dequantize(q, scale, out_dtype)
        # unpack: q [..., H, d_v//2, d_k] -> [..., H, d_v, d_k]
        lo = q & 0x0F
        hi = (q >> 4) & 0x0F
        lo = torch.where(lo > 7, lo - 16, lo)
        hi = torch.where(hi > 7, hi - 16, hi)
        q = torch.stack([lo, hi], dim=-2).flatten(-2, -2)
        return (q.to(torch.float32) * scale.to(torch.float32)).to(out_dtype)

    # ---- store / load (caller supplies slot indices) ----

    def store(self, slots: torch.Tensor, state: torch.Tensor) -> None:
        """Quantize and write states. state: [L, N, H, d_v, d_k] for the N slots
        (or [L, H, d_v, d_k] when slots is a scalar/len-1)."""
        if state.dim() == 4:
            state = state.unsqueeze(1)
        q, scale = self.quantize_mode(state)
        self.qdata[:, slots] = q
        self.scale[:, slots] = scale.to(self.scale.dtype)

    def load(self, slots: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """Dequantize states at slots -> [L, N, H, d_v, d_k] in out_dtype."""
        return self.dequantize_mode(
            self.qdata[:, slots], self.scale[:, slots], out_dtype
        )

    def copy_to_pool(
        self,
        dst_temporal: torch.Tensor,
        src_slots: torch.Tensor,
        dst_slots: torch.Tensor,
    ) -> None:
        """Dequantize checkpoints at ``src_slots`` directly into the active pool
        tensor ``dst_temporal`` [L, pool_slots, H, d_v, d_k] at ``dst_slots`` (the
        copy-on-write on a cache hit). Output dtype follows ``dst_temporal``."""
        dst_temporal[:, dst_slots] = self.load(src_slots, dst_temporal.dtype)

    def store_from_pool(
        self,
        src_temporal: torch.Tensor,
        src_slots: torch.Tensor,
        dst_slots: torch.Tensor,
    ) -> None:
        """Quantize states from an active pool tensor into checkpoint slots (cache
        store / donate)."""
        self.store(dst_slots, src_temporal[:, src_slots])

    def mem_usage_bytes(self) -> int:
        return (
            self.qdata.numel() * self.qdata.element_size()
            + self.scale.numel() * self.scale.element_size()
        )

    def bytes_per_slot(self) -> int:
        return self.mem_usage_bytes() // max(1, self.num_slots)


class MambaCheckpointPool:
    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_shapes: List[tuple],
        conv_dtype: torch.dtype,
        device: str,
        temporal_dtype: Optional[torch.dtype] = None,
    ):
        self.num_slots = num_slots
        self.device = device
        self.temporal = Int8CheckpointStore(
            num_layers=num_layers,
            num_slots=num_slots + 1,  # slot 0 reserved (matches MambaSlotAllocator)
            num_heads=num_heads,
            head_v_dim=head_v_dim,
            head_k_dim=head_k_dim,
            device=device,
            # store the scale in the temporal state's own dtype so quantize and
            # dequantize use the identical scale (not hard-coded to bf16)
            scale_dtype=(
                temporal_dtype if temporal_dtype is not None else torch.bfloat16
            ),
        )
        # conv windows stay at their native dtype (small); one buffer per conv
        # tensor in the State
        self.conv = [
            torch.empty(
                (num_layers, num_slots + 1) + tuple(shape),
                dtype=conv_dtype,
                device=device,
            )
            for shape in conv_shapes
        ]
        self.allocator = MambaSlotAllocator(size=num_slots, device=device)

    # ---- lifecycle (delegates to the embedded allocator) ----

    def alloc(self, n: int = 1):
        return self.allocator.alloc(n)

    def free(self, slots: torch.Tensor):
        self.allocator.free(slots)

    def available_size(self) -> int:
        return self.allocator.available_size()

    def clear(self) -> None:
        """Release every checkpoint slot (radix flush/reset). The int8 qdata is
        left as-is; slots are reused/overwritten on the next store."""
        self.allocator.clear()

    # ---- state transfer between the active MambaPool and this store ----

    def store_from_active(self, active_mamba_pool, active_slots, ckpt_slots) -> None:
        """Quantize temporal + copy conv from the active pool into checkpoint slots
        (the radix donate / cache-store)."""
        cache = active_mamba_pool.mamba_cache
        self.temporal.store_from_pool(cache.temporal, active_slots, ckpt_slots)
        for i, c in enumerate(self.conv):
            c[:, ckpt_slots] = cache.conv[i][:, active_slots]

    def load_to_active(self, active_mamba_pool, ckpt_slots, active_slots) -> None:
        """Dequantize temporal + copy conv from checkpoint slots into the active pool
        (the cache-hit copy-on-write)."""
        cache = active_mamba_pool.mamba_cache
        self.temporal.copy_to_pool(cache.temporal, ckpt_slots, active_slots)
        for i, c in enumerate(self.conv):
            cache.conv[i][:, active_slots] = c[:, ckpt_slots].to(cache.conv[i].dtype)
        if _ckpt_load_enabled():
            # int8 is a lossy-quant, not a unit-drop store -> dropped_units=0.
            _record_ckpt_load(int(ckpt_slots.numel()), 0, kind="int8")

    @staticmethod
    def estimate_mem_usage_bytes(
        *,
        num_layers: int,
        num_slots: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_shapes: List[tuple],
        conv_dtype: torch.dtype,
        temporal_dtype: torch.dtype,
    ) -> dict:
        """Estimate the pool's HBM footprint (bytes) WITHOUT allocating, so a
        caller can check it against free memory before construction. Mirrors the
        real layout: int8 qdata + per-(head,k) scale + bf16 conv windows, including
        the reserved slot 0."""
        slots = num_slots + 1  # slot 0 reserved (matches MambaSlotAllocator)
        scale_isz = torch.empty((), dtype=temporal_dtype).element_size()
        conv_isz = torch.empty((), dtype=conv_dtype).element_size()
        qdata = num_layers * slots * num_heads * head_v_dim * head_k_dim  # int8 = 1B
        scale = num_layers * slots * num_heads * head_k_dim * scale_isz
        conv = 0
        for shape in conv_shapes:
            n = 1
            for s in shape:
                n *= int(s)
            conv += num_layers * slots * n * conv_isz
        return {
            "qdata": qdata,
            "scale": scale,
            "conv": conv,
            "total": qdata + scale + conv,
        }

    def mem_usage_bytes(self) -> int:
        conv_bytes = sum(c.numel() * c.element_size() for c in self.conv)
        return self.temporal.mem_usage_bytes() + conv_bytes


def maybe_init_int8_mamba_checkpoint_pool(
    *,
    mamba_size: int,
    cache_params,
    mamba_layer_ids: List[int],
    device: str,
) -> Optional[MambaCheckpointPool]:
    """Build the optional int8 ``MambaCheckpointPool`` when
    ``--enable-int8-mamba-checkpoint`` is set (and a global server-args context
    exists), else return ``None``. The radix caches states here (int8) instead of
    in the active bf16 pool -> ~2x cached-prefix capacity at fixed memory.

    Estimates the pool's HBM footprint and checks it against free memory BEFORE
    allocating, so an oversized ``--int8-mamba-ckpt-size`` fails with an actionable
    message instead of a cryptic mid-allocation CUDA OOM.
    """
    from sglang.srt.server_args import get_global_server_args

    try:
        _sa = get_global_server_args()
    except ValueError:
        # Some unit-test / mock runners construct HybridReqToTokenPool directly
        # without a global server-args context. The int8 checkpoint pool is opt-in
        # via a CLI flag, so an unset context unambiguously means it is off.
        _sa = None
    if not getattr(_sa, "enable_int8_mamba_checkpoint", False):
        return None

    GB = 1 << 30
    H, d_v, d_k = cache_params.shape.temporal
    ckpt_size = _sa.int8_mamba_ckpt_size or (2 * mamba_size)
    kwargs = dict(
        num_layers=len(mamba_layer_ids),
        num_slots=ckpt_size,
        num_heads=H,
        head_v_dim=d_v,
        head_k_dim=d_k,
        conv_shapes=list(cache_params.shape.conv),
        conv_dtype=cache_params.dtype.conv,
        temporal_dtype=cache_params.dtype.temporal,
    )

    est = MambaCheckpointPool.estimate_mem_usage_bytes(**kwargs)
    free_bytes = None
    if isinstance(device, str) and device.startswith("cuda"):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except Exception:
            free_bytes = None
    logger.info(
        f"int8 mamba checkpoint pool: {ckpt_size} slots, "
        f"{est['total'] / GB:.2f}GB (qdata {est['qdata'] / GB:.2f} + scale "
        f"{est['scale'] / GB:.2f} + conv {est['conv'] / GB:.2f}); active mamba "
        f"pool {mamba_size} slots"
        + (f"; free HBM {free_bytes / GB:.2f}GB" if free_bytes is not None else "")
    )
    if free_bytes is not None and est["total"] >= free_bytes:
        raise RuntimeError(
            f"int8 mamba checkpoint pool needs ~{est['total'] / GB:.2f}GB but only "
            f"{free_bytes / GB:.2f}GB HBM is free. Lower --int8-mamba-ckpt-size "
            f"(currently {ckpt_size}) or --mem-fraction-static."
        )

    pool = MambaCheckpointPool(device=device, **kwargs)
    # NOTE: this pool's HBM is NOT subtracted from the KV-cache budget
    # (max_total_num_tokens); it is allocated from --mem-fraction-static headroom.
    # The estimate check above guards against an oversized pool; accounting it in
    # the KV budget is a follow-up.
    logger.warning(
        f"int8 mamba checkpoint pool ({est['total'] / GB:.2f}GB) is allocated from "
        f"--mem-fraction-static headroom and is not reflected in "
        f"max_total_num_tokens; ensure headroom covers it."
    )
    return pool


# ---------------------------------------------------------------------------
# Head-aware checkpoint store (Route A)
# ---------------------------------------------------------------------------
#
# Where int8 shrinks EVERY cached (head, k-channel) uniformly to 1 byte, the
# head-aware store shrinks by DROPPING whole heads whose decay tau does not
# outlive the reconstruction window W:
#
#   * global head (tau_h > W): memory outlives the window -> store the EXACT
#     [d_v, d_k] state (what the radix caches today).
#   * local head  (tau_h <= W): the state is ~the contribution of the last W
#     tokens -> do NOT store the [d_v, d_k] state. On a hit re-prefill the
#     last-W prefix tokens through the full model (scheduler-side; the store
#     leaves local rows zeroed and reports which heads need re-prefill).
#
# The plan (which heads are global) is per GDN LAYER (each layer has its own
# A_log / dt_bias), so the packed layout pads to G_max = max over layers of the
# global-head count and keeps a per-layer head-id map. bytes/slot < the dense
# [L, HV, d_v, d_k] baseline -> more checkpoint slots at fixed HBM == the real
# capacity gain this experiment measures.

try:
    import msgspec

    _StructBase = msgspec.Struct
except Exception:  # msgspec is a hard dep in sglang; guard only for odd envs
    _StructBase = object


class HeadAwarePlan(_StructBase):
    """Static per-(layer, head) classification for the head-aware store.

    Built ONCE from the model's stacked GDN weights (``build_plan``).

    Fields:
      route       : "A".
      w_head      : [L, HV] int64 — 0 = global (store exact state); W>0 = local.
      global_idx  : [L, G_max] int64 — head-id of each packed global row (-1 pad).
      local_idx   : [L, Lwin_max] int64 — head-id of each local row (-1 pad).
      local_w     : [L, Lwin_max] int64 — per-local window length W (0 pad).
      G_max, Lwin_max, W_max : packed dimensions (max over layers).
      HV, d_v, d_k : per-head state dims.
    """

    route: str
    w_head: torch.Tensor
    global_idx: torch.Tensor
    local_idx: torch.Tensor
    local_w: torch.Tensor
    G_max: int
    Lwin_max: int
    W_max: int
    HV: int
    d_v: int
    d_k: int
    # ---- KDA per-channel extension (all defaulted -> GDN construction unchanged) ----
    # KDA decay is per (head, d_k-channel), so the keep/drop unit is a single
    # d_k COLUMN of a head's [d_v, d_k] state (a d_v-vector), not a whole head row.
    # When ``per_channel`` is set the per-head fields above (w_head/global_idx/...)
    # are left as GDN-shaped placeholders and these carry the real plan:
    #   w_chan   : [L, HV, d_k] int64 — 0 = global column (store exact); W>0 = local.
    #   global_hk: [L, GU_max, 2] int64 — (head, col) of each packed global unit
    #              (-1 pad). GU_max = max over layers of #global columns.
    per_channel: bool = False
    w_chan: Optional[torch.Tensor] = None
    global_hk: Optional[torch.Tensor] = None
    GU_max: int = 0

    @classmethod
    def build_plan(
        cls,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        route: str,
        d_k: int,
        d_v: int,
        eps: float = 1e-3,
        w_max: int = 16,
        w_min: int = 4,
        a_margin: float = 0.3,
    ) -> HeadAwarePlan:
        """Compute the per-layer window plan from stacked GDN weights.

        A_log / dt_bias: [L, HV] (per layer, per v-head; the static decay weights
        read from each GDN layer, e.g. Qwen3-Next ``*.A_log`` / ``*.dt_bias``).
        Classify at a slightly NEGATIVE ``a`` (= -a_margin, slower decay) so the
        window covers real per-token a<0 fluctuations (mirrors the offline test).

        KDA dispatch: when ``dt_bias`` is PER-CHANNEL ([L, HV*d_k], i.e. wider than
        ``A_log`` [L, HV]) the decay is elementwise per (head, d_k-channel), so we
        build a per-COLUMN plan (``build_plan_per_channel``) instead of the per-head
        GDN plan. GDN (dt_bias [L, HV], shape-equal) keeps the exact path below.
        """
        from sglang.srt.layers.attention.fla.gdn_head_aware import (
            _next_pow2,
            gdn_gate,
            tau_from_g,
        )

        assert route == "A", f"head_aware_route must be A, got {route!r}"
        assert (
            A_log.dim() == 2
            and dt_bias.dim() == 2
            and A_log.shape[0] == dt_bias.shape[0]
        )
        L, HV = A_log.shape
        if dt_bias.shape[1] != HV:
            # KDA per-channel: dt_bias is [L, HV*d_k]. The per-column decay is
            # elementwise (delta-rule couples columns) -> per-channel plan.
            assert (
                dt_bias.shape[1] % HV == 0
            ), f"KDA dt_bias width {dt_bias.shape[1]} not a multiple of HV={HV}"
            K = dt_bias.shape[1] // HV
            assert K == d_k, f"KDA per-channel expects d_k={d_k} columns, got {K}"
            assert route == "A", "KDA per-channel head-aware supports Route A only"
            return cls.build_plan_per_channel(
                A_log=A_log,
                dt_bias=dt_bias.reshape(L, HV, K),
                d_k=d_k,
                d_v=d_v,
                eps=eps,
                w_max=w_max,
                w_min=w_min,
                a_margin=a_margin,
            )
        assert A_log.shape == dt_bias.shape
        # the plan is a one-time static computation; do it on CPU so it never
        # depends on the caller's device (build_plan tensors are CPU scratch).
        A_log = A_log.detach().cpu().to(torch.float64)
        dt_bias = dt_bias.detach().cpu().to(torch.float64)

        w_rows: List[torch.Tensor] = []
        for l in range(L):
            g_repr = gdn_gate(
                torch.full((HV,), -a_margin, dtype=torch.float64),
                A_log[l],
                dt_bias[l],
            )
            # Route A stores NOTHING for a local head (re-prefilled on a hit),
            # so a dropped head always costs 0 bytes -> no break-even gate: drop
            # EVERY head whose memory is coverable by a re-prefill window
            # (tau <= w_max). w_max is the global/local threshold knob (heads
            # with tau > w_max keep the exact state).
            tau = tau_from_g(g_repr, eps)
            w = torch.zeros(HV, dtype=torch.int64)
            local = torch.isfinite(tau) & (tau <= float(w_max))
            for h in torch.nonzero(local, as_tuple=False).flatten().tolist():
                w[h] = min(max(_next_pow2(tau[h].item()), w_min), w_max)
            w_rows.append(w)
        w_head = torch.stack(w_rows, dim=0)  # [L, HV] int64

        if _random_drop_enabled():
            # ABLATION: keep the same #local heads per layer, randomize which heads.
            w_head = _randomize_local_positions(w_head)

        is_global = w_head == 0
        n_global = is_global.sum(dim=1)  # [L]
        is_local = w_head > 0
        n_local = is_local.sum(dim=1)  # [L]
        G_max = int(n_global.max().item()) if L else 0
        Lwin_max = int(n_local.max().item()) if L else 0
        W_max = int(w_head.max().item()) if bool(is_local.any()) else 0

        global_idx = torch.full((L, max(G_max, 1)), -1, dtype=torch.int64)
        local_idx = torch.full((L, max(Lwin_max, 1)), -1, dtype=torch.int64)
        local_w = torch.zeros((L, max(Lwin_max, 1)), dtype=torch.int64)
        for l in range(L):
            g = torch.nonzero(is_global[l], as_tuple=False).flatten()
            global_idx[l, : g.numel()] = g
            lo = torch.nonzero(is_local[l], as_tuple=False).flatten()
            local_idx[l, : lo.numel()] = lo
            local_w[l, : lo.numel()] = w_head[l, lo]

        return cls(
            route=route,
            w_head=w_head,
            global_idx=global_idx,
            local_idx=local_idx,
            local_w=local_w,
            G_max=G_max,
            Lwin_max=Lwin_max,
            W_max=W_max,
            HV=HV,
            d_v=d_v,
            d_k=d_k,
        )

    @classmethod
    def build_plan_per_channel(
        cls,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        d_k: int,
        d_v: int,
        eps: float = 1e-3,
        w_max: int = 16,
        w_min: int = 4,
        a_margin: float = 0.3,
    ) -> HeadAwarePlan:
        """KDA per-(head, d_k-channel) Route-A plan.

        A_log   : [L, HV] per-head scalar. dt_bias : [L, HV, d_k] per-channel.
        Decay g[l,h,k] = -exp(A_log[l,h]) * softplus(-a_margin + dt_bias[l,h,k]);
        tau = ln(eps)/g. A COLUMN k of head h's [d_v, d_k] state is:
          * global (w_chan==0) if tau > w_max (or non-finite) -> store the d_v-vector
            exact.
          * local  (w_chan=W>0) if tau <= w_max -> drop it (Route A re-prefills the
            last W prefix tokens on a hit). W = clamp(next_pow2(tau), w_min, w_max).
        The packed unit is a global (head, col) pair; ``global_hk`` [L, GU_max, 2]
        lists them (GU_max = max over layers of #global columns). The per-head
        fields are GDN-shaped placeholders (unused when ``per_channel``).
        """
        from sglang.srt.layers.attention.fla.gdn_head_aware import (
            _next_pow2,
            gdn_gate,
            tau_from_g,
        )

        L, HV, K = dt_bias.shape
        assert K == d_k
        A_log = A_log.detach().cpu().to(torch.float64)
        dt_bias = dt_bias.detach().cpu().to(torch.float64)

        w_chan = torch.zeros((L, HV, K), dtype=torch.int64)
        for l in range(L):
            # g_repr[h,k] with a = -a_margin (broadcast A_log[l] over K columns).
            a_off = torch.full((HV, K), -a_margin, dtype=torch.float64)
            g_repr = gdn_gate(a_off, A_log[l][:, None], dt_bias[l])  # [HV, K]
            tau = tau_from_g(g_repr, eps)  # [HV, K] float64
            local = torch.isfinite(tau) & (tau <= float(w_max))
            for h, k in torch.nonzero(local, as_tuple=False).tolist():
                w_chan[l, h, k] = min(max(_next_pow2(tau[h, k].item()), w_min), w_max)

        if _random_drop_enabled():
            # ABLATION: keep the same #local columns per layer, randomize which
            # (head, d_k-col) units are dropped. Same compression ratio.
            w_chan = _randomize_local_positions(w_chan)

        is_local = w_chan > 0
        n_global = (~is_local).sum(dim=(1, 2))  # [L] (global columns per layer)
        GU_max = int(n_global.max().item()) if L else 0
        W_max = int(w_chan.max().item()) if bool(is_local.any()) else 0

        global_hk = torch.full((L, max(GU_max, 1), 2), -1, dtype=torch.int64)
        for l in range(L):
            gu = torch.nonzero(~is_local[l], as_tuple=False)  # [n_global, 2] (h, k)
            global_hk[l, : gu.shape[0]] = gu

        placeholder = torch.zeros((L, HV), dtype=torch.int64)
        return cls(
            route="A",
            w_head=placeholder,
            global_idx=placeholder,
            local_idx=placeholder,
            local_w=placeholder,
            G_max=0,
            Lwin_max=0,
            W_max=W_max,
            HV=HV,
            d_v=d_v,
            d_k=d_k,
            per_channel=True,
            w_chan=w_chan,
            global_hk=global_hk,
            GU_max=GU_max,
        )


class StateQuantizer:
    """Per-group symmetric fake quantizer for checkpoint state (INT8/INT4).

    Quantize on store, dequantize on load.  Tests the accuracy impact of uniform
    quantization at matched compression ratios vs DASC (which drops units
    selectively).  The computation kernel still gets BF16 (dequantized), so
    latency per decode-step is unchanged — measures accuracy + bytes/slot, not
    kernel speedup.

    Group size defaults to d_v (128), giving one scale per d_v-vector.  INT4
    values are stored packed two-per-byte in int8 storage.
    """

    def __init__(self, mode: str, group_size: int, d_v: int):
        self.mode = mode
        self.group_size = group_size
        self.d_v = d_v
        self.groups_per_dv = max(1, d_v // group_size)
        if mode == "int8":
            self.qmax = 127
            self.packed_factor = 1  # 1 int8 per value
        elif mode == "int4":
            self.qmax = 7
            self.packed_factor = 2  # 2 values per int8 storage byte
        else:
            raise ValueError(f"unsupported StateQuantizer mode: {mode}")

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def storage_len(self, d_v: int) -> int:
        """Number of int8 storage elements per d_v-vector."""
        return d_v // self.packed_factor

    def quantize(self, state: torch.Tensor):
        """state [..., d_v] (BF16) -> (qdata int8, scale BF16).

        qdata: [..., d_v//packed_factor] int8 (packed for INT4)
        scale: [..., groups_per_dv] BF16
        """
        state_fp32 = state.to(torch.float32)
        shape = state_fp32.shape
        d_v = shape[-1]
        g = self.groups_per_dv
        gs = d_v // g
        # per-group amax -> scale
        state_grp = state_fp32.reshape(*shape[:-1], g, gs)
        amax = state_grp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (amax / self.qmax).to(state.dtype).squeeze(-1)  # [.., g]
        q = (
            torch.round(state_fp32 / scale.to(torch.float32))
            .clamp(-self.qmax, self.qmax)
            .to(torch.int8)
        )  # [.., d_v]
        if self.packed_factor == 2:
            # pack pairs of int4 values into one int8 byte (two's complement 4-bit)
            q_even = q[..., 0::2] & 0x0F  # low nibble
            q_odd = q[..., 1::2] & 0x0F  # high nibble
            qdata = q_even | (q_odd << 4)  # [.., d_v//2] int8
        else:
            qdata = q
        return qdata, scale

    def dequantize(
        self, qdata: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
    ) -> torch.Tensor:
        """qdata + scale -> [..., d_v] in out_dtype."""
        if self.packed_factor == 2:
            lo = qdata & 0x0F
            hi = (qdata >> 4) & 0x0F
            # sign-extend from 4-bit two's complement
            lo = torch.where(lo > 7, lo - 16, lo)
            hi = torch.where(hi > 7, hi - 16, hi)
            q = torch.stack([lo, hi], dim=-1).flatten(-2)  # interleave
        else:
            q = qdata
        return (q.to(torch.float32) * scale.to(torch.float32)).to(out_dtype)


class HeadAwareCheckpointStore:
    """Packed head-aware store for cached multi-layer GDN states.

    Layout (slot index handed out by the caller's allocator):
        state_buf : [L, num_slots, G_max, d_v, d_k]  — exact state, GLOBAL rows only

    ``load`` returns the full dense [L, N, HV, d_v, d_k] state: global rows copied
    verbatim; local rows left zero with a ``local_needs_reprefill`` mask (Route A,
    filled by the scheduler re-prefill).
    """

    def __init__(
        self,
        *,
        plan: HeadAwarePlan,
        num_slots: int,
        device: str,
        state_dtype: torch.dtype = torch.bfloat16,
    ):
        self.plan = plan
        self.num_slots = num_slots
        self.device = device
        self.state_dtype = state_dtype
        L = plan.global_idx.shape[0]
        self.L = L
        self.per_channel = plan.per_channel
        self.quantizer = None  # set only for per-channel (KDA) path below
        # move static plan tensors onto the store's device once
        self.global_idx = plan.global_idx.to(device)
        self.local_idx = plan.local_idx.to(device)
        self.local_w = plan.local_w.to(device)
        # valid (non-pad) global rows per layer, for scatter on load
        self._g_valid = self.global_idx >= 0  # [L, G_max]

        if self.per_channel:
            # KDA: the packed unit is a global (head, d_k-col) pair -> a d_v-vector.
            self.global_hk = plan.global_hk.to(device)  # [L, GU_max, 2]
            self.w_chan = plan.w_chan.to(device)  # [L, HV, d_k]
            self._u_valid = self.global_hk[..., 0] >= 0  # [L, GU_max]
            from sglang.srt.environ import envs

            # Fake quantization (INT8/INT4): quantize on store, dequantize on load.
            # Experiment-only; when "none" (default), uses BF16 state_buf_pc unchanged.
            qmode = envs.SGLANG_STATE_QUANT_MODE.get()
            self.quantizer = (
                StateQuantizer(
                    qmode, envs.SGLANG_STATE_QUANT_GROUP_SIZE.get(), plan.d_v
                )
                if qmode != "none"
                else None
            )

            # Ragged (per-layer packed) per-channel buffer: drop the GU_max padding so
            # each layer stores exactly its own #valid (head, col) units in a flat
            # [num_slots, sum_l(#units), d_v] buffer with a per-layer row offset. Off ->
            # legacy GU_max-padded [L, num_slots, GU_max, d_v] (byte-identical).
            self.ragged = bool(envs.SGLANG_HEAD_AWARE_RAGGED.get())
            if self.ragged:
                n_units = self._u_valid.sum(dim=1)  # [L] #valid (h,col) units per layer
                off = torch.zeros(L + 1, dtype=torch.int64, device=device)
                off[1:] = torch.cumsum(n_units, dim=0)
                self._layer_off_pc = (
                    off  # [L+1] prefix-sum row offsets into flat buffer
                )
                total_units = int(off[-1].item())
                if self.quantizer:
                    qlen = self.quantizer.storage_len(plan.d_v)
                    self.state_buf_pc = torch.empty(
                        num_slots,
                        max(total_units, 1),
                        qlen,
                        dtype=torch.int8,
                        device=device,
                    )
                    self.qscale_pc = torch.empty(
                        num_slots,
                        max(total_units, 1),
                        self.quantizer.groups_per_dv,
                        dtype=state_dtype,
                        device=device,
                    )
                else:
                    self.state_buf_pc = torch.empty(
                        num_slots,
                        max(total_units, 1),
                        plan.d_v,
                        dtype=state_dtype,
                        device=device,
                    )
            else:
                if self.quantizer:
                    qlen = self.quantizer.storage_len(plan.d_v)
                    self.state_buf_pc = torch.empty(
                        L,
                        num_slots,
                        max(plan.GU_max, 1),
                        qlen,
                        dtype=torch.int8,
                        device=device,
                    )
                    self.qscale_pc = torch.empty(
                        L,
                        num_slots,
                        max(plan.GU_max, 1),
                        self.quantizer.groups_per_dv,
                        dtype=state_dtype,
                        device=device,
                    )
                else:
                    self.state_buf_pc = torch.empty(
                        L,
                        num_slots,
                        max(plan.GU_max, 1),
                        plan.d_v,
                        dtype=state_dtype,
                        device=device,
                    )
            return

        # Ragged (per-layer packed) GLOBAL-state buffer: drop the G_max padding so
        # each layer stores exactly its own #global rows in a flat
        # [num_slots, sum_l(#global), d_v, d_k] buffer with a per-layer offset.
        # Off -> legacy G_max-padded layout (byte-identical).
        from sglang.srt.environ import envs

        self.ragged = bool(envs.SGLANG_HEAD_AWARE_RAGGED.get())
        if self.ragged:
            n_glob = self._g_valid.sum(dim=1)  # [L] #global heads per layer
            off = torch.zeros(L + 1, dtype=torch.int64, device=device)
            off[1:] = torch.cumsum(n_glob, dim=0)
            self._layer_off = off  # [L+1] prefix-sum row offsets into the flat buffer
            total_global = int(off[-1].item())
            self.state_buf = torch.empty(
                num_slots,
                max(total_global, 1),
                plan.d_v,
                plan.d_k,
                dtype=state_dtype,
                device=device,
            )
        else:
            self.state_buf = torch.empty(
                L,
                num_slots,
                max(plan.G_max, 1),
                plan.d_v,
                plan.d_k,
                dtype=state_dtype,
                device=device,
            )

    # ---- tensor-level store / load (offline-testable; caller supplies slots) --

    @staticmethod
    def _as_slots(slots) -> torch.Tensor:
        if not torch.is_tensor(slots):
            slots = torch.as_tensor([slots], dtype=torch.int64)
        return slots.flatten()

    def store(
        self,
        slots,
        states: torch.Tensor,
    ) -> None:
        """Pack states into checkpoint ``slots``.

        states  : [L, N, HV, d_v, d_k] (or [L, HV, d_v, d_k] for a single slot).
        """
        slots = self._as_slots(slots)
        if states.dim() == 4:
            states = states.unsqueeze(1)
        N = slots.numel()
        assert states.shape[1] == N, f"{states.shape[1]} states for {N} slots"
        if self.per_channel:
            self._store_per_channel(slots, states)
            return
        if self.ragged:
            # per-layer packed: write exactly this layer's #global rows at its offset
            for l in range(self.L):
                valid = self._g_valid[l]  # [G_max] bool
                o0 = int(self._layer_off[l])
                o1 = int(self._layer_off[l + 1])
                if o1 > o0:
                    gi = self.global_idx[l][valid]  # [n_l] real head ids
                    self.state_buf[slots, o0:o1] = states[l][:, gi].to(self.state_dtype)
        else:
            for l in range(self.L):
                gi = self.global_idx[l].clamp(min=0)  # [G_max]; pad rows -> head 0
                # states[l]: [N, HV, d_v, d_k] -> gather global -> [N, G_max, d_v, d_k]
                self.state_buf[l, slots] = states[l][:, gi].to(self.state_dtype)

    def _store_per_channel(self, slots: torch.Tensor, states: torch.Tensor) -> None:
        """Pack GLOBAL (head, d_k-col) units' d_v-vectors. states: [L, N, HV, d_v, d_k].

        For each layer, gather ``states[l][:, h, :, k]`` over the layer's global
        (h, k) units into ``state_buf_pc[l, slots]`` = [N, GU_max, d_v]. Pad units
        (h==-1) are clamped to (0,0) and their packed rows are never read on load
        (masked by ``_u_valid``), so the junk they gather is harmless.
        """
        for l in range(self.L):
            if self.ragged:
                valid = self._u_valid[l]  # [GU_max]
                if not valid.any():
                    continue
                h_idx = self.global_hk[l, valid, 0]  # real heads [n_l]
                k_idx = self.global_hk[l, valid, 1]  # real d_k-cols [n_l]
                picked = states[l][:, h_idx, :, k_idx]  # [n_l, N, d_v]
                picked = picked.permute(1, 0, 2)  # [N, n_l, d_v]
                o0 = int(self._layer_off_pc[l])
                o1 = int(self._layer_off_pc[l + 1])
                if self.quantizer:
                    q, scale = self.quantizer.quantize(picked)
                    self.state_buf_pc[slots, o0:o1] = q
                    self.qscale_pc[slots, o0:o1] = scale
                else:
                    self.state_buf_pc[slots, o0:o1] = picked.to(self.state_dtype)
            else:
                h_idx = self.global_hk[l, :, 0].clamp(min=0)  # [GU_max]
                k_idx = self.global_hk[l, :, 1].clamp(min=0)  # [GU_max]
                # states[l][:, h_idx, :, k_idx]: advanced indices on non-adjacent axes
                # 1 and 3 -> broadcast dim moves to front -> [GU_max, N, d_v].
                picked = states[l][:, h_idx, :, k_idx]  # [GU_max, N, d_v]
                picked = picked.permute(1, 0, 2)  # [N, GU_max, d_v]
                if self.quantizer:
                    q, scale = self.quantizer.quantize(picked)
                    self.state_buf_pc[l, slots] = q
                    self.qscale_pc[l, slots] = scale
                else:
                    self.state_buf_pc[l, slots] = picked.to(self.state_dtype)

    def load(
        self, slots, out_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Reconstruct dense [L, N, HV, d_v, d_k] states at ``slots``.

        Returns (states, local_needs_reprefill). The second element is
        a [L, HV] bool mask of local heads left zeroed (the scheduler re-prefills
        them).
        """
        slots = self._as_slots(slots)
        N = slots.numel()
        HV, d_v, d_k = self.plan.HV, self.plan.d_v, self.plan.d_k
        out = torch.zeros(self.L, N, HV, d_v, d_k, dtype=out_dtype, device=self.device)
        if self.per_channel:
            return self._load_per_channel(slots, out, out_dtype)
        if self.ragged:
            for l in range(self.L):
                valid = self._g_valid[l]
                if valid.any():
                    heads = self.global_idx[l][valid]  # real head ids [n_l]
                    o0 = int(self._layer_off[l])
                    o1 = int(self._layer_off[l + 1])
                    out[l][:, heads] = self.state_buf[slots, o0:o1].to(out_dtype)
        else:
            for l in range(self.L):
                valid = self._g_valid[l]
                if valid.any():
                    heads = self.global_idx[l][valid]  # real head ids
                    out[l][:, heads] = self.state_buf[l, slots][:, valid].to(out_dtype)
        # Route A: local heads stay zero; report the mask for scheduler re-prefill
        return out, (self.plan.w_head.to(self.device) > 0)

    def _load_per_channel(
        self, slots: torch.Tensor, out: torch.Tensor, out_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Scatter packed GLOBAL (head, d_k-col) d_v-vectors back into the dense
        state; LOCAL columns stay zero and are reported per-(head, col) so the
        Route-A re-prefill masked copy-back rebuilds only those columns.

        ``state_buf_pc[l, slots]`` is [N, GU_max, d_v]; only the ``_u_valid`` units
        (real (h, k) pairs, not pad) are scattered to ``out[l][:, h, :, k]``.
        Returns (states, local_needs_reprefill) with the mask shaped [L, HV, d_k].
        """
        for l in range(self.L):
            valid = self._u_valid[l]  # [GU_max]
            if not valid.any():
                continue
            h_idx = self.global_hk[l, valid, 0]  # real head ids [nvalid]
            k_idx = self.global_hk[l, valid, 1]  # real d_k-col ids [nvalid]
            if self.ragged:
                o0 = int(self._layer_off_pc[l])
                o1 = int(self._layer_off_pc[l + 1])
                if self.quantizer:
                    q = self.state_buf_pc[slots, o0:o1]  # [N, nvalid, qlen]
                    sc = self.qscale_pc[slots, o0:o1]  # [N, nvalid, groups]
                    src = self.quantizer.dequantize(
                        q, sc, out_dtype
                    )  # [N, nvalid, d_v]
                else:
                    src = self.state_buf_pc[slots, o0:o1].to(
                        out_dtype
                    )  # [N, nvalid, d_v]
            else:
                if self.quantizer:
                    q = self.state_buf_pc[l, slots][:, valid]
                    sc = self.qscale_pc[l, slots][:, valid]
                    src = self.quantizer.dequantize(q, sc, out_dtype)
                else:
                    src = self.state_buf_pc[l, slots][:, valid].to(out_dtype)
            # scatter to out[l][:, h_idx, :, k_idx] (adv idx axes 1,3 -> front dim)
            out[l][:, h_idx, :, k_idx] = src.permute(1, 0, 2)
        # Route A: local (dropped) columns reported as a [L, HV, d_k] bool mask.
        return out, (self.w_chan > 0)

    # ---- byte accounting (capacity) ----

    def mem_usage_bytes(self) -> int:
        if self.per_channel:
            total = self.state_buf_pc.numel() * self.state_buf_pc.element_size()
            if self.quantizer:
                total += self.qscale_pc.numel() * self.qscale_pc.element_size()
            return total
        return self.state_buf.numel() * self.state_buf.element_size()

    def bytes_per_slot(self) -> int:
        return self.mem_usage_bytes() // max(1, self.num_slots)

    @staticmethod
    def dense_bytes_per_slot(
        *,
        num_layers: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        state_dtype: torch.dtype,
    ) -> int:
        isz = torch.empty((), dtype=state_dtype).element_size()
        return num_layers * num_heads * head_v_dim * head_k_dim * isz


class HeadAwareCheckpointPool:
    """Radix checkpoint pool backed by ``HeadAwareCheckpointStore``.

    Duck-types ``MambaCheckpointPool`` (same alloc/free/available_size/clear/
    store_from_active/load_to_active) so it drops into the SAME radix seam sites
    (``_commit_int8_checkpoint`` -> store_from_active, ``_free_mamba_value`` ->
    free, the COW load -> load_to_active) with no hook changes — memory_pool just
    assigns whichever pool is enabled to ``mamba_ckpt_pool``.

    The GDN decay plan needs the model's per-layer A_log/dt_bias, which are NOT
    available at pool construction (before weights load). So the plan / packed
    buffers are built lazily via ``set_plan`` (called once from the model runner
    after weights land). Until then the pool holds only the allocator + conv
    buffers and reports ``available_size()==0`` so the radix never donates into it.

    Local-head reconstruction inputs (the last-W prefix token ids) are a transient
    GDN-forward product, not resident in the active MambaPool. They are supplied by
    a re-prefill hook, which the scheduler populates during prefill — the GPU-side
    M2 integration. Global-head state + conv (both pool-resident) are transferred
    faithfully here regardless.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        num_layers: int,
        num_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        conv_shapes: List[tuple],
        conv_dtype: torch.dtype,
        device: str,
        route: str,
        temporal_dtype: torch.dtype,
        eps: float = 1e-3,
        w_max: int = 16,
    ):
        self.num_slots = num_slots
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_v_dim = head_v_dim
        self.head_k_dim = head_k_dim
        self.device = device
        self.route = route
        self.temporal_dtype = temporal_dtype
        self.eps = eps
        self.w_max = w_max
        self.store: Optional[HeadAwareCheckpointStore] = None
        # conv windows are plan-independent (tiny; not head-classified) -> allocate now
        self.conv = [
            torch.empty(
                (num_layers, num_slots + 1) + tuple(shape),
                dtype=conv_dtype,
                device=device,
            )
            for shape in conv_shapes
        ]
        self.allocator = MambaSlotAllocator(size=num_slots, device=device)

    # ---- lazy plan (needs model weights) ----

    def set_plan(self, A_log: torch.Tensor, dt_bias: torch.Tensor) -> None:
        """Build the per-layer head plan + packed buffers from stacked GDN weights.
        A_log / dt_bias: [num_layers, num_heads]."""
        plan = HeadAwarePlan.build_plan(
            A_log=A_log,
            dt_bias=dt_bias,
            route=self.route,
            d_k=self.head_k_dim,
            d_v=self.head_v_dim,
            eps=self.eps,
            w_max=self.w_max,
        )
        self.store = HeadAwareCheckpointStore(
            plan=plan,
            num_slots=self.num_slots + 1,  # slot 0 reserved (matches allocator)
            device=self.device,
            state_dtype=self.temporal_dtype,
        )
        GB = 1 << 30
        dense = HeadAwareCheckpointStore.dense_bytes_per_slot(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_v_dim=self.head_v_dim,
            head_k_dim=self.head_k_dim,
            state_dtype=self.temporal_dtype,
        )
        logger.info(
            f"head-aware mamba checkpoint pool (route {self.route}): plan built, "
            f"bytes/slot={self.store.bytes_per_slot()} vs dense {dense} "
            f"({dense / max(1, self.store.bytes_per_slot()):.2f}x capacity); "
            f"{self.mem_usage_bytes() / GB:.2f}GB over {self.num_slots} slots"
        )
        # DIAGNOSTIC (no layout change): per-layer #global units to quantify the
        # G_max-padding waste vs a hypothetical ragged (per-layer variable-length)
        # layout. Current buffer pads every layer to G_max=max_l(n_global) so
        # capacity == units / G_max; ragged would give units / mean(n_global).
        # A large max/mean gap = ragged upside; max≈mean = ragged is a no-op.
        try:
            if self.store.per_channel:
                # KDA: packed unit = one (head, d_k-col) d_v-vector; _u_valid [L, GU_max]
                per_layer = self.store._u_valid.sum(dim=1).tolist()
                pad_width = int(self.store._u_valid.shape[1])  # GU_max
                unit = "global (head,col) units"
            else:
                # GDN: packed unit = one head's d_v x d_k state; _g_valid [L, G_max]
                per_layer = self.store._g_valid.sum(dim=1).tolist()
                pad_width = int(self.store._g_valid.shape[1])  # G_max
                unit = "global heads"
            per_layer = [int(x) for x in per_layer]
            g_max = max(per_layer) if per_layer else 0
            n_mean = (sum(per_layer) / len(per_layer)) if per_layer else 0.0
            # current buffer packs L*pad_width rows (pad_width==G_max); a ragged
            # per-layer layout would pack sum(per_layer) rows => the relative extra
            # capacity is (L*G_max)/sum(per_layer) == G_max/mean(per_layer).
            ragged_upside = (g_max / n_mean) if n_mean else float("nan")
            logger.info(
                f"[ragged-diag] per-layer #{unit}: {per_layer}; "
                f"G_max(pad)={pad_width} mean={n_mean:.2f} "
                f"-> ragged (per-layer packed) = {ragged_upside:.3f}x more capacity "
                f"than current G_max-padded (max/mean gap; ~1.0 => not worth it)"
            )
        except Exception as e:  # diagnostic only; never break pool build
            logger.info(f"[ragged-diag] skipped: {e}")

    # ---- lifecycle (delegates to the embedded allocator) ----

    def alloc(self, n: int = 1):
        if self.store is None:  # plan not set yet -> never donate
            return None
        return self.allocator.alloc(n)

    def free(self, slots: torch.Tensor):
        self.allocator.free(slots)

    def available_size(self) -> int:
        if self.store is None:
            return 0
        return self.allocator.available_size()

    def clear(self) -> None:
        self.allocator.clear()

    # ---- state transfer between the active MambaPool and this store ----

    def store_from_active(self, active_mamba_pool, active_slots, ckpt_slots) -> None:
        assert self.store is not None, "head-aware pool used before set_plan()"
        cache = active_mamba_pool.mamba_cache
        states = cache.temporal[:, active_slots]  # [L, N, HV, d_v, d_k]
        self.store.store(ckpt_slots, states)
        for i, c in enumerate(self.conv):
            c[:, ckpt_slots] = cache.conv[i][:, active_slots]

    def load_to_active(self, active_mamba_pool, ckpt_slots, active_slots) -> None:
        assert self.store is not None, "head-aware pool used before set_plan()"
        cache = active_mamba_pool.mamba_cache
        states, needs_reprefill = self.store.load(ckpt_slots, cache.temporal.dtype)
        cache.temporal[:, active_slots] = states
        for i, c in enumerate(self.conv):
            cache.conv[i][:, active_slots] = c[:, ckpt_slots].to(cache.conv[i].dtype)
        # Route A leaves local-head rows zeroed; the scheduler must re-prefill the
        # last-W prefix tokens through the full model to fill them (M2). Surfaced
        # here as the needs_reprefill mask for the scheduler seam to consume.
        if needs_reprefill is not None:
            active_mamba_pool.mamba_head_reprefill_mask = needs_reprefill
        if _ckpt_load_enabled():
            # dropped_units = #local (head or per-channel) units this plan does NOT
            # store exact = the head-aware compression that just ran on this restore.
            # dense (W_max=0) -> all-global -> 0; idea1 -> >0. Direct proof.
            dropped = (
                int(needs_reprefill.sum().item()) if needs_reprefill is not None else 0
            )
            _record_ckpt_load(int(ckpt_slots.numel()), dropped, kind="head-aware")

    def copy_local_rows_from_scratch(
        self, active_mamba_pool, scratch_slots, active_slots
    ) -> None:
        """Route-A reconstruction step (M2): copy the local-head rows that
        ``load_to_active`` left zeroed from a scratch mamba slot (window-rolled
        state@P via a full-model re-prefill of the last-W tokens) into the real
        active slot. Global-head rows keep the exact checkpoint@P written by
        ``load_to_active``. Consumes and clears ``mamba_head_reprefill_mask``.

        ``scratch_slots`` / ``active_slots`` are aligned [N] index tensors (one
        scratch slot per Route-A hit). ``mask[l]`` is a [HV] bool selecting the
        local heads for layer ``l`` (GDN) — or, for KDA per-channel, a [HV, d_k]
        bool selecting the local (head, d_k-column) pairs, so only those columns of
        each head's [d_v, d_k] state are copied and global columns keep the exact
        checkpoint@P.
        """
        mask = active_mamba_pool.mamba_head_reprefill_mask
        if mask is None:
            return
        assert self.store is not None, "head-aware pool used before set_plan()"
        temporal = active_mamba_pool.mamba_cache.temporal  # [L, slots, HV, d_v, d_k]
        scratch_slots = self.store._as_slots(scratch_slots).to(temporal.device)
        active_slots = self.store._as_slots(active_slots).to(temporal.device)
        if self.store.per_channel:
            # KDA: mask[l] is [HV, d_k] — copy only the local (head, col) d_v-vectors
            # (a single column of the [d_v, d_k] state), leaving global columns exact.
            for l in range(self.store.L):
                hk = mask[l].nonzero(as_tuple=False)  # [n_local, 2] (head, col)
                if hk.shape[0] == 0:
                    continue
                h_idx, k_idx = hk[:, 0], hk[:, 1]
                # adv-index axes 0,1,3 (slice on the d_v axis 2) -> [N, n_local, d_v]
                src = temporal[l][scratch_slots.unsqueeze(1), h_idx, :, k_idx]
                temporal[l][active_slots.unsqueeze(1), h_idx, :, k_idx] = src
            active_mamba_pool.mamba_head_reprefill_mask = None
            return
        for l in range(self.store.L):
            heads = mask[l].nonzero(as_tuple=False).flatten()  # local head ids
            if heads.numel() == 0:
                continue
            # temporal[l] is a view; broadcasting [N,1] x [n_local] writes back
            # the [N, n_local, d_v, d_k] block into the underlying storage.
            src = temporal[l][scratch_slots.unsqueeze(1), heads]
            temporal[l][active_slots.unsqueeze(1), heads] = src
        active_mamba_pool.mamba_head_reprefill_mask = None

    def mem_usage_bytes(self) -> int:
        conv_bytes = sum(c.numel() * c.element_size() for c in self.conv)
        store_bytes = self.store.mem_usage_bytes() if self.store is not None else 0
        return store_bytes + conv_bytes


def maybe_init_head_aware_mamba_checkpoint_pool(
    *,
    mamba_size: int,
    cache_params,
    mamba_layer_ids: List[int],
    device: str,
) -> Optional[HeadAwareCheckpointPool]:
    """Build the head-aware ``HeadAwareCheckpointPool`` when
    ``--enable-head-aware-mamba-checkpoint`` is set, else None. Mutually exclusive
    with the int8 pool (enforced in ServerArgs). Sized so its HBM budget matches
    what the int8 pool would use (``2 * mamba_size`` dense-equivalent slots), then
    scaled up by the head-aware byte savings once ``set_plan`` runs is left to a
    follow-up; for now it allocates ``head_aware_mamba_ckpt_size`` slots (default
    ``2 * mamba_size``)."""
    from sglang.srt.server_args import get_global_server_args

    try:
        _sa = get_global_server_args()
    except ValueError:
        _sa = None
    if not getattr(_sa, "enable_head_aware_mamba_checkpoint", False):
        return None

    H, d_v, d_k = cache_params.shape.temporal
    ckpt_size = getattr(_sa, "head_aware_mamba_ckpt_size", None) or (2 * mamba_size)
    # Expert A/B override of the global/local tau threshold. w_max=0 forces every
    # head global -> an exact same-precision dense checkpoint (the fair capacity
    # baseline vs Route A/B selective storage). Unset keeps the plan default.
    from sglang.srt.environ import envs

    _forced_wmax = envs.SGLANG_FORCE_HEAD_AWARE_WMAX.get()
    _pool_kwargs = {} if _forced_wmax is None else {"w_max": int(_forced_wmax)}
    pool = HeadAwareCheckpointPool(
        num_slots=ckpt_size,
        num_layers=len(mamba_layer_ids),
        num_heads=H,
        head_v_dim=d_v,
        head_k_dim=d_k,
        conv_shapes=list(cache_params.shape.conv),
        conv_dtype=cache_params.dtype.conv,
        device=device,
        route=getattr(_sa, "head_aware_route", "A"),
        temporal_dtype=cache_params.dtype.temporal,
        **_pool_kwargs,
    )
    logger.info(
        f"head-aware mamba checkpoint pool: {ckpt_size} slots, route "
        f"{pool.route}; plan/buffers built lazily on set_plan() after weights load"
    )
    return pool
