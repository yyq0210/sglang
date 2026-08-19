"""EXPERIMENT-ONLY causal ablation of hybrid-model memory tiers (Phase A2).

Part of the "State as a Memory Tier" study
(``docs/hybrid_state_tiering_experiment_plan.md``, Phase A2). We independently
DESTROY each long-range memory pathway of a hybrid linear-attention model and
measure needle recall:

  * linear tier  -- the GDN recurrent ``ssm_state`` that folds the whole prefix.
  * full-attn tier -- the exact prefix KV cached by the few full-attention layers.

Hypothesis: on a hybrid model the full-attention layers carry long-range recall,
so zeroing the linear state barely moves needle accuracy, while hiding the prefix
from full-attn collapses it. That 2x2 is the causal evidence underwriting both
Idea 1 (no-recon prefix caching) and Idea 2 (fold-to-state eviction).

Everything is gated by BARE env vars (matching the existing NORECON / SEAM local-
experiment style). With them unset this module is a no-op and imports nothing heavy
on the hot path. This is INSTRUMENTATION, not a product feature -- do NOT wire it
into any release/default code path or commit it as an enabled feature.

  ABLATE_LINEAR_STATE = off | zero | noise   corrupt GDN ssm_state before each kernel
  ABLATE_FULL_KV      = off | zero | noise   corrupt full-attn PREFIX KV per forward
  ABLATE_NOISE_SCALE  = float (default 1.0)  noise std as a multiple of the tensor std
  ABLATE_SEED         = int   (default 0)    rng seed for reproducible noise

Semantics note (asymmetric by design, both faithful):
  * linear: corrupted in place before every kernel read -> the recurrent memory is
    destroyed continuously (GDN behaves as if it carries no cross-step history).
  * full-KV: snapshot -> corrupt -> restore around one forward -> non-persistent, so
    the KV pool is byte-identical afterward and other requests / later questions in
    the same needle group are unaffected.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_VALID = ("off", "zero", "noise")


def _read_mode(var: str) -> str:
    mode = os.environ.get(var, "off").lower()
    if mode not in _VALID:
        raise ValueError(f"{var}={mode!r} not in {_VALID}")
    return mode


# Env is frozen for the process lifetime -> resolve once at import (the launch
# driver sets these before the server starts).
_LINEAR_MODE = _read_mode("ABLATE_LINEAR_STATE")
_FULL_KV_MODE = _read_mode("ABLATE_FULL_KV")
_NOISE_SCALE = float(os.environ.get("ABLATE_NOISE_SCALE", "1.0"))
_SEED = int(os.environ.get("ABLATE_SEED", "0"))

_LINEAR_ON = _LINEAR_MODE != "off"
_FULL_KV_ON = _FULL_KV_MODE != "off"


def _read_layer_set(var: str):
    """Parse a comma-separated layer-id allowlist. Empty / "all" -> None (every layer).

    Phase A1: restrict the full-KV corruption to a subset of full-attn layers so we
    can causally localize WHICH layers carry the needle (ablate a depth band, watch
    recall drop). ``ABLATE_FULL_KV_LAYERS=all`` (default) ablates every full-attn layer
    (== Phase A2 behaviour).
    """
    raw = os.environ.get(var, "all").strip().lower()
    if raw in ("", "all"):
        return None
    return frozenset(int(x) for x in raw.replace(" ", "").split(",") if x != "")


_FULL_KV_LAYERS = _read_layer_set("ABLATE_FULL_KV_LAYERS")


def linear_state_enabled() -> bool:
    return _LINEAR_ON


def full_kv_enabled() -> bool:
    return _FULL_KV_ON


def _corrupted(ref: torch.Tensor, mode: str) -> torch.Tensor:
    """Return a corrupted copy of ``ref`` (same shape/dtype/device)."""
    if mode == "zero":
        return torch.zeros_like(ref)
    # noise: compute in fp32 (fp8/quantized KV can't take randn/add directly), then cast.
    f = ref.float()
    std = f.std()
    std = float(std) if torch.isfinite(std) and float(std) > 0 else 1.0
    gen = torch.Generator(device=ref.device)
    gen.manual_seed(_SEED)
    noise = torch.randn(ref.shape, generator=gen, device=ref.device) * (
        std * _NOISE_SCALE
    )
    return (f + noise).to(ref.dtype)


def corrupt_linear_state_(
    ssm_states: torch.Tensor, cache_indices: torch.Tensor
) -> None:
    """In-place corrupt the GDN recurrent-state rows for the active slots.

    Called before each GDN kernel read, so the linear tier carries no usable
    long-range memory for the duration of the probe.
    """
    if not _LINEAR_ON:
        return
    sel = ssm_states[cache_indices]
    ssm_states[cache_indices] = _corrupted(sel, _LINEAR_MODE)


def _prefix_kv_locs(
    req_to_token: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    """Cache locations of the PREFIX tokens (exclude tokens written this step).

    Decode: prefix = every position before the current (last) token per request.
    Extend: prefix = the cached shared prefix (where the needle lives); the tokens
    being extended this step are written to ``out_cache_loc`` and left intact.

    ``req_to_token`` is the ``[num_reqs, max_ctx]`` slot table (``ForwardBatch`` does
    not expose the pool, so the caller passes ``req_to_token_pool.req_to_token``).
    """
    req_idx = forward_batch.req_pool_indices
    if forward_batch.forward_mode.is_decode():
        lens = forward_batch.seq_lens - 1
    else:
        pl = forward_batch.extend_prefix_lens
        lens = (
            pl
            if pl is not None
            else (forward_batch.seq_lens - forward_batch.extend_seq_lens)
        )
    locs: List[torch.Tensor] = []
    for r, n in zip(req_idx.tolist(), lens.tolist()):
        if n > 0:
            locs.append(req_to_token[r, :n])
    if not locs:
        return torch.empty(0, dtype=torch.long, device=req_to_token.device)
    return torch.cat(locs)


@contextmanager
def corrupt_full_kv(
    token_to_kv_pool, req_to_token_pool, layer_id: int, forward_batch: ForwardBatch
):
    """Hide the PREFIX from one full-attn layer for THIS forward, then restore.

    Snapshot -> corrupt -> yield -> restore, so the KV pool is byte-identical after
    the forward (non-persistent; safe across the probe's repeated questions).
    """
    if not _FULL_KV_ON:
        yield
        return
    # Phase A1: only corrupt the layers in the allowlist (None -> all full-attn layers).
    if _FULL_KV_LAYERS is not None and layer_id not in _FULL_KV_LAYERS:
        yield
        return
    locs = _prefix_kv_locs(req_to_token_pool.req_to_token, forward_batch)
    if locs.numel() == 0:
        yield
        return
    kbuf = token_to_kv_pool.get_key_buffer(layer_id)
    vbuf = token_to_kv_pool.get_value_buffer(layer_id)
    saved_k = kbuf[locs].clone()
    saved_v = vbuf[locs].clone()
    try:
        kbuf[locs] = _corrupted(saved_k, _FULL_KV_MODE)
        vbuf[locs] = _corrupted(saved_v, _FULL_KV_MODE)
        yield
    finally:
        kbuf[locs] = saved_k
        vbuf[locs] = saved_v
