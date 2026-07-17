"""EXPERIMENT-ONLY capture of the REAL GDN gate (and GDN k/v) for Phase C1.

Part of the "State as a Memory Tier" study
(``docs/hybrid_state_tiering_experiment_plan.md``, Phase C1). Phase C0 showed the
fold-to-state eviction beats drop, BUT a FIXED decay < 1 zeroes deep content
(0.99^900 ~ 1e-4 kills a 900-token-deep needle). The proposed fix is to fold with
the model's OWN learned, content-adaptive gate instead of a hand-picked constant.

Qwen3-Next's GDN forget gate is exactly such a signal:

    g = -exp(A_log) * softplus(a + dt_bias)   (per token, per head; g <= 0)

so the per-step retention is exp(g) in (0, 1]: for a salient token the model drives
``a`` very negative -> softplus -> 0 -> g -> 0 -> retention ~ 1 (kept); filler ->
g < 0 -> decays. This module snapshots that REAL gate over a real needle prefill so
we can measure offline whether the model keeps deep salient tokens alive (retention
>> constant-decay) and drive the C0 fold with the captured gate.

Everything is gated by a BARE env var (matching the NORECON / SEAM / ABLATE local-
experiment style). With ``CAPTURE_TIER_DIR`` unset this module is a no-op and costs
nothing on the hot path. This is INSTRUMENTATION, not a product feature -- do NOT
wire it into any release/default code path or commit it as an enabled feature.

  CAPTURE_TIER_DIR     path to dump *.pt captures (unset -> disabled)
  CAPTURE_GDN_LAYERS   comma list of GDN layer ids to dump (default "0")
  CAPTURE_MIN_SEQLEN   only fire on a prefill with >= this many tokens (default 2000)
                       so the big needle-doc prime is captured, not question tails.

Each captured layer is dumped exactly once (the first qualifying prefill).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Set

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_CAPTURE_DIR: Optional[str] = os.environ.get("CAPTURE_TIER_DIR") or None
_ON = _CAPTURE_DIR is not None


def _read_layer_set(var: str, default: str):
    raw = os.environ.get(var, default).strip().lower()
    if raw in ("", "all"):
        return None
    return frozenset(int(x) for x in raw.replace(" ", "").split(",") if x != "")


_GDN_LAYERS = _read_layer_set("CAPTURE_GDN_LAYERS", "0")
_MIN_SEQLEN = int(os.environ.get("CAPTURE_MIN_SEQLEN", "2000"))

_DONE_GDN: Set[int] = set()


def gdn_gate_enabled() -> bool:
    return _ON


def capture_gdn_gate(
    layer_id: int,
    g: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    forward_batch: "ForwardBatch",
) -> None:
    """Dump the real per-token GDN gate (+ k/v, positions, input_ids) once per layer.

    Called from ``GDNAttnBackend.forward_extend`` right after ``fused_gdn_gating``.
    Shapes on entry: ``g`` [1, T, H], ``key``/``value`` [1, T, H, D].
    """
    if not _ON:
        return
    if _GDN_LAYERS is not None and layer_id not in _GDN_LAYERS:
        return
    if layer_id in _DONE_GDN:
        return
    seq_len = g.shape[1] if g.dim() == 3 else g.shape[0]
    if seq_len < _MIN_SEQLEN:
        return

    os.makedirs(_CAPTURE_DIR, exist_ok=True)
    positions = getattr(forward_batch, "positions", None)
    input_ids = getattr(forward_batch, "input_ids", None)
    blob = {
        "layer_id": layer_id,
        "seq_len": int(seq_len),
        # gate: [T, H] fp32 (per-token per-head negative log-decay)
        "g": g.squeeze(0).detach().to(torch.float32).cpu(),
        # k/v: [T, H, D] (keep native dtype, just off-device)
        "key": key.squeeze(0).detach().cpu(),
        "value": value.squeeze(0).detach().cpu(),
        "positions": (
            positions.detach().cpu() if positions is not None else None
        ),
        "input_ids": (
            input_ids.detach().cpu() if input_ids is not None else None
        ),
    }
    path = os.path.join(_CAPTURE_DIR, f"gdn_L{layer_id}.pt")
    torch.save(blob, path)
    _DONE_GDN.add(layer_id)
    print(
        f"[tier_capture] dumped GDN gate layer={layer_id} seq_len={seq_len} "
        f"g{tuple(blob['g'].shape)} -> {path}",
        flush=True,
    )
