"""Route-A head-aware GDN checkpoint re-prefill plumbing (M2 load side).

On a prefix-cache hit, ``HeadAwareCheckpointPool.load_to_active`` writes the exact
global-head state@P into the active mamba slot and leaves the local-head rows zeroed
(Route A drops them at store time), surfacing a ``[L, HV]`` bool re-prefill mask. This
module reconstructs those local rows by re-prefilling the last ``window_len`` prefix
tokens through the FULL model into a scratch mamba slot (all heads zero-started at the
window base), then copying only the masked local rows back into the active slot
(``copy_local_rows_from_scratch``). Global rows keep the exact checkpoint@P.

Ordering (critical): ``load_to_active`` runs inside the MAIN (suffix) forward's
``init_forward_metadata`` -> ``_execute_deferred_mamba_cow_and_clear``, and the GDN
kernel reads the active mamba slot LATER in that same forward. So the scratch slot must
already be filled (a SEPARATE recon forward that runs first) and the masked copy is
appended to that same deferred-COW hook, immediately after ``load_to_active``. The
recon forward and the copy are both forward-stream ops, keeping the triple
recon -> load_to_active+copy -> suffix strictly forward-stream-ordered inside one
``_forward_isolation`` context.

Everything here is gated by ``SGLANG_ENABLE_HEAD_AWARE_REPREFILL`` and is inert unless
the pool is head-aware Route A with a re-prefill window (``plan.W_max > 0``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# One-shot guard for the C2 seam-window observability log (see build_reprefill_batch).
_logged_first_recon = False

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

try:
    import msgspec

    _StructBase = msgspec.Struct
except Exception:  # msgspec is a hard dep in sglang; guard only for odd envs
    _StructBase = object


class HeadAwareReprefillBatch(_StructBase):
    """Per-extend-batch Route-A reconstruction bookkeeping.

    Fields:
      window_len          : W — number of trailing prefix tokens re-prefilled through the
                            full model (``plan.W_max``, uniform per plan). Actual per-req
                            span is ``[max(0, P - W), P)`` so short prefixes forward fewer
                            tokens.
      scratch_mamba_slots : [N] int64 — the dedicated mamba slots the recon forward wrote
                            window-rolled state into. Aligned with ``active_mamba_slots``.
                            Freed after the masked copy.
      active_mamba_slots  : [N] int64 — the real active mamba slots (``req.mamba_pool_idx``)
                            holding global@P; masked local rows are copied in from the
                            scratch slots, then the re-prefill mask is cleared.

    The throwaway req-pool + KV slots the recon forward allocates are freed per recon
    CHUNK right after that chunk's forward (see ``free_recon_scratch_kv``); they are not
    tracked here because the scratch mamba slots (aligned N-wide with the active slots)
    are the only state the suffix masked copy needs.
    """

    window_len: int
    scratch_mamba_slots: torch.Tensor
    active_mamba_slots: torch.Tensor

    def num_hits(self) -> int:
        return int(self.active_mamba_slots.numel())

    def is_empty(self) -> bool:
        return self.num_hits() == 0


def get_head_aware_route_a_pool(model_runner: ModelRunner):
    """Return the head-aware ``HeadAwareCheckpointPool`` iff it is Route A with a live
    re-prefill window (``plan.W_max > 0``), else None. Used to gate the whole seam."""
    pool = getattr(model_runner.req_to_token_pool, "mamba_ckpt_pool", None)
    if pool is None:
        return None
    store = getattr(pool, "store", None)
    if store is None:  # plan not built yet
        return None
    if getattr(pool, "route", None) != "A":
        return None
    if getattr(store.plan, "W_max", 0) <= 0:
        return None
    return pool


def _route_a_hit_reqs(batch: ScheduleBatch) -> List[Req]:
    """The Route-A prefix-cache hits in this extend batch.

    ``prepare_for_extend`` -> ``_collect_deferred_mamba_cow_and_clear`` runs in the
    scheduler BEFORE the worker sees the batch and clears each ``req.mamba_cow_src_index``
    to None (schedule_batch.py:2370), so the per-req field is gone by now. The batch-level
    ``mamba_cow_dst_indices`` survives, and each entry is exactly a hit req's
    ``mamba_pool_idx`` (the active slot the COW load targets, appended at :2369). Map those
    back to reqs (non-empty matched prefix), preserving batch order.
    """
    dst = getattr(batch, "mamba_cow_dst_indices", None)
    if dst is None or dst.numel() == 0:
        return []
    dst_set = {int(x) for x in dst.tolist()}
    return [
        r
        for r in batch.reqs
        if r.mamba_pool_idx is not None
        and int(r.mamba_pool_idx) in dst_set
        and len(r.prefix_indices) > 0
    ]


def build_reprefill_batch(
    batch: ScheduleBatch,
    model_runner: ModelRunner,
) -> Optional[Tuple[List[Req], List[int], HeadAwareReprefillBatch]]:
    """Derive the Route-A reconstruction req set from an extend batch.

    For each Route-A hit req at matched prefix ``P``, build a throwaway extend req that
    re-prefills the window ``[base, P)`` (``base = max(0, P - W)``) through the full
    model into a freshly-allocated scratch mamba slot (zeroed at ``base`` via
    ``mamba_needs_clear``). The window token KV is written to throwaway KV slots; the
    ``[0, base)`` context attends to the real cached prefix KV (correct attention). The
    recon logits are discarded — only the final scratch mamba state@P is kept.

    Returns ``(recon_reqs, recon_win_lens, reprefill)`` — a FLAT list of throwaway Reqs
    (each with its scratch ``mamba_pool_idx`` pre-set) plus their window lengths, to be
    scheduled in waves by ``iter_recon_chunks`` — or None if there are no reconstructable
    hits. The scratch mamba slots are recorded on ``reprefill`` and MUST be freed after the
    suffix masked copy (see ``free_recon_scratch_mamba``). The per-wave req-pool + KV slots
    are freed by the caller after each wave's forward (see ``free_recon_scratch_kv``).
    """
    from array import array

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    pool = get_head_aware_route_a_pool(model_runner)
    if pool is None:
        return None

    hits = _route_a_hit_reqs(batch)
    if not hits:
        return None

    W = int(pool.store.plan.W_max)
    # Hypic C2 seam window: decouple the reconstruction span from the tau split.
    # The tau global/local split above already used plan.W_max (capacity unchanged);
    # here we optionally shrink ONLY the re-prefill window to a small seam so t_recon
    # drops from O(W_max) to O(seam). Local heads with tau > seam are then partially
    # reconstructed (accuracy judged empirically). Unset/<=0 -> full W_max (legacy).
    from sglang.srt.environ import envs

    _seam = envs.SGLANG_HEAD_AWARE_SEAM_WINDOW.get()
    if _seam is not None and _seam > 0:
        W = min(int(_seam), W)
    req_to_token_pool = model_runner.req_to_token_pool
    mamba_allocator = req_to_token_pool.mamba_allocator

    recon_reqs: List[Req] = []
    recon_win_lens: List[int] = []
    scratch_slots: List[torch.Tensor] = []
    active_slots: List[torch.Tensor] = []
    for r in hits:
        P = len(r.prefix_indices)
        base = max(0, P - W)
        if P - base <= 0:  # empty window (P == 0 already filtered, guard anyway)
            continue
        fill_ids = r.get_fill_ids()
        # window token ids [base, P); origin/full ids up to P so extend_range is valid.
        prefix_ids = array("q", fill_ids[:P])

        scratch = mamba_allocator.alloc(1)
        assert scratch is not None, (
            "head-aware Route-A re-prefill ran out of scratch mamba slots; "
            "increase --mamba-full-memory-ratio or --max-mamba-cache-size."
        )
        # scratch must be distinct from the real active slot it reconstructs.
        assert int(scratch[0]) != int(
            r.mamba_pool_idx
        ), "reprefill scratch slot aliases the active slot"

        rr = Req(
            rid=f"__head_aware_reprefill__{r.rid}",
            origin_input_text="",
            origin_input_ids=prefix_ids,
            sampling_params=SamplingParams(max_new_tokens=0),
            vocab_size=r.vocab_size,
        )
        rr.full_untruncated_fill_ids = array("q", prefix_ids)
        # Cached KV for [0, base): a slice of the real hit's matched prefix.
        rr.prefix_indices = r.prefix_indices[:base]
        rr.set_extend_range(base, P)
        rr.mamba_pool_idx = scratch[0]
        rr.mamba_needs_clear = True  # zero scratch at the window base
        rr.mamba_cow_src_index = None  # do NOT COW-load global@P into scratch

        recon_reqs.append(rr)
        recon_win_lens.append(P - base)
        scratch_slots.append(scratch)
        active_slots.append(r.mamba_pool_idx.unsqueeze(0))

    if not recon_reqs:
        return None

    device = req_to_token_pool.device
    scratch_t = torch.cat(scratch_slots).to(device)
    active_t = torch.cat(active_slots).to(device)
    reprefill = HeadAwareReprefillBatch(
        window_len=W,
        scratch_mamba_slots=scratch_t,
        active_mamba_slots=active_t,
    )
    # One-shot observability (C2 seam experiment): confirm the reconstruction path
    # fires and with which window (seam W vs plan.W_max). Logs on the first hit only.
    global _logged_first_recon
    if not _logged_first_recon:
        _logged_first_recon = True
        logger.info(
            "[head-aware Route-A C2] first re-prefill hit: hits=%d W(recon)=%d "
            "plan.W_max=%d recon_tokens=%s",
            len(recon_reqs),
            W,
            int(pool.store.plan.W_max),
            recon_win_lens,
        )
    return recon_reqs, recon_win_lens, reprefill


def iter_recon_chunks(
    recon_reqs: List[Req],
    recon_win_lens: List[int],
    batch: ScheduleBatch,
    model_runner: ModelRunner,
):
    """Yield the recon reqs as prepared ``ScheduleBatch`` WAVES, one at a time.

    Each wave is bounded by BOTH the eager input-buffer token ceiling
    (``max_prefill_buffer_tokens()``, so the cuda-graph-buffer-registry copy fits) AND the
    number of ``req_to_token`` slots free RIGHT NOW (so recon never out-competes the live
    batch for req-pool slots -> ``alloc_req_slots`` OOM). The caller must forward and then
    free each yielded wave (``free_recon_scratch_kv``) BEFORE requesting the next, so the
    freed slots are restored for the following wave. A single window is W <= ceiling and
    each wave holds >=1 req, so all reqs drain as long as >=1 req-pool slot is free at entry
    (the server should run ``--max-running-requests`` below the req-pool size to guarantee
    headroom; the bench passes MAX_RUNNING > MAX_CONC for exactly this).
    """
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    req_to_token_pool = model_runner.req_to_token_pool
    device = req_to_token_pool.device
    token_ceiling = int(model_runner.server_args.max_prefill_buffer_tokens())

    i = 0
    n = len(recon_reqs)
    while i < n:
        # A throwaway recon req needs one fresh req-pool slot; keep a small margin so a
        # concurrent chunked-prefill continuation is never starved.
        slot_cap = max(1, req_to_token_pool.available_size() - 1)
        chunk_reqs: List[Req] = []
        chunk_tokens = 0
        while i < n and len(chunk_reqs) < slot_cap:
            win = recon_win_lens[i]
            if chunk_reqs and chunk_tokens + win > token_ceiling:
                break
            chunk_reqs.append(recon_reqs[i])
            chunk_tokens += win
            i += 1

        recon_batch = ScheduleBatch.init_new(
            reqs=chunk_reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            tree_cache=batch.tree_cache,
            model_config=model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        recon_batch.prepare_for_extend()

        # prepare_for_extend stages input_ids as pinned CPU (prefill_input_ids_cpu),
        # normally H2D'd by the scheduler's resolve_forward_inputs. This throwaway recon
        # batch bypasses that, so materialize input_ids here.
        if (
            recon_batch.input_ids is None
            and recon_batch.prefill_input_ids_cpu is not None
        ):
            recon_batch.input_ids = recon_batch.prefill_input_ids_cpu.to(
                device, non_blocking=True
            )
            recon_batch.prefill_input_ids_cpu = None
        yield recon_batch


def free_recon_scratch_kv(
    recon_batch: ScheduleBatch,
    model_runner: ModelRunner,
) -> None:
    """Release ONE recon chunk's throwaway req-pool + KV slots. Safe to call right after
    that chunk's recon forward: these are NOT read again. The scratch MAMBA slots are held
    until after the suffix forward's masked copy (see ``free_recon_scratch_mamba``)."""
    req_to_token_pool = model_runner.req_to_token_pool
    req_pool_indices = recon_batch.req_pool_indices
    # Under extra_buffer, prepare_for_extend -> alloc allocated ping-pong track
    # buffers for each throwaway recon req (recorded in the pool's
    # req_index_to_mamba_ping_pong_track_buffer_mapping). Nothing keeps the recon
    # tracked state, so release those slots now. This mirrors the ping-pong release
    # in free_mamba_cache but leaves the scratch mamba_pool_idx (the authoritative
    # window-rolled state) alive for the suffix masked copy.
    if getattr(req_to_token_pool, "enable_mamba_extra_buffer", False):
        mapping = req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping
        mamba_allocator = req_to_token_pool.mamba_allocator
        for req_idx in req_pool_indices.tolist():
            buf = mapping[req_idx]
            to_free = buf[buf != -1]
            if to_free.numel() > 0:
                mamba_allocator.free(to_free)
    # req_to_token free-list is a python list of ints; return slots individually.
    for idx in req_pool_indices.tolist():
        req_to_token_pool.free_slots.append(idx)
    model_runner.token_to_kv_pool_allocator.free(recon_batch.out_cache_loc)


def free_recon_scratch_mamba(
    reprefill: HeadAwareReprefillBatch,
    model_runner: ModelRunner,
) -> None:
    """Release the scratch mamba slots. Call AFTER the suffix forward, since the masked
    copy (inside the suffix forward's deferred-COW hook) reads them."""
    model_runner.req_to_token_pool.mamba_allocator.free(reprefill.scratch_mamba_slots)
