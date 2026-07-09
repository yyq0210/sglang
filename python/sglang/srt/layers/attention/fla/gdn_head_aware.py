"""GDN head-aware prefix-cache reconstruction — Route B prototype (reference impl).

Design doc: ``docs/gdn_head_aware_prefix_cache_design.md`` (§2.3 Route B, §3.3).

Route B splits a linear-attention (GDN) recurrent state, cached at a prefix-cache
node, into two head classes decided by each head's decay:

  * **global head** (tau_h > w): memory outlives the window -> store the EXACT
    state matrix S_h in [d_v, d_k] (what the radix does today).
  * **local head**  (tau_h <= w): state ~= contribution of the last ``w`` tokens
    -> do NOT store S_h; instead store the last-w window of (k, v, g) and, on a
    cache hit, REPLAY that window through THIS layer's recurrence to rebuild S_h.

This module is a *pure-torch reference* that mirrors the shipped GDN step in
``fused_recurrent.py:242-265`` exactly (per-head scalar gate). It is the thing the
validation script (``test/manual/test_gdn_head_aware_prefix.py``) checks against,
and the spec for a future kernel/mem-pool integration. It does NOT touch the
radix cache or any pool — it operates on plain tensors.

Faithful per-step recurrence (one token), state S is [V, K], per v-head:
    k = k / ||k||_2                      # if use_qk_l2norm_in_kernel
    g = -exp(A_log) * softplus(a + dt_bias)
    beta = sigmoid(b)
    S = exp(g) * S                       # decay
    v = v - (S @ k)                      # delta-rule correction  (sum over K)
    v = v * beta
    S = S + outer(v, k)                  # rank-1 update
alpha_h = exp(g_h) is the per-step decay; tau_h = ln(eps)/g_h is effective memory.
"""

from __future__ import annotations

import math

import msgspec
import torch
import torch.nn.functional as F

SOFTPLUS_THRESHOLD = 20.0  # matches fused_gdn_gating_kernel / fused_recurrent.py:252


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x, beta=1.0, threshold=SOFTPLUS_THRESHOLD)


def gdn_gate(a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor):
    """Per-head scalar decay g = -exp(A_log)*softplus(a+dt_bias).

    a: [..., HV] input-dependent projection; A_log/dt_bias: [HV] static weights.
    Returns g with the same shape as ``a`` (broadcast over leading dims).
    """
    return -torch.exp(A_log) * _softplus(a + dt_bias)


def tau_from_g(g: torch.Tensor, eps: float) -> torch.Tensor:
    """Effective memory length tau = ln(eps)/g (tokens). g==0 -> +inf (global)."""
    ln_eps = math.log(eps)
    g = g.to(torch.float64)
    return torch.where(g < 0, ln_eps / g, torch.full_like(g, float("inf")))


def _next_pow2(x) -> int:
    x = int(math.ceil(x))
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def plan_windows(
    g_repr: torch.Tensor,
    eps: float,
    d_k: int,
    d_v: int,
    w_max: int = 4096,
    w_min: int = 4,
) -> torch.Tensor:
    """Per-head Route-B plan. Returns ``w_head`` [HV] int:

      * ``w_head[h] = 0``  -> store the EXACT state for head h (global, OR local
        but a window would not beat the [d_v, d_k] state).
      * ``w_head[h] = W>0`` -> windowize: store the last W tokens and replay them
        on a hit. W = clamp(next_pow2(tau_h), w_min, w_max). Chosen only when it
        is BOTH correct (W >= tau_h, so recon error <= eps) AND cheaper than the
        state:  W * (d_k + d_v + 2) < d_v * d_k  (break-even ~64 at 128x128).

    ``w_min`` gives ultra-fast heads (tau ~ 1-2) a small safety window so a single
    atypically-slow token cannot under-cover them; its cost is negligible.
    ``g_repr`` [HV] is a representative per-head g. Classify at a slightly
    NEGATIVE a (slower decay) rather than a=0, since a is zero-mean and a<0 tokens
    decay slower — a=0 is only conservative when a>=0.
    """
    tau = tau_from_g(g_repr, eps)
    state_cost = d_v * d_k
    per_tok = d_k + d_v + 2  # k + v + g + beta
    w_head = torch.zeros(g_repr.numel(), dtype=torch.int64)
    finite = torch.isfinite(tau)
    for h in torch.nonzero(finite, as_tuple=False).flatten().tolist():
        W = min(max(_next_pow2(tau[h].item()), w_min), w_max)
        if W * per_tok < state_cost:
            w_head[h] = W
    return w_head


def gdn_step(
    S: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    l2norm_k: bool = True,
) -> torch.Tensor:
    """One faithful GDN recurrence step, batched over heads.

    S:    [HV, V, K]   state (float32)
    k:    [HV, K]      key    (pre-l2norm allowed; l2norm applied here if flagged)
    v:    [HV, V]      value
    g:    [HV]         per-head scalar log-decay (already = gdn_gate(...))
    beta: [HV]         per-head gate = sigmoid(b)
    Returns the updated S [HV, V, K]. Mirrors fused_recurrent.py:242-265.
    """
    if l2norm_k:
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    alpha = torch.exp(g)  # [HV]
    S = alpha[:, None, None] * S  # decay
    # v -= S @ k  (sum over K)
    Sk = (S * k[:, None, :]).sum(-1)  # [HV, V]
    v = v - Sk
    v = v * beta[:, None]  # gate
    S = S + v[:, :, None] * k[:, None, :]  # rank-1 update
    return S


def run_prefix_recurrence(
    S0: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    g_seq: torch.Tensor,
    beta_seq: torch.Tensor,
    l2norm_k: bool = True,
) -> torch.Tensor:
    """Roll the GDN recurrence over T tokens from initial state S0.

    S0:      [HV, V, K]
    *_seq:   k,v [T, HV, K]/[T, HV, V]; g,beta [T, HV]
    Returns the final state S_T [HV, V, K]. This is the EXACT reference the
    head-aware reconstruction must match (for global heads) / approximate within
    eps^(w/tau) (for local heads).
    """
    S = S0
    T = k_seq.shape[0]
    for t in range(T):
        S = gdn_step(S, k_seq[t], v_seq[t], g_seq[t], beta_seq[t], l2norm_k)
    return S


class HeadAwareCheckpoint(msgspec.Struct):
    """Route-B cached representation of one prefix node's GDN state.

    Instead of the dense [HV, V, K] exact state (today's radix ``mamba_value``),
    store:
      * ``w_head`` [HV] int  — per-head plan (0 = store exact state; W>0 =
        windowize with last-W tokens), from ``plan_windows``.
      * ``state`` [HV, V, K] — exact state; rows for windowized heads are unused
                               (left as-is) and NOT counted in ``mem_elements``.
      * the last ``W_max`` tokens' (k,v,g,beta) window (full-HV width for
        prototype simplicity; each windowized head only *reads* its own last
        ``w_head[h]`` rows, and only those are counted in ``mem_elements``).

    Reconstruction (``rebuild``) returns a full [HV, V, K] state: exact rows for
    stored heads, window-replayed rows for windowized heads (started from ZERO,
    since a windowized head has forgotten everything before its window).
    """

    w_head: torch.Tensor
    state: torch.Tensor
    k_win: torch.Tensor
    v_win: torch.Tensor
    g_win: torch.Tensor
    beta_win: torch.Tensor
    d_k: int
    d_v: int

    def rebuild(self, l2norm_k: bool = True) -> torch.Tensor:
        """Reconstruct the full [HV, V, K] state. Windowized heads sharing the
        same window length W are replayed together (few distinct pow2 lengths)."""
        HV, V, K = self.state.shape
        W_max = self.k_win.shape[0]
        S = self.state.clone()
        win = self.w_head
        for W in sorted({int(x) for x in win.tolist() if x > 0}):
            idx = torch.nonzero(win == W, as_tuple=False).flatten()
            S_g = torch.zeros(idx.numel(), V, K, dtype=S.dtype, device=S.device)
            for t in range(W_max - W, W_max):  # last W stored tokens
                S_g = gdn_step(
                    S_g,
                    self.k_win[t, idx],
                    self.v_win[t, idx],
                    self.g_win[t, idx],
                    self.beta_win[t, idx],
                    l2norm_k,
                )
            S[idx] = S_g
        return S

    def mem_elements(self) -> dict:
        """Element counts for the *packed* Route-B layout vs today's dense
        baseline: exact state only for stored heads, window only for windowized
        heads (each its own w_head[h] length)."""
        HV, V, K = self.state.shape
        win = self.w_head
        n_win = int((win > 0).sum().item())
        state = (HV - n_win) * V * K
        window = int(win.sum().item()) * (K + V + 2)  # sum of per-head w_head
        baseline = HV * V * K
        return {
            "baseline": baseline,
            "route_b": state + window,
            "state_stored": state,
            "window": window,
            "n_windowized": n_win,
            "n_stored": HV - n_win,
        }


def build_checkpoint(
    S_exact: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    g_seq: torch.Tensor,
    beta_seq: torch.Tensor,
    w_head: torch.Tensor,
    d_k: int,
    d_v: int,
) -> HeadAwareCheckpoint:
    """Assemble a Route-B checkpoint from a prefix's exact final state + the
    tail window of its (k,v,g,beta) inputs. Stores the last W_max = max(w_head)
    tokens (windowized heads read only their own suffix).

    S_exact: [HV, V, K] exact final state (stored rows kept verbatim).
    *_seq:   k,v [T, HV, ...]; g,beta [T, HV]; T >= max(w_head).
    """
    T = k_seq.shape[0]
    w_max = int(w_head.max().item()) if (w_head > 0).any() else 0
    assert T >= w_max, f"need at least w_max={w_max} tokens, got T={T}"
    sl = slice(T - w_max, T) if w_max else slice(0, 0)
    return HeadAwareCheckpoint(
        w_head=w_head,
        state=S_exact,
        k_win=k_seq[sl].contiguous(),
        v_win=v_seq[sl].contiguous(),
        g_win=g_seq[sl].contiguous(),
        beta_win=beta_seq[sl].contiguous(),
        d_k=d_k,
        d_v=d_v,
    )
