"""GDN head-aware prefix-cache utility functions.

Decay math helpers shared by the head-aware checkpoint plan builder
(``mamba_checkpoint_pool.HeadAwarePlan.build_plan`` / ``build_plan_per_channel``).

Faithful per-step decay gate (one token), per v-head:
    g = -exp(A_log) * softplus(a + dt_bias)
alpha_h = exp(g_h) is the per-step decay; tau_h = ln(eps)/g_h is effective memory.
"""

from __future__ import annotations

import math

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
