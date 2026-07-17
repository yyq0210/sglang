"""Phase C2-a: fused windowed-exact + folded-state decode GPU primitive.

EXPERIMENT-ONLY (part of "State as a Memory Tier",
``docs/hybrid_state_tiering_experiment_plan.md``, Phase C2). C0 (GATE-C0) proved
OFFLINE that folding an evicted middle KV segment into a q-independent linear state
beats dropping it at small exact-KV budget; C1 proved the model's own GDN gate keeps
deep content alive so gated fold rescues the deep needle. C2-a turns that validated
fold math into a real GPU decode primitive and unit-tests it against the C0 python
reference BEFORE any serving run (the mandated gate).

Per full-attention layer the decode output is the LSE-merge of two substrates:

    o = merge( windowed_softmax(q, K_win, V_win),   # exact [sink + recent window]
               state_readout(q, S, z) )             # folded middle -> linear state

with a fixed exact-KV budget B = |sink| + |window|. The state substrate is the
q-independent fold built at eviction time:

    S = sum_i g_i phi(k_i) v_i^T   [d, dv]
    z = sum_i g_i phi(k_i)         [d]

read back at decode as  N = phi(q) @ S,  D = phi(q) @ z,  o_state = N / D,
lse_state = log D  (natural-log LSE, absolute exp(scale*q.k) frame). The window
substrate emits  o_win = softmax(scale q.K_win) V_win,  lse_win = logsumexp(scale q.K_win).
Merging with the shared ``merge_state`` (natural-log LSE) reproduces C0's
absolute-exp-frame fold merge exactly:

    merge_state(o_win, lse_win, o_state, lse_state)
      = (N_win + N_mid) / (D_win + D_mid)              # == fold_vs_drop_c0.eval_budget["fold"]

Feature map is df=d (serving-viable, matches GDN's own [V,K] delta-rule state):
elu1 (phi(x)=elu(sqrt(s) x)+1, positive) or l2norm-identity. taylor2 (df=d^2) is
an OFFLINE accuracy reference only (fold_vs_drop_c0.py) and is NOT built here.

Nothing here is wired into any release/default path; C2-a is a standalone GPU test.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import triton
import triton.language as tl

# Shared natural-log LSE merge (sgl_kernel merge_state_v2, triton fallback).
from sglang.srt.layers.attention.merge_state import merge_state

# phi ids (kept in sync with the host feature maps below and the kernels).
PHI_ELU1 = 0
PHI_L2NORM = 1
PHI_IDS = {"elu1": PHI_ELU1, "l2norm": PHI_L2NORM}


# --------------------------------------------------------------------------- #
# Host feature maps (df=d). MUST match the in-kernel phi exactly so the fold
# built here (fold_state_at_evict) and the readout kernel agree numerically.
# --------------------------------------------------------------------------- #
def phi_elu1(x: torch.Tensor, scale: float) -> torch.Tensor:
    """phi(x) = elu(sqrt(scale) * x) + 1  (positive; matches fold_vs_drop_c0.phi_elu1)."""
    return torch.nn.functional.elu(math.sqrt(scale) * x) + 1.0


def phi_l2norm(x: torch.Tensor, scale: float) -> torch.Tensor:
    """phi(x) = x / ||x||_2  (df=d identity feature; GDN delta-rule flavour)."""
    return torch.nn.functional.normalize(x, dim=-1)


PHI = {"elu1": phi_elu1, "l2norm": phi_l2norm}


# --------------------------------------------------------------------------- #
# Eviction-time fold: build q-independent linear state (df=d). query UNKNOWN.
# --------------------------------------------------------------------------- #
def fold_state_at_evict(
    k_mid: torch.Tensor,
    v_mid: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    phi: str = "elu1",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold the evicted middle segment into (S, z). df=d.

    Shapes (batched):
      k_mid [B, H, M, d], v_mid [B, H, M, dv], g [B, H, M] (or [M], broadcast).
    Returns S [B, H, d, dv], z [B, H, d].

    g_i is the per-token retention weight (decay^age or the real GDN gate product).
    Matches fold_vs_drop_c0.fold_state at df=d.
    """
    if k_mid.shape[-2] == 0:
        B, H, _, d = k_mid.shape
        dv = v_mid.shape[-1]
        z = k_mid.new_zeros(B, H, d)
        S = k_mid.new_zeros(B, H, d, dv)
        return S, z
    pk = PHI[phi](k_mid, scale)  # [B,H,M,d]
    if g.dim() == 1:
        g = g.view(1, 1, -1)
    gpk = (g.unsqueeze(-1) * pk).to(torch.float32)  # [B,H,M,d]
    v = v_mid.to(torch.float32)
    S = torch.einsum("bhmd,bhme->bhde", gpk, v)  # [B,H,d,dv]
    z = gpk.sum(dim=-2)  # [B,H,d]
    return S, z


def decay_weights(M: int, decay: float, device, dtype=torch.float32) -> torch.Tensor:
    """g_i = decay^age, age = M-1 (oldest) .. 0 (newest, closest to window).

    Matches fold_vs_drop_c0.fold_state's ``ages`` ordering.
    """
    ages = torch.arange(M - 1, -1, -1, device=device, dtype=dtype)
    return decay**ages


# --------------------------------------------------------------------------- #
# state_readout triton kernel: N = phi(q) @ S, D = phi(q) @ z, o = N/D, lse=log D
# --------------------------------------------------------------------------- #
@triton.jit
def _state_readout_kernel(
    q_ptr,  # [B, H, d]
    S_ptr,  # [B, H, d, dv] fp32
    z_ptr,  # [B, H, d]     fp32
    o_ptr,  # [B, H, dv]    out
    lse_ptr,  # [B, H]      out (natural log D)
    scale,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_sb,
    stride_sh,
    stride_sd,
    stride_sv,
    stride_zb,
    stride_zh,
    stride_zd,
    stride_ob,
    stride_oh,
    stride_ov,
    stride_lb,
    stride_lh,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    PHI: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    off_d = tl.arange(0, BLOCK_D)
    off_v = tl.arange(0, BLOCK_DV)
    mask_d = off_d < D
    mask_v = off_v < DV

    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + off_d * stride_qd, mask=mask_d, other=0.0)
    q = q.to(tl.float32)

    # feature map phi(q), df=d
    sq = tl.sqrt(scale)
    if PHI == 0:  # elu1: elu(sqrt(scale) q) + 1
        a = sq * q
        phi_q = tl.where(a > 0, a, tl.exp(a) - 1.0) + 1.0
    else:  # l2norm: q / ||q||_2
        nrm = tl.sqrt(tl.sum(q * q, axis=0) + 1e-12)
        phi_q = q / nrm
    phi_q = tl.where(mask_d, phi_q, 0.0)

    # S tile [BLOCK_D, BLOCK_DV]; N[dv] = sum_d phi_q[d] * S[d, dv]
    s_off = (
        b * stride_sb
        + h * stride_sh
        + off_d[:, None] * stride_sd
        + off_v[None, :] * stride_sv
    )
    S = tl.load(s_off + S_ptr, mask=mask_d[:, None] & mask_v[None, :], other=0.0)
    N = tl.sum(phi_q[:, None] * S, axis=0)  # [BLOCK_DV]

    z = tl.load(z_ptr + b * stride_zb + h * stride_zh + off_d * stride_zd, mask=mask_d, other=0.0)
    Dnum = tl.sum(phi_q * z, axis=0)  # scalar
    Dnum = tl.maximum(Dnum, 1e-20)

    o = N / Dnum
    tl.store(o_ptr + b * stride_ob + h * stride_oh + off_v * stride_ov, o, mask=mask_v)
    tl.store(lse_ptr + b * stride_lb + h * stride_lh, tl.log(Dnum))


def state_readout(
    q: torch.Tensor,
    S: torch.Tensor,
    z: torch.Tensor,
    scale: float,
    phi: str = "elu1",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read the folded state. q [B,H,d], S [B,H,d,dv], z [B,H,d].

    Returns o_state [B,H,dv], lse_state [B,H] (natural log D, absolute exp frame).
    """
    B, H, d = q.shape
    dv = S.shape[-1]
    S = S.to(torch.float32).contiguous()
    z = z.to(torch.float32).contiguous()
    o = q.new_empty(B, H, dv, dtype=torch.float32)
    lse = q.new_empty(B, H, dtype=torch.float32)
    grid = (B, H)
    _state_readout_kernel[grid](
        q,
        S,
        z,
        o,
        lse,
        float(scale),
        q.stride(0), q.stride(1), q.stride(2),
        S.stride(0), S.stride(1), S.stride(2), S.stride(3),
        z.stride(0), z.stride(1), z.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lse.stride(0), lse.stride(1),
        D=d,
        DV=dv,
        BLOCK_D=triton.next_power_of_2(d),
        BLOCK_DV=triton.next_power_of_2(dv),
        PHI=PHI_IDS[phi],
    )
    return o, lse


# --------------------------------------------------------------------------- #
# Windowed exact flash-decode kernel over the B window tokens (sink + recent).
# Emits o_win [B,H,dv] and lse_win [B,H] = logsumexp(scale q.K_win) (natural log).
# --------------------------------------------------------------------------- #
@triton.jit
def _window_decode_kernel(
    q_ptr,  # [B, H, d]
    k_ptr,  # [B, H, T, d]
    v_ptr,  # [B, H, T, dv]
    o_ptr,  # [B, H, dv]
    lse_ptr,  # [B, H]
    scale,
    T,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ov,
    stride_lb, stride_lh,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # int64 offsets: large N * batch * H * d can exceed int32 (2^31).
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)

    off_d = tl.arange(0, BLOCK_D).to(tl.int64)
    off_v = tl.arange(0, BLOCK_DV).to(tl.int64)
    mask_d = off_d < D
    mask_v = off_v < DV

    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + off_d * stride_qd, mask=mask_d, other=0.0)
    q = q.to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    for t0 in range(0, T, BLOCK_T):
        off_t = (t0 + tl.arange(0, BLOCK_T)).to(tl.int64)
        mask_t = off_t < T
        # K tile [BLOCK_T, BLOCK_D]
        k_off = (
            b * stride_kb + h * stride_kh
            + off_t[:, None] * stride_kt + off_d[None, :] * stride_kd
        )
        k = tl.load(k_ptr + k_off, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1) * scale  # [BLOCK_T]
        logits = tl.where(mask_t, logits, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(logits, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new)  # [BLOCK_T]
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha

        v_off = (
            b * stride_vb + h * stride_vh
            + off_t[:, None] * stride_vt + off_v[None, :] * stride_vd
        )
        v = tl.load(v_ptr + v_off, mask=mask_t[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(p[:, None] * v, axis=0)  # [BLOCK_DV]
        m_i = m_new

    o = acc / l_i
    lse = m_i + tl.log(l_i)  # natural-log logsumexp
    tl.store(o_ptr + b * stride_ob + h * stride_oh + off_v * stride_ov, o, mask=mask_v)
    tl.store(lse_ptr + b * stride_lb + h * stride_lh, lse)


def window_decode(
    q: torch.Tensor,
    k_win: torch.Tensor,
    v_win: torch.Tensor,
    scale: float,
    block_t: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact windowed softmax decode. q [B,H,d], k_win [B,H,T,d], v_win [B,H,T,dv].

    Returns o_win [B,H,dv], lse_win [B,H] (natural-log logsumexp of scale*q.k).
    """
    B, H, d = q.shape
    T = k_win.shape[2]
    dv = v_win.shape[-1]
    o = q.new_empty(B, H, dv, dtype=torch.float32)
    lse = q.new_empty(B, H, dtype=torch.float32)
    grid = (B, H)
    _window_decode_kernel[grid](
        q, k_win, v_win, o, lse,
        float(scale),
        T,
        q.stride(0), q.stride(1), q.stride(2),
        k_win.stride(0), k_win.stride(1), k_win.stride(2), k_win.stride(3),
        v_win.stride(0), v_win.stride(1), v_win.stride(2), v_win.stride(3),
        o.stride(0), o.stride(1), o.stride(2),
        lse.stride(0), lse.stride(1),
        D=d,
        DV=dv,
        BLOCK_D=triton.next_power_of_2(d),
        BLOCK_DV=triton.next_power_of_2(dv),
        BLOCK_T=block_t,
    )
    return o, lse


# --------------------------------------------------------------------------- #
# Fused fold-decode driver: window substrate + state substrate, LSE-merged.
# --------------------------------------------------------------------------- #
def fold_decode(
    q: torch.Tensor,
    k_win: torch.Tensor,
    v_win: torch.Tensor,
    S: torch.Tensor,
    z: torch.Tensor,
    scale: float,
    phi: str = "elu1",
) -> torch.Tensor:
    """Fused decode over [sink+window] exact KV + folded (S,z) middle state.

    q [B,H,d], k_win/v_win [B,H,T,d]/[B,H,T,dv] (sink prepended into the window),
    S [B,H,d,dv], z [B,H,d]. Returns o [B,H,dv].
    """
    o_win, lse_win = window_decode(q, k_win, v_win, scale)
    o_state, lse_state = state_readout(q, S, z, scale, phi)

    # merge_state expects [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] & [NUM_TOKENS, NUM_HEADS].
    # Flatten (B,H) -> tokens=B, heads=H (already that shape).
    o_merged, _ = merge_state(
        o_win.contiguous(),
        lse_win.contiguous(),
        o_state.to(o_win.dtype).contiguous(),
        lse_state.contiguous(),
    )
    return o_merged


# --------------------------------------------------------------------------- #
# Pure-torch reference of the exact same fused path (for the kernel's own check).
# --------------------------------------------------------------------------- #
def fold_decode_ref(q, k_win, v_win, S, z, scale, phi="elu1"):
    """Torch reference: same two-substrate LSE merge, no triton. For test cross-check."""
    # window
    logits = scale * torch.einsum("bhd,bhtd->bht", q.float(), k_win.float())
    m = logits.max(dim=-1, keepdim=True).values
    p = torch.exp(logits - m)
    l = p.sum(dim=-1)
    o_win = torch.einsum("bht,bhtv->bhv", p, v_win.float()) / l.unsqueeze(-1)
    lse_win = (m.squeeze(-1) + torch.log(l))
    # state
    pq = PHI[phi](q.float(), scale)
    N = torch.einsum("bhd,bhdv->bhv", pq, S.float())
    Dnum = torch.einsum("bhd,bhd->bh", pq, z.float()).clamp_min(1e-20)
    o_state = N / Dnum.unsqueeze(-1)
    lse_state = torch.log(Dnum)
    # merge in absolute frame
    mmax = torch.maximum(lse_win, lse_state)
    sw = torch.exp(lse_win - mmax)
    ss = torch.exp(lse_state - mmax)
    o = (o_win * sw.unsqueeze(-1) + o_state * ss.unsqueeze(-1)) / (sw + ss).unsqueeze(-1)
    return o
