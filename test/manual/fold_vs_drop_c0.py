"""Phase C0 (GATE-C0): OFFLINE fold-vs-drop eviction feasibility for hybrid
linear-attention KV eviction. EXPERIMENT-ONLY, zero new kernel, CPU-runnable.

Part of "State as a Memory Tier" (docs/hybrid_state_tiering_experiment_plan.md,
Phase C0). The structural form of the eviction (forced by the gated-decay path) is,
per full-attention layer:

    [ sink tokens (exact) ] + [ middle contiguous segment ] + [ recent window (exact) ]

with a fixed EXACT-KV budget B = |sink| + |window|. The middle segment is either
  * DROP  — discarded (SnapKV-style: only sink + recent window survive), or
  * FOLD  — folded at eviction time (query UNKNOWN) into a q-independent linear
            state S = sum_i g_i phi(k_i) v_i^T, z = sum_i g_i phi(k_i), then at decode
            read back N_mid = phi(q) @ S, D_mid = phi(q) @ z and LSE-MERGED with the
            exact softmax over sink+window.

GATE-C0 question: at small exact-KV budget, does FOLD recover the (dropped) middle's
contribution — most importantly a needle token that lives in the middle — better than
DROP? If the fold curve does NOT beat drop, do NOT build the kernel (C2).

C-risk (load-bearing): the two-substrate merge must be numerically correct. This file
FIRST unit-tests the merge on two EXACT partitions against full-attention ground truth
(must be ~0 error), and only then measures the lossy feature-map fold.

The feature map phi approximates the softmax kernel so the folded (linear) partition
lives in the same scale as the exact softmax partition:
  * taylor2 ("based"): <phi(q),phi(k)> = 1 + (s q.k) + (s q.k)^2/2  ~ exp(s q.k),
    phi(x) = [1, sqrt(s) x, (sqrt(s) x) outer (sqrt(s) x)/sqrt(2)] (always > 0).
  * elu1: phi(x) = elu(x)+1 (linear-attention standard; sharper-agnostic honest floor).
This is INSTRUMENTATION/analysis, not a product path.
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple

import torch

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Exact softmax attention (ground truth) and its online-softmax partition merge
# --------------------------------------------------------------------------- #
def softmax_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
    """Full softmax attention. q:[d]  k:[N,d]  v:[N,dv] -> o:[dv]."""
    logits = scale * (k @ q)  # [N]
    w = torch.softmax(logits, dim=0)
    return w @ v, logits, w


def partition_stats(q, k, v, scale):
    """Max-shifted (numerator, denom, max) for a token set, flash-attention style.
    Returns n:[dv] unnormalized-in-exp(-m) frame, d:scalar, m:scalar."""
    if k.numel() == 0:
        return (
            torch.zeros(v.shape[-1] if v.ndim == 2 else v.shape[-1]),
            torch.tensor(0.0),
            torch.tensor(float("-inf")),
        )
    logits = scale * (k @ q)
    m = logits.max()
    e = torch.exp(logits - m)  # [N]
    n = e @ v  # [dv]
    d = e.sum()  # scalar
    return n, d, m


def merge_partitions(parts):
    """Exact online-softmax merge of disjoint partitions [(n,d,m), ...] -> o:[dv].
    o = (sum_p exp(m_p - M) n_p) / (sum_p exp(m_p - M) d_p),  M = max m_p."""
    ms = torch.stack([p[2] for p in parts])
    M = ms.max()
    if torch.isinf(M):  # all empty
        return torch.zeros_like(parts[0][0])
    num = torch.zeros_like(parts[0][0])
    den = torch.tensor(0.0)
    for n, d, m in parts:
        if torch.isinf(m):
            continue
        s = torch.exp(m - M)
        num = num + s * n
        den = den + s * d
    return num / den


# --------------------------------------------------------------------------- #
# Feature maps phi (positive; approximate exp(scale * q.k))
# --------------------------------------------------------------------------- #
def phi_taylor2(x: torch.Tensor, scale: float) -> torch.Tensor:
    """x:[...,d] -> [..., 1 + d + d*d]. <phi(a),phi(b)> = 1 + s(a.b) + (s(a.b))^2/2."""
    a = math.sqrt(scale) * x
    d = a.shape[-1]
    ones = torch.ones(a.shape[:-1] + (1,), dtype=a.dtype)
    outer = (a.unsqueeze(-1) * a.unsqueeze(-2)).reshape(a.shape[:-1] + (d * d,)) / math.sqrt(2.0)
    return torch.cat([ones, a, outer], dim=-1)


def phi_elu1(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.nn.functional.elu(math.sqrt(scale) * x) + 1.0


PHI = {"taylor2": phi_taylor2, "elu1": phi_elu1}


def fold_state(k_mid, v_mid, scale, phi_name, decay):
    """Fold middle tokens into q-independent linear state at eviction (q unknown).
    S = sum_i g_i phi(k_i) v_i^T  [df,dv];  z = sum_i g_i phi(k_i)  [df].
    g_i = decay^(age): most-recent middle token (closest to window) gets decay^0."""
    if k_mid.numel() == 0:
        return None, None
    phi = PHI[phi_name]
    pk = phi(k_mid, scale)  # [M, df]
    M = pk.shape[0]
    ages = torch.arange(M - 1, -1, -1, dtype=torch.float32)  # oldest..newest -> newest age 0
    g = decay ** ages  # [M]
    gpk = g.unsqueeze(-1) * pk  # [M, df]
    S = gpk.t() @ v_mid  # [df, dv]
    z = gpk.sum(dim=0)  # [df]
    return S, z


def fold_readout(q, S, z, scale, phi_name):
    """Read the folded state at decode: N_mid = phi(q)@S, D_mid = phi(q)@z (abs exp frame)."""
    if S is None:
        return torch.zeros(z.shape[-1] if z is not None else 1), torch.tensor(0.0)
    pq = PHI[phi_name](q, scale)  # [df]
    N = pq @ S  # [dv]
    D = pq @ z  # scalar
    return N, D.clamp_min(0.0)  # clamp guards tiny negative from elu tails


# --------------------------------------------------------------------------- #
# Eviction schemes at a fixed exact-KV budget B = sink + window
# --------------------------------------------------------------------------- #
def eval_budget(q, k, v, scale, budget, sink, phi_name, decay):
    """Return dict of outputs {full, drop, fold} for one exact-KV budget."""
    N = k.shape[0]
    window = max(0, budget - sink)
    win_lo = max(sink, N - window)  # window = [win_lo, N)
    # partitions
    k_sink, v_sink = k[:sink], v[:sink]
    k_win, v_win = k[win_lo:], v[win_lo:]
    k_mid, v_mid = k[sink:win_lo], v[sink:win_lo]

    o_full, logits_full, w_full = softmax_attn(q, k, v, scale)

    # exact partition = sink + window
    ps = partition_stats(q, k_sink, v_sink, scale)
    pw = partition_stats(q, k_win, v_win, scale)
    o_drop = merge_partitions([ps, pw])

    # fold: middle -> state, read back, merge in ABSOLUTE exp frame with exact part
    S, z = fold_state(k_mid, v_mid, scale, phi_name, decay)
    N_mid, D_mid = fold_readout(q, S, z, scale, phi_name)
    # bring exact (sink+window) into absolute exp frame, then add folded absolute-frame part
    M_exact = torch.stack([ps[2], pw[2]])
    Mx = M_exact[torch.isfinite(M_exact)].max() if torch.isfinite(M_exact).any() else torch.tensor(0.0)
    n_exact = torch.zeros_like(o_full)
    d_exact = torch.tensor(0.0)
    for n, d, m in (ps, pw):
        if torch.isinf(m):
            continue
        s = torch.exp(m - Mx)
        n_exact = n_exact + s * n
        d_exact = d_exact + s * d
    # fold readout is ~exp(scale q.k) in ABSOLUTE frame; rescale to the Mx frame
    scale_fold = torch.exp(-Mx)
    o_fold = (n_exact + scale_fold * N_mid) / (d_exact + scale_fold * D_mid)
    return dict(full=o_full, drop=o_drop, fold=o_fold, w_full=w_full,
                mid_lo=sink, mid_hi=win_lo)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def relerr(a, b):
    return (torch.norm(a - b) / (torch.norm(b) + 1e-9)).item()


# --------------------------------------------------------------------------- #
# Unit test: two EXACT partitions must merge to full softmax (GATE precondition)
# --------------------------------------------------------------------------- #
def unit_test_merge(d=128, dv=128, N=2048, scale=None, trials=20):
    scale = scale or 1.0 / math.sqrt(d)
    worst = 0.0
    for _ in range(trials):
        q = torch.randn(d)
        k = torch.randn(N, d)
        v = torch.randn(N, dv)
        o_full, _, _ = softmax_attn(q, k, v, scale)
        cut = torch.randint(1, N, (1,)).item()
        pa = partition_stats(q, k[:cut], v[:cut], scale)
        pb = partition_stats(q, k[cut:], v[cut:], scale)
        o_merge = merge_partitions([pa, pb])
        worst = max(worst, relerr(o_merge, o_full))
    return worst


# --------------------------------------------------------------------------- #
# Needle-in-haystack synthetic: one middle token strongly attended, rest diffuse
# --------------------------------------------------------------------------- #
def make_needle(d=128, dv=128, N=2048, needle_pos=1024, logit_gap=6.0, scale=None):
    """Build q,k,v where token[needle_pos] has a query-aligned key (top-1 logit) and a
    distinctive value e_0. Distractors are diffuse. Returns q,k,v,needle_pos,v_needle."""
    scale = scale or 1.0 / math.sqrt(d)
    q = torch.randn(d)
    q = q / q.norm()
    k = 0.3 * torch.randn(N, d)  # low-logit distractors
    v = torch.randn(N, dv)
    # needle key aligned with q so scale*(q.k) ~ logit_gap above the field
    k[needle_pos] = (logit_gap / scale) * q / (q @ q)
    v_needle = torch.zeros(dv)
    v_needle[0] = 3.0  # distinctive, large along e_0
    v[needle_pos] = v_needle
    return q, k, v, needle_pos, v_needle, scale


def needle_signal(o, v_needle):
    """Projection of output onto the needle value direction (its recovered mass)."""
    u = v_needle / (v_needle.norm() + 1e-9)
    return (o @ u).item()


def run_needle_sweep(args):
    q, k, v, npos, v_needle, scale = make_needle(
        d=args.d, dv=args.d, N=args.N, needle_pos=args.needle_pos,
        logit_gap=args.logit_gap,
    )
    o_full, _, w_full = softmax_attn(q, k, v, scale)
    full_sig = needle_signal(o_full, v_needle)
    needle_w = w_full[npos].item()
    print(f"[needle] N={args.N} pos={npos} logit_gap={args.logit_gap} "
          f"phi={args.phi} decay={args.decay}")
    print(f"[needle] full-attn: needle softmax weight={needle_w:.4f}  "
          f"needle-signal(o.u)={full_sig:.4f}")
    print(f"{'budget':>7} {'win_lo':>7} {'needle_in':>9} "
          f"{'sig_drop':>9} {'sig_fold':>9} {'sig_full':>9} "
          f"{'cos_drop':>9} {'cos_fold':>9} {'rel_drop':>9} {'rel_fold':>9}")
    for B in args.budgets:
        r = eval_budget(q, k, v, scale, B, args.sink, args.phi, args.decay)
        in_mid = r["mid_lo"] <= npos < r["mid_hi"]
        sd = needle_signal(r["drop"], v_needle)
        sf = needle_signal(r["fold"], v_needle)
        print(f"{B:>7} {r['mid_hi']:>7} {str(in_mid):>9} "
              f"{sd:>9.4f} {sf:>9.4f} {full_sig:>9.4f} "
              f"{cos(r['drop'], o_full):>9.4f} {cos(r['fold'], o_full):>9.4f} "
              f"{relerr(r['drop'], o_full):>9.4f} {relerr(r['fold'], o_full):>9.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--needle-pos", dest="needle_pos", type=int, default=1024)
    ap.add_argument("--logit-gap", dest="logit_gap", type=float, default=6.0)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--phi", choices=list(PHI), default="taylor2")
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 1152, 1536, 2048])
    ap.add_argument("--skip-unit", action="store_true")
    args = ap.parse_args()

    if not args.skip_unit:
        worst = unit_test_merge(d=args.d, dv=args.d, N=args.N)
        status = "PASS" if worst < 1e-4 else "FAIL"
        print(f"[unit] exact 2-partition merge vs full softmax: worst rel-err={worst:.2e} -> {status}")
        print(f"[unit] (GATE precondition: the LSE merge algebra is numerically correct)\n")

    run_needle_sweep(args)


if __name__ == "__main__":
    main()
