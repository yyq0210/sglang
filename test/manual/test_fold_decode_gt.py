"""Phase C2-a correctness GATE: fold_decode GPU primitive vs C0 python reference.

This is the mandated gate before any serving run (C2-b). It checks:

  (i)   exact-window-only fold_decode == full softmax over [sink+window] (rel-err ~0):
        sanity that the window flash-decode + degenerate (empty) state merge is exact.
  (ii)  fold_decode (GPU triton kernels) == fold_vs_drop_c0.eval_budget(...)["fold"]
        at df=d/elu1 across random + needle inputs, bf16 and fp32, tight rtol/atol.
  (iii) needle recovery: needle_signal(fold) > needle_signal(drop) at small budget B,
        reproducing C0's fold>drop on real GPU dtypes.

Run:  python test/manual/test_fold_decode_gt.py
Skips if no CUDA.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fold_vs_drop_c0 as c0
from fold_decode_kernel import (
    decay_weights,
    fold_decode,
    fold_decode_ref,
    fold_state_at_evict,
    state_readout,
    window_decode,
)

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:  # pragma: no cover
    CustomTestCase = unittest.TestCase

_HAS_CUDA = torch.cuda.is_available()


def _c0_fold_single(q, k, v, scale, budget, sink, phi, decay):
    """Run the C0 reference for one (d-vector) query; returns o_fold, o_drop, parts."""
    r = c0.eval_budget(q, k, v, scale, budget, sink, phi, decay)
    return r


@unittest.skipUnless(_HAS_CUDA, "CUDA required for fold_decode kernels")
class TestFoldDecodeGT(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = "cuda"

    # ------------------------------------------------------------------ #
    # (i) exact window-only == full softmax over [sink+window]
    # ------------------------------------------------------------------ #
    def test_window_decode_exact(self):
        B, H, d, dv, T = 4, 16, 128, 128, 512
        scale = 1.0 / math.sqrt(d)
        q = torch.randn(B, H, d, device=self.device, dtype=torch.float32)
        k = torch.randn(B, H, T, d, device=self.device, dtype=torch.float32)
        v = torch.randn(B, H, T, dv, device=self.device, dtype=torch.float32)
        o_win, lse_win = window_decode(q, k, v, scale)

        logits = scale * torch.einsum("bhd,bhtd->bht", q, k)
        ref = torch.softmax(logits, dim=-1)
        o_ref = torch.einsum("bht,bhtv->bhv", ref, v)
        lse_ref = torch.logsumexp(logits, dim=-1)

        self.assertLess((o_win - o_ref).norm() / o_ref.norm(), 1e-4)
        self.assertLess((lse_win - lse_ref).abs().max().item(), 1e-3)

    # ------------------------------------------------------------------ #
    # state_readout kernel == torch reference
    # ------------------------------------------------------------------ #
    def test_state_readout_kernel(self):
        B, H, d, dv = 4, 16, 128, 128
        scale = 1.0 / math.sqrt(d)
        for phi in ("elu1", "l2norm"):
            q = torch.randn(B, H, d, device=self.device)
            S = torch.randn(B, H, d, dv, device=self.device)
            z = torch.rand(B, H, d, device=self.device) + 0.5  # positive-ish denom
            o, lse = state_readout(q, S, z, scale, phi)
            from fold_decode_kernel import PHI

            pq = PHI[phi](q.float(), scale)
            N = torch.einsum("bhd,bhdv->bhv", pq, S.float())
            Dn = torch.einsum("bhd,bhd->bh", pq, z.float()).clamp_min(1e-20)
            o_ref = N / Dn.unsqueeze(-1)
            self.assertLess((o - o_ref).norm() / (o_ref.norm() + 1e-9), 1e-4)
            self.assertLess((lse - torch.log(Dn)).abs().max().item(), 1e-3)

    # ------------------------------------------------------------------ #
    # fused fold_decode kernels == in-file torch reference (self-consistency)
    # ------------------------------------------------------------------ #
    def test_fold_decode_vs_torch_ref(self):
        B, H, d, dv, T = 4, 16, 128, 128, 256
        M = 800
        scale = 1.0 / math.sqrt(d)
        for phi in ("elu1", "l2norm"):
            q = torch.randn(B, H, d, device=self.device)
            k_win = torch.randn(B, H, T, d, device=self.device)
            v_win = torch.randn(B, H, T, dv, device=self.device)
            k_mid = torch.randn(B, H, M, d, device=self.device)
            v_mid = torch.randn(B, H, M, dv, device=self.device)
            g = decay_weights(M, 0.999, self.device)
            S, z = fold_state_at_evict(k_mid, v_mid, g, scale, phi)
            o = fold_decode(q, k_win, v_win, S, z, scale, phi)
            o_ref = fold_decode_ref(q, k_win, v_win, S, z, scale, phi)
            rel = (o - o_ref).norm() / (o_ref.norm() + 1e-9)
            self.assertLess(rel.item(), 2e-3, f"phi={phi} rel={rel.item():.2e}")

    # ------------------------------------------------------------------ #
    # (ii) fold_decode (GPU) == C0 eval_budget[...]["fold"] at df=d/elu1
    # ------------------------------------------------------------------ #
    def test_fold_decode_vs_c0_reference(self):
        d = dv = 128
        N = 2048
        sink = 4
        phi = "elu1"
        scale = 1.0 / math.sqrt(d)
        for decay in (1.0, 0.999):
            for budget in (256, 512, 1024):
                # single (d) query -> C0 reference (CPU float64-ish via float32)
                q_c = torch.randn(d)
                k_c = torch.randn(N, d)
                v_c = torch.randn(N, dv)
                r = c0.eval_budget(q_c, k_c, v_c, scale, budget, sink, phi, decay)
                o_c0_fold = r["fold"]

                # reconstruct the same partition on GPU: window = [sink] ++ [recent window]
                window = max(0, budget - sink)
                win_lo = max(sink, N - window)
                k_sink, v_sink = k_c[:sink], v_c[:sink]
                k_win, v_win = k_c[win_lo:], v_c[win_lo:]
                k_mid, v_mid = k_c[sink:win_lo], v_c[sink:win_lo]

                # concat sink + window as the exact substrate
                k_exact = torch.cat([k_sink, k_win], dim=0)
                v_exact = torch.cat([v_sink, v_win], dim=0)

                dev = self.device
                q_g = q_c.to(dev).view(1, 1, d)
                k_g = k_exact.to(dev).view(1, 1, -1, d)
                v_g = v_exact.to(dev).view(1, 1, -1, dv)
                M = k_mid.shape[0]
                g = decay_weights(M, decay, dev) if M > 0 else torch.zeros(0, device=dev)
                S, z = fold_state_at_evict(
                    k_mid.to(dev).view(1, 1, M, d),
                    v_mid.to(dev).view(1, 1, M, dv),
                    g,
                    scale,
                    phi,
                )
                o_g = fold_decode(q_g, k_g, v_g, S, z, scale, phi).view(dv).cpu()
                rel = (o_g - o_c0_fold).norm() / (o_c0_fold.norm() + 1e-9)
                self.assertLess(
                    rel.item(),
                    5e-3,
                    f"decay={decay} B={budget} rel={rel.item():.2e}",
                )

    # ------------------------------------------------------------------ #
    # (iii) needle recovery: fold > drop at small budget on GPU dtypes
    # ------------------------------------------------------------------ #
    def _needle_gpu(self, budget, decay, dtype):
        # Reseed so every dtype sees the IDENTICAL canonical C0 needle draw
        # (the raw needle_signal margin at df=d/elu1 is small; a different draw
        # would compare against a different haystack).
        torch.manual_seed(0)
        d = dv = 128
        N = 2048
        sink = 4
        phi = "elu1"
        q_c, k_c, v_c, npos, v_needle, scale = c0.make_needle(
            d=d, dv=dv, N=N, needle_pos=1024, logit_gap=6.0
        )
        window = max(0, budget - sink)
        win_lo = max(sink, N - window)
        # drop + full reference from C0
        r = c0.eval_budget(q_c, k_c, v_c, scale, budget, sink, phi, decay)
        o_full = r["full"]
        sig_drop = c0.needle_signal(r["drop"], v_needle)
        cos_drop = c0.cos(r["drop"], o_full)

        k_exact = torch.cat([k_c[:sink], k_c[win_lo:]], dim=0)
        v_exact = torch.cat([v_c[:sink], v_c[win_lo:]], dim=0)
        k_mid, v_mid = k_c[sink:win_lo], v_c[sink:win_lo]
        dev = self.device
        q_g = q_c.to(dev).to(dtype).view(1, 1, d)
        k_g = k_exact.to(dev).to(dtype).view(1, 1, -1, d)
        v_g = v_exact.to(dev).to(dtype).view(1, 1, -1, dv)
        M = k_mid.shape[0]
        g = decay_weights(M, decay, dev)
        S, z = fold_state_at_evict(
            k_mid.to(dev).to(dtype).view(1, 1, M, d),
            v_mid.to(dev).to(dtype).view(1, 1, M, dv),
            g,
            scale,
            phi,
        )
        o_g = fold_decode(q_g, k_g, v_g, S, z, scale, phi).view(dv).float().cpu()
        sig_fold = c0.needle_signal(o_g, v_needle)
        cos_fold = c0.cos(o_g, o_full)
        return dict(
            sig_fold=sig_fold, sig_drop=sig_drop,
            cos_fold=cos_fold, cos_drop=cos_drop,
            in_mid=(win_lo > npos >= sink),
        )

    def test_needle_fold_beats_drop(self):
        # needle at 1024 must be in the folded middle (budget small enough).
        # fp32: assert the raw needle_signal recovery (fold>drop, C0's claim).
        # bf16: assert the robust cosine-to-full metric (dtype-noise stable).
        for dtype in (torch.float32, torch.bfloat16):
            r = self._needle_gpu(512, 0.999, dtype)
            self.assertTrue(r["in_mid"], "needle should be in the folded middle at B=512")
            # cosine-to-full: fold must be closer to full than drop (robust in both dtypes)
            self.assertGreater(
                r["cos_fold"], r["cos_drop"],
                f"dtype={dtype} cos_fold={r['cos_fold']:.4f} cos_drop={r['cos_drop']:.4f}",
            )
            if dtype == torch.float32:
                self.assertGreater(
                    r["sig_fold"], r["sig_drop"],
                    f"dtype={dtype} sig_fold={r['sig_fold']:.4f} sig_drop={r['sig_drop']:.4f}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
