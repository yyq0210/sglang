"""Phase C2-a microbench: fold-decode (window B + state readout) vs full dense decode.

Mirrors bench_replayssm_amortized.py's cuda-event manual-loop style. Measures the
per-step decode latency of:

  * fold-decode : windowed exact attention over B [sink+recent] tokens + a df=d state
                  readout (phi(q)@S, phi(q)@z), LSE-merged. Cost ~ O(B + d*dv), so it
                  is CONSTANT in the true context length N (the middle is folded once).
  * full-decode : dense flash-decode over the entire context of N tokens (the baseline
                  a normal decode pays every step). Cost ~ O(N).

Swept over exact-KV budget B, context length N, and batch. Reports the crossover N
beyond which fold-decode is cheaper (the whole point: fold trades an O(N) scan for an
O(B) window + O(d*dv) matvec, so it wins as N grows / B shrinks).

Run:  python test/manual/bench_fold_decode.py
Requires a GPU (Triton).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fold_decode_kernel import fold_decode, window_decode


def _time(fn, n_steps, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_steps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_steps  # ms/step


def _full_decode(q, k_full, v_full, scale):
    """Baseline: dense flash-decode over the whole context (reuses window_decode)."""
    return window_decode(q, k_full, v_full, scale)


def _bench_cfg(B_win, N, batch, H, d, dv, scale, phi, dtype, device, warmup, rep):
    q = torch.randn(batch, H, d, device=device, dtype=dtype)

    # fold-decode inputs: exact window of B_win tokens + folded state (S,z).
    k_win = torch.randn(batch, H, B_win, d, device=device, dtype=dtype)
    v_win = torch.randn(batch, H, B_win, dv, device=device, dtype=dtype)
    S = torch.randn(batch, H, d, dv, device=device, dtype=torch.float32)
    z = torch.rand(batch, H, d, device=device, dtype=torch.float32) + 0.5

    t_fold = _time(lambda: fold_decode(q, k_win, v_win, S, z, scale, phi), rep, warmup)

    # full-decode inputs: the entire context of N tokens (what a normal decode scans).
    # This dense KV can be huge (that is the point of folding); tolerate OOM.
    try:
        k_full = torch.randn(batch, H, N, d, device=device, dtype=dtype)
        v_full = torch.randn(batch, H, N, dv, device=device, dtype=dtype)
        t_full = _time(lambda: _full_decode(q, k_full, v_full, scale), rep, warmup)
        del k_full, v_full
    except torch.cuda.OutOfMemoryError:
        t_full = float("inf")  # dense context does not even fit -> fold wins by capacity
    del q, k_win, v_win, S, z
    torch.cuda.empty_cache()
    return t_fold, t_full


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--ns", type=int, nargs="+", default=[4096, 16384, 65536])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64, 256])
    ap.add_argument("--h", type=int, default=16)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--phi", choices=["elu1", "l2norm"], default="elu1")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rep", type=int, default=200)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA / Triton required.")
    device = "cuda"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    scale = 1.0 / (args.d**0.5)

    print(
        f"FOLD-DECODE microbench  phi={args.phi} H={args.h} d={args.d} dv={args.dv} "
        f"dtype={args.dtype}\n"
        "fold = window(B) + state-readout (O(B)+O(d*dv), constant in N);  "
        "full = dense decode over N (O(N))."
    )
    for batch in args.batch_sizes:
        print(f"\nbatch={batch}")
        print(
            f"  {'N':>7} {'B':>6} {'fold(ms)':>10} {'full(ms)':>10} "
            f"{'speedup':>9} {'winner':>8}"
        )
        for N in args.ns:
            for B in args.budgets:
                if B >= N:
                    continue
                t_fold, t_full = _bench_cfg(
                    B, N, batch, args.h, args.d, args.dv, scale, args.phi,
                    dtype, device, args.warmup, args.rep,
                )
                sp = t_full / t_fold
                winner = "fold" if sp > 1.0 else "full"
                full_str = "  OOM" if t_full == float("inf") else f"{t_full:>10.4f}"
                sp_str = "  cap∞" if t_full == float("inf") else f"{sp:>8.2f}x"
                print(
                    f"  {N:>7} {B:>6} {t_fold:>10.4f} {full_str:>10} "
                    f"{sp_str:>9} {winner:>8}"
                )


if __name__ == "__main__":
    main()
