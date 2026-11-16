"""Benchmark expand_to_mhc fwd/bwd vs torch_npu (aclnn) baselines.

Baselines (situation A of the migration task):
  fwd: o.copy_(x.unsqueeze(1).expand(n, mhc, h))   (aclnn Expand / copy path)
  bwd: o_grad.sum(dim=1)                            (aclnn ReduceSum path)

Acceptance: tilelang time <= aclnn time / 0.6.
"""

import argparse

import torch

from tilelang.profiler import do_bench

from expand_to_mhc_kernel import expand_to_mhc_bwd, expand_to_mhc_fwd

BENCH_CASES = [
    (512, 4, 7168),
    (512, 4, 5120),
    (512, 4, 4096),
    (8192, 4, 7168),
    (8192, 4, 5120),
    (8192, 4, 4096),
]


def bench_fwd(n: int, mhc: int, h: int) -> dict:
    device = "npu"
    x = torch.randn(n, h, dtype=torch.bfloat16, device=device)
    o = torch.empty(n, mhc, h, dtype=torch.bfloat16, device=device)

    t_tl = do_bench(lambda: expand_to_mhc_fwd(x, o))
    x_exp = x.unsqueeze(1).expand(n, mhc, h)
    t_aclnn = do_bench(lambda: o.copy_(x_exp))

    io_bytes = (n * h + n * mhc * h) * 2
    bw = io_bytes / (t_tl * 1e-3) / 1e9
    return {"t_tl_ms": t_tl, "t_aclnn_ms": t_aclnn, "speedup": t_aclnn / t_tl, "bw_gbs": bw}


def bench_bwd(n: int, mhc: int, h: int) -> dict:
    device = "npu"
    o_grad = torch.randn(n, mhc, h, dtype=torch.bfloat16, device=device)
    x_grad = torch.empty(n, h, dtype=torch.bfloat16, device=device)

    t_tl = do_bench(lambda: expand_to_mhc_bwd(o_grad, x_grad))
    t_aclnn = do_bench(lambda: o_grad.sum(dim=1))

    io_bytes = (n * mhc * h + n * h) * 2
    bw = io_bytes / (t_tl * 1e-3) / 1e9
    return {"t_tl_ms": t_tl, "t_aclnn_ms": t_aclnn, "speedup": t_aclnn / t_tl, "bw_gbs": bw}


def measure_bw_peak() -> float:
    device = "npu"
    x = torch.randn(128 * 1024 * 1024 // 4, dtype=torch.float32, device=device)
    y = torch.empty_like(x)
    t = do_bench(lambda: y.copy_(x))
    return 2 * x.numel() * 4 / (t * 1e-3) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, nargs="*", default=None, help="indices into BENCH_CASES")
    args = parser.parse_args()

    cases = BENCH_CASES if args.cases is None else [BENCH_CASES[i] for i in args.cases]

    bw_peak = measure_bw_peak()
    print(f"measured copy peak bandwidth: {bw_peak:.1f} GB/s")

    print("\nfwd: tilelang vs aclnn Expand/copy (accept: tl <= aclnn/0.6)")
    for n, mhc, h in cases:
        r = bench_fwd(n, mhc, h)
        ok = r["t_tl_ms"] <= r["t_aclnn_ms"] / 0.6
        print(
            f"  n={n:5d} mhc={mhc} h={h:5d}: tl={r['t_tl_ms']:.4f}ms "
            f"aclnn={r['t_aclnn_ms']:.4f}ms speedup={r['speedup']:.2f}x "
            f"bw={r['bw_gbs']:.0f}GB/s {'OK' if ok else 'FAIL'}"
        )

    print("\nbwd: tilelang vs aclnn ReduceSum (accept: tl <= aclnn/0.6)")
    for n, mhc, h in cases:
        r = bench_bwd(n, mhc, h)
        ok = r["t_tl_ms"] <= r["t_aclnn_ms"] / 0.6
        print(
            f"  n={n:5d} mhc={mhc} h={h:5d}: tl={r['t_tl_ms']:.4f}ms "
            f"aclnn={r['t_aclnn_ms']:.4f}ms speedup={r['speedup']:.2f}x "
            f"bw={r['bw_gbs']:.0f}GB/s {'OK' if ok else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
