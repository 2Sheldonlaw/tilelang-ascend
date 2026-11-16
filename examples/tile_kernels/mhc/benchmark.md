# MHC Expand (expand_to_mhc) Benchmark & Optimization Record

## 1. Operator

```
fwd: o[n, m, h]      = x[n, h]                    # broadcast mhc copies
bwd: x_grad[n, h]    = sum_m o_grad[n, m, h]      # fp32 accumulate -> bf16
```

- fwd input `x [n, h]` bf16, output `o [n, mhc, h]` bf16
- bwd input `o_grad [n, mhc, h]` bf16, output `x_grad [n, h]` bf16
- mhc = 4 (tested), h must be 16-aligned after host-side padding (kernel tail requires 16bf16 = 32B granule)

## 2. Hardware & Software

| Item | Value |
|------|-------|
| NPU | Ascend 910B |
| CANN | 9.0.0 |
| Tool | `tilelang.profiler.do_bench` (torch.npu.Event, auto warmup + L2 flush) |
| Dtype | bf16 in/out, fp32 accumulate (bwd) |

## 3. Kernel Design

| Aspect | Implementation |
|--------|----------------|
| Parallelism | `T.Kernel(ceildiv(n, VEC_NUM), is_npu=True)`, VEC_NUM=2, one token row per vector core |
| Tiling | hidden split by `h_blk` (`T.Pipelined(num_stages=2)`); tail tile handled as a compile-time-sized epilogue |
| fwd compute | `T.copy` GM->UB (x once) -> `T.tile.broadcast` (mhc copies in UB) -> single 2D `T.copy` UB->GM |
| bwd compute | single 2D `T.copy` GM->UB -> `T.tile.cast` bf16->fp32 -> `T.reduce_sum(dim=0)` -> `T.tile.cast` fp32->bf16 -> `T.copy` UB->GM |
| h_blk selection | candidates `(4096, 3584, 3072, 2560, 2048, 1024, 512)`, largest divisor of h; else smallest padded length |
| Non-aligned h | host-side pad to 16-multiple (`_align_up_16`), result sliced back |

## 4. Correctness

`test_expand_to_mhc.py` (ported from TileKernels `tests/mhc/test_expand.py`, same grid):
n0 in {1,2} x n1 in {1024,4096} x mhc=4 x h in {1280,1920,2560,4224,5120,7168} = **24/24 passed**.

- fwd: bit-exact vs `x.unsqueeze(1).expand(...).contiguous()`
- bwd: max diff 0.0 vs `torch.autograd.backward` reference

## 5. Performance (910B, do_bench)

Baselines (situation A, aclnn names confirmed via torch profiler):
- fwd baseline: `o.copy_(x.unsqueeze(1).expand(n, mhc, h))` -> **aclnnInplaceCopy**
- bwd baseline: `o_grad.sum(dim=1)` -> **aclnnReduceSum**
- Acceptance: tilelang <= aclnn / 0.6
- Measured copy peak bandwidth: **894 GB/s**

### fwd

| n | mhc | h | tilelang (ms) | aclnn (ms) | ratio (aclnn/tl) | eff. BW (GB/s) | verdict |
|---|-----|---|---------------|------------|------------------|----------------|---------|
| 512 | 4 | 7168 | 0.0476 | 0.0369 | 0.78x | 772 | OK |
| 512 | 4 | 5120 | 0.0397 | 0.0278 | 0.70x | 660 | OK |
| 512 | 4 | 4096 | 0.0262 | 0.0229 | 0.87x | 801 | OK |
| 8192 | 4 | 7168 | 0.6778 | 0.4771 | 0.70x | 866 | OK |
| 8192 | 4 | 5120 | 0.5564 | 0.3504 | 0.63x | 754 | OK |
| 8192 | 4 | 4096 | 0.3443 | 0.2870 | 0.83x | 975 | OK |

### bwd

| n | mhc | h | tilelang (ms) | aclnn (ms) | ratio (aclnn/tl) | eff. BW (GB/s) | verdict |
|---|-----|---|---------------|------------|------------------|----------------|---------|
| 512 | 4 | 7168 | 0.0619 | 0.0624 | 1.01x | 593 | OK |
| 512 | 4 | 5120 | 0.0486 | 0.0480 | 0.99x | 540 | OK |
| 512 | 4 | 4096 | 0.0360 | 0.0415 | 1.15x | 582 | OK |
| 8192 | 4 | 7168 | 0.7451 | 0.6754 | 0.91x | 788 | OK |
| 8192 | 4 | 5120 | 0.6143 | 0.5220 | 0.85x | 683 | OK |
| 8192 | 4 | 4096 | 0.4615 | 0.4582 | 0.99x | 727 | OK |

**12/12 cases pass the 0.6x acceptance line.**

## 6. Tuning Log

| # | Issue | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | fwd/bwd all-NaN | `out_idx=[1]` combined with a pre-allocated output argument (JIT ignores/aliases the passed tensor) | drop `out_idx`, pass output explicitly (matches TileKernels `fwd(x, o)` interface) |
| 2 | `Cannot find pre-allocated address for buffer: o_ub` | `TL_ASCEND_MEMORY_PLANNING=True` incompatible with the Pipelined-loop + tail-epilogue structure | set `TL_ASCEND_MEMORY_PLANNING=False` |
| 3 | NaN for h=1000/1016 | non-16-aligned tail (UB transfers need 32B granules) | host-side pad to 16-multiple; kernel tail supports 16-aligned |
| 4 | fwd (8192,4096) 0.57x < 0.6x | h_blk=2048 (2 tiles) loop overhead | add 4096 to h_blk candidates -> 0.83x |
