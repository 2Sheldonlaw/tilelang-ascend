"""expand_to_mhc kernel for Ascend NPU (TileLang-Ascend).

Forward:  o[n, m, h] = x[n, h]                     (broadcast mhc copies)
Backward: x_grad[n, h] = sum_m o_grad[n, m, h]     (fp32 accumulate -> bf16)

Port of TileKernels tile_kernels/mhc/expand_{kernel,asc,cuda}.py to the
tilelang-ascend high-level API (T.alloc_ub + T.copy + T.tile.*).

Strategy (borrowed from the A5 reference):
  - one vector-core pair per token row (grid = ceildiv(n, VEC_NUM), each
    block owns VEC_NUM rows via (cid, vid));
  - hidden dim is tiled by h_blk with double buffering (T.Pipelined);
  - fwd reads x once, replicates mhc times in UB (T.tile.broadcast), then
    writes the (mhc, h_blk) block with a single T.copy (MTE3);
  - bwd reads the (mhc, h_blk) block with a single T.copy, casts bf16 ->
    fp32, reduces along the mhc axis in fp32, casts back to bf16;
  - a non-divisible tail of h is handled with a compile-time-sized tail
    tile, so no host-side padding is needed.
"""

import tilelang
import torch
from tilelang import language as T

VEC_NUM = 2
H_BLK_CANDIDATES = (4096, 3584, 3072, 2560, 2048, 1024, 512)
_DTYPE = "bfloat16"
_ACCUM_DTYPE = "float32"

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_TAIL_MASK: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _expand_fwd_kernel(mhc: int, h: int, h_blk: int) -> tilelang.JITKernel:
    n = T.symbolic("n")
    num_full = h // h_blk
    tail = h - num_full * h_blk

    @T.prim_func
    def expand_fwd_kernel(
        x: T.Tensor((n, h), _DTYPE),
        o: T.Tensor((n, mhc, h), _DTYPE),
    ) -> None:
        with T.Kernel(T.ceildiv(n, VEC_NUM), is_npu=True) as (cid, vid):
            bid = cid * VEC_NUM + vid
            if bid < n:
                x_ub = T.alloc_ub((h_blk,), _DTYPE)
                o_ub = T.alloc_ub((mhc, h_blk), _DTYPE)

                if num_full > 0:
                    for i_h in T.Pipelined(num_full, num_stages=2):
                        h_start = i_h * h_blk
                        T.copy(x[bid, h_start : h_start + h_blk], x_ub)
                        T.tile.broadcast(o_ub, x_ub, axis=0)
                        T.copy(o_ub, o[bid, 0:mhc, h_start : h_start + h_blk])

                if tail > 0:
                    x_tail_ub = T.alloc_ub((tail,), _DTYPE)
                    o_tail_ub = T.alloc_ub((mhc, tail), _DTYPE)
                    T.copy(x[bid, num_full * h_blk : h], x_tail_ub)
                    T.tile.broadcast(o_tail_ub, x_tail_ub, axis=0)
                    T.copy(o_tail_ub, o[bid, 0:mhc, num_full * h_blk : h])

    return expand_fwd_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _expand_bwd_kernel(mhc: int, h: int, h_blk: int) -> tilelang.JITKernel:
    n = T.symbolic("n")
    num_full = h // h_blk
    tail = h - num_full * h_blk

    @T.prim_func
    def expand_bwd_kernel(
        o_grad: T.Tensor((n, mhc, h), _DTYPE),
        x_grad: T.Tensor((n, h), _DTYPE),
    ) -> None:
        with T.Kernel(T.ceildiv(n, VEC_NUM), is_npu=True) as (cid, vid):
            bid = cid * VEC_NUM + vid
            if bid < n:
                og_ub = T.alloc_ub((mhc, h_blk), _DTYPE)
                og_f32_ub = T.alloc_ub((mhc, h_blk), _ACCUM_DTYPE)
                xg_f32_ub = T.alloc_ub((1, h_blk), _ACCUM_DTYPE)
                xg_ub = T.alloc_ub((h_blk,), _DTYPE)

                if num_full > 0:
                    for i_h in T.Pipelined(num_full, num_stages=2):
                        h_start = i_h * h_blk
                        T.copy(o_grad[bid, 0:mhc, h_start : h_start + h_blk], og_ub)
                        T.tile.cast(og_f32_ub, og_ub, "CAST_NONE", mhc * h_blk)
                        T.reduce_sum(og_f32_ub, xg_f32_ub, dim=0)
                        T.tile.cast(xg_ub, xg_f32_ub, "CAST_RINT", h_blk)
                        T.copy(xg_ub, x_grad[bid, h_start : h_start + h_blk])

                if tail > 0:
                    og_tail_ub = T.alloc_ub((mhc, tail), _DTYPE)
                    og_f32_tail_ub = T.alloc_ub((mhc, tail), _ACCUM_DTYPE)
                    xg_f32_tail_ub = T.alloc_ub((1, tail), _ACCUM_DTYPE)
                    xg_tail_ub = T.alloc_ub((tail,), _DTYPE)
                    T.copy(o_grad[bid, 0:mhc, num_full * h_blk : h], og_tail_ub)
                    T.tile.cast(og_f32_tail_ub, og_tail_ub, "CAST_NONE", mhc * tail)
                    T.reduce_sum(og_f32_tail_ub, xg_f32_tail_ub, dim=0)
                    T.tile.cast(xg_tail_ub, xg_f32_tail_ub, "CAST_RINT", tail)
                    T.copy(xg_tail_ub, x_grad[bid, num_full * h_blk : h])

    return expand_bwd_kernel


def _select_h_blk(h: int) -> int:
    """Largest candidate dividing h, else the candidate with the smallest padded length."""
    for blk in H_BLK_CANDIDATES:
        if h % blk == 0:
            return blk
    if h < H_BLK_CANDIDATES[-1]:
        return h
    return min(H_BLK_CANDIDATES, key=lambda blk: -(-h // blk) * blk)


def _align_up_16(h: int) -> int:
    """Ascend UB transfers work on 32B (16 bf16) granules; tails must be 16-aligned."""
    return -(-h // 16) * 16


_fwd_kernel_cache = {}
_bwd_kernel_cache = {}


def _get_fwd_kernel(mhc: int, h: int, h_blk: int) -> tilelang.JITKernel:
    key = (mhc, h, h_blk)
    if key not in _fwd_kernel_cache:
        _fwd_kernel_cache[key] = _expand_fwd_kernel(mhc, h, h_blk)
    return _fwd_kernel_cache[key]


def _get_bwd_kernel(mhc: int, h: int, h_blk: int) -> tilelang.JITKernel:
    key = (mhc, h, h_blk)
    if key not in _bwd_kernel_cache:
        _bwd_kernel_cache[key] = _expand_bwd_kernel(mhc, h, h_blk)
    return _bwd_kernel_cache[key]


def expand_to_mhc_fwd(x: torch.Tensor, o: torch.Tensor | None = None) -> torch.Tensor:
    """x: (n, h) bf16 -> o: (n, mhc, h) bf16, each row replicated mhc times."""
    assert x.dim() == 2, f"x must be 2D (n, h), got {tuple(x.shape)}"
    assert x.dtype == torch.bfloat16, f"x must be bfloat16, got {x.dtype}"
    n, h = x.shape
    if o is None:
        mhc = 4
        o = torch.empty((n, mhc, h), dtype=x.dtype, device=x.device)
    mhc = o.shape[1]
    assert tuple(o.shape) == (n, mhc, h), f"o must be ({n}, {mhc}, {h}), got {tuple(o.shape)}"

    h_pad = _align_up_16(h)
    x = x.contiguous()
    if h_pad == h:
        kernel = _get_fwd_kernel(mhc, h, _select_h_blk(h))
        kernel(x, o)
    else:
        x_pad = torch.zeros(n, h_pad, dtype=x.dtype, device=x.device)
        x_pad[:, :h] = x
        o_pad = torch.empty(n, mhc, h_pad, dtype=x.dtype, device=x.device)
        kernel = _get_fwd_kernel(mhc, h_pad, _select_h_blk(h_pad))
        kernel(x_pad, o_pad)
        o.copy_(o_pad[:, :, :h])
    return o


def expand_to_mhc_bwd(o_grad: torch.Tensor, x_grad: torch.Tensor | None = None) -> torch.Tensor:
    """o_grad: (n, mhc, h) bf16 -> x_grad: (n, h) bf16 (fp32 sum over mhc)."""
    assert o_grad.dim() == 3, f"o_grad must be 3D (n, mhc, h), got {tuple(o_grad.shape)}"
    assert o_grad.dtype == torch.bfloat16, f"o_grad must be bfloat16, got {o_grad.dtype}"
    n, mhc, h = o_grad.shape
    if x_grad is None:
        x_grad = torch.empty((n, h), dtype=o_grad.dtype, device=o_grad.device)
    assert tuple(x_grad.shape) == (n, h), f"x_grad must be ({n}, {h}), got {tuple(x_grad.shape)}"

    h_pad = _align_up_16(h)
    o_grad = o_grad.contiguous()
    if h_pad == h:
        kernel = _get_bwd_kernel(mhc, h, _select_h_blk(h))
        kernel(o_grad, x_grad)
    else:
        o_grad_pad = torch.zeros(n, mhc, h_pad, dtype=o_grad.dtype, device=o_grad.device)
        o_grad_pad[:, :, :h] = o_grad
        x_grad_pad = torch.empty(n, h_pad, dtype=o_grad.dtype, device=o_grad.device)
        kernel = _get_bwd_kernel(mhc, h_pad, _select_h_blk(h_pad))
        kernel(o_grad_pad, x_grad_pad)
        x_grad.copy_(x_grad_pad[:, :h])
    return x_grad


def expand_to_mhc_fwd_ref(x: torch.Tensor, mhc_mult: int) -> torch.Tensor:
    return x.unsqueeze(-2).expand(*x.shape[:-1], mhc_mult, x.shape[-1]).contiguous()


def expand_to_mhc_bwd_ref(o_grad: torch.Tensor) -> torch.Tensor:
    return o_grad.float().sum(dim=-2).to(o_grad.dtype)
