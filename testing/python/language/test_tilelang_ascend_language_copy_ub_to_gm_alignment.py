"""UB->GM ``T.copy`` alignment coverage (issues #1682 / #1304).

``copy_ub_to_gm`` used to always emit one multi-burst ``DataCopyPad``. The MTE3
engine on dav-c220 requires every burst's GM destination to be 32B aligned and
advances the UB source by whole 32B blocks per burst, so two shapes of input
silently corrupted memory:

* rows packed tighter than 32B with more than one row -- e.g. the ``(M, 1)``
  fp32 keepdim output of ``T.reduce_max/min/sum`` written straight back to GM
  (issue #1682);
* GM rows whose stride is not a 32B multiple, e.g. a ``(B, C, H, 126)`` fp32
  tensor (issue #1304).

The helper now picks between a single flat burst (packed src+dst), the classic
multi-burst (32B-multiple row pitches), and a scalar fallback for everything
the MTE3 engine cannot express. These tests pin each path down, plus the
loop-carried UB reuse that needs an explicit producer sync once the copy sits
under a runtime conditional.

Note: the scalar fallback makes a *single core's* writes correct. Adjacent
GM lines shared between rows owned by *different* cores are a separate,
pre-existing hazard of scalar GM stores on this platform and are not covered
here (a pure-scalar multi-block kernel over ``(N, C, 126, 126)`` already
corrupts without any ``T.copy`` involved).
"""

import pytest
import torch

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[0], target="ascendc")
def copy_packed(M: int = 16, N: int = 1, dtype: str = "float32"):
    @T.prim_func
    def main(B: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, N), dtype)
            T.tile.arith_progression(a_ub, 0.0, 1.0, M * N)
            T.copy(a_ub, B)

    return main


@tilelang.jit(out_idx=[0], target="ascendc")
def copy_strided(M: int = 16, N: int = 96, src_pitch: int = 96, dst_pitch: int = 128, dtype: str = "float32"):
    """Copy an (M, N) tile: the UB buffer and the GM rows both have their own
    pitch, so realdstN/srcN exercise the stride arithmetic."""

    @T.prim_func
    def main(B: T.Tensor((M, dst_pitch), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, src_pitch), dtype)
            T.tile.arith_progression(a_ub, 0.0, 1.0, M * src_pitch)
            T.copy(a_ub, B[0:M, 0:N])

    return main


@tilelang.jit(out_idx=[0], target="ascendc")
def copy_rows(M: int = 8, W: int = 126, pitch: int = 126, dtype: str = "float32"):
    """Copy a 1D row into each row of a (M, pitch) tensor. With W=126 fp32 the
    row destinations alternate between aligned (multi-burst) and misaligned
    (scalar fallback) addresses."""

    @T.prim_func
    def main(B: T.Tensor((M, pitch), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((W,), dtype)
            for m in T.serial(M):
                T.tile.arith_progression(a_ub, T.cast(m, dtype), 1.0, W)
                T.copy(a_ub, B[m, 0])

    return main


@tilelang.jit(out_idx=[0], target="ascendc")
def copy_keepdim_reduce(M: int = 16, N: int = 128, dtype: str = "float32"):
    """The issue #1682 repro: reduce_max keepdim output copied straight to GM
    with 4-byte rows (blockLen < 32B)."""

    @T.prim_func
    def main(B: T.Tensor((M, 1), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((M, N), dtype)
            b_ub = T.alloc_ub((M, 1), dtype)
            T.tile.arith_progression(a_ub, 0.0, 1.0, M * N)
            T.reduce_max(a_ub, b_ub, dim=-1)
            T.copy(b_ub, B)

    return main


@tilelang.jit(out_idx=[0], target="ascendc")
def copy_reuse_ub_in_loop(M: int = 512, N: int = 128, blk: int = 64, dtype: str = "float32"):
    """One UB buffer reused across loop iterations: iteration i+1's producer
    must not overwrite data iteration i's copy still reads."""

    @T.prim_func
    def main(B: T.Tensor((M, N), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((blk, N), dtype)
            for it in T.serial(M // blk):
                T.tile.arith_progression(a_ub, T.cast(it, dtype), 1.0, blk * N)
                T.copy(a_ub, B[it * blk : (it + 1) * blk, :])

    return main


@tilelang.jit(out_idx=[0], target="ascendc")
def fill_rows_with_tails(oc_n: int = 8, h: int = 12, w: int = 126, main_w: int = 112, dtype: str = "float32"):
    """The single-core issue #1304 pattern: an aligned bulk T.copy per row plus
    scalar tail stores, with w % 8 != 0 so consecutive rows share GM lines."""

    @T.prim_func
    def main(Y: T.Tensor((oc_n, h, w), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            row_ub = T.alloc_ub((main_w,), dtype)
            with T.Scope("V"):
                for oc in T.serial(oc_n):
                    for oh in T.serial(h):
                        for ow in T.serial(main_w):
                            row_ub[ow] = (
                                T.float32(0.125)
                                + T.cast(oc, "float32") * T.float32(0.0001)
                                + T.cast(oh, "float32") * T.float32(0.00001)
                                + T.cast(ow, "float32") * T.float32(0.000001)
                            )
                        T.copy(row_ub, Y[oc, oh, 0])
                        for tw in T.serial(w - main_w):
                            ow_tail = main_w + tw
                            Y[oc, oh, ow_tail] = (
                                T.float32(0.125)
                                + T.cast(oc, "float32") * T.float32(0.0001)
                                + T.cast(oh, "float32") * T.float32(0.00001)
                                + T.cast(ow_tail, "float32") * T.float32(0.000001)
                            )

    return main


@pytest.mark.parametrize(
    "M,N,dtype",
    [
        (16, 1, "float32"),  # issue #1682: 4-byte rows
        (3, 1, "float32"),  # flat copy below 32B total
        (5, 3, "float32"),  # odd width, neither side 32B-aligned
        (16, 100, "float32"),  # packed, non-32B-multiple row
        (16, 1, "float16"),  # 2-byte rows
    ],
)
def test_copy_ub_to_gm_packed(M, N, dtype):
    out = copy_packed(M, N, dtype)()
    torch.npu.synchronize()
    ref = torch.arange(M * N, dtype=torch.float32).reshape(M, N)
    if dtype == "float16":
        ref = ref.half()
    assert torch.equal(out.cpu(), ref)


def test_copy_ub_to_gm_keepdim_reduce():
    out = copy_keepdim_reduce(16, 128)()
    torch.npu.synchronize()
    ref = torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128)
    ref = ref.max(dim=-1, keepdim=True).values
    assert torch.equal(out.cpu(), ref)


def test_copy_ub_to_gm_aligned_strided():
    out = copy_strided(16, 96, 96, 128)()
    torch.npu.synchronize()
    ref = torch.arange(16 * 96, dtype=torch.float32).reshape(16, 96)
    assert torch.equal(out.cpu()[:, :96], ref)


def test_copy_ub_to_gm_misaligned_dst_pitch():
    out = copy_strided(16, 112, 112, 126)()
    torch.npu.synchronize()
    ref = torch.arange(16 * 112, dtype=torch.float32).reshape(16, 112)
    assert torch.equal(out.cpu()[:, :112], ref)


def test_copy_ub_to_gm_sub32b_src_pitch():
    out = copy_strided(16, 100, 100, 128)()
    torch.npu.synchronize()
    ref = torch.arange(16 * 100, dtype=torch.float32).reshape(16, 100)
    assert torch.equal(out.cpu()[:, :100], ref)


def test_copy_ub_to_gm_rows_mixed_alignment():
    M, W = 8, 126
    out = copy_rows(M, W, W)()
    torch.npu.synchronize()
    ref = torch.arange(M, dtype=torch.float32).unsqueeze(1) + torch.arange(W, dtype=torch.float32).unsqueeze(0)
    assert torch.equal(out.cpu(), ref)


def test_copy_ub_to_gm_reuse_ub_in_loop():
    M, N, blk = 512, 128, 64
    out = copy_reuse_ub_in_loop(M, N, blk)()
    torch.npu.synchronize()
    ref = torch.stack([torch.arange(blk * N, dtype=torch.float32).reshape(blk, N) + it for it in range(M // blk)]).reshape(M, N)
    assert torch.equal(out.cpu(), ref)


def test_copy_ub_to_gm_with_scalar_tails_w126():
    oc_n, h, w = 8, 12, 126
    out = fill_rows_with_tails(oc_n, h, w)()
    torch.npu.synchronize()
    oc_g = torch.arange(oc_n, dtype=torch.float32).view(oc_n, 1, 1)
    oh_g = torch.arange(h, dtype=torch.float32).view(1, h, 1)
    ow_g = torch.arange(w, dtype=torch.float32).view(1, 1, w)
    ref = 0.125 + oc_g * 0.0001 + oh_g * 0.00001 + ow_g * 0.000001
    assert torch.equal(out.cpu(), ref)
