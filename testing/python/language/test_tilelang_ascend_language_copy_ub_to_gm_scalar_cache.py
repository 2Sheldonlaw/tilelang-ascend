"""Issue #1304 regression: bulk T.copy + scalar tails over W=126 fp32.

A GM buffer written by both scalar stores (S pipe, write-back cache) and
MTE3 DMA needs cache-coherence treatment: the scalar store's cache-line fill
(read-modify-write) can snapshot GM before the DMA lands, and the dirty
line's later eviction stamps stale bytes over the freshly copied data. The
AscendSyncInsert pass detects this mixed-write pattern and inserts a dcci
before the DMA plus an MTE3_S ordering after it. The pass only runs with
TL_ASCEND_AUTO_SYNC=True, so these tests require it.
"""

import torch

import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[0],
    target="ascendc",
    pass_configs={tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True},
)
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


def test_copy_ub_to_gm_with_scalar_tails_w126():
    oc_n, h, w = 8, 12, 126
    out = fill_rows_with_tails(oc_n, h, w)()
    torch.npu.synchronize()
    oc_g = torch.arange(oc_n, dtype=torch.float32).view(oc_n, 1, 1)
    oh_g = torch.arange(h, dtype=torch.float32).view(1, h, 1)
    ow_g = torch.arange(w, dtype=torch.float32).view(1, 1, w)
    ref = 0.125 + oc_g * 0.0001 + oh_g * 0.00001 + ow_g * 0.000001
    assert torch.equal(out.cpu(), ref)
