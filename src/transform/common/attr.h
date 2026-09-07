// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file attr.h
 * \brief Check attributes of the IR
 */

namespace tvm {
namespace tl {

constexpr const char *MainBlockName = "tilelang_root";

constexpr const char *tilelang_is_cpu_kernel_frame =
    "tilelang.is_cpu_kernel_frame";

constexpr const char *tilelang_is_npu_kernel_frame =
    "tilelang.is_npu_kernel_frame";

constexpr const char *tilelang_is_npu_kernel_frame_dev_mode =
    "tilelang.is_npu_kernel_frame_dev_mode";

constexpr const char *cv_1_1 = "cv_1_1";

constexpr const char *cv_1_2 = "cv_1_2";

constexpr const char *kInitialBufferShapes = "initial_buffer_shapes";

constexpr const char *kLogicBufferShapes = "logic_buffer_shapes";

// Trailing two dims of each buffer's shape as observed *before* the
// Flatten2DBuffer pass collapses leading dimensions into the row count.
// For L1 Mat buffers this is the logical 2D tile (e.g. [4, 128, 128] ->
// [128, 128]) whose fractal layout both the GM->L1 loads and the L1->L0
// extracts must agree on.
constexpr const char *kLogicBufferTileShapes = "logic_buffer_tile_shapes";

} // namespace tl
} // namespace tvm
