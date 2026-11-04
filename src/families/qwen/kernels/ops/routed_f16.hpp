#pragma once

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Routed expert GEMMs on the F16 WMMA cores for the Qwen3.6-35B-A3B and
/// Ornith MoE layers (Strycast). They began as Flash-Next's routed GEMMs and
/// add Q6_K, Q8_0, 64-row tiles and a paired gate/up launch for every one of
/// those formats. Flash-Next keeps its own, separately evolved kernels; the
/// routing layout (RoutedCompact's `pad_bounds`, `rows_in`, `rows_out` and
/// the expert | tile << 16 tile map) is the same.
namespace gufo::hip::moe_f16 {

/// GGUF type ids these kernels decode.
enum class WeightType : std::uint32_t {
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
};

/// Routed expert GEMM on the F16 WMMA cores over F16 activation rows
/// `x` ([rows][k]): the weights are dequantized to F16 after the LDS read
/// and the whole K extent accumulates in F32. `tiles` (n_tiles entries)
/// packs each launched (expert, token tile) as expert | tile << 16, so empty
/// tiles cost nothing. Exactly one of `out` (F32) and `out_half` (F16)
/// receives the result; a non-null `swiglu_gate` (the gate projection's F32
/// output over the same rows) turns it into silu(gate) * result.
/// `tile_rows` is the token rows per tile the map was built with: 16 or 48,
/// or 64 for Q5_1/Q6_K/Q8_0.
/// Q4_K, Q5_K and Q6_K need k % 256 == 0, Q5_1 and Q8_0 k % 64 == 0; other
/// types and tile widths return false.
bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream);

/// Whether RoutedGatedF16Gemm decodes `type`.
bool RoutedGatedF16Supported(WeightType type);

/// Whether RoutedF16Gemm takes `type` with 64- and 128-row tiles; Q4_K and
/// Q5_K stop at 48.
bool RoutedF16WideTilesSupported(WeightType type);

/// Paired Q4_K/Q5_K/Q6_K/Q8_0 gate/up GEMM with SwiGLU and F16 output. Uses
/// the same routing layout as RoutedF16Gemm, with 64 or 128 token rows per
/// tile. The two projections share one launch and never materialize the gate
/// output.
bool RoutedGatedF16Gemm(const void* gate, const void* up, WeightType type,
                        const __half* x, const std::int32_t* tiles,
                        std::uint32_t n_tiles, std::uint32_t tile_rows,
                        const std::int32_t* pad_bounds,
                        const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out,
                        std::size_t m, std::size_t k, hipStream_t stream);

}  // namespace gufo::hip::moe_f16

#endif  // ENGINE_ENABLE_HIP
