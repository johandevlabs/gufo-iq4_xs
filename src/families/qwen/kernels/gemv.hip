#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "qfn_mmq.h"
#include "src/core/gguf_reader.hpp"
#include <gufo/ext/experimental/dispatch_telemetry.hpp>
#include "src/families/qwen/hip_common/gemm_route.hpp"
#include "src/families/qwen/kernels/ops/gemm.hpp"
#include "src/families/qwen/kernels/fast_gemv_row.hpp"

namespace gufo::hip {

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32GEMVKernel_1Row(const hip_bfloat16* __restrict__ A,
                               const float* __restrict__ x,
                               float* __restrict__ y, std::size_t M,
                               std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  const auto* row = A + (m * K);
  const auto* row_vec = reinterpret_cast<const uint4*>(row);
  const std::size_t K_vec = K / 8;

  float dot = 0.0F;
  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const uint4 v = row_vec[vi];
    const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    dot += (static_cast<float>(vals[0]) * x0.x) +
           (static_cast<float>(vals[1]) * x0.y) +
           (static_cast<float>(vals[2]) * x0.z) +
           (static_cast<float>(vals[3]) * x0.w) +
           (static_cast<float>(vals[4]) * x1.x) +
           (static_cast<float>(vals[5]) * x1.y) +
           (static_cast<float>(vals[6]) * x1.z) +
           (static_cast<float>(vals[7]) * x1.w);
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    dot += static_cast<float>(row[k]) * x[k];
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  if (lane_id == 0) {
    const float r = 0.0F;
    y[m] = dot + r;
  }
}

// Grid row `blockIdx.y` is one input vector: the block reads x + row * K and
// writes y + row * M with the arithmetic of a single-vector launch, so a
// batch of vectors costs one launch instead of one per vector.
__global__ void FastGEMVBlockKernel(const void* __restrict__ A,
                                    core::GgmlType a_type,
                                    const float* __restrict__ x_rows,
                                    float* __restrict__ y_rows, std::size_t M,
                                    std::size_t K) {
  const std::size_t m = blockIdx.x;
  if (m >= M)
    return;
  const float* x = x_rows + (static_cast<std::size_t>(blockIdx.y) * K);
  float* y = y_rows + (static_cast<std::size_t>(blockIdx.y) * M);
  FastGEMVBlockRow(A, a_type, x, y, m, K);
}

namespace {

void ExecuteDenseGemv(models::qwen::QwenGemmRoute route, const void* A,
                      core::GgmlType a_type, const float* x, float* y,
                      std::size_t M, std::size_t K, hipStream_t stream) {
  constexpr std::uint32_t kWavesPerBlock = 4;
  constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;
  if (route == models::qwen::QwenGemmRoute::kHipBf16Wave32Single) {
    detail::EmitGemvDispatch(M, K, true,
                             models::qwen::QwenGemmRouteName(route));
    const auto* A_bf16 = static_cast<const hip_bfloat16*>(A);
    const std::size_t grid = (M + kWavesPerBlock - 1) / kWavesPerBlock;
    Wave32GEMVKernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(A_bf16, x, y, M, K);
    return;
  }

  detail::EmitGemvDispatch(M, K, a_type == core::GgmlType::kBF16,
                           models::qwen::QwenGemmRouteName(route));
  const std::size_t block_size =
      route == models::qwen::QwenGemmRoute::kHipF32Baseline512 ||
              route == models::qwen::QwenGemmRoute::kHipBf16Baseline512
          ? 512
          : 256;
  FastGEMVBlockKernel<<<M, block_size, 0, stream>>>(A, a_type, x, y, M, K);
}

// The vendored vector kernel reads a routed expert id per row; a dense weight
// is expert 0 of one for every row.
constexpr std::size_t kMaxMmvqRows = 64;
__device__ std::int32_t gDenseExpertIds[kMaxMmvqRows] = {};

void ExecuteMmvqGemv(const void* A, core::GgmlType a_type, const float* x,
                     float* y, std::size_t rows, std::size_t M, std::size_t K,
                     hipStream_t stream) {
  static const std::int32_t* const kExpertIds = [] {
    void* address = nullptr;
    if (hipGetSymbolAddress(&address, HIP_SYMBOL(gDenseExpertIds)) !=
        hipSuccess) {
      std::abort();
    }
    return static_cast<const std::int32_t*>(address);
  }();
  if (rows > kMaxMmvqRows) {
    std::abort();
  }
  detail::EmitGemvDispatch(M, K, false,
                           models::qwen::QwenGemmRouteName(
                               models::qwen::QwenGemmRoute::kHipQuantMmvq));
  if (qfn_mmq_moe_vec(static_cast<int>(a_type), A, x, kExpertIds, y,
                      static_cast<int>(M), static_cast<int>(K),
                      static_cast<int>(rows), 1, 1, stream) != 0) {
    std::abort();
  }
}

}  // namespace

void LaunchGEMVRows(const void* A, core::GgmlType a_type, const float* x,
                    float* y, std::size_t rows, std::size_t M, std::size_t K,
                    hipStream_t stream, models::qwen::QwenGemmMode mode) {
  if (rows == 0) {
    return;
  }
  const auto resolution = models::qwen::ResolveQwenGemmRoute(
      {.type = a_type, .batch_size = 1, .m = M, .k = K, .mode = mode});
  switch (resolution.route) {
    case models::qwen::QwenGemmRoute::kHipQuantMmvq:
      // Each launch quantizes and multiplies up to kMaxMmvqRows rows; each row
      // keeps the single-row arithmetic.
      for (std::size_t row = 0; row < rows; row += kMaxMmvqRows) {
        ExecuteMmvqGemv(A, a_type, x + (row * K), y + (row * M),
                        std::min(rows - row, kMaxMmvqRows), M, K, stream);
      }
      return;
    case models::qwen::QwenGemmRoute::kHipF32Baseline256:
    case models::qwen::QwenGemmRoute::kHipF32Baseline512:
    case models::qwen::QwenGemmRoute::kHipBf16Baseline256:
    case models::qwen::QwenGemmRoute::kHipBf16Baseline512: {
      detail::EmitGemvDispatch(
          M, K, a_type == core::GgmlType::kBF16,
          models::qwen::QwenGemmRouteName(resolution.route));
      const unsigned int block_size =
          resolution.route == models::qwen::QwenGemmRoute::kHipF32Baseline512 ||
                  resolution.route ==
                      models::qwen::QwenGemmRoute::kHipBf16Baseline512
              ? 512
              : 256;
      const dim3 grid(static_cast<unsigned int>(M),
                      static_cast<unsigned int>(rows));
      FastGEMVBlockKernel<<<grid, block_size, 0, stream>>>(A, a_type, x, y, M,
                                                           K);
      return;
    }
    default:
      for (std::size_t row = 0; row < rows; ++row) {
        LaunchGEMV(A, a_type, x + (row * K), y + (row * M), M, K, stream, mode);
      }
      return;
  }
}

void LaunchGEMV(const void* A, core::GgmlType a_type, const float* x, float* y,
                std::size_t M, std::size_t K, hipStream_t stream,
                models::qwen::QwenGemmMode mode) {
  const auto resolution = models::qwen::ResolveQwenGemmRoute(
      {.type = a_type, .batch_size = 1, .m = M, .k = K, .mode = mode});
  switch (resolution.route) {
    case models::qwen::QwenGemmRoute::kHipQuantDirect:
      LaunchQ8KBlockGEMV(A, a_type, x, y, M, K, stream);
      return;
    case models::qwen::QwenGemmRoute::kHipQuantMmvq:
      ExecuteMmvqGemv(A, a_type, x, y, 1, M, K, stream);
      return;
    case models::qwen::QwenGemmRoute::kHipF32Baseline256:
    case models::qwen::QwenGemmRoute::kHipF32Baseline512:
    case models::qwen::QwenGemmRoute::kHipBf16Baseline256:
    case models::qwen::QwenGemmRoute::kHipBf16Baseline512:
    case models::qwen::QwenGemmRoute::kHipBf16Wave32Single:
      ExecuteDenseGemv(resolution.route, A, a_type, x, y, M, K, stream);
      return;
    default:
      std::abort();
  }
}

}  // namespace gufo::hip
#endif
