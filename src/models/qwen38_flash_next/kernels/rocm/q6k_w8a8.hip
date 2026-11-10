#include "src/families/qwen/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

void Q6KW8A8Gemm(const void* w, const void* x_tiled, float* out,
                 std::size_t batch, std::size_t m, std::size_t k,
                 hipStream_t stream) {
  // One tile configuration for every pass width: each output is the same
  // chain of K16 half-block updates, so a row does not depend on its pass.
  constexpr int kBM = 128;
  constexpr int kBN = 128;
  const dim3 grid(static_cast<unsigned>((batch + kBN - 1) / kBN),
                  static_cast<unsigned>((m + kBM - 1) / kBM));
  gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2,
                                            core::GgmlType::kQ6_K>
      <<<grid, 256, 0, stream>>>(w, x_tiled, out, batch, m, k);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
