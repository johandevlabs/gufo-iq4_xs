#pragma once

#include "gufo/ext/gguf_reader.hpp"

namespace gufo::models::qwen {

/// Storage formats the Qwen projections load and the HIP kernels decode:
/// dense F32/BF16, Q8_K/Q8_0, the K-quants from Q3_K, and the IQ formats of
/// the Unsloth UD mixes (IQ3_S, IQ4_XS, IQ4_NL). The loader and the provider
/// probe share this list, so a file the loader would reject is declined at
/// probe time and left to another provider.
[[nodiscard]] constexpr bool IsQwenWeightFormat(core::GgmlType type) noexcept {
  using core::GgmlType;
  return type == GgmlType::kF32 || type == GgmlType::kBF16 ||
         type == GgmlType::kQ8_K || type == GgmlType::kQ8_0 ||
         type == GgmlType::kQ5_K || type == GgmlType::kQ6_K ||
         type == GgmlType::kQ4_K || type == GgmlType::kQ3_K ||
         type == GgmlType::kIQ4_NL || type == GgmlType::kIQ4_XS ||
         type == GgmlType::kIQ3_S;
}

}  // namespace gufo::models::qwen
