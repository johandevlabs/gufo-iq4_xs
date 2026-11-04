// Synthetic op-level checks for the qwen35moe GPU path: router top-k,
// routed expert projections (MMQ and the per-slot fallback), the shared
// expert and the combine epilogue, each compared against the CPU reference
// on fixed pseudo-random weights. No model file required.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <gufo/ext/experimental/hip_utils.hpp>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "qfn_mmq.h"
#include "src/core/gguf_reader.hpp"
#include "src/core/model_config.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/families/qwen/kernels/ops/gemm.hpp"
#include "src/families/qwen/kernels/ops/moe.hpp"
#include "src/families/qwen/kernels/ops/norm_residual.hpp"
#include "src/families/qwen/kernels/ops/routed_f16.hpp"
#include "src/families/qwen/kernels/ops/ssm.hpp"
#include "src/families/qwen/kernels/ops/swiglu.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/modules/moe.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::models::QwenTensorRef;
using Q8_0Block = gufo::quant::block_q8_0;

std::uint32_t seed = 20260926U;
std::uint32_t Rnd() {
  seed = seed * 1664525U + 1013904223U;
  return seed;
}
float RndFloat(float lo, float hi) {
  const float u = static_cast<float>(Rnd() & 0xFFFFU) / 65535.0F;
  return lo + u * (hi - lo);
}

std::uint16_t FloatToFp16Bits(float v) {
  // Round-to-nearest fp32 -> fp16 for the test's block scales.
  const std::uint32_t bits = [&] {
    std::uint32_t b;
    std::memcpy(&b, &v, sizeof(b));
    return b;
  }();
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xFF) - 127;
  std::uint32_t mantissa = bits & 0x7FFFFFU;
  if (exponent > 15) {
    return static_cast<std::uint16_t>(sign | 0x7BFFU);
  }
  if (exponent < -14) {
    return static_cast<std::uint16_t>(sign);
  }
  std::uint16_t out = 0;
  if (exponent >= -14) {
    out = static_cast<std::uint16_t>((exponent + 15) << 10);
  }
  const std::uint32_t rounded = (mantissa >> 13) + ((mantissa >> 12) & 1U);
  return static_cast<std::uint16_t>(sign | out | rounded);
}

/// Quantizes a dense row-major matrix into Q8_0 blocks.
std::vector<Q8_0Block> QuantizeQ8_0(const float* w, std::size_t rows,
                                    std::size_t cols) {
  const std::size_t blocks_per_row = cols / 32;
  std::vector<Q8_0Block> out(rows * blocks_per_row);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < blocks_per_row; ++b) {
      float max_abs = 1e-8F;
      for (int i = 0; i < 32; ++i) {
        max_abs = std::max(max_abs, std::fabs(w[r * cols + b * 32 + i]));
      }
      const float scale = max_abs / 127.0F;
      out[r * blocks_per_row + b].d = FloatToFp16Bits(scale);
      for (int i = 0; i < 32; ++i) {
        const float v = w[r * cols + b * 32 + i] / scale;
        out[r * blocks_per_row + b].qs[i] =
            static_cast<std::int8_t>(std::lround(v));
      }
    }
  }
  return out;
}

/// Decodes one row of a Q8_0 matrix.
float DequantRowDot(const std::vector<Q8_0Block>& q, std::size_t row,
                    std::size_t cols, const std::vector<float>& x) {
  const std::size_t blocks_per_row = cols / 32;
  float sum = 0.0F;
  for (std::size_t b = 0; b < blocks_per_row; ++b) {
    const Q8_0Block& blk = q[row * blocks_per_row + b];
    const float d = gufo::quant::Fp16ToFloat(blk.d);
    float dot = 0.0F;
    for (int i = 0; i < 32; ++i) {
      dot += static_cast<float>(blk.qs[i]) * x[b * 32 + i];
    }
    sum += d * dot;
  }
  return sum;
}

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Fixture {
  static constexpr std::size_t kHidden = 2048;
  static constexpr std::size_t kExpertFf = 512;
  static constexpr std::size_t kSharedFf = 512;
  static constexpr std::uint32_t kExperts = 256;
  static constexpr std::uint32_t kUsed = 8;

  std::vector<float> x;               // [hidden]
  std::vector<float> router;          // [experts x hidden]
  std::vector<float> shexp_gate_inp;  // [hidden]
  std::vector<float> gate_src;        // [experts x ff x hidden]
  std::vector<float> up_src;          // [experts x ff x hidden]
  std::vector<float> down_src;        // [experts x hidden x ff]
  std::vector<float> shexp_src;       // dense shexp trio source
  std::vector<Q8_0Block> gate_q, up_q, down_q, shexp_gate_q, shexp_up_q,
      shexp_down_q;

  gufo::core::ModelConfig config;
  std::vector<gufo::models::QwenLayerWeights> layers;

  Fixture()
      : x(kHidden),
        router(kExperts * kHidden),
        shexp_gate_inp(kHidden),
        gate_src(kExperts * kExpertFf * kHidden),
        up_src(kExperts * kExpertFf * kHidden),
        down_src(kExperts * kHidden * kExpertFf),
        shexp_src(3 * kSharedFf * kHidden) {
    for (auto& v : x)
      v = RndFloat(-1.0F, 1.0F);
    for (auto& v : router)
      v = RndFloat(-1.0F, 1.0F) * 0.05F;
    for (auto& v : shexp_gate_inp)
      v = RndFloat(-1.0F, 1.0F) * 0.05F;
    for (auto& v : gate_src)
      v = RndFloat(-1.0F, 1.0F) * 0.1F;
    for (auto& v : up_src)
      v = RndFloat(-1.0F, 1.0F) * 0.1F;
    for (auto& v : down_src)
      v = RndFloat(-1.0F, 1.0F) * 0.1F;
    for (auto& v : shexp_src)
      v = RndFloat(-1.0F, 1.0F) * 0.1F;
    gate_q = QuantizeQ8_0(gate_src.data(), kExperts * kExpertFf, kHidden);
    up_q = QuantizeQ8_0(up_src.data(), kExperts * kExpertFf, kHidden);
    down_q = QuantizeQ8_0(down_src.data(), kExperts * kHidden, kExpertFf);
    shexp_gate_q = QuantizeQ8_0(shexp_src.data(), kSharedFf, kHidden);
    shexp_up_q = QuantizeQ8_0(shexp_src.data() + kSharedFf * kHidden, kSharedFf,
                              kHidden);
    shexp_down_q = QuantizeQ8_0(shexp_src.data() + 2 * kSharedFf * kHidden,
                                kHidden, kSharedFf);

    config.hidden_size = kHidden;
    config.expert_count = kExperts;
    config.expert_used_count = kUsed;
    config.expert_ff_length = kExpertFf;
    config.expert_shared_ff_length = kSharedFf;
    config.intermediate_size = kSharedFf;
    layers.resize(1);
    auto& l = layers.front();
    l.ffn_gate_inp = {.data = router.data(),
                      .type = GgmlType::kF32,
                      .num_elements = router.size()};
    l.ffn_gate_inp_shexp = {.data = shexp_gate_inp.data(),
                            .type = GgmlType::kF32,
                            .num_elements = shexp_gate_inp.size()};
    l.ffn_gate_exps = {.data = gate_q.data(),
                       .type = GgmlType::kQ8_0,
                       .num_elements = kExperts * kExpertFf * kHidden};
    l.ffn_up_exps = {.data = up_q.data(),
                     .type = GgmlType::kQ8_0,
                     .num_elements = kExperts * kExpertFf * kHidden};
    l.ffn_down_exps = {.data = down_q.data(),
                       .type = GgmlType::kQ8_0,
                       .num_elements = kExperts * kHidden * kExpertFf};
    l.ffn_gate_shexp = {.data = shexp_gate_q.data(),
                        .type = GgmlType::kQ8_0,
                        .num_elements = kSharedFf * kHidden};
    l.ffn_up_shexp = {.data = shexp_up_q.data(),
                      .type = GgmlType::kQ8_0,
                      .num_elements = kSharedFf * kHidden};
    l.ffn_down_shexp = {.data = shexp_down_q.data(),
                        .type = GgmlType::kQ8_0,
                        .num_elements = kHidden * kSharedFf};
  }
};

void CpuRouterTopK(const std::vector<float>& logits, std::uint32_t n_experts,
                   std::uint32_t k, std::vector<std::int32_t>& ids,
                   std::vector<float>& weights) {
  std::vector<float> probs(n_experts);
  float max_logit = -std::numeric_limits<float>::infinity();
  for (std::uint32_t e = 0; e < n_experts; ++e)
    max_logit = std::max(max_logit, logits[e]);
  float denom = 0.0F;
  for (std::uint32_t e = 0; e < n_experts; ++e) {
    probs[e] = std::exp(logits[e] - max_logit);
    denom += probs[e];
  }
  for (std::uint32_t e = 0; e < n_experts; ++e)
    probs[e] /= denom;
  ids.assign(k, 0);
  weights.assign(k, 0.0F);
  for (std::uint32_t slot = 0; slot < k; ++slot) {
    float best = -1.0F;
    std::uint32_t index = n_experts;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      if (probs[e] > best) {
        best = probs[e];
        index = e;
      }
    }
    ids[slot] = static_cast<std::int32_t>(index);
    weights[slot] = best;
    probs[index] = -1.0F;
  }
  float sum = 0.0F;
  for (std::uint32_t slot = 0; slot < k; ++slot)
    sum += weights[slot];
  sum = std::max(sum, 6.103515625e-5F);
  for (std::uint32_t slot = 0; slot < k; ++slot)
    weights[slot] /= sum;
}

void CheckRouterTopK(const Fixture& f, hipStream_t stream) {
  std::vector<float> logits(f.kExperts);
  for (auto& v : logits)
    v = RndFloat(-3.0F, 3.0F);
  gufo::test::DeviceBuffer<float> d_logits(logits);
  gufo::test::DeviceBuffer<std::int32_t> d_ids(
      std::vector<std::int32_t>(f.kUsed, -1));
  gufo::test::DeviceBuffer<float> d_weights(std::vector<float>(f.kUsed, -1.0F));
  gufo::hip::LaunchMoeRouterTopK(d_logits.data(), f.kExperts, d_ids.data(),
                                 d_weights.data(), 1, f.kExperts, f.kUsed,
                                 stream);
  const auto h_ids = d_ids.CopyToHost();
  const auto h_weights = d_weights.CopyToHost();
  std::vector<std::int32_t> ref_ids;
  std::vector<float> ref_weights;
  CpuRouterTopK(logits, f.kExperts, f.kUsed, ref_ids, ref_weights);
  for (std::uint32_t s = 0; s < f.kUsed; ++s) {
    Expect(h_ids[s] == ref_ids[s],
           "top-k id mismatch at slot " + std::to_string(s) + ": gpu " +
               std::to_string(h_ids[s]) + " ref " + std::to_string(ref_ids[s]));
    Expect(std::fabs(h_weights[s] - ref_weights[s]) < 1e-5F,
           "top-k weight mismatch at slot " + std::to_string(s));
  }
  std::cout << "router top-k: OK\n";
}

void CheckDecodeMoe(const Fixture& f, hipStream_t stream, bool force_fallback) {
  const auto view = gufo::models::qwen::MakeMoeView(f.layers.front(), f.config);

  gufo::test::DeviceBuffer<float> d_x(f.x);
  gufo::test::DeviceBuffer<float> d_router_w(f.router);
  gufo::test::DeviceBuffer<float> d_shexp_gi(f.shexp_gate_inp);
  gufo::test::DeviceBuffer<float> d_logits(
      std::vector<float>(f.kExperts, 0.0F));
  gufo::test::DeviceBuffer<float> d_shexp_gate(std::vector<float>(1, 0.0F));
  gufo::test::DeviceBuffer<std::int32_t> d_ids(
      std::vector<std::int32_t>(f.kUsed, 0));
  gufo::test::DeviceBuffer<float> d_weights(std::vector<float>(f.kUsed, 0.0F));
  gufo::test::DeviceBuffer<float> d_gate_e(
      std::vector<float>(f.kUsed * f.kExpertFf, 0.0F));
  gufo::test::DeviceBuffer<float> d_down_e(
      std::vector<float>(f.kUsed * f.kHidden, 0.0F));
  gufo::test::DeviceBuffer<float> d_shexp_act(
      std::vector<float>(f.kSharedFf, 0.0F));
  gufo::test::DeviceBuffer<float> d_shexp_out(
      std::vector<float>(f.kHidden, 0.0F));
  gufo::test::DeviceBuffer<float> d_out(std::vector<float>(f.kHidden, 0.0F));
  gufo::test::DeviceBuffer<Q8_0Block> d_gate_q(f.gate_q);
  gufo::test::DeviceBuffer<Q8_0Block> d_up_q(f.up_q);
  gufo::test::DeviceBuffer<Q8_0Block> d_down_q(f.down_q);
  gufo::test::DeviceBuffer<Q8_0Block> d_shexp_gate_q(f.shexp_gate_q);
  gufo::test::DeviceBuffer<Q8_0Block> d_shexp_up_q(f.shexp_up_q);
  gufo::test::DeviceBuffer<Q8_0Block> d_shexp_down_q(f.shexp_down_q);

  // Router logits and the shared gate scalar.
  gufo::hip::LaunchGEMV(d_router_w.data(), GgmlType::kF32, d_x.data(),
                        d_logits.data(), f.kExperts, f.kHidden, stream);
  gufo::hip::LaunchGEMV(d_shexp_gi.data(), GgmlType::kF32, d_x.data(),
                        d_shexp_gate.data(), 1, f.kHidden, stream);
  gufo::hip::LaunchMoeRouterTopK(d_logits.data(), f.kExperts, d_ids.data(),
                                 d_weights.data(), 1, f.kExperts, f.kUsed,
                                 stream);
  const auto gpu_logits = d_logits.CopyToHost();
  for (std::uint32_t e = 0; e < f.kExperts; ++e) {
    float ref = 0.0F;
    for (std::size_t i = 0; i < f.kHidden; ++i)
      ref += f.router[e * f.kHidden + i] * f.x[i];
    Expect(std::fabs(gpu_logits[e] - ref) < 1e-2F,
           "router logit mismatch at expert " + std::to_string(e) + ": gpu " +
               std::to_string(gpu_logits[e]) + " ref " + std::to_string(ref));
  }
  const auto h_ids = d_ids.CopyToHost();
  std::vector<std::int32_t> ref_ids;
  std::vector<float> ref_weights;
  CpuRouterTopK(gpu_logits, f.kExperts, f.kUsed, ref_ids, ref_weights);
  for (std::uint32_t s = 0; s < f.kUsed; ++s) {
    Expect(h_ids[s] == ref_ids[s],
           "decode top-k id mismatch at slot " + std::to_string(s) + ": gpu " +
               std::to_string(h_ids[s]) + " ref " + std::to_string(ref_ids[s]));
  }
  std::cout << "decode router: OK\n";

  // Shared expert.
  gufo::hip::LaunchFusedSwiGLUGEMV(d_shexp_gate_q.data(), GgmlType::kQ8_0,
                                   d_shexp_up_q.data(), GgmlType::kQ8_0,
                                   d_x.data(), d_shexp_act.data(), f.kSharedFf,
                                   f.kHidden, stream);
  gufo::hip::LaunchGEMV(d_shexp_down_q.data(), GgmlType::kQ8_0,
                        d_shexp_act.data(), d_shexp_out.data(), f.kHidden,
                        f.kSharedFf, stream);

  // Routed experts.
  if (!force_fallback) {
    const int rc = qfn_mmq_moe_gated_vec(
        static_cast<int>(GgmlType::kQ8_0), d_gate_q.data(), d_up_q.data(),
        d_x.data(), d_ids.data(), d_gate_e.data(),
        static_cast<int>(f.kExpertFf), static_cast<int>(f.kHidden), 1,
        static_cast<int>(f.kExperts), static_cast<int>(f.kUsed), stream);
    Expect(rc == 0, "qfn_mmq_moe_gated_vec failed");
  } else {
    gufo::hip::LaunchMoeSlotSwigluGemv(
        d_gate_q.data(), GgmlType::kQ8_0, d_up_q.data(), GgmlType::kQ8_0,
        d_x.data(), d_ids.data(), d_gate_e.data(), f.kExpertFf, f.kHidden,
        f.kUsed, f.kUsed, stream);
  }
  const auto gate_e = d_gate_e.CopyToHost();
  for (std::uint32_t s = 0; s < f.kUsed; ++s) {
    const std::size_t expert = static_cast<std::size_t>(h_ids[s]);
    for (std::size_t r = 0; r < f.kExpertFf; ++r) {
      const float g =
          DequantRowDot(f.gate_q, expert * f.kExpertFf + r, f.kHidden, f.x);
      const float u =
          DequantRowDot(f.up_q, expert * f.kExpertFf + r, f.kHidden, f.x);
      const float ref = (g / (1.0F + std::exp(-g))) * u;
      const float got = gate_e[static_cast<std::size_t>(s) * f.kExpertFf + r];
      Expect(std::fabs(got - ref) < 0.02F + 0.05F * std::fabs(ref),
             "expert activation mismatch slot " + std::to_string(s) + " row " +
                 std::to_string(r) + ": gpu " + std::to_string(got) + " ref " +
                 std::to_string(ref));
    }
  }
  std::cout << (force_fallback ? "fallback" : "mmq") << " gated experts: OK\n";

  if (!force_fallback) {
    const int rc = qfn_mmq_moe_vec(
        static_cast<int>(GgmlType::kQ8_0), d_down_q.data(), d_gate_e.data(),
        d_ids.data(), d_down_e.data(), static_cast<int>(f.kHidden),
        static_cast<int>(f.kExpertFf), static_cast<int>(f.kUsed),
        static_cast<int>(f.kExperts), 1, stream, nullptr, nullptr);
    Expect(rc == 0, "qfn_mmq_moe_vec failed");
  } else {
    gufo::hip::LaunchMoeSlotGemv(d_down_q.data(), GgmlType::kQ8_0,
                                 d_gate_e.data(), d_ids.data(), d_down_e.data(),
                                 f.kHidden, f.kExpertFf, f.kUsed, 1, stream);
  }

  gufo::hip::LaunchMoeEpilogue(d_down_e.data(), d_weights.data(),
                               d_shexp_out.data(), d_shexp_gate.data(),
                               d_out.data(), 1, f.kUsed, f.kHidden, stream);
  const auto gpu_out = d_out.CopyToHost();

  gufo::models::QwenScratchArena arena(f.config);
  std::vector<float> cpu_out(f.kHidden, 0.0F);
  const gufo::models::qwen::CpuModuleContext ctx;
  gufo::models::qwen::MoeForward(ctx, view, f.x, arena, cpu_out);

  double max_abs = 0.0;
  double sum_sq = 0.0;
  std::size_t worst = 0;
  for (std::size_t i = 0; i < f.kHidden; ++i) {
    const double d = std::fabs(static_cast<double>(gpu_out[i]) - cpu_out[i]);
    if (d > max_abs) {
      max_abs = d;
      worst = i;
    }
    sum_sq += d * d;
  }
  std::cout << "decode MoE (" << (force_fallback ? "fallback" : "mmq")
            << ") vs CPU: max_abs=" << max_abs
            << " rmse=" << std::sqrt(sum_sq / f.kHidden)
            << " worst_idx=" << worst << " gpu=" << gpu_out[worst]
            << " cpu=" << cpu_out[worst] << "\n";
  Expect(max_abs < 0.02 + 0.05 * std::fabs(cpu_out[worst]),
         "decode MoE output mismatch");
}

std::uint16_t FloatToBf16Bits(float v) {
  std::uint32_t u;
  std::memcpy(&u, &v, sizeof(u));
  u += 0x7FFFU + ((u >> 16) & 1U);
  return static_cast<std::uint16_t>(u >> 16);
}

/// Grouped BF16 expert GEMM (prefill) against the per-slot GEMV, which reads
/// the same BF16 weights with full FP32 activations. Routing is skewed so hot
/// experts span several row tiles while others receive no slot.
void CheckGroupedBf16(hipStream_t stream) {
  constexpr std::uint32_t kExperts = 64;
  constexpr std::uint32_t kUsed = 8;
  struct Shape {
    const char* name;
    std::size_t m;
    std::size_t k;
    bool per_slot_input;
  };
  for (const Shape shape :
       {Shape{"gate/up", 512, 2048, false}, Shape{"down", 2048, 512, true}}) {
    Expect(gufo::hip::IsMoeGroupedBf16GemmSupported(shape.m, shape.k, kExperts),
           "grouped BF16 shape rejected");
    std::vector<std::uint16_t> w(static_cast<std::size_t>(kExperts) * shape.m *
                                 shape.k);
    for (auto& v : w) {
      v = FloatToBf16Bits(RndFloat(-0.05F, 0.05F));
    }
    gufo::test::DeviceBuffer<std::uint16_t> d_w(w);
    for (const std::uint32_t tokens : {1U, 7U, 100U, 512U}) {
      const std::uint32_t slots = tokens * kUsed;
      std::vector<std::int32_t> ids(slots);
      for (std::uint32_t s = 0; s < slots; ++s) {
        ids[s] = static_cast<std::int32_t>(
            (Rnd() % 3 == 0) ? (s % 3) : (3 + Rnd() % (kExperts / 2)));
      }
      const std::size_t x_rows = shape.per_slot_input ? slots : tokens;
      std::vector<float> x(x_rows * shape.k);
      for (auto& v : x) {
        v = RndFloat(-1.0F, 1.0F);
      }
      gufo::test::DeviceBuffer<std::int32_t> d_ids(ids);
      gufo::test::DeviceBuffer<float> d_x(x);
      const std::vector<float> nan_fill(
          static_cast<std::size_t>(slots) * shape.m,
          std::numeric_limits<float>::quiet_NaN());
      gufo::test::DeviceBuffer<float> d_ref(nan_fill);
      gufo::test::DeviceBuffer<float> d_got(nan_fill);
      gufo::test::DeviceBuffer<std::int32_t> d_sorted(slots);
      gufo::test::DeviceBuffer<std::int32_t> d_tiles(slots);
      gufo::test::DeviceBuffer<std::int32_t> d_bounds(kExperts + 1);
      const gufo::hip::MoeGroupedScratch scratch{
          .sorted_slots = d_sorted.data(),
          .tiles = d_tiles.data(),
          .expert_bounds = d_bounds.data()};
      const std::uint32_t divisor = shape.per_slot_input ? 1U : kUsed;

      gufo::hip::LaunchMoeSlotGemv(d_w.data(), GgmlType::kBF16, d_x.data(),
                                   d_ids.data(), d_ref.data(), shape.m, shape.k,
                                   slots, divisor, stream);
      gufo::hip::LaunchMoeGroupSlots(d_ids.data(), slots, kExperts, scratch,
                                     stream);
      gufo::hip::LaunchMoeGroupedBf16Gemm(d_w.data(), d_x.data(), d_got.data(),
                                          shape.m, shape.k, slots, kExperts,
                                          divisor, scratch, stream);
      HIP_CHECK(hipStreamSynchronize(stream));
      const auto ref = d_ref.CopyToHost();
      const auto got = d_got.CopyToHost();

      double max_abs = 0.0;
      double magnitude = 0.0;
      std::size_t non_finite = 0;
      for (std::size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) {
          ++non_finite;
          continue;
        }
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i]) -
                                              static_cast<double>(ref[i])));
        magnitude = std::max(magnitude, std::fabs(static_cast<double>(ref[i])));
      }
      const double rel = magnitude > 0.0 ? max_abs / magnitude : max_abs;
      std::cout << "grouped BF16 " << shape.name << " tokens=" << tokens
                << ": max_abs=" << max_abs << " rel=" << rel
                << " non_finite=" << non_finite << "\n";
      Expect(non_finite == 0, "grouped BF16 GEMM left slots unwritten");
      // Only the activations are rounded to BF16 (8-bit mantissa).
      Expect(rel < 1e-2, "grouped BF16 GEMM disagrees with the slot GEMV");

      // The expert-major slot order is built with atomics; the result must
      // not depend on it.
      gufo::test::DeviceBuffer<float> d_again(nan_fill);
      gufo::hip::LaunchMoeGroupSlots(d_ids.data(), slots, kExperts, scratch,
                                     stream);
      gufo::hip::LaunchMoeGroupedBf16Gemm(
          d_w.data(), d_x.data(), d_again.data(), shape.m, shape.k, slots,
          kExperts, divisor, scratch, stream);
      HIP_CHECK(hipStreamSynchronize(stream));
      const auto again = d_again.CopyToHost();
      Expect(std::memcmp(again.data(), got.data(),
                         got.size() * sizeof(float)) == 0,
             "grouped BF16 GEMM is not deterministic");
    }
  }
  std::cout << "grouped BF16 experts: OK\n";
}

/// The tiled-activation GEMM must reproduce LaunchBf16WmmaGemm bit for bit:
/// the same BF16 operands in the same WMMA chain, only a different layout.
void CheckDenseBf16Tiled(hipStream_t stream) {
  constexpr std::size_t kK = 2048;
  for (const std::size_t m : {256U, 64U, 192U}) {
    std::vector<std::uint16_t> w(m * kK);
    for (auto& v : w) {
      v = FloatToBf16Bits(RndFloat(-0.05F, 0.05F));
    }
    gufo::test::DeviceBuffer<std::uint16_t> d_w(w);
    for (const std::uint32_t rows : {1U, 100U, 129U, 512U, 4096U}) {
      std::vector<float> x(rows * kK);
      for (auto& v : x) {
        v = RndFloat(-1.0F, 1.0F);
      }
      std::vector<std::uint16_t> x_tiled(rows * kK);
      for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < kK; ++c) {
          x_tiled[((((c / 64) * rows) + r) * 64) + (c % 64)] =
              FloatToBf16Bits(x[(r * kK) + c]);
        }
      }
      gufo::test::DeviceBuffer<float> d_x(x);
      gufo::test::DeviceBuffer<std::uint16_t> d_x_tiled(x_tiled);
      const std::vector<float> nan_fill(
          rows * m, std::numeric_limits<float>::quiet_NaN());
      gufo::test::DeviceBuffer<float> d_ref(nan_fill);
      gufo::test::DeviceBuffer<float> d_got(nan_fill);
      gufo::hip::LaunchBf16WmmaGemm(d_w.data(), d_x.data(), d_ref.data(), rows,
                                    m, kK, stream);
      gufo::hip::LaunchBf16WmmaGemmTiled(d_w.data(), d_x_tiled.data(),
                                         d_got.data(), rows, m, kK, stream);
      HIP_CHECK(hipStreamSynchronize(stream));
      const auto ref = d_ref.CopyToHost();
      const auto got = d_got.CopyToHost();
      Expect(
          std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0,
          "tiled BF16 WMMA GEMM differs from LaunchBf16WmmaGemm");
    }
  }
  std::cout << "tiled dense BF16 WMMA: OK\n";
}

/// Dense mode of the same WMMA kernel (BF16 dense projections in prefill),
/// against the per-slot GEMV with every row routed to one matrix.
void CheckDenseBf16(hipStream_t stream) {
  constexpr std::size_t kM = 4096;
  constexpr std::size_t kK = 2048;
  std::vector<std::uint16_t> w(kM * kK);
  for (auto& v : w) {
    v = FloatToBf16Bits(RndFloat(-0.05F, 0.05F));
  }
  gufo::test::DeviceBuffer<std::uint16_t> d_w(w);
  for (const std::uint32_t rows : {1U, 100U, 512U}) {
    std::vector<float> x(rows * kK);
    for (auto& v : x) {
      v = RndFloat(-1.0F, 1.0F);
    }
    gufo::test::DeviceBuffer<float> d_x(x);
    gufo::test::DeviceBuffer<std::int32_t> d_ids(
        std::vector<std::int32_t>(rows, 0));
    const std::vector<float> nan_fill(rows * kM,
                                      std::numeric_limits<float>::quiet_NaN());
    gufo::test::DeviceBuffer<float> d_ref(nan_fill);
    gufo::test::DeviceBuffer<float> d_got(nan_fill);
    gufo::hip::LaunchMoeSlotGemv(d_w.data(), GgmlType::kBF16, d_x.data(),
                                 d_ids.data(), d_ref.data(), kM, kK, rows, 1,
                                 stream);
    gufo::hip::LaunchBf16WmmaGemm(d_w.data(), d_x.data(), d_got.data(), rows,
                                  kM, kK, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto ref = d_ref.CopyToHost();
    const auto got = d_got.CopyToHost();
    double max_abs = 0.0;
    double magnitude = 0.0;
    std::size_t non_finite = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      if (!std::isfinite(got[i])) {
        ++non_finite;
        continue;
      }
      max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i]) -
                                            static_cast<double>(ref[i])));
      magnitude = std::max(magnitude, std::fabs(static_cast<double>(ref[i])));
    }
    const double rel = magnitude > 0.0 ? max_abs / magnitude : max_abs;
    std::cout << "dense BF16 WMMA rows=" << rows << ": max_abs=" << max_abs
              << " rel=" << rel << " non_finite=" << non_finite << "\n";
    Expect(non_finite == 0, "dense BF16 WMMA GEMM left rows unwritten");
    Expect(rel < 1e-2, "dense BF16 WMMA GEMM disagrees with the GEMV");
  }
  std::cout << "dense BF16 WMMA: OK\n";
}

/// Random block_q6_K payload for `rows` x `cols`: arbitrary codes, int8
/// sub-block scales and a small F16 superblock scale.
std::vector<std::uint8_t> RandomQ6K(std::size_t rows, std::size_t cols) {
  constexpr std::size_t kBlockBytes = 210;
  std::vector<std::uint8_t> out(rows * (cols / 256) * kBlockBytes);
  for (std::size_t b = 0; b < out.size() / kBlockBytes; ++b) {
    std::uint8_t* blk = out.data() + (b * kBlockBytes);
    for (std::size_t i = 0; i < 192; ++i) {
      blk[i] = static_cast<std::uint8_t>(Rnd() & 0xFFU);
    }
    for (std::size_t i = 192; i < 208; ++i) {
      blk[i] = static_cast<std::uint8_t>(
          static_cast<std::int8_t>(static_cast<int>(Rnd() % 81) - 40));
    }
    const std::uint16_t d = FloatToFp16Bits(RndFloat(0.0002F, 0.0012F));
    std::memcpy(blk + 208, &d, sizeof(d));
  }
  return out;
}

/// Random block_q4_K (`with_high_bits` false) or block_q5_K payload for
/// `rows` x `cols`: arbitrary codes and packed 6-bit scales/mins, small F16
/// superblock scale and min.
std::vector<std::uint8_t> RandomQ45K(std::size_t rows, std::size_t cols,
                                     bool with_high_bits) {
  const std::size_t block_bytes = with_high_bits
                                      ? sizeof(gufo::quant::block_q5_K)
                                      : sizeof(gufo::quant::block_q4_K);
  std::vector<std::uint8_t> out(rows * (cols / 256) * block_bytes);
  for (std::size_t b = 0; b < out.size() / block_bytes; ++b) {
    std::uint8_t* blk = out.data() + (b * block_bytes);
    const std::uint16_t d = FloatToFp16Bits(RndFloat(0.0002F, 0.0012F));
    const std::uint16_t dmin = FloatToFp16Bits(RndFloat(0.0F, 0.0006F));
    std::memcpy(blk, &d, sizeof(d));
    std::memcpy(blk + 2, &dmin, sizeof(dmin));
    for (std::size_t i = 4; i < block_bytes; ++i) {
      blk[i] = static_cast<std::uint8_t>(Rnd() & 0xFFU);
    }
  }
  return out;
}

/// Expert weights of `type` for `rows` x `cols`: Q8_0 quantized from uniform
/// values, the K-quants as random payloads.
std::vector<std::uint8_t> RandomExperts(GgmlType type, std::size_t rows,
                                        std::size_t cols) {
  switch (type) {
    case GgmlType::kQ6_K:
      return RandomQ6K(rows, cols);
    case GgmlType::kQ4_K:
      return RandomQ45K(rows, cols, false);
    case GgmlType::kQ5_K:
      return RandomQ45K(rows, cols, true);
    default: {
      std::vector<float> src(rows * cols);
      for (auto& v : src) {
        v = RndFloat(-0.05F, 0.05F);
      }
      const auto q = QuantizeQ8_0(src.data(), rows, cols);
      std::vector<std::uint8_t> bytes(q.size() * sizeof(Q8_0Block));
      std::memcpy(bytes.data(), q.data(), bytes.size());
      return bytes;
    }
  }
}

namespace routed = gufo::models::qwen38_flash_next::rocm;
namespace moe_f16 = gufo::hip::moe_f16;

struct RoutedCase {
  GgmlType type;
  moe_f16::WeightType weight_type;
  const char* name;
};

constexpr std::array<RoutedCase, 4> kRoutedCases{{
    {GgmlType::kQ8_0, moe_f16::WeightType::kQ8_0, "Q8_0"},
    {GgmlType::kQ6_K, moe_f16::WeightType::kQ6_K, "Q6_K"},
    {GgmlType::kQ5_K, moe_f16::WeightType::kQ5_K, "Q5_K"},
    {GgmlType::kQ4_K, moe_f16::WeightType::kQ4_K, "Q4_K"},
}};

/// Flash-Next's routed F16 expert GEMM on this model's expert shapes, against
/// the per-slot GEMV (FP32 activations, same weights). The difference is the
/// F16 rounding of the activation rows (and, for the K-quants, of the scales).
/// Tiles of 64 and 128 rows, which the paired prefill sends the down
/// projection over, are checked for the types that support them; Q4_K and
/// Q5_K must reject them.
void CheckRoutedF16(hipStream_t stream) {
  for (const RoutedCase& routed_case : kRoutedCases) {
    const GgmlType type = routed_case.type;
    const auto weight_type = routed_case.weight_type;
    const char* type_name = routed_case.name;
    const bool wide = moe_f16::RoutedF16WideTilesSupported(weight_type);
    Expect(wide == (type == GgmlType::kQ8_0 || type == GgmlType::kQ6_K),
           std::string("unexpected wide-tile support for ") + type_name);
    constexpr std::uint32_t kExperts = 64;
    constexpr std::uint32_t kUsed = 8;
    struct Shape {
      const char* name;
      std::size_t m;
      std::size_t k;
      bool per_slot_input;
    };
    std::vector<Shape> shapes{Shape{"gate/up", 512, 2048, false},
                              Shape{"down", 2048, 512, true}};
    if (type == GgmlType::kQ4_K) {
      // Ornith 1.5 122B's Q4_K experts: hidden 3072, expert FFN 1024.
      shapes.push_back(Shape{"Ornith gate/up", 1024, 3072, false});
      shapes.push_back(Shape{"Ornith down", 3072, 1024, true});
    }
    for (const Shape& shape : shapes) {
      gufo::test::DeviceBuffer<std::uint8_t> d_w(
          RandomExperts(type, kExperts * shape.m, shape.k));
      for (const std::uint32_t tokens : {1U, 7U, 100U, 512U}) {
        const std::uint32_t slots = tokens * kUsed;
        std::vector<std::int32_t> ids(slots);
        for (std::uint32_t t = 0; t < tokens; ++t) {
          // Distinct experts per token, skewed toward the first few.
          for (std::uint32_t s = 0; s < kUsed; ++s) {
            ids[t * kUsed + s] = static_cast<std::int32_t>(
                (Rnd() % 3 == 0) ? s
                                 : (kUsed + (t + s * 5) % (kExperts - kUsed)));
          }
        }
        const std::size_t x_rows = shape.per_slot_input ? slots : tokens;
        std::vector<float> x(x_rows * shape.k);
        for (auto& v : x) {
          v = RndFloat(-1.0F, 1.0F);
        }
        gufo::test::DeviceBuffer<std::int32_t> d_ids(ids);
        gufo::test::DeviceBuffer<float> d_x(x);
        const std::vector<float> nan_fill(
            static_cast<std::size_t>(slots) * shape.m,
            std::numeric_limits<float>::quiet_NaN());
        gufo::test::DeviceBuffer<float> d_ref(nan_fill);
        gufo::test::DeviceBuffer<float> d_got(nan_fill);
        gufo::hip::LaunchMoeSlotGemv(d_w.data(), type, d_x.data(), d_ids.data(),
                                     d_ref.data(), shape.m, shape.k, slots,
                                     shape.per_slot_input ? 1U : kUsed, stream);

        gufo::test::DeviceBuffer<std::uint32_t> d_counts(kExperts);
        const std::size_t rows = gufo::hip::RoutedRows(slots, kExperts);
        gufo::test::DeviceBuffer<std::int32_t> d_bounds(kExperts + 1);
        gufo::test::DeviceBuffer<std::int32_t> d_cursors(kExperts);
        gufo::test::DeviceBuffer<std::int32_t> d_rows_token(rows);
        gufo::test::DeviceBuffer<std::int32_t> d_rows_slot(rows);
        gufo::test::DeviceBuffer<std::uint16_t> d_x_half(x_rows * shape.k);
        routed::ExpertCounts(d_ids.data(), d_counts.data(), tokens, kExperts,
                             kUsed, stream);
        const auto counts = d_counts.CopyToHost();
        for (const std::uint32_t tile_rows : {16U, 48U, 64U, 128U}) {
          std::vector<std::int32_t> tiles;
          for (std::uint32_t e = 0; e < kExperts; ++e) {
            const std::uint32_t padded = (counts[e] + 15U) / 16U * 16U;
            for (std::uint32_t j = 0; j * tile_rows < padded; ++j) {
              tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
            }
          }
          gufo::test::DeviceBuffer<std::int32_t> d_tiles(tiles);
          d_got.CopyFrom(std::span<const float>(nan_fill));
          routed::RoutedCompact(d_ids.data(), d_counts.data(), d_bounds.data(),
                                d_cursors.data(), d_rows_token.data(),
                                d_rows_slot.data(), tokens, kUsed, kExperts,
                                stream);
          routed::NarrowActivations(d_x.data(), d_x_half.data(), false,
                                    x_rows * shape.k, stream);
          const auto* rows_in =
              shape.per_slot_input ? d_rows_slot.data() : d_rows_token.data();
          const bool launched = moe_f16::RoutedF16Gemm(
              d_w.data(), weight_type,
              reinterpret_cast<const __half*>(d_x_half.data()), d_tiles.data(),
              static_cast<std::uint32_t>(tiles.size()), tile_rows,
              d_bounds.data(), rows_in, d_rows_slot.data(), nullptr,
              d_got.data(), nullptr, shape.m, shape.k, stream);
          if (tile_rows > 48U && !wide) {
            Expect(!launched, std::string("routed F16 GEMM took a wide ") +
                                  type_name + " tile");
            continue;
          }
          Expect(launched, "routed F16 GEMM rejected the shape");
          HIP_CHECK(hipStreamSynchronize(stream));
          const auto ref = d_ref.CopyToHost();
          const auto got = d_got.CopyToHost();
          double max_abs = 0.0;
          double magnitude = 0.0;
          std::size_t non_finite = 0;
          for (std::size_t i = 0; i < ref.size(); ++i) {
            if (!std::isfinite(got[i])) {
              ++non_finite;
              continue;
            }
            max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i]) -
                                                  static_cast<double>(ref[i])));
            magnitude =
                std::max(magnitude, std::fabs(static_cast<double>(ref[i])));
          }
          const double rel = magnitude > 0.0 ? max_abs / magnitude : max_abs;
          std::cout << "routed F16 " << type_name << " " << shape.name
                    << " tokens=" << tokens << " tile_rows=" << tile_rows
                    << ": rel=" << rel << " non_finite=" << non_finite << "\n";
          Expect(non_finite == 0, "routed F16 GEMM left slots unwritten");
          Expect(rel < 1e-2, "routed F16 GEMM disagrees with the slot GEMV");
        }
      }
    }
  }
  std::cout << "routed F16 experts: OK\n";
}

float Fp16BitsToFloat(std::uint16_t h) {
  const std::uint32_t sign = (h & 0x8000U) << 16;
  std::uint32_t exponent = (h >> 10) & 0x1FU;
  std::uint32_t mantissa = h & 0x3FFU;
  std::uint32_t bits = 0;
  if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13);
  } else if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400U) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      bits = sign | (exponent << 23) | ((mantissa & 0x3FFU) << 13);
    }
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

/// Paired routed gate/up (SwiGLU in the epilogue, F16 out) against
/// silu(gate) * up from two per-slot GEMVs on the same weights.
void CheckRoutedPair(hipStream_t stream) {
  constexpr std::uint32_t kExperts = 64;
  constexpr std::uint32_t kUsed = 8;
  constexpr std::size_t kM = 512;
  constexpr std::size_t kK = 2048;
  for (const RoutedCase& routed_case : kRoutedCases) {
    const GgmlType type = routed_case.type;
    const auto weight_type = routed_case.weight_type;
    const char* type_name = routed_case.name;
    gufo::test::DeviceBuffer<std::uint8_t> d_gate(
        RandomExperts(type, kExperts * kM, kK));
    gufo::test::DeviceBuffer<std::uint8_t> d_up(
        RandomExperts(type, kExperts * kM, kK));
    for (const std::uint32_t tokens : {100U, 512U, 1024U}) {
      const std::uint32_t slots = tokens * kUsed;
      std::vector<std::int32_t> ids(slots);
      for (std::uint32_t t = 0; t < tokens; ++t) {
        for (std::uint32_t s = 0; s < kUsed; ++s) {
          ids[t * kUsed + s] = static_cast<std::int32_t>(
              (Rnd() % 3 == 0) ? s
                               : (kUsed + (t + s * 5) % (kExperts - kUsed)));
        }
      }
      std::vector<float> x(tokens * kK);
      for (auto& v : x) {
        v = RndFloat(-1.0F, 1.0F);
      }
      gufo::test::DeviceBuffer<std::int32_t> d_ids(ids);
      gufo::test::DeviceBuffer<float> d_x(x);
      gufo::test::DeviceBuffer<float> d_gate_ref(slots * kM);
      gufo::test::DeviceBuffer<float> d_up_ref(slots * kM);
      gufo::hip::LaunchMoeSlotGemv(d_gate.data(), type, d_x.data(),
                                   d_ids.data(), d_gate_ref.data(), kM, kK,
                                   slots, kUsed, stream);
      gufo::hip::LaunchMoeSlotGemv(d_up.data(), type, d_x.data(), d_ids.data(),
                                   d_up_ref.data(), kM, kK, slots, kUsed,
                                   stream);
      gufo::test::DeviceBuffer<std::uint32_t> d_counts(kExperts);
      const std::size_t rows = gufo::hip::RoutedRows(slots, kExperts);
      gufo::test::DeviceBuffer<std::int32_t> d_bounds(kExperts + 1);
      gufo::test::DeviceBuffer<std::int32_t> d_cursors(kExperts);
      gufo::test::DeviceBuffer<std::int32_t> d_rows_token(rows);
      gufo::test::DeviceBuffer<std::int32_t> d_rows_slot(rows);
      gufo::test::DeviceBuffer<std::uint16_t> d_x_half(tokens * kK);
      routed::ExpertCounts(d_ids.data(), d_counts.data(), tokens, kExperts,
                           kUsed, stream);
      routed::RoutedCompact(d_ids.data(), d_counts.data(), d_bounds.data(),
                            d_cursors.data(), d_rows_token.data(),
                            d_rows_slot.data(), tokens, kUsed, kExperts,
                            stream);
      routed::NarrowActivations(d_x.data(), d_x_half.data(), false, tokens * kK,
                                stream);
      const auto counts = d_counts.CopyToHost();
      const auto gate_ref = d_gate_ref.CopyToHost();
      const auto up_ref = d_up_ref.CopyToHost();
      for (const std::uint32_t pair_rows : {64U, 128U}) {
        std::vector<std::int32_t> tiles;
        for (std::uint32_t e = 0; e < kExperts; ++e) {
          const std::uint32_t padded = (counts[e] + 15U) / 16U * 16U;
          for (std::uint32_t j = 0; j * pair_rows < padded; ++j) {
            tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
          }
        }
        gufo::test::DeviceBuffer<std::int32_t> d_tiles(tiles);
        gufo::test::DeviceBuffer<std::uint16_t> d_out(
            std::vector<std::uint16_t>(slots * kM, 0x7E00U));  // F16 NaN
        Expect(moe_f16::RoutedGatedF16Gemm(
                   d_gate.data(), d_up.data(), weight_type,
                   reinterpret_cast<const __half*>(d_x_half.data()),
                   d_tiles.data(), static_cast<std::uint32_t>(tiles.size()),
                   pair_rows, d_bounds.data(), d_rows_token.data(),
                   d_rows_slot.data(), reinterpret_cast<__half*>(d_out.data()),
                   kM, kK, stream),
               "paired routed GEMM rejected the shape");
        HIP_CHECK(hipStreamSynchronize(stream));
        const auto out = d_out.CopyToHost();
        double max_abs = 0.0;
        double magnitude = 0.0;
        std::size_t non_finite = 0;
        for (std::size_t i = 0; i < out.size(); ++i) {
          const float got = Fp16BitsToFloat(out[i]);
          if (!std::isfinite(got)) {
            ++non_finite;
            continue;
          }
          const double g = gate_ref[i];
          const double want = g / (1.0 + std::exp(-g)) * up_ref[i];
          max_abs = std::max(max_abs, std::fabs(got - want));
          magnitude = std::max(magnitude, std::fabs(want));
        }
        const double rel = magnitude > 0.0 ? max_abs / magnitude : max_abs;
        std::cout << "routed pair " << type_name << " tokens=" << tokens
                  << " pair_rows=" << pair_rows << ": rel=" << rel
                  << " non_finite=" << non_finite << "\n";
        Expect(non_finite == 0, "paired routed GEMM left slots unwritten");
        Expect(rel < 5e-3, "paired routed GEMM disagrees with the GEMVs");
      }
    }
  }
  std::cout << "routed pair: OK\n";
}

/// The router, shared-expert gate and shared expert as single-token decode
/// and batched decode / verification launch them; each batched row must equal
/// that row decoded alone, bit for bit.
void CheckSharedPathBatchInvariance(const Fixture& f, hipStream_t stream) {
  constexpr std::size_t kRows = 8;
  std::vector<float> x(kRows * f.kHidden);
  for (auto& v : x) {
    v = RndFloat(-1.0F, 1.0F);
  }
  gufo::test::DeviceBuffer<float> d_x(x);
  gufo::test::DeviceBuffer<float> d_router(f.router);
  gufo::test::DeviceBuffer<float> d_shexp_gi(f.shexp_gate_inp);
  gufo::test::DeviceBuffer<Q8_0Block> d_gate(f.shexp_gate_q);
  gufo::test::DeviceBuffer<Q8_0Block> d_up(f.shexp_up_q);
  gufo::test::DeviceBuffer<Q8_0Block> d_down(f.shexp_down_q);
  const auto compare = [](const char* name, gufo::test::DeviceBuffer<float>& a,
                          gufo::test::DeviceBuffer<float>& b) {
    HIP_CHECK(hipDeviceSynchronize());
    const auto ha = a.CopyToHost();
    const auto hb = b.CopyToHost();
    std::size_t differing = 0;
    for (std::size_t i = 0; i < ha.size(); ++i) {
      differing += std::memcmp(&ha[i], &hb[i], sizeof(float)) != 0 ? 1 : 0;
    }
    std::cout << "shared path " << name << ": " << differing << "/" << ha.size()
              << " differ\n";
    return differing;
  };
  std::size_t differing = 0;

  // Router and shared-expert gate: F32 GEMV against the row GEMV.
  for (const auto& [name, weights, m] :
       {std::tuple{"router", d_router.data(), std::size_t{f.kExperts}},
        std::tuple{"shexp gate", d_shexp_gi.data(), std::size_t{1}}}) {
    gufo::test::DeviceBuffer<float> one(kRows * m);
    gufo::test::DeviceBuffer<float> rows(kRows * m);
    for (std::size_t r = 0; r < kRows; ++r) {
      gufo::hip::LaunchGEMV(weights, GgmlType::kF32, d_x.data() + r * f.kHidden,
                            one.data() + r * m, m, f.kHidden, stream);
    }
    gufo::hip::LaunchGEMVRows(weights, GgmlType::kF32, d_x.data(), rows.data(),
                              kRows, m, f.kHidden, stream);
    differing += compare(name, one, rows);
  }

  // Shared expert gate/up: fused SwiGLU GEMV against two projections.
  gufo::test::DeviceBuffer<float> act_one(kRows * f.kSharedFf);
  gufo::test::DeviceBuffer<float> gate(kRows * f.kSharedFf);
  gufo::test::DeviceBuffer<float> up(kRows * f.kSharedFf);
  gufo::test::DeviceBuffer<float> act(kRows * f.kSharedFf);
  for (std::size_t r = 0; r < kRows; ++r) {
    gufo::hip::LaunchFusedSwiGLUGEMV(
        d_gate.data(), GgmlType::kQ8_0, d_up.data(), GgmlType::kQ8_0,
        d_x.data() + r * f.kHidden, act_one.data() + r * f.kSharedFf,
        f.kSharedFf, f.kHidden, stream);
  }
  gufo::hip::LaunchBatchedQuantGEMMFp32(GgmlType::kQ8_0, d_gate.data(),
                                        d_x.data(), gate.data(), kRows,
                                        f.kSharedFf, f.kHidden, stream);
  gufo::hip::LaunchBatchedQuantGEMMFp32(GgmlType::kQ8_0, d_up.data(),
                                        d_x.data(), up.data(), kRows,
                                        f.kSharedFf, f.kHidden, stream);
  gufo::hip::LaunchBatchedSwiGLUActivation(
      gate.data(), up.data(), act.data(), nullptr, kRows * f.kSharedFf, stream);
  differing += compare("shexp gate/up", act_one, act);

  // Shared expert down: Q8_0 GEMV against the batched projection, on the
  // same activation rows.
  gufo::test::DeviceBuffer<float> down_one(kRows * f.kHidden);
  gufo::test::DeviceBuffer<float> down(kRows * f.kHidden);
  for (std::size_t r = 0; r < kRows; ++r) {
    gufo::hip::LaunchGEMV(
        d_down.data(), GgmlType::kQ8_0, act_one.data() + r * f.kSharedFf,
        down_one.data() + r * f.kHidden, f.kHidden, f.kSharedFf, stream);
  }
  gufo::hip::LaunchBatchedQuantGEMMFp32(GgmlType::kQ8_0, d_down.data(),
                                        act_one.data(), down.data(), kRows,
                                        f.kHidden, f.kSharedFf, stream);
  differing += compare("shexp down", down_one, down);
  Expect(differing == 0,
         "batched router or shared expert differs from single-token decode");
  std::cout << "shared path batch invariance: OK\n";
}

/// The gated-delta-net input/output projections and RMSNorm at this model's
/// shapes, as single-token decode and batched decode / verification launch
/// them; each batched row must equal that row decoded alone, bit for bit.
void CheckSsmPathBatchInvariance(hipStream_t stream) {
  constexpr std::size_t kRows = 4;
  constexpr std::size_t kHidden = 2048;
  constexpr std::size_t kQkv = 8192;
  constexpr std::size_t kInner = 4096;
  constexpr std::size_t kRank = 32;
  const auto random = [](std::size_t n, float scale) {
    std::vector<float> v(n);
    for (auto& e : v) {
      e = RndFloat(-1.0F, 1.0F) * scale;
    }
    return v;
  };
  const auto x = random(kRows * kHidden, 1.0F);
  const auto inner_x = random(kRows * kInner, 1.0F);
  const auto qkv_src = random(kQkv * kHidden, 0.05F);
  const auto gate_src = random(kInner * kHidden, 0.05F);
  const auto out_src = random(kHidden * kInner, 0.05F);
  gufo::test::DeviceBuffer<float> d_x(x);
  gufo::test::DeviceBuffer<float> d_inner_x(inner_x);
  gufo::test::DeviceBuffer<Q8_0Block> d_qkv(
      QuantizeQ8_0(qkv_src.data(), kQkv, kHidden));
  gufo::test::DeviceBuffer<Q8_0Block> d_gate(
      QuantizeQ8_0(gate_src.data(), kInner, kHidden));
  gufo::test::DeviceBuffer<Q8_0Block> d_out(
      QuantizeQ8_0(out_src.data(), kHidden, kInner));
  gufo::test::DeviceBuffer<float> d_alpha(random(kRank * kHidden, 0.05F));
  gufo::test::DeviceBuffer<float> d_beta(random(kRank * kHidden, 0.05F));
  gufo::test::DeviceBuffer<float> d_norm(random(kHidden, 1.0F));
  std::size_t total = 0;
  const auto compare = [&](const char* name, gufo::test::DeviceBuffer<float>& a,
                           gufo::test::DeviceBuffer<float>& b) {
    HIP_CHECK(hipDeviceSynchronize());
    const auto ha = a.CopyToHost();
    const auto hb = b.CopyToHost();
    std::size_t differing = 0;
    for (std::size_t i = 0; i < ha.size(); ++i) {
      differing += std::memcmp(&ha[i], &hb[i], sizeof(float)) != 0 ? 1 : 0;
    }
    std::cout << "ssm path " << name << ": " << differing << "/" << ha.size()
              << " differ\n";
    total += differing;
  };

  gufo::test::DeviceBuffer<float> qkv_one(kRows * kQkv);
  gufo::test::DeviceBuffer<float> gate_one(kRows * kInner);
  gufo::test::DeviceBuffer<float> alpha_one(kRows * kRank);
  gufo::test::DeviceBuffer<float> beta_one(kRows * kRank);
  gufo::test::DeviceBuffer<float> out_one(kRows * kHidden);
  gufo::test::DeviceBuffer<float> norm_one(kRows * kHidden);
  for (std::size_t r = 0; r < kRows; ++r) {
    gufo::hip::LaunchFusedSSMInputProjections(
        d_qkv.data(), GgmlType::kQ8_0, d_gate.data(), GgmlType::kQ8_0,
        d_alpha.data(), GgmlType::kF32, d_beta.data(), GgmlType::kF32,
        d_x.data() + r * kHidden, qkv_one.data() + r * kQkv,
        gate_one.data() + r * kInner, alpha_one.data() + r * kRank,
        beta_one.data() + r * kRank, kHidden, kQkv, kInner, kRank, stream);
    gufo::hip::LaunchGEMV(
        d_out.data(), GgmlType::kQ8_0, d_inner_x.data() + r * kInner,
        out_one.data() + r * kHidden, kHidden, kInner, stream);
    gufo::hip::LaunchRMSNorm(d_x.data() + r * kHidden, d_norm.data(),
                             norm_one.data() + r * kHidden, kHidden, 1e-6F,
                             stream);
  }
  gufo::test::DeviceBuffer<float> qkv(kRows * kQkv);
  gufo::test::DeviceBuffer<float> gate(kRows * kInner);
  gufo::test::DeviceBuffer<float> alpha(kRows * kRank);
  gufo::test::DeviceBuffer<float> beta(kRows * kRank);
  gufo::test::DeviceBuffer<float> out(kRows * kHidden);
  gufo::test::DeviceBuffer<float> norm(kRows * kHidden);
  gufo::hip::LaunchBatchedQuantGEMMFp32(GgmlType::kQ8_0, d_qkv.data(),
                                        d_x.data(), qkv.data(), kRows, kQkv,
                                        kHidden, stream);
  gufo::hip::LaunchBatchedQuantGEMMFp32(GgmlType::kQ8_0, d_gate.data(),
                                        d_x.data(), gate.data(), kRows, kInner,
                                        kHidden, stream);
  Expect(
      gufo::hip::LaunchSsmControlProjectionsRows(
          d_alpha.data(), GgmlType::kF32, d_beta.data(), GgmlType::kF32,
          d_x.data(), alpha.data(), beta.data(), kRows, kHidden, kRank, stream),
      "SSM control rows rejected the hidden size");
  gufo::hip::LaunchBatchedQuantGEMMFp32(GgmlType::kQ8_0, d_out.data(),
                                        d_inner_x.data(), out.data(), kRows,
                                        kHidden, kInner, stream);
  gufo::hip::LaunchBatchedRMSNorm(d_x.data(), d_norm.data(), norm.data(),
                                  nullptr, kRows, kHidden, 1e-6F, stream);
  compare("qkv", qkv_one, qkv);
  compare("gate", gate_one, gate);
  compare("alpha", alpha_one, alpha);
  compare("beta", beta_one, beta);
  compare("ssm_out", out_one, out);
  compare("rms norm", norm_one, norm);
  Expect(total == 0, "batched SSM-layer projection differs from decode");
  std::cout << "ssm path batch invariance: OK\n";
}

/// Routed experts as single-token decode (ExecuteMoeDecodeStep) and batched
/// decode / speculative verification (LaunchMoeFfn) launch them. Verification
/// must reproduce single-token decode, so each batched row's SwiGLU activation
/// and down projection must equal that row decoded alone, bit for bit.
void CheckDecodeBatchInvariance(hipStream_t stream) {
  constexpr std::uint32_t kExperts = 32;
  constexpr std::uint32_t kUsed = 8;
  constexpr std::size_t kFf = 512;
  constexpr std::size_t kHidden = 2048;
  constexpr std::size_t kGatedVecMaxRows = 8;
  for (const RoutedCase& routed_case : kRoutedCases) {
    const GgmlType type = routed_case.type;
    const int ggml_type = static_cast<int>(type);
    gufo::test::DeviceBuffer<std::uint8_t> d_gate(
        RandomExperts(type, kExperts * kFf, kHidden));
    gufo::test::DeviceBuffer<std::uint8_t> d_up(
        RandomExperts(type, kExperts * kFf, kHidden));
    gufo::test::DeviceBuffer<std::uint8_t> d_down(
        RandomExperts(type, kExperts * kHidden, kFf));
    const auto gate_up = [&](const float* x, const std::int32_t* ids,
                             float* act, std::size_t rows) {
      if (gufo::hip::IsMoeMmqGatedVecType(type)) {
        for (std::size_t row = 0; row < rows; row += kGatedVecMaxRows) {
          const std::size_t n = std::min(kGatedVecMaxRows, rows - row);
          Expect(qfn_mmq_moe_gated_vec(
                     ggml_type, d_gate.data(), d_up.data(), x + row * kHidden,
                     ids + row * kUsed, act + row * kUsed * kFf,
                     static_cast<int>(kFf), static_cast<int>(kHidden),
                     static_cast<int>(n), static_cast<int>(kExperts),
                     static_cast<int>(kUsed), stream) == 0,
                 "qfn_mmq_moe_gated_vec failed");
        }
      } else {
        gufo::hip::LaunchMoeSlotSwigluGemv(
            d_gate.data(), type, d_up.data(), type, x, ids, act, kFf, kHidden,
            static_cast<std::uint32_t>(rows * kUsed), kUsed, stream);
      }
    };
    const auto down = [&](const float* act, const std::int32_t* ids, float* out,
                          std::size_t rows) {
      const auto slots = static_cast<std::uint32_t>(rows * kUsed);
      if (gufo::hip::IsMoeMmqVecType(type)) {
        Expect(
            qfn_mmq_moe_vec(ggml_type, d_down.data(), act, ids, out,
                            static_cast<int>(kHidden), static_cast<int>(kFf),
                            static_cast<int>(slots), static_cast<int>(kExperts),
                            1, stream, nullptr, nullptr) == 0,
            "qfn_mmq_moe_vec failed");
      } else {
        gufo::hip::LaunchMoeSlotGemv(d_down.data(), type, act, ids, out,
                                     kHidden, kFf, slots, 1, stream);
      }
    };
    for (const std::size_t rows :
         std::array<std::size_t, 6>{2, 3, 8, 9, 23, 64}) {
      // Few experts, so rows share experts as consecutive draft tokens do.
      std::vector<std::int32_t> ids(rows * kUsed);
      for (std::size_t r = 0; r < rows; ++r) {
        for (std::uint32_t s = 0; s < kUsed; ++s) {
          ids[r * kUsed + s] = static_cast<std::int32_t>(
              (s * 4 + (Rnd() % 2 == 0 ? 0 : r)) % kExperts);
        }
      }
      std::vector<float> x(rows * kHidden);
      for (auto& v : x) {
        v = RndFloat(-1.0F, 1.0F);
      }
      gufo::test::DeviceBuffer<std::int32_t> d_ids(ids);
      gufo::test::DeviceBuffer<float> d_x(x);
      gufo::test::DeviceBuffer<float> d_act(rows * kUsed * kFf);
      gufo::test::DeviceBuffer<float> d_out(rows * kUsed * kHidden);
      gufo::test::DeviceBuffer<float> d_act_one(rows * kUsed * kFf);
      gufo::test::DeviceBuffer<float> d_out_one(rows * kUsed * kHidden);
      gate_up(d_x.data(), d_ids.data(), d_act.data(), rows);
      down(d_act.data(), d_ids.data(), d_out.data(), rows);
      for (std::size_t r = 0; r < rows; ++r) {
        gate_up(d_x.data() + r * kHidden, d_ids.data() + r * kUsed,
                d_act_one.data() + r * kUsed * kFf, 1);
        down(d_act_one.data() + r * kUsed * kFf, d_ids.data() + r * kUsed,
             d_out_one.data() + r * kUsed * kHidden, 1);
      }
      HIP_CHECK(hipStreamSynchronize(stream));
      const auto act = d_act.CopyToHost();
      const auto act_one = d_act_one.CopyToHost();
      const auto out = d_out.CopyToHost();
      const auto out_one = d_out_one.CopyToHost();
      const auto differing = [](const std::vector<float>& a,
                                const std::vector<float>& b) {
        std::size_t count = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
          count += std::memcmp(&a[i], &b[i], sizeof(float)) != 0 ? 1 : 0;
        }
        return count;
      };
      const std::size_t act_diff = differing(act, act_one);
      const std::size_t out_diff = differing(out, out_one);
      std::cout << "decode batch " << routed_case.name << " rows=" << rows
                << ": activation " << act_diff << "/" << act.size()
                << " differ, down " << out_diff << "/" << out.size()
                << " differ\n";
      Expect(act_diff == 0,
             std::string("batched routed gate/up differs from single-token "
                         "decode for ") +
                 routed_case.name);
      Expect(out_diff == 0,
             std::string("batched routed down differs from single-token "
                         "decode for ") +
                 routed_case.name);
    }
  }
  std::cout << "decode batch invariance: OK\n";
}

}  // namespace

int main() {
  const int gate = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kRequired, "qwen35ba3b_moe_ops_test");
  if (gate != gufo::test::kHipTestSuccess) {
    return gate;
  }
  try {
    Expect(qfn_mmq_init(0) == 0, "qfn_mmq_init failed");
    Fixture f;
    CheckRouterTopK(f, nullptr);
    CheckDecodeMoe(f, nullptr, /*force_fallback=*/false);
    CheckDecodeMoe(f, nullptr, /*force_fallback=*/true);
    CheckGroupedBf16(nullptr);
    CheckDenseBf16(nullptr);
    CheckDenseBf16Tiled(nullptr);
    CheckRoutedF16(nullptr);
    CheckRoutedPair(nullptr);
    CheckDecodeBatchInvariance(nullptr);
    CheckSharedPathBatchInvariance(f, nullptr);
    CheckSsmPathBatchInvariance(nullptr);
  } catch (const std::exception& e) {
    std::cerr << "qwen35ba3b_moe_ops_test: " << e.what() << "\n";
    return 1;
  }
  std::cout << "qwen35ba3b_moe_ops_test: PASS\n";
  return 0;
}
