// IQ3_S/IQ4_NL/IQ4_XS routed experts and the Q6_K logits head (UD-IQ4_XS
// target): accuracy against an F64 dot product over the CPU dequantizers, and
// the exactness contracts of the vector paths (batched rows, paired and fused
// gate/up all match single-row decode bit for bit).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace q = gufo::models::qwen38_flash_next::rocm;
using gufo::core::GgmlType;

namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

float Uniform(std::uint32_t* state, float scale) {
  return scale *
         static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                            32768) /
         32768.0F;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  T* device = nullptr;
  CheckHip(hipMalloc(&device, host.size() * sizeof(T) + 4096), "hipMalloc");
  CheckHip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
  return device;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  CheckHip(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return host;
}

struct Format {
  GgmlType type;
  const char* name;
  std::size_t block;
  std::size_t bytes;
  std::size_t scale_offset;  ///< byte offset of the F16 block scale
  void (*dequantize)(const void*, float*, std::size_t);
};

const Format kIQ3_S{
    GgmlType::kIQ3_S, "IQ3_S", 256, 110, 0, gufo::quant::DequantizeIQ3_S};
const Format kIQ4_NL{
    GgmlType::kIQ4_NL, "IQ4_NL", 32, 18, 0, gufo::quant::DequantizeIQ4_NL};
const Format kIQ4_XS{
    GgmlType::kIQ4_XS, "IQ4_XS", 256, 136, 0, gufo::quant::DequantizeIQ4_XS};
const Format kQ6_K{
    GgmlType::kQ6_K, "Q6_K", 256, 210, 208, gufo::quant::DequantizeQ6_K};

/// Random blocks of `format` (every code byte random, a moderate positive
/// F16 scale) plus their dequantized values, row-major [matrices][m][k].
struct Weights {
  std::vector<std::uint8_t> packed;
  std::vector<float> values;
};

Weights Make(const Format& f, std::size_t matrices, std::size_t m,
             std::size_t k, std::uint32_t seed) {
  Weights w;
  const std::size_t rows = matrices * m;
  const std::size_t blocks_per_row = k / f.block;
  w.packed.resize(rows * blocks_per_row * f.bytes);
  w.values.resize(rows * k);
  for (auto& byte : w.packed)
    byte = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFFU);
  for (std::size_t b = 0; b < rows * blocks_per_row; ++b) {
    // Q6_K's int8 sub-block scales multiply its scale; keep products sane.
    const float magnitude = f.type == GgmlType::kQ6_K ? 0.0004F : 0.004F;
    const __half d = __float2half(Uniform(&seed, magnitude) + 1.5F * magnitude);
    std::memcpy(w.packed.data() + b * f.bytes + f.scale_offset, &d, 2);
  }
  for (std::size_t r = 0; r < rows; ++r)
    f.dequantize(w.packed.data() + r * blocks_per_row * f.bytes,
                 w.values.data() + r * k, k);
  return w;
}

double Dot(const float* w, const float* x, std::size_t k) {
  double acc = 0.0;
  for (std::size_t i = 0; i < k; ++i)
    acc += static_cast<double>(w[i]) * x[i];
  return acc;
}

/// Largest error against `reference`, relative to its largest magnitude.
double RelativeError(const std::vector<float>& actual,
                     const std::vector<double>& reference) {
  double scale = 0.0, error = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    scale = std::max(scale, std::abs(reference[i]));
    if (!std::isfinite(actual[i]))
      return INFINITY;
    error = std::max(error, std::abs(actual[i] - reference[i]));
  }
  return scale > 0.0 ? error / scale : error;
}

void Expect(bool ok, const std::string& what) {
  if (!ok)
    throw std::runtime_error(what);
}

/// Routed vector, paired, fused gate/up and tiled MMQ projections of one
/// format: all against F64, the vector paths against single-row decode.
void CheckRouted(const Format& f, int rows, int cols, int tokens, int used) {
  constexpr int experts = 8;
  const std::string tag = std::string(f.name) + " m=" + std::to_string(rows) +
                          " k=" + std::to_string(cols);
  const Weights a = Make(f, experts, rows, cols, 0x1234U + rows);
  const Weights b = Make(f, experts, rows, cols, 0x8765U + rows);
  std::vector<float> x(static_cast<std::size_t>(tokens) * cols);
  std::uint32_t seed = 37;
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = Uniform(&seed, (i / cols) % 2 ? 1.0F / 128.0F : 2.0F);
  std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens) * used);
  for (int t = 0; t < tokens; ++t)
    for (int j = 0; j < used; ++j)
      ids[t * used + j] = used > 1 && j == 1 ? -1 : (t + j) % experts;
  if (used > 2)
    ids[2] = ids[0];  // a duplicate expert within one token

  const std::size_t count = static_cast<std::size_t>(tokens) * used * rows;
  std::vector<double> ref_a(count), ref_b(count);
  for (int t = 0; t < tokens; ++t)
    for (int j = 0; j < used; ++j)
      for (int r = 0; r < rows; ++r) {
        const int e = ids[t * used + j];
        const std::size_t o =
            (static_cast<std::size_t>(t) * used + j) * rows + r;
        if (e < 0)
          continue;
        const std::size_t row = static_cast<std::size_t>(e) * rows + r;
        ref_a[o] = Dot(a.values.data() + row * cols, x.data() + t * cols, cols);
        ref_b[o] = Dot(b.values.data() + row * cols, x.data() + t * cols, cols);
      }

  auto* wa = Upload(a.packed);
  auto* wb = Upload(b.packed);
  auto* dx = Upload(x);
  auto* di = Upload(ids);
  const std::vector<float> zeros(count);
  auto* single_a = Upload(zeros);
  auto* single_b = Upload(zeros);
  auto* out = Upload(zeros);
  auto* out_b = Upload(zeros);
  const int type = static_cast<int>(f.type);

  // Decode: one row per launch.
  for (int t = 0; t < tokens; ++t) {
    const std::size_t o = static_cast<std::size_t>(t) * used * rows;
    Expect(
        !qfn_mmq_moe_vec(type, wa, dx + t * cols, di + t * used, single_a + o,
                         rows, cols, 1, experts, used, nullptr) &&
            !qfn_mmq_moe_vec(type, wb, dx + t * cols, di + t * used,
                             single_b + o, rows, cols, 1, experts, used,
                             nullptr),
        tag + ": single-row vector launch failed");
  }
  const auto decode_a = Download(single_a, count);
  const auto decode_b = Download(single_b, count);
  for (std::size_t i = 0; i < count; ++i)
    Expect(ids[i / rows] >= 0 || decode_a[i] == 0.0F,
           tag + ": inactive expert row is not zero");
  const double vec_error = RelativeError(decode_a, ref_a);
  Expect(vec_error < 1e-2, tag + ": vector error " + std::to_string(vec_error));

  // Batched rows and the paired gate/up keep the decode bits.
  for (int n : {2, 3, 7, 8, 9, 16, 31, 32}) {
    if (n > tokens)
      break;
    const std::size_t bytes = static_cast<std::size_t>(n) * used * rows * 4;
    Expect(!qfn_mmq_moe_vec(type, wa, dx, di, out, rows, cols, n, experts, used,
                            nullptr, wb, out_b),
           tag + ": batched vector launch failed");
    Expect(
        !std::memcmp(Download(out, count).data(), decode_a.data(), bytes) &&
            !std::memcmp(Download(out_b, count).data(), decode_b.data(), bytes),
        tag + ": batched vector width " + std::to_string(n) +
            " changed decode bits");
  }

  // Fused gate/up (decode and verification widths) against SwiGLU of the
  // separate decode projections.
  q::Swiglu(single_a, single_b, count, nullptr);
  const auto swiglu = Download(single_a, count);
  for (int n = 1; n <= std::min(tokens, 8); ++n) {
    Expect(!qfn_mmq_moe_gated_vec(type, wa, wb, dx, di, out, rows, cols, n,
                                  experts, used, nullptr),
           tag + ": gated vector launch failed");
    Expect(!std::memcmp(Download(out, count).data(), swiglu.data(),
                        static_cast<std::size_t>(n) * used * rows * 4),
           tag + ": gated vector width " + std::to_string(n) +
               " changed SwiGLU bits");
  }

  // Tiled MMQ (prefill fallback).
  using Raw = int (*)(const void*, const float*, const std::int32_t*, float*,
                      int, int, int, int, int, hipStream_t);
  const Raw raw = f.type == GgmlType::kIQ3_S    ? qfn_mmq_iq3_s_moe_raw
                  : f.type == GgmlType::kIQ4_NL ? qfn_mmq_iq4_nl_moe_raw
                                                : qfn_mmq_iq4_xs_moe_raw;
  CheckHip(hipMemset(out, 0, count * sizeof(float)), "clear MMQ output");
  Expect(!raw(wa, dx, di, out, rows, cols, tokens, experts, used, nullptr),
         tag + ": tiled MMQ launch failed");
  CheckHip(hipDeviceSynchronize(), "tiled MMQ");
  const double mmq_error = RelativeError(Download(out, count), ref_a);
  Expect(mmq_error < 1e-2, tag + ": MMQ error " + std::to_string(mmq_error));

  std::cout << tag << " tokens=" << tokens << " used=" << used
            << ": vector rel err " << vec_error << ", MMQ rel err " << mmq_error
            << '\n';
  for (void* p : {static_cast<void*>(wa), static_cast<void*>(wb),
                  static_cast<void*>(dx), static_cast<void*>(di),
                  static_cast<void*>(single_a), static_cast<void*>(single_b),
                  static_cast<void*>(out), static_cast<void*>(out_b)})
    CheckHip(hipFree(p), "free");
}

/// Q6_K head: 1–8 rows per launch, each row bitwise equal to a one-row
/// launch, all within tolerance of F64.
void CheckQ6KHead(int rows, int cols) {
  const std::string tag =
      "Q6_K head m=" + std::to_string(rows) + " k=" + std::to_string(cols);
  constexpr int tokens = 8;
  const Weights w = Make(kQ6_K, 1, rows, cols, 0x6060U + rows);
  std::vector<float> x(static_cast<std::size_t>(tokens) * cols);
  std::uint32_t seed = 91;
  for (auto& v : x)
    v = Uniform(&seed, 3.0F);
  std::vector<double> ref(static_cast<std::size_t>(tokens) * rows);
  for (int t = 0; t < tokens; ++t)
    for (int r = 0; r < rows; ++r)
      ref[static_cast<std::size_t>(t) * rows + r] =
          Dot(w.values.data() + static_cast<std::size_t>(r) * cols,
              x.data() + static_cast<std::size_t>(t) * cols, cols);
  auto* dw = Upload(w.packed);
  auto* dx = Upload(x);
  void* xq = nullptr;
  CheckHip(hipMalloc(&xq, qfn_mmq_q8_1_bytes(tokens, cols)), "hipMalloc");
  Expect(!qfn_mmq_quantize_q8_1(dx, xq, tokens, cols, nullptr),
         tag + ": quantize failed");
  const std::size_t row_q8 = qfn_mmq_q8_1_bytes(1, cols);
  const std::vector<float> zeros(static_cast<std::size_t>(tokens) * rows);
  auto* single = Upload(zeros);
  auto* out = Upload(zeros);
  const int type = static_cast<int>(kQ6_K.type);
  for (int t = 0; t < tokens; ++t)
    Expect(!qfn_mmq_dense_vec_preq(
               type, dw, static_cast<const std::uint8_t*>(xq) + t * row_q8,
               single + static_cast<std::size_t>(t) * rows, rows, 1, cols,
               nullptr),
           tag + ": one-row launch failed");
  const auto decode = Download(single, zeros.size());
  const double error = RelativeError(decode, ref);
  Expect(error < 1e-2, tag + ": error " + std::to_string(error));
  for (int n = 2; n <= tokens; ++n) {
    Expect(!qfn_mmq_dense_vec_preq(type, dw, xq, out, rows, n, cols, nullptr),
           tag + ": batched launch failed");
    Expect(!std::memcmp(Download(out, zeros.size()).data(), decode.data(),
                        static_cast<std::size_t>(n) * rows * 4),
           tag + ": width " + std::to_string(n) + " changed decode bits");
  }
  std::cout << tag << ": rel err " << error << '\n';
  for (void* p : {static_cast<void*>(dw), static_cast<void*>(dx), xq,
                  static_cast<void*>(single), static_cast<void*>(out)})
    CheckHip(hipFree(p), "free");
}

}  // namespace

int main() {
  try {
    if (qfn_mmq_init(0) != 0)
      throw std::runtime_error("qfn_mmq_init failed");
    // Gate/up view (top-k slots), with a ragged row count.
    for (int rows : {64, 65}) {
      CheckRouted(kIQ3_S, rows, 2560, 33, 3);
      CheckRouted(kIQ4_XS, rows, 2560, 33, 3);
      CheckRouted(kIQ4_NL, rows, 2560, 33, 3);
    }
    // Down view: one slot per row, K 640.
    CheckRouted(kIQ4_NL, 2560, 640, 40, 1);
    CheckRouted(kIQ4_NL, 2561, 640, 9, 1);
    CheckQ6KHead(300, 2560);
    CheckQ6KHead(301, 2560);
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << '\n';
    return 1;
  }
  std::cout << "IQ routed experts and Q6_K head: all checks passed\n";
  return 0;
}
