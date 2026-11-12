// IQ3_S/IQ4_NL/IQ4_XS (UD-IQ4_XS), IQ2_S/IQ3_XXS/Q2_0 (UD-IQ3_XXS) and
// IQ2_XS/IQ2_XXS (GSQ-RCO IQ3_XXS) routed experts, the Q6_K logits head and
// GSQ-RCO's K-quant, IQ and Q2_0 dense projections: accuracy against an F64 dot
// product over the CPU dequantizers, and the exactness contracts of the vector
// paths (batched rows, paired and fused gate/up all match single-row decode bit
// for bit).
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
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
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
  std::size_t min_offset{0};  ///< byte offset of a K-quant's F16 dmin, or 0
};

const Format kIQ3_S{
    GgmlType::kIQ3_S, "IQ3_S", 256, 110, 0, gufo::quant::DequantizeIQ3_S};
const Format kIQ4_NL{
    GgmlType::kIQ4_NL, "IQ4_NL", 32, 18, 0, gufo::quant::DequantizeIQ4_NL};
const Format kIQ4_XS{
    GgmlType::kIQ4_XS, "IQ4_XS", 256, 136, 0, gufo::quant::DequantizeIQ4_XS};
const Format kIQ2_S{
    GgmlType::kIQ2_S, "IQ2_S", 256, 82, 0, gufo::quant::DequantizeIQ2_S};
const Format kIQ2_XS{
    GgmlType::kIQ2_XS, "IQ2_XS", 256, 74, 0, gufo::quant::DequantizeIQ2_XS};
const Format kIQ2_XXS{
    GgmlType::kIQ2_XXS, "IQ2_XXS", 256, 66, 0, gufo::quant::DequantizeIQ2_XXS};
const Format kIQ3_XXS{
    GgmlType::kIQ3_XXS, "IQ3_XXS", 256, 98, 0, gufo::quant::DequantizeIQ3_XXS};
const Format kQ2_0{
    GgmlType::kQ2_0, "Q2_0", 64, 18, 0, gufo::quant::DequantizeQ2_0};
const Format kQ5_0{
    GgmlType::kQ5_0, "Q5_0", 32, 22, 0, gufo::quant::DequantizeQ5_0};
const Format kQ6_K{
    GgmlType::kQ6_K, "Q6_K", 256, 210, 208, gufo::quant::DequantizeQ6_K};
const Format kQ5_K{
    GgmlType::kQ5_K, "Q5_K", 256, 176, 0, gufo::quant::DequantizeQ5_K, 2};
const Format kQ4_K{
    GgmlType::kQ4_K, "Q4_K", 256, 144, 0, gufo::quant::DequantizeQ4_K, 2};

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
    if (f.min_offset != 0) {
      const __half dmin = __float2half(Uniform(&seed, magnitude));
      std::memcpy(w.packed.data() + b * f.bytes + f.min_offset, &dmin, 2);
    }
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
  // IQ3_S/IQ4_XS gate/up also group up to 64 rows of a request batch by
  // expert (the wave64 path). Q2_0 and Q5_0 are only down projection formats.
  std::vector<int> gated_widths{1, 2, 3, 4, 5, 6, 7, 8};
  if (f.type == GgmlType::kQ2_0 || f.type == GgmlType::kQ5_0)
    gated_widths.clear();
  if (f.type == GgmlType::kIQ3_S || f.type == GgmlType::kIQ4_XS)
    gated_widths.insert(gated_widths.end(), {9, 16, 31, 32, 33});
  for (int n : gated_widths) {
    if (n > tokens)
      break;
    Expect(!qfn_mmq_moe_gated_vec(type, wa, wb, dx, di, out, rows, cols, n,
                                  experts, used, nullptr),
           tag + ": gated vector launch failed");
    Expect(!std::memcmp(Download(out, count).data(), swiglu.data(),
                        static_cast<std::size_t>(n) * used * rows * 4),
           tag + ": gated vector width " + std::to_string(n) +
               " changed SwiGLU bits");
  }

  std::cout << tag << " tokens=" << tokens << " used=" << used
            << ": vector rel err " << vec_error << '\n';
  for (void* p : {static_cast<void*>(wa), static_cast<void*>(wb),
                  static_cast<void*>(dx), static_cast<void*>(di),
                  static_cast<void*>(single_a), static_cast<void*>(single_b),
                  static_cast<void*>(out), static_cast<void*>(out_b)})
    CheckHip(hipFree(p), "free");
}

/// Layouts the expert kernels can run: IQ formats must take the routed F16 path
/// at prefill, because they have no tiled MMQ route.
void CheckIqExpertLayouts() {
  gufo::models::qwen38_flash_next::Config c;
  c.hidden_size = 2560;
  c.expert_ff = 640;
  const auto ok = [&](GgmlType gate, GgmlType up, GgmlType down) {
    return q::IqExpertLayoutSupported(c, gate, up, down);
  };
  using enum GgmlType;
  Expect(ok(kIQ3_S, kIQ3_S, kIQ4_NL), "IQ3_S gate/up with IQ4_NL down");
  Expect(ok(kIQ3_S, kIQ3_S, kQ8_0), "IQ3_S gate/up with Q8_0 down");
  Expect(ok(kIQ4_XS, kIQ4_XS, kIQ4_NL), "IQ4_XS gate/up with IQ4_NL down");
  Expect(ok(kIQ2_S, kIQ2_S, kIQ4_NL), "IQ2_S gate/up with IQ4_NL down");
  Expect(ok(kIQ2_S, kIQ2_S, kQ2_0), "IQ2_S gate/up with Q2_0 down");
  Expect(ok(kIQ3_XXS, kIQ3_XXS, kIQ4_NL), "IQ3_XXS gate/up with IQ4_NL down");
  Expect(ok(kQ4_K, kQ4_K, kIQ4_NL), "Q4_K gate/up with IQ4_NL down");
  Expect(ok(kQ8_0, kQ8_0, kQ8_0), "layouts without IQ formats stay allowed");
  Expect(ok(kQ4_K, kQ8_0, kQ5_1), "mixed layouts without IQ stay allowed");
  Expect(!ok(kIQ3_S, kIQ4_XS, kIQ4_NL), "mismatched IQ gate/up");
  Expect(!ok(kIQ3_S, kQ4_K, kIQ4_NL), "IQ gate with another up format");
  Expect(!ok(kIQ4_NL, kIQ4_NL, kIQ4_NL), "IQ4_NL gate/up");
  Expect(!ok(kIQ3_S, kIQ3_S, kIQ4_XS), "IQ4_XS down");
  Expect(!ok(kQ4_K, kQ4_K, kIQ3_S), "IQ3_S down");
  Expect(!ok(kQ2_0, kQ2_0, kIQ4_NL), "Q2_0 gate/up");
  Expect(ok(kIQ3_XXS, kIQ3_XXS, kQ5_0), "IQ3_XXS gate/up with Q5_0 down");
  Expect(ok(kIQ4_XS, kIQ4_XS, kQ5_0), "IQ4_XS gate/up with Q5_0 down");
  Expect(!ok(kIQ2_S, kIQ2_S, kIQ3_XXS), "IQ3_XXS down");
  c.expert_ff = 641;
  Expect(!ok(kIQ3_S, kIQ3_S, kIQ4_NL), "IQ experts outside the WMMA shape");
}

/// Embedding lookup over IQ4_XS (UD-IQ4_XS), Q6_K (UD-IQ3_XXS) and IQ3_S
/// (GSQ-RCO IQ3_XXS) tables: every element equal to the CPU decoder's value,
/// copied to each stream.
void CheckEmbedding(const Format& f) {
  constexpr std::size_t kHidden = 2560, kVocab = 97, kStreams = 2;
  const Weights w = Make(f, 1, kVocab, kHidden, 0xE3BEDU);
  const std::vector<std::int32_t> tokens{0, 96, 5, 5, 40};
  auto* table = Upload(w.packed);
  auto* ids = Upload(tokens);
  auto* res = Upload(std::vector<float>(tokens.size() * kStreams * kHidden));
  q::EmbedTokens(table, static_cast<q::WeightType>(f.type), ids, res,
                 static_cast<std::uint32_t>(tokens.size()), kHidden, kStreams,
                 nullptr);
  const auto out = Download(res, tokens.size() * kStreams * kHidden);
  for (std::size_t t = 0; t < tokens.size(); ++t)
    for (std::size_t s = 0; s < kStreams; ++s)
      for (std::size_t i = 0; i < kHidden; ++i) {
        const float want = w.values[tokens[t] * kHidden + i];
        const float got = out[(t * kStreams + s) * kHidden + i];
        Expect(std::fabs(got - want) <= 1e-6F * std::fabs(want),
               std::string(f.name) + " embedding differs from the decoder");
      }
  std::cout << f.name << " embedding: exact against the CPU decoder\n";
  for (void* p : {static_cast<void*>(table), static_cast<void*>(ids),
                  static_cast<void*>(res)})
    CheckHip(hipFree(p), "free");
}

/// Dense GEMV (the Q6_K head, GSQ-RCO's dense projections in decode): 1–8
/// rows per launch, each row bitwise equal to a one-row launch, all within
/// tolerance of F64.
void CheckDenseGemv(const Format& f, int rows, int cols) {
  const std::string tag = std::string(f.name) +
                          " GEMV m=" + std::to_string(rows) +
                          " k=" + std::to_string(cols);
  constexpr int tokens = 8;
  const Weights w = Make(f, 1, rows, cols, 0x6060U + rows);
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
  const int type = static_cast<int>(f.type);
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

/// Dense prefill through the routed F16 GEMM: one expert whose bucket is
/// every row, in order, as Executor::DenseRoutedF16 launches it (128-token
/// tiles). Within tolerance of F64 over the F16 activations, each row's bits
/// independent of the pass width, and equal to 48-token tiles.
void CheckDenseRouted(const Format& f, int rows, int cols) {
  const std::string tag = std::string(f.name) +
                          " dense F16 m=" + std::to_string(rows) +
                          " k=" + std::to_string(cols);
  constexpr int kTileRows = 128, kTokens = 213;
  const Weights w = Make(f, 1, rows, cols, 0xD00DU + rows);
  std::vector<__half> x(static_cast<std::size_t>(kTokens) * cols);
  std::uint32_t seed = 17;
  for (auto& v : x)
    v = __float2half(Uniform(&seed, 3.0F));
  std::vector<float> xf(x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    xf[i] = __half2float(x[i]);
  std::vector<double> ref(static_cast<std::size_t>(kTokens) * rows);
  for (int t = 0; t < kTokens; ++t)
    for (int r = 0; r < rows; ++r)
      ref[static_cast<std::size_t>(t) * rows + r] =
          Dot(w.values.data() + static_cast<std::size_t>(r) * cols,
              xf.data() + static_cast<std::size_t>(t) * cols, cols);
  std::vector<std::int32_t> identity(kTokens), tiles;
  for (int i = 0; i < kTokens; ++i)
    identity[i] = i;
  for (int i = 0; i * 16 < kTokens; ++i)
    tiles.push_back(i << 16);
  auto* dw = Upload(w.packed);
  auto* dx = Upload(x);
  auto* rows_map = Upload(identity);
  auto* tile_map = Upload(tiles);
  const std::vector<float> zeros(static_cast<std::size_t>(kTokens) * rows);
  std::vector<float> full;
  for (int n : {kTokens, 1, 47, 49, 127, 129}) {
    auto* bounds = Upload(std::vector<std::int32_t>{0, n});
    auto* out = Upload(zeros);
    Expect(q::RoutedF16Gemm(
               dw, static_cast<q::WeightType>(f.type), dx, tile_map,
               static_cast<std::uint32_t>((n + kTileRows - 1) / kTileRows),
               kTileRows, bounds, rows_map, rows_map, nullptr, out, nullptr,
               rows, cols, nullptr),
           tag + ": launch rejected");
    const auto got = Download(out, zeros.size());
    if (n == kTokens) {
      const double error = RelativeError(got, ref);
      Expect(error < 2e-3, tag + ": error " + std::to_string(error));
      std::cout << tag << ": rel err " << error << '\n';
      full = got;
    } else {
      Expect(!std::memcmp(got.data(), full.data(),
                          static_cast<std::size_t>(n) * rows * 4),
             tag + ": " + std::to_string(n) + " rows changed the bits");
    }
    for (void* p : {static_cast<void*>(bounds), static_cast<void*>(out)})
      CheckHip(hipFree(p), "free");
  }
  // The 48-token tiles of the expert route give the same bits.
  {
    auto* bounds = Upload(std::vector<std::int32_t>{0, kTokens});
    auto* out = Upload(zeros);
    Expect(
        q::RoutedF16Gemm(dw, static_cast<q::WeightType>(f.type), dx, tile_map,
                         (kTokens + 47) / 48, 48, bounds, rows_map, rows_map,
                         nullptr, out, nullptr, rows, cols, nullptr),
        tag + ": 48-token launch rejected");
    Expect(!std::memcmp(Download(out, zeros.size()).data(), full.data(),
                        zeros.size() * 4),
           tag + ": 48- and 128-token tiles differ");
    for (void* p : {static_cast<void*>(bounds), static_cast<void*>(out)})
      CheckHip(hipFree(p), "free");
  }
  // A prefill chunk boundary: the last rows as their own pass.
  for (int first : {kTokens - 9, 128, 64, 17}) {
    const int n = kTokens - first;
    auto* bounds = Upload(std::vector<std::int32_t>{0, n});
    auto* out = Upload(zeros);
    Expect(q::RoutedF16Gemm(
               dw, static_cast<q::WeightType>(f.type),
               dx + static_cast<std::size_t>(first) * cols, tile_map,
               static_cast<std::uint32_t>((n + kTileRows - 1) / kTileRows),
               kTileRows, bounds, rows_map, rows_map, nullptr, out, nullptr,
               rows, cols, nullptr),
           tag + ": launch rejected");
    const auto got = Download(out, static_cast<std::size_t>(n) * rows);
    Expect(!std::memcmp(got.data(),
                        full.data() + static_cast<std::size_t>(first) * rows,
                        got.size() * 4),
           tag + ": rows from " + std::to_string(first) +
               " changed bits as their own pass");
    for (void* p : {static_cast<void*>(bounds), static_cast<void*>(out)})
      CheckHip(hipFree(p), "free");
  }
  for (void* p : {static_cast<void*>(dw), static_cast<void*>(dx),
                  static_cast<void*>(rows_map), static_cast<void*>(tile_map)})
    CheckHip(hipFree(p), "free");
}

/// BF16 projections in prefill (GSQ-RCO's hyper-connection mixers and PLE
/// key/value): the rows of a pass split at a chunk boundary keep their bits,
/// also where the whole pass and its pieces take different tile plans.
void CheckBf16Chunks(int rows, int cols) {
  const std::string tag =
      "BF16 GEMM m=" + std::to_string(rows) + " k=" + std::to_string(cols);
  constexpr int kTokens = 1100;
  std::uint32_t seed = 0xBF16U + rows;
  const auto bf16 = [](float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    return static_cast<std::uint16_t>(bits >> 16U);
  };
  std::vector<std::uint16_t> w(static_cast<std::size_t>(rows) * cols);
  std::vector<std::uint16_t> x(static_cast<std::size_t>(kTokens) * cols);
  for (auto& v : w)
    v = bf16(Uniform(&seed, 0.05F));
  for (auto& v : x)
    v = bf16(Uniform(&seed, 3.0F));
  auto* dw = Upload(w);
  auto* dx = Upload(x);
  const std::vector<float> zeros(static_cast<std::size_t>(kTokens) * rows);
  auto* full = Upload(zeros);
  auto* part = Upload(zeros);
  Expect(q::DenseBf16Gemm(dw, dx, full, kTokens, rows, cols, nullptr),
         tag + ": launch rejected");
  const auto whole = Download(full, zeros.size());
  for (int first : {1091, 1036, 76, 1}) {
    const int n = kTokens - first;
    Expect(q::DenseBf16Gemm(dw, dx + static_cast<std::size_t>(first) * cols,
                            part, n, rows, cols, nullptr),
           tag + ": launch rejected");
    const auto got = Download(part, static_cast<std::size_t>(n) * rows);
    Expect(!std::memcmp(got.data(),
                        whole.data() + static_cast<std::size_t>(first) * rows,
                        got.size() * 4),
           tag + ": rows from " + std::to_string(first) +
               " changed bits as their own pass");
  }
  std::cout << tag << ": chunked passes exact\n";
  for (void* p : {static_cast<void*>(dw), static_cast<void*>(dx),
                  static_cast<void*>(full), static_cast<void*>(part)})
    CheckHip(hipFree(p), "free");
}

}  // namespace

int main() {
  try {
    if (qfn_mmq_init(0) != 0)
      throw std::runtime_error("qfn_mmq_init failed");
    CheckIqExpertLayouts();
    // Gate/up view (top-k slots), with a ragged row count.
    for (int rows : {64, 65}) {
      CheckRouted(kIQ3_S, rows, 2560, 33, 3);
      CheckRouted(kIQ4_XS, rows, 2560, 33, 3);
      CheckRouted(kIQ4_NL, rows, 2560, 33, 3);
      CheckRouted(kIQ2_S, rows, 2560, 33, 3);
      CheckRouted(kIQ3_XXS, rows, 2560, 33, 3);
      CheckRouted(kIQ2_XS, rows, 2560, 33, 3);
      CheckRouted(kIQ2_XXS, rows, 2560, 33, 3);
    }
    // Down view: one slot per row, K 640.
    CheckRouted(kIQ4_NL, 2560, 640, 40, 1);
    CheckRouted(kIQ4_NL, 2561, 640, 9, 1);
    CheckRouted(kQ2_0, 2560, 640, 40, 1);
    CheckRouted(kQ2_0, 2561, 640, 9, 1);
    CheckRouted(kQ5_0, 2560, 640, 40, 1);
    CheckRouted(kQ5_0, 2561, 640, 9, 1);
    CheckEmbedding(kIQ4_XS);
    CheckEmbedding(kQ6_K);
    CheckEmbedding(kIQ3_S);
    CheckDenseGemv(kQ6_K, 300, 2560);
    CheckDenseGemv(kQ6_K, 301, 2560);
    // GSQ-RCO dense projections: K 2560 (attention, linear attention, shared
    // gate/up), 6144 (output projections) and 640 (shared down).
    for (const Format* f : {&kQ5_K, &kQ4_K, &kIQ4_XS, &kIQ3_S}) {
      CheckDenseGemv(*f, 301, 2560);
      CheckDenseGemv(*f, 257, 6144);
      CheckDenseRouted(*f, 301, 2560);
      CheckDenseRouted(*f, 257, 6144);
    }
    for (const Format* f : {&kIQ4_NL, &kQ2_0}) {
      CheckDenseGemv(*f, 2561, 640);
      CheckDenseRouted(*f, 2561, 640);
    }
    // Hyper-connection down/up (HC 10240, low rank 320).
    CheckBf16Chunks(320, 10240);
    CheckBf16Chunks(10240, 320);
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << '\n';
    return 1;
  }
  std::cout << "IQ routed experts, dense projections and Q6_K head: all "
               "checks passed\n";
  return 0;
}
