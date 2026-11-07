// Oracle test for the 8-bit attention KV cache (--kv-cache q8_0) on the
// Qwen3.6-35B-A3B head shape: sixteen query heads over two 256-wide KV heads.
//
// Writers: the batched (prefill and verification) and single-token fused
// writers and the unfused decode writer store the rows a CPU Q8_0
// quantization gives, within one step of rounding, and the three GPU writers
// agree byte for byte, so a position has the same bytes however it was
// written. Readers: decode, verification and prefill attention over the
// 8-bit cache agree with a double-precision reference over the dequantized
// cache, and a verification row is byte-identical to the same position
// decoded alone.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <gufo/ext/experimental/hip_utils.hpp>

#include "src/families/qwen/hip_common/attention_policy.hpp"
#include "src/families/qwen/kernels/kv_q8.hpp"
#include "src/families/qwen/kernels/ops.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

namespace {

namespace kv_q8 = gufo::hip::kv_q8;

constexpr std::uint32_t kHeads = 16;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kRotaryDim = 64;
constexpr std::size_t kKvWidth = std::size_t{kKvHeads} * kHeadDim;
constexpr std::size_t kAttnWidth = std::size_t{kHeads} * kHeadDim;
constexpr std::size_t kTokenBytes = std::size_t{kKvHeads} * kv_q8::kRowBytes;

class Rng {
public:
  explicit Rng(std::uint32_t seed) : state_(seed) {}
  float Uniform(float lo, float hi) {
    state_ = (state_ * 1664525U) + 1013904223U;
    const float u = static_cast<float>((state_ >> 8) & 0xFFFFU) / 65535.0F;
    return lo + (u * (hi - lo));
  }
  std::uint8_t Byte() {
    state_ = (state_ * 1664525U) + 1013904223U;
    return static_cast<std::uint8_t>(state_ >> 24);
  }

private:
  std::uint32_t state_;
};

void Fail(const char* message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

template<typename T>
gufo::test::DeviceBuffer<T> Upload(const std::vector<T>& host) {
  gufo::test::DeviceBuffer<T> device(host.size());
  HIP_CHECK(hipMemcpy(device.data(), host.data(), host.size() * sizeof(T),
                      hipMemcpyHostToDevice));
  return device;
}

template<typename T>
std::vector<T> Download(const gufo::test::DeviceBuffer<T>& device,
                        std::size_t count) {
  std::vector<T> host(count);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(host.data(), device.data(), count * sizeof(T),
                      hipMemcpyDeviceToHost));
  return host;
}

float Scale(const std::uint8_t* row, std::uint32_t block) {
  __half d;
  std::memcpy(&d, row + kv_q8::kScaleOffset + block * sizeof(__half),
              sizeof(d));
  return __half2float(d);
}

// The FP16 value the kernels stage for dimension `dim` of a row.
float Dequantized(const std::uint8_t* row, std::uint32_t dim) {
  const auto q = static_cast<std::int8_t>(row[dim]);
  return __half2float(
      __float2half(static_cast<float>(q) * Scale(row, dim / kv_q8::kBlock)));
}

// CPU Q8_0 of one 256-wide head.
void Quantize(const float* x, std::uint8_t* row) {
  for (std::uint32_t block = 0; block < kv_q8::kBlocks; ++block) {
    float amax = 0.0F;
    for (std::uint32_t i = 0; i < kv_q8::kBlock; ++i)
      amax = std::max(amax, std::fabs(x[block * kv_q8::kBlock + i]));
    const __half d = __float2half(amax / 127.0F);
    const float scale = __half2float(d);
    const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
    for (std::uint32_t i = 0; i < kv_q8::kBlock; ++i) {
      const long q = std::lrint(x[block * kv_q8::kBlock + i] * inverse);
      row[block * kv_q8::kBlock + i] = static_cast<std::uint8_t>(
          static_cast<std::int8_t>(std::clamp<long>(q, -127, 127)));
    }
    std::memcpy(row + kv_q8::kScaleOffset + block * sizeof(__half), &d,
                sizeof(d));
  }
}

// A GPU row matches the CPU quantization of `x` up to one rounding step:
// scales within one FP16 ulp, values within one quantization step, and every
// dequantized value within one scale of the input.
void ExpectRow(const std::uint8_t* got, const float* x, const char* what) {
  std::uint8_t want[kv_q8::kRowBytes];
  Quantize(x, want);
  for (std::uint32_t block = 0; block < kv_q8::kBlocks; ++block) {
    const float a = Scale(got, block), b = Scale(want, block);
    if (std::fabs(a - b) > std::ldexp(std::max(a, b), -10))
      Fail(what);
  }
  for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
    const int a = static_cast<std::int8_t>(got[dim]);
    const int b = static_cast<std::int8_t>(want[dim]);
    if (std::abs(a - b) > 1 ||
        std::fabs(Dequantized(got, dim) - x[dim]) >
            Scale(got, dim / kv_q8::kBlock) * 1.01F + 1e-6F)
      Fail(what);
  }
}

void TestWriters() {
  constexpr std::uint32_t kLayer = 1, kContext = 64, kStart = 37, kBatch = 5;
  const std::size_t cache_bytes = 2 * kContext * kTokenBytes;
  Rng rng(7);
  std::vector<float> q(kBatch * kAttnWidth), k(kBatch * kKvWidth),
      v(kBatch * kKvWidth), q_weight(kHeadDim), k_weight(kHeadDim);
  for (auto& x : q)
    x = rng.Uniform(-2.0F, 2.0F);
  for (auto& x : k)
    x = rng.Uniform(-2.0F, 2.0F);
  for (auto& x : v)
    x = rng.Uniform(-3.0F, 3.0F);
  // A large outlier in one block, as RoPE dimensions of real keys have.
  k[3] = 40.0F;
  v[kKvWidth + 300] = -25.0F;
  for (auto& x : q_weight)
    x = rng.Uniform(0.5F, 1.5F);
  for (auto& x : k_weight)
    x = rng.Uniform(0.5F, 1.5F);
  std::vector<std::uint8_t> fill(cache_bytes);
  for (auto& b : fill)
    b = rng.Byte();

  const auto d_q = Upload(q), d_k = Upload(k), d_v = Upload(v),
             d_qw = Upload(q_weight), d_kw = Upload(k_weight);
  gufo::test::DeviceBuffer<float> d_q_out(q.size()), d_k_out(k.size());
  auto batched_k = Upload(fill), batched_v = Upload(fill);
  gufo::hip::LaunchBatchedFusedQKNormRoPEKvWrite(
      d_q.data(), d_k.data(), d_v.data(), d_qw.data(), d_kw.data(),
      d_q_out.data(), d_k_out.data(), nullptr, nullptr, batched_k.data(),
      batched_v.data(), kLayer, kStart, kBatch, kContext, kHeads, kKvHeads,
      kHeadDim, kRotaryDim, 1.0e7F, 1e-6F, nullptr, nullptr, true);
  const auto k_rot = Download(d_k_out, k.size());
  const auto got_k = Download(batched_k, cache_bytes);
  const auto got_v = Download(batched_v, cache_bytes);

  const auto row_at = [&](std::uint32_t pos, std::uint32_t head) {
    return kv_q8::RowOffset(kLayer, pos, head, kContext, kKvHeads);
  };
  for (std::uint32_t b = 0; b < kBatch; ++b) {
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
      const std::size_t src = b * kKvWidth + head * kHeadDim;
      ExpectRow(&got_k[row_at(kStart + b, head)], &k_rot[src],
                "batched writer K row differs from the CPU quantization");
      ExpectRow(&got_v[row_at(kStart + b, head)], &v[src],
                "batched writer V row differs from the CPU quantization");
    }
  }
  const std::size_t first = row_at(kStart, 0);
  const std::size_t last = row_at(kStart + kBatch, 0);
  for (std::size_t i = 0; i < cache_bytes; ++i)
    if ((i < first || i >= last) &&
        (got_k[i] != fill[i] || got_v[i] != fill[i]))
      Fail("the batched writer touched bytes outside its rows");

  // Single-token fused writer, one position at a time.
  auto single_k = Upload(fill), single_v = Upload(fill);
  gufo::test::DeviceBuffer<std::uint32_t> d_pos(1);
  for (std::uint32_t b = 0; b < kBatch; ++b) {
    const std::uint32_t pos = kStart + b;
    HIP_CHECK(
        hipMemcpy(d_pos.data(), &pos, sizeof(pos), hipMemcpyHostToDevice));
    gufo::hip::LaunchFusedQKNormRoPEKvWrite(
        d_q.data() + b * kAttnWidth, d_k.data() + b * kKvWidth,
        d_v.data() + b * kKvWidth, d_qw.data(), d_kw.data(), d_q_out.data(),
        d_k_out.data(), nullptr, nullptr, single_k.data(), single_v.data(),
        kLayer, d_pos.data(), kContext, kHeads, kKvHeads, kHeadDim, kRotaryDim,
        1.0e7F, 1e-6F, nullptr, nullptr, true);
    HIP_CHECK(hipDeviceSynchronize());
  }
  if (Download(single_k, cache_bytes) != got_k ||
      Download(single_v, cache_bytes) != got_v)
    Fail("the single-token writer differs from the batched writer");

  // Unfused decode writer, fed the rotated keys the fused writers stored.
  auto plain_k = Upload(fill), plain_v = Upload(fill);
  const auto d_k_rot = Upload(k_rot);
  const std::size_t row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  gufo::test::DeviceBuffer<float> d_out(kAttnWidth), d_scratch(row_scratch);
  for (std::uint32_t b = 0; b < kBatch; ++b) {
    gufo::hip::LaunchAttention(
        d_q_out.data(), d_k_rot.data() + b * kKvWidth,
        d_v.data() + b * kKvWidth, nullptr, nullptr, nullptr, plain_k.data(),
        plain_v.data(), d_out.data(), kLayer, kStart + b, kContext, kHeads,
        kKvHeads, kHeadDim, nullptr, d_scratch.data(), false, true);
  }
  // Rows before kStart hold random bytes, which may decode to infinities in
  // the attention output; only the cache bytes are checked here.
  if (Download(plain_k, cache_bytes) != got_k ||
      Download(plain_v, cache_bytes) != got_v)
    Fail("the unfused decode writer differs from the fused writers");
  std::cout << "8-bit KV writers: rows match the CPU quantization, all three "
               "writers byte-identical\n";
}

struct Cache {
  std::vector<std::uint8_t> k, v;
  std::uint32_t context;
};

// Layer 1 of a two-layer cache; every row is quantized from random values.
Cache MakeCache(std::uint32_t context, Rng& rng) {
  Cache cache{std::vector<std::uint8_t>(2 * context * kTokenBytes),
              std::vector<std::uint8_t>(2 * context * kTokenBytes), context};
  std::vector<float> x(kHeadDim);
  for (std::uint32_t pos = 0; pos < context; ++pos) {
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
      const std::size_t at = kv_q8::RowOffset(1, pos, head, context, kKvHeads);
      for (auto& value : x)
        value = rng.Uniform(-1.0F, 1.0F);
      x[pos % kHeadDim] = 6.0F;
      Quantize(x.data(), &cache.k[at]);
      for (auto& value : x)
        value = rng.Uniform(-1.0F, 1.0F);
      Quantize(x.data(), &cache.v[at]);
    }
  }
  return cache;
}

// Double-precision gated attention of query rows at positions start_pos + r
// over the dequantized layer-1 cache; returns the largest error and output.
void Reference(const Cache& cache, const std::vector<float>& q,
               const std::vector<float>& gate, std::uint32_t start_pos,
               std::uint32_t rows, const std::vector<float>& got,
               double* max_error, double* magnitude) {
  *max_error = 0.0;
  *magnitude = 0.0;
  const auto at = [&](std::uint32_t pos, std::uint32_t head) {
    return kv_q8::RowOffset(1, pos, head, cache.context, kKvHeads);
  };
  for (std::uint32_t row = 0; row < rows; ++row) {
    const std::uint32_t pos = start_pos + row;
    for (std::uint32_t head = 0; head < kHeads; ++head) {
      const std::uint32_t kv_head = head / (kHeads / kKvHeads);
      const float* qh = &q[row * kAttnWidth + head * kHeadDim];
      std::vector<double> scores(pos + 1);
      double max_score = -INFINITY;
      for (std::uint32_t key = 0; key <= pos; ++key) {
        double dot = 0.0;
        for (std::uint32_t d = 0; d < kHeadDim; ++d)
          dot += static_cast<double>(qh[d]) *
                 Dequantized(&cache.k[at(key, kv_head)], d);
        scores[key] = dot / 16.0;
        max_score = std::max(max_score, scores[key]);
      }
      double sum = 0.0;
      for (auto& s : scores)
        sum += (s = std::exp(s - max_score));
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        double acc = 0.0;
        for (std::uint32_t key = 0; key <= pos; ++key)
          acc += scores[key] * Dequantized(&cache.v[at(key, kv_head)], d);
        const std::size_t i = row * kAttnWidth + head * kHeadDim + d;
        const double want = acc / sum / (1.0 + std::exp(-gate[i]));
        *max_error = std::max(*max_error, std::fabs(want - got[i]));
        *magnitude = std::max(*magnitude, std::fabs(want));
      }
    }
  }
}

std::vector<float> RunDecode(const Cache& cache, const std::vector<float>& q,
                             const std::vector<float>& gate,
                             std::uint32_t start_pos, std::uint32_t rows,
                             std::uint32_t valid_end, Rng& rng) {
  auto k = cache.k;
  auto v = cache.v;
  // Positions past valid_end are not written yet in single-token decoding.
  for (std::uint32_t pos = valid_end; pos < cache.context; ++pos) {
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
      const std::size_t at =
          kv_q8::RowOffset(1, pos, head, cache.context, kKvHeads);
      for (std::size_t i = 0; i < kv_q8::kHeadDim; ++i) {
        k[at + i] = rng.Byte();
        v[at + i] = rng.Byte();
      }
    }
  }
  const std::size_t row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  auto d_k = Upload(k), d_v = Upload(v);
  const auto d_q = Upload(q), d_gate = Upload(gate);
  gufo::test::DeviceBuffer<float> d_out(rows * kAttnWidth),
      d_scratch(rows * row_scratch);
  gufo::hip::LaunchCausalDecodeAttention(
      d_q.data(), d_gate.data(), nullptr, nullptr, d_k.data(), d_v.data(),
      d_out.data(), 1, start_pos, rows, cache.context, kHeads, kKvHeads,
      kHeadDim, nullptr, {d_scratch.data(), rows * row_scratch}, true);
  return Download(d_out, rows * kAttnWidth);
}

void TestDecode(std::uint32_t start_pos, std::uint32_t rows) {
  Rng rng(start_pos * 31U + rows);
  const auto cache = MakeCache(start_pos + rows + 32, rng);
  std::vector<float> q(rows * kAttnWidth), gate(rows * kAttnWidth);
  for (auto& x : q)
    x = rng.Uniform(-3.0F, 3.0F);
  for (auto& x : gate)
    x = rng.Uniform(-2.0F, 2.0F);
  const auto batched =
      RunDecode(cache, q, gate, start_pos, rows, start_pos + rows, rng);
  double max_error = 0.0, magnitude = 0.0;
  Reference(cache, q, gate, start_pos, rows, batched, &max_error, &magnitude);
  if (!(max_error <= 4e-3 * std::max(magnitude, 1e-3)))
    Fail("8-bit decode attention is far from the reference");
  for (std::uint32_t row = 0; row < rows; ++row) {
    const std::vector<float> q_row(q.begin() + row * kAttnWidth,
                                   q.begin() + (row + 1) * kAttnWidth);
    const std::vector<float> gate_row(gate.begin() + row * kAttnWidth,
                                      gate.begin() + (row + 1) * kAttnWidth);
    const auto alone = RunDecode(cache, q_row, gate_row, start_pos + row, 1,
                                 start_pos + row + 1, rng);
    if (std::memcmp(alone.data(), &batched[row * kAttnWidth],
                    kAttnWidth * sizeof(float)) != 0)
      Fail("an 8-bit verification row differs from the same position alone");
  }
  std::cout << "8-bit decode attention start " << start_pos << " rows " << rows
            << ": max error " << max_error << " of " << magnitude
            << ", rows identical alone\n";
}

// packed_heads > 0 gives the launcher room to pack that many KV heads into
// FP16 head-major scratch, the route of chunks of 1024 rows or more.
void TestPrefill(std::uint32_t start_pos, std::uint32_t rows,
                 std::uint32_t packed_heads = 0) {
  Rng rng(start_pos * 17U + rows + 3U);
  const auto cache = MakeCache(start_pos + rows + 16, rng);
  std::vector<float> q(rows * kAttnWidth), gate(rows * kAttnWidth);
  for (auto& x : q)
    x = rng.Uniform(-3.0F, 3.0F);
  for (auto& x : gate)
    x = rng.Uniform(-2.0F, 2.0F);
  auto d_k = Upload(cache.k), d_v = Upload(cache.v);
  const auto d_q = Upload(q), d_gate = Upload(gate);
  gufo::test::DeviceBuffer<float> d_out(rows * kAttnWidth);
  // One head's padded FP16 keys or values, in floats.
  const std::size_t head_floats =
      ((start_pos + rows + 15U) & ~15U) * kHeadDim / 2;
  gufo::test::DeviceBuffer<float> d_packed_k(
      std::max<std::size_t>(packed_heads * head_floats, 1)),
      d_packed_v(std::max<std::size_t>(packed_heads * head_floats, 1));
  const std::span<float> packed_k{d_packed_k.data(),
                                  packed_heads * head_floats};
  const std::span<float> packed_v{d_packed_v.data(),
                                  packed_heads * head_floats};
  if (!gufo::hip::LaunchQwenWmmaAttention(
          d_q.data(), nullptr, nullptr, d_gate.data(), nullptr, nullptr,
          d_k.data(), d_v.data(), d_out.data(), 1, start_pos, rows,
          cache.context, kHeads, kKvHeads, kHeadDim, nullptr, nullptr, 0,
          /*skip_kv_write=*/true, packed_k, packed_v, 0, /*kv_q8=*/true))
    Fail("8-bit prefill attention rejected the 35B shape");
  const auto got = Download(d_out, rows * kAttnWidth);
  double max_error = 0.0, magnitude = 0.0;
  Reference(cache, q, gate, start_pos, rows, got, &max_error, &magnitude);
  if (!(max_error <= 4e-3 * std::max(magnitude, 1e-3)))
    Fail("8-bit prefill attention is far from the reference");
  std::cout << "8-bit prefill attention start " << start_pos << " rows " << rows
            << " packed heads " << packed_heads << ": max error " << max_error
            << " of " << magnitude << '\n';
}

}  // namespace
#endif

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen 8-bit KV cache ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  TestWriters();
  // Below the FP16 split-K threshold, one partition, many partitions, and
  // rows on both sides of the 256-to-512 partition step.
  TestDecode(5, 3);
  TestDecode(140, 1);
  TestDecode(150, 4);
  TestDecode(5000, 8);
  TestDecode(8190, 4);
  TestPrefill(0, 33);
  TestPrefill(300, 70);
  TestPrefill(2000, 64);
  // Packed into FP16 scratch: all heads at once, then one head at a time.
  TestPrefill(0, 1024, kKvHeads);
  TestPrefill(1000, 1024, 1);
  std::cout << "Qwen 8-bit KV cache ops test passed on gfx1151.\n";
#endif
  return 0;
}
