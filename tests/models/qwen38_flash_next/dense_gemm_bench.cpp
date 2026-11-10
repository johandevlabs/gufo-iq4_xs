// Times the dense F16 prefill projections on synthetic weights
// (diagnostic): the Q8_0 2560 x 6144 output projection, the 2560 x 640
// shared-expert down projection, the fused SSM projection (16384 x 2560 with
// its convolution), the stacked attention projection (13312 x 2560 with norms
// and RoPE), the F16 router (513 x 2560) and recurrent gates (96 x 2560),
// and the Dynamic 3.1 Q6_K trunk shapes on the Q6_K F16 route beside Q8_0's
// W8A8 and F16 routes.
//
// dense_gemm_bench [tokens] [reps]
//
// Each line prints the median launch time, TFLOPS and an FNV-1a hash of the
// output, so launch-order changes can be checked for identical results.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, host.size() * sizeof(T)), "hipMalloc");
  CheckHip(
      hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
      "hipMemcpy");
  return static_cast<T*>(p);
}

void* Zeros(std::size_t bytes) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, bytes), "hipMalloc");
  CheckHip(hipMemset(p, 0, bytes), "hipMemset");
  return p;
}

// Q8_0 rows: per 32 codes a F16 scale near 1/256 and random codes.
std::vector<std::uint8_t> Q8Weights(std::size_t rows, std::size_t k,
                                    std::uint32_t seed) {
  const std::size_t blocks = rows * (k / 32);
  std::vector<std::uint8_t> w(blocks * 34);
  for (std::size_t b = 0; b < blocks; ++b) {
    const __half d = __float2half(
        (1.0F + static_cast<float>(Next(&seed) % 64) / 64.0F) / 256.0F);
    std::memcpy(&w[b * 34], &d, 2);
    for (int i = 0; i < 32; ++i)
      w[b * 34 + 2 + i] = static_cast<std::uint8_t>(Next(&seed));
  }
  return w;
}

std::vector<__half> Activations(std::size_t count, std::uint32_t seed) {
  std::vector<__half> x(count);
  for (auto& v : x)
    v = __float2half(
        static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
        1000.0F);
  return x;
}

std::uint64_t Hash(const void* device, std::size_t bytes) {
  std::vector<std::uint8_t> host(bytes);
  CheckHip(hipMemcpy(host.data(), device, bytes, hipMemcpyDeviceToHost),
           "hipMemcpy");
  std::uint64_t h = 1469598103934665603ULL;
  for (std::uint8_t c : host)
    h = (h ^ c) * 1099511628211ULL;
  return h;
}

double MedianMs(const std::function<void()>& launch, int reps) {
  hipEvent_t start;
  hipEvent_t stop;
  CheckHip(hipEventCreate(&start), "hipEventCreate");
  CheckHip(hipEventCreate(&stop), "hipEventCreate");
  for (int i = 0; i < 2; ++i)
    launch();
  std::vector<float> ms(reps);
  for (int i = 0; i < reps; ++i) {
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
    const auto host_start = std::chrono::steady_clock::now();
    CheckHip(hipEventRecord(start, nullptr), "hipEventRecord");
    launch();
    CheckHip(hipEventRecord(stop, nullptr), "hipEventRecord");
    CheckHip(hipEventSynchronize(stop), "hipEventSynchronize");
    CheckHip(hipEventElapsedTime(&ms[i], start, stop), "hipEventElapsedTime");
    // Some driver states report zero for every event pair; time the
    // synchronized launch on the host clock then.
    if (ms[i] <= 0.0F)
      ms[i] = std::chrono::duration<float, std::milli>(
                  std::chrono::steady_clock::now() - host_start)
                  .count();
  }
  std::sort(ms.begin(), ms.end());
  return ms[reps / 2];
}

}  // namespace

int main(int argc, char** argv) try {
  const std::uint32_t tokens = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 10;
  const auto report = [&](const char* name, double ms, double flops,
                          std::uint64_t hash) {
    std::printf("%-5s tokens %u  %8.3f ms  %6.2f TFLOPS  hash %016llx\n", name,
                tokens, ms, flops / ms / 1e9,
                static_cast<unsigned long long>(hash));
  };

  // Output projection: [tokens][6144] -> [tokens][2560].
  {
    constexpr std::size_t m = 2560;
    constexpr std::size_t k = 6144;
    auto* w = Upload(Q8Weights(m, k, 11U));
    auto* x = Upload(Activations(tokens * k, 12U));
    auto* y = static_cast<float*>(Zeros(tokens * m * sizeof(float)));
    const double ms = MedianMs(
        [&] {
          if (!q::DenseF16Gemm(w, x, y, tokens, m, k, nullptr))
            throw std::runtime_error("DenseF16Gemm");
        },
        reps);
    report("out", ms, 2.0 * tokens * m * k,
           Hash(y, tokens * m * sizeof(float)));
    if (tokens >= 1024) {
      constexpr auto stride = k + 32;
      auto* padded =
          static_cast<__half*>(Zeros(tokens * stride * sizeof(__half)));
      CheckHip(
          hipMemcpy2D(padded, stride * sizeof(__half), x, k * sizeof(__half),
                      k * sizeof(__half), tokens, hipMemcpyDeviceToDevice),
          "padded activation");
      const auto expected = Hash(y, tokens * m * sizeof(float));
      const double padded_ms = MedianMs(
          [&] {
            if (!q::DenseF16Gemm(w, padded, y, tokens, m, k, nullptr, stride))
              throw std::runtime_error("strided DenseF16Gemm");
          },
          reps);
      const auto actual = Hash(y, tokens * m * sizeof(float));
      if (actual != expected)
        throw std::runtime_error("strided projection changed output");
      report("out-padded", padded_ms, 2.0 * tokens * m * k, actual);
      CheckHip(hipFree(padded), "hipFree");
    }
    CheckHip(hipFree(w), "hipFree");
    CheckHip(hipFree(x), "hipFree");
    CheckHip(hipFree(y), "hipFree");
  }

  // Shared-expert down projection: [tokens][640] -> [tokens][2560].
  {
    constexpr std::size_t m = 2560;
    constexpr std::size_t k = 640;
    auto* w = Upload(Q8Weights(m, k, 41U));
    auto* x = Upload(Activations(tokens * k, 42U));
    auto* y = static_cast<float*>(Zeros(tokens * m * sizeof(float)));
    const double ms = MedianMs(
        [&] {
          if (!q::DenseF16Gemm(w, x, y, tokens, m, k, nullptr))
            throw std::runtime_error("DenseF16Gemm");
        },
        reps);
    report("down", ms, 2.0 * tokens * m * k,
           Hash(y, tokens * m * sizeof(float)));
    CheckHip(hipFree(w), "hipFree");
    CheckHip(hipFree(x), "hipFree");
    CheckHip(hipFree(y), "hipFree");
  }

  // F16 router and recurrent gates (unquantized weights).
  for (const std::size_t m : {std::size_t{513}, std::size_t{96}}) {
    constexpr std::size_t k = 2560;
    auto* w = Upload(Activations(m * k, 51U));
    auto* x = Upload(Activations(tokens * k, 52U));
    auto* y = static_cast<float*>(Zeros(tokens * m * sizeof(float)));
    const double ms = MedianMs(
        [&] {
          if (!q::UnquantizedF16Gemm(w, x, y, tokens, m, k, nullptr))
            throw std::runtime_error("UnquantizedF16Gemm");
        },
        reps);
    report(m == 513 ? "route" : "gates", ms, 2.0 * tokens * m * k,
           Hash(y, tokens * m * sizeof(float)));
    CheckHip(hipFree(w), "hipFree");
    CheckHip(hipFree(x), "hipFree");
    CheckHip(hipFree(y), "hipFree");
  }

  // HC up projection fused with the stream mixer: [tokens][320] gates over
  // four 2560-wide F16 streams into F16 and tiled Q8 mixer rows.
  {
    constexpr std::uint32_t hidden = 2560;
    constexpr std::uint32_t rank = 320;
    auto* w = Upload(Q8Weights(4 * hidden, rank, 51U));
    auto* low = Upload(Activations(std::size_t{tokens} * rank, 52U));
    auto* xn = Upload(Activations(std::size_t{tokens} * 4 * hidden, 53U));
    auto* mixed = static_cast<__half*>(
        Zeros(std::size_t{tokens} * hidden * sizeof(__half)));
    const std::size_t q8_bytes =
        ((tokens + 15) / 16) * std::size_t{hidden / 32} * 576;
    auto* mixed_q8 = Zeros(q8_bytes);
    const double ms = MedianMs(
        [&] {
          if (!q::HcMixF16Gemm(w, low, xn, nullptr, nullptr, mixed, mixed_q8,
                               nullptr, tokens, hidden, rank, nullptr))
            throw std::runtime_error("HcMixF16Gemm");
        },
        reps);
    report("hcmix", ms, 2.0 * tokens * 4 * hidden * rank,
           Hash(mixed, std::size_t{tokens} * hidden * sizeof(__half)) ^
               Hash(mixed_q8, q8_bytes));
    for (void* p :
         {static_cast<void*>(w), static_cast<void*>(low),
          static_cast<void*>(xn), static_cast<void*>(mixed), mixed_q8})
      CheckHip(hipFree(p), "hipFree");
  }

  // SSM projection fused with the four-tap convolution.
  {
    constexpr std::uint32_t m = 16384;
    constexpr std::uint32_t k = 2560;
    constexpr std::uint32_t channels = 10240;
    auto* w = Upload(Q8Weights(m, k, 21U));
    auto* x = Upload(Activations(std::size_t{tokens} * k, 22U));
    std::vector<float> conv(channels * 4);
    std::uint32_t seed = 23U;
    for (auto& v : conv)
      v = static_cast<float>(Next(&seed) % 1000) / 2000.0F;
    auto* conv_w = Upload(conv);
    auto* history =
        static_cast<float*>(Zeros(std::size_t{3} * channels * sizeof(float)));
    auto* qkvz =
        static_cast<float*>(Zeros(std::size_t{tokens} * m * sizeof(float)));
    auto* convolved = static_cast<float*>(
        Zeros(std::size_t{tokens} * channels * sizeof(float)));
    const double ms = MedianMs(
        [&] {
          if (!q::DenseF16SsmGemm(w, x, conv_w, history, qkvz, convolved,
                                  tokens, m, k, channels, 4, nullptr))
            throw std::runtime_error("DenseF16SsmGemm");
        },
        reps);
    report("ssm", ms, 2.0 * tokens * m * k,
           Hash(convolved, std::size_t{tokens} * channels * sizeof(float)));
    for (void* p : {static_cast<void*>(w), static_cast<void*>(x),
                    static_cast<void*>(conv_w), static_cast<void*>(history),
                    static_cast<void*>(qkvz), static_cast<void*>(convolved)})
      CheckHip(hipFree(p), "hipFree");
  }

  // Stacked attention projection: 24 query, 24 gate, 2 key and 2 value heads.
  {
    constexpr std::uint32_t heads = 24;
    constexpr std::uint32_t kv_heads = 2;
    constexpr std::uint32_t m = 2 * (heads + kv_heads) * 256;
    constexpr std::uint32_t k = 2560;
    auto* w = Upload(Q8Weights(m, k, 31U));
    auto* x = Upload(Activations(std::size_t{tokens} * k, 32U));
    std::vector<float> gamma(256, 1.0F);
    auto* q_gamma = Upload(gamma);
    auto* k_gamma = Upload(gamma);
    auto* position = Upload(std::vector<std::uint32_t>{0U});
    const std::size_t q_bytes =
        std::size_t{tokens} * heads * 256 * sizeof(float);
    const std::size_t kv_bytes =
        std::size_t{tokens} * kv_heads * 256 * sizeof(__half);
    auto* query = static_cast<float*>(Zeros(q_bytes));
    auto* gate = static_cast<float*>(Zeros(q_bytes));
    auto* keys = static_cast<__half*>(Zeros(kv_bytes));
    auto* values = static_cast<__half*>(Zeros(kv_bytes));
    const double ms = MedianMs(
        [&] {
          if (!q::AttentionF16Gemm(w, x, q_gamma, k_gamma, query, gate, keys,
                                   values, tokens, position, 1.0e7F, 1.0e-6F,
                                   nullptr))
            throw std::runtime_error("AttentionF16Gemm");
        },
        reps);
    const std::uint64_t hash = Hash(query, q_bytes) ^ Hash(gate, q_bytes) ^
                               Hash(keys, kv_bytes) ^ Hash(values, kv_bytes);
    report("attn", ms, 2.0 * tokens * m * k, hash);
    for (void* p : {static_cast<void*>(w), static_cast<void*>(x),
                    static_cast<void*>(q_gamma), static_cast<void*>(k_gamma),
                    static_cast<void*>(position), static_cast<void*>(query),
                    static_cast<void*>(gate), static_cast<void*>(keys),
                    static_cast<void*>(values)})
      CheckHip(hipFree(p), "hipFree");
  }

  // BF16 indexer projections (query 512 x 2560, key 128 x 2560, and a TP2
  // rank's halves): the library kernel against BlasLt::Gemm's own kernel,
  // which sums every token in K order. Timed at the requested tokens; the two
  // agree to rounding, and the own kernel's rows must not depend on where a
  // token sits in the batch.
  {
    std::string error;
    auto blas = q::BlasLt::Create(nullptr, &error);
    if (!blas)
      throw std::runtime_error(error);
    constexpr int k = 2560;
    const auto bf16 = [](std::size_t count, std::uint32_t seed, float scale) {
      std::vector<std::uint16_t> v(count);
      for (auto& e : v) {
        const float f =
            scale *
            static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
            1000.0F;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &f, 4);
        e = static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >>
                                       16);
      }
      return v;
    };
    const auto host = [](const float* d, std::size_t count) {
      std::vector<float> h(count);
      CheckHip(hipMemcpy(h.data(), d, count * 4, hipMemcpyDeviceToHost),
               "hipMemcpy");
      return h;
    };
    for (const int m : {512, 128, 256, 64}) {
      auto* w = Upload(bf16(std::size_t{512} * k, 51U + m, 0.05F));
      auto* x = Upload(bf16(std::size_t{tokens} * k, 52U, 3.0F));
      const std::size_t bytes = std::size_t{tokens} * m * sizeof(float);
      auto* y_lib = static_cast<float*>(Zeros(bytes));
      auto* y_own = static_cast<float*>(Zeros(bytes));
      const auto lib = [&](int n) {
        if (!blas->LibraryGemm(w, x, y_lib, HIP_R_16BF, m, n, k, &error))
          throw std::runtime_error(error);
      };
      const auto own = [&](int n) {
        if (!blas->Gemm(w, x, y_own, HIP_R_16BF, m, n, k, &error))
          throw std::runtime_error(error);
      };
      const int t = static_cast<int>(tokens);
      char name[16];
      std::snprintf(name, sizeof(name), "bf%d", m);
      std::printf("%s-lib", name);
      report("", MedianMs([&] { lib(t); }, reps), 2.0 * tokens * m * k,
             Hash(y_lib, bytes));
      std::printf("%s-own", name);
      report("", MedianMs([&] { own(t); }, reps), 2.0 * tokens * m * k,
             Hash(y_own, bytes));
      if (!q::BlasLt::UsesOwnKernel(HIP_R_16BF, m, k))
        throw std::runtime_error(std::string(name) + " runs the library");
      lib(t);
      own(t);
      CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
      const std::size_t count = std::size_t{static_cast<unsigned>(t)} * m;
      const auto library = host(y_lib, count);
      const auto whole = host(y_own, count);
      double worst = 0.0;
      for (std::size_t i = 0; i < count; ++i)
        worst = std::max(worst, std::abs(double{library[i]} - whole[i]));
      // The same rows at other offsets and batch widths.
      int moved = 0;
      for (const int offset : {1, 31, 37, 128, 777}) {
        if (offset >= t)
          continue;
        const int n = t - offset;
        if (!blas->Gemm(
                w, static_cast<std::uint16_t*>(x) + std::size_t(offset) * k,
                y_own, HIP_R_16BF, m, n, k, &error))
          throw std::runtime_error(error);
        CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
        const auto part =
            host(y_own, std::size_t{static_cast<unsigned>(n)} * m);
        if (std::memcmp(part.data(), whole.data() + std::size_t(offset) * m,
                        part.size() * 4) != 0) {
          ++moved;
          std::printf("%s offset %d DIFF\n", name, offset);
        }
      }
      std::printf(
          "%s n %d: max |own - library| %.3g; rows changed with their "
          "batch offset at %d of 5 offsets\n",
          name, t, worst, moved);
      for (void* p : {static_cast<void*>(w), static_cast<void*>(x),
                      static_cast<void*>(y_lib), static_cast<void*>(y_own)})
        CheckHip(hipFree(p), "hipFree");
    }
  }
  // Q6_K trunk (Dynamic 3.1): each shape on the Q6_K F16 route, and the
  // same shape's Q8_0 W8A8 and F16 routes for comparison.
  {
    struct Shape {
      const char* name;
      std::size_t m;
      std::size_t k;
    };
    const Shape shapes[] = {{"qkv", 10240, 2560}, {"gate", 6144, 2560},
                            {"q", 12288, 2560},   {"kv", 512, 2560},
                            {"shexp", 640, 2560}, {"ssmout", 2560, 6144}};
    for (const Shape& sh : shapes) {
      std::vector<std::uint8_t> q6((sh.m * sh.k / 256) * 210);
      std::uint32_t seed = 31U;
      for (std::size_t b = 0; b < q6.size(); b += 210) {
        for (std::size_t i = 0; i < 208; ++i)
          q6[b + i] = static_cast<std::uint8_t>(Next(&seed));
        const __half d = __float2half(1.0F / 4096.0F);
        std::memcpy(&q6[b + 208], &d, 2);
      }
      auto* w6 = Upload(q6);
      auto* w8 = Upload(Q8Weights(sh.m, sh.k, 13U));
      std::vector<float> xf(static_cast<std::size_t>(tokens) * sh.k);
      for (auto& v : xf)
        v = static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
            1000.0F;
      auto* x = Upload(xf);
      auto* xh = Upload(Activations(tokens * sh.k, 12U));
      auto* tiled = Zeros(q::Q8TiledBytes(tokens, sh.k));
      auto* y = static_cast<float*>(Zeros(tokens * sh.m * sizeof(float)));
      const double flops = 2.0 * tokens * sh.m * sh.k;
      q::QuantizeQ8Tiled(x, tiled, tokens, sh.k, nullptr);
      const double quant_ms = MedianMs(
          [&] { q::QuantizeQ8Tiled(x, tiled, tokens, sh.k, nullptr); }, reps);
      std::printf("%-8s quantize %8.3f ms\n", sh.name, quant_ms);

      const double q8_ms = MedianMs(
          [&] {
            if (!q::W8A8Gemm(w8, tiled, y, tokens, sh.m, sh.k, nullptr))
              throw std::runtime_error("W8A8Gemm");
          },
          reps);
      report("q8w8", q8_ms, flops, Hash(y, tokens * sh.m * sizeof(float)));
      const double q6f16_ms = MedianMs(
          [&] {
            if (!q::DenseQ6F16Gemm(w6, xh, y, tokens, sh.m, sh.k, nullptr))
              throw std::runtime_error("DenseQ6F16Gemm");
          },
          reps);
      report("q6f16", q6f16_ms, flops, Hash(y, tokens * sh.m * sizeof(float)));
      if (sh.k <= 2560) {
        const double f16_ms = MedianMs(
            [&] {
              if (!q::DenseF16Gemm(w8, xh, y, tokens, sh.m, sh.k, nullptr))
                throw std::runtime_error("DenseF16Gemm");
            },
            reps);
        report("q8f16", f16_ms, flops, Hash(y, tokens * sh.m * sizeof(float)));
      }
      std::printf("^ %s m=%zu k=%zu\n", sh.name, sh.m, sh.k);
      for (void* p : {static_cast<void*>(w6), static_cast<void*>(w8),
                      static_cast<void*>(x), static_cast<void*>(xh), tiled,
                      static_cast<void*>(y)})
        CheckHip(hipFree(p), "hipFree");
    }
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "dense_gemm_bench: %s\n", e.what());
  return 1;
}
