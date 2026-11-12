// Times the routed gate/up vector kernels of decode and verification
// (diagnostic): qfn_mmq_moe_gated_vec_preq over 512 experts of 640 x 2560
// with ten experts per token drawn at random each repetition, so the active
// weights stream from memory as in the model. Prints the median time, the
// streamed bandwidth of the active experts and an FNV-1a hash of the output,
// so a kernel change can be checked for identical results.
//
// moe_vec_bench [reps] [types...]   types: iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace {

void CheckHip(hipError_t error, const char* what) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* state) {
  *state = *state * 1664525U + 1013904223U;
  return *state;
}

struct Format {
  const char* name;
  int type;
  std::size_t block_bytes;
};

constexpr Format kFormats[] = {{"iq2_xxs", 16, 66},
                               {"iq2_xs", 17, 74},
                               {"iq2_s", 22, 82},
                               {"iq3_xxs", 18, 98},
                               {"iq3_s", 21, 110}};

std::uint64_t Hash(const float* device, std::size_t count) {
  std::vector<float> host(count);
  CheckHip(hipMemcpy(host.data(), device, count * sizeof(float),
                     hipMemcpyDeviceToHost),
           "hipMemcpy");
  std::uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < count * sizeof(float); ++i)
    h = (h ^ reinterpret_cast<const std::uint8_t*>(host.data())[i]) *
        1099511628211ULL;
  return h;
}

}  // namespace

int main(int argc, char** argv) try {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 50;
  std::vector<std::string> wanted(argv + std::min(argc, 2), argv + argc);
  constexpr int kExperts = 512, kUsed = 10, kRows = 640, kCols = 2560;
  if (qfn_mmq_init(0) != 0)
    throw std::runtime_error("qfn_mmq_init failed");
  for (const Format& f : kFormats) {
    if (!wanted.empty() &&
        std::find(wanted.begin(), wanted.end(), f.name) == wanted.end())
      continue;
    const std::size_t matrix =
        std::size_t{kExperts} * kRows * (kCols / 256) * f.block_bytes;
    std::vector<std::uint8_t> host(matrix);
    std::uint32_t seed = 7U + static_cast<std::uint32_t>(f.type);
    for (auto& b : host)
      b = static_cast<std::uint8_t>(Next(&seed) >> 24);
    for (std::size_t b = 0; b < host.size(); b += f.block_bytes) {
      const std::uint16_t d = 0x1400;  // F16 2^-10
      std::memcpy(&host[b], &d, 2);
    }
    void* gate = nullptr;
    void* up = nullptr;
    CheckHip(hipMalloc(&gate, matrix), "hipMalloc");
    CheckHip(hipMalloc(&up, matrix), "hipMalloc");
    CheckHip(hipMemcpy(gate, host.data(), matrix, hipMemcpyHostToDevice),
             "upload");
    CheckHip(hipMemcpy(up, host.data(), matrix, hipMemcpyHostToDevice),
             "upload");
    for (const int tokens : {1, 2, 4, 8}) {
      std::vector<float> x(std::size_t{tokens} * kCols);
      for (auto& v : x)
        v = static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
            1000.0F;
      float* dx = nullptr;
      void* xq = nullptr;
      float* out = nullptr;
      std::int32_t* ids = nullptr;
      // Expert sets for one timed batch of launches, each a fresh draw: the
      // host clock times the batch (GPU timestamps are unreliable on some
      // driver states), and no launch reuses cached experts.
      constexpr int kBatch = 20;
      const std::size_t set = std::size_t{tokens} * kUsed;
      CheckHip(hipMalloc(&dx, x.size() * sizeof(float)), "hipMalloc");
      CheckHip(hipMalloc(&xq, qfn_mmq_q8_1_bytes(tokens, kCols)), "hipMalloc");
      CheckHip(
          hipMalloc(&out, std::size_t{tokens} * kUsed * kRows * sizeof(float)),
          "hipMalloc");
      CheckHip(hipMalloc(&ids, set * kBatch * sizeof(int)), "hipMalloc");
      CheckHip(hipMemcpy(dx, x.data(), x.size() * sizeof(float),
                         hipMemcpyHostToDevice),
               "upload");
      if (qfn_mmq_quantize_q8_1(dx, xq, tokens, kCols, nullptr) != 0)
        throw std::runtime_error("quantize failed");
      std::vector<int> pool(kExperts);
      std::vector<std::int32_t> chosen(set * kBatch);
      const auto draw = [&](std::uint32_t pick, std::int32_t* dst) {
        for (int t = 0; t < tokens; ++t) {
          std::iota(pool.begin(), pool.end(), 0);
          for (int i = 0; i < kUsed; ++i) {
            pick = pick * 1664525U + 1013904223U;
            std::swap(pool[i], pool[i + (pick >> 8) % (kExperts - i)]);
            dst[std::size_t{t} * kUsed + i] = pool[i];
          }
        }
      };
      const auto launch = [&](const std::int32_t* expert_ids) {
        if (qfn_mmq_moe_gated_vec_preq(f.type, gate, up, xq, expert_ids, out,
                                       kRows, kCols, tokens, kExperts, kUsed,
                                       nullptr) != 0)
          throw std::runtime_error("gated vector launch failed");
      };
      // The hashed launch uses fixed experts.
      draw(99U, chosen.data());
      CheckHip(hipMemcpy(ids, chosen.data(), set * sizeof(int),
                         hipMemcpyHostToDevice),
               "ids");
      launch(ids);
      CheckHip(hipDeviceSynchronize(), "sync");
      const std::uint64_t hash = Hash(out, std::size_t{tokens} * kUsed * kRows);
      std::vector<double> ms;
      for (int r = -2; r < reps; ++r) {
        for (int b = 0; b < kBatch; ++b)
          draw(Next(&seed), chosen.data() + set * b);
        CheckHip(hipMemcpy(ids, chosen.data(), chosen.size() * sizeof(int),
                           hipMemcpyHostToDevice),
                 "ids");
        CheckHip(hipDeviceSynchronize(), "sync");
        const auto t0 = std::chrono::steady_clock::now();
        for (int b = 0; b < kBatch; ++b)
          launch(ids + set * b);
        CheckHip(hipDeviceSynchronize(), "sync");
        if (r >= 0)
          ms.push_back(std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t0)
                           .count() /
                       kBatch);
      }
      std::sort(ms.begin(), ms.end());
      const double median = ms[ms.size() / 2];
      // Distinct active experts per launch: up to tokens x 10, both matrices.
      const double bytes = 2.0 * std::min(tokens * kUsed, kExperts) * kRows *
                           (kCols / 256) * f.block_bytes;
      std::printf("%-8s tokens %d  %8.1f us  %6.1f GB/s  hash %016llx\n",
                  f.name, tokens, median * 1000.0, bytes / median / 1e6,
                  static_cast<unsigned long long>(hash));
      for (void* p : {static_cast<void*>(dx), xq, static_cast<void*>(out),
                      static_cast<void*>(ids)})
        CheckHip(hipFree(p), "hipFree");
    }
    CheckHip(hipFree(gate), "hipFree");
    CheckHip(hipFree(up), "hipFree");
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "FAILED: %s\n", e.what());
  return 1;
}
