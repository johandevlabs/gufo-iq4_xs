// 8-bit attention KV cache (--kv-cache q8_0) for 256-wide KV heads.
//
// Each (position, KV head) row keeps the FP16 cache's token-major order and
// holds 256 int8 values followed by 8 FP16 scales, one per block of 32
// dimensions (the ggml Q8_0 rule: d = amax / 127, q = round(x / d)). A row is
// 272 bytes instead of 512, and stays 16-byte aligned, so a 16-byte load
// reads 16 values of one block. Rows of one position are contiguous like the
// FP16 cache, so prompt-cache rows and session forks copy them unchanged.
#ifndef GUFO_FAMILIES_QWEN_KERNELS_KV_Q8_HPP_
#define GUFO_FAMILIES_QWEN_KERNELS_KV_Q8_HPP_

#include <cstddef>
#include <cstdint>

namespace gufo::hip::kv_q8 {

inline constexpr std::uint32_t kHeadDim = 256;
inline constexpr std::uint32_t kBlock = 32;
inline constexpr std::uint32_t kBlocks = kHeadDim / kBlock;
inline constexpr std::uint32_t kScaleOffset = kHeadDim;
inline constexpr std::uint32_t kRowBytes =
    kHeadDim + (kBlocks * sizeof(std::uint16_t));
static_assert(kRowBytes == 272 && kRowBytes % 16 == 0);

/// Byte offset of one (position, KV head) row inside a K or V plane.
[[nodiscard]] constexpr std::size_t RowOffset(std::uint32_t layer,
                                              std::uint32_t pos,
                                              std::uint32_t kv_head,
                                              std::uint32_t max_context,
                                              std::uint32_t num_kv_heads) {
  return (((static_cast<std::size_t>(layer) * max_context + pos) *
           num_kv_heads) +
          kv_head) *
         kRowBytes;
}

/// Bytes of one layer in a K or V plane.
[[nodiscard]] constexpr std::size_t LayerBytes(std::uint32_t max_context,
                                               std::uint32_t num_kv_heads) {
  return static_cast<std::size_t>(max_context) * num_kv_heads * kRowBytes;
}

}  // namespace gufo::hip::kv_q8

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace gufo::hip::kv_q8 {

/// 16 consecutive values of one block and their scale.
struct Slice16 {
  uint4 q;
  float d;
};

/// Loads dimensions [dim, dim + 16) of a row; dim is a multiple of 16.
__device__ __forceinline__ Slice16 LoadSlice16(const std::uint8_t* row,
                                               std::uint32_t dim) {
  Slice16 slice;
  slice.q = *reinterpret_cast<const uint4*>(row + dim);
  slice.d = __half2float(
      reinterpret_cast<const __half*>(row + kScaleOffset)[dim / kBlock]);
  return slice;
}

/// Expands a slice into 16 FP16 values, as two 16-byte words.
__device__ __forceinline__ void Dequantize16(const Slice16& slice,
                                             uint4 out[2]) {
  union {
    uint4 u[2];
    __half h[16];
  } dst;
  union {
    uint4 u;
    std::int8_t b[16];
  } src;
  src.u = slice.q;
#pragma unroll
  for (std::uint32_t i = 0; i < 16; ++i)
    dst.h[i] = __float2half_rn(static_cast<float>(src.b[i]) * slice.d);
  out[0] = dst.u[0];
  out[1] = dst.u[1];
}

/// Quantizes one block of 32 values held one per lane of a wave32. Every lane
/// of the wave must call it; lane `lane` holds dimension 32 * block + lane.
__device__ __forceinline__ void StoreBlock(float x, std::uint32_t lane,
                                           std::uint8_t* row,
                                           std::uint32_t block) {
  float amax = fabsf(x);
#pragma unroll
  for (std::uint32_t offset = 16; offset > 0; offset /= 2)
    amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
  const __half d = __float2half_rn(amax / 127.0F);
  const float scale = __half2float(d);
  const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
  const int q = min(127, max(-127, __float2int_rn(x * inverse)));
  row[(block * kBlock) + lane] =
      static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
  if (lane == 0)
    reinterpret_cast<__half*>(row + kScaleOffset)[block] = d;
}

}  // namespace gufo::hip::kv_q8
#endif  // defined(__HIPCC__)

#endif  // GUFO_FAMILIES_QWEN_KERNELS_KV_Q8_HPP_
