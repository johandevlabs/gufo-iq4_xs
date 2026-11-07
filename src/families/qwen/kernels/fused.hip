#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/detail/gemv_dispatcher.hpp"
#include "src/families/qwen/kernels/ops.hpp"
#include "src/families/qwen/hip_common/quant_ops.hpp"
#include "src/families/qwen/kernels/kv_q8.hpp"

namespace gufo::hip {
namespace {

// kQ8: k_cache_f16 and v_cache_f16 are 8-bit planes. K is staged in LDS so
// that wave w holds block w, then each wave quantizes its block.
template<bool kQ8>
__launch_bounds__(256, 4) __global__ void FusedQKNormRoPEKvWriteKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ q_weight,
    const float* __restrict__ k_weight, float* __restrict__ q_out,
    float* __restrict__ k_out, float* __restrict__ k_cache,
    float* __restrict__ v_cache, __half* __restrict__ k_cache_f16,
    __half* __restrict__ v_cache_f16, std::uint32_t layer_idx,
    const std::uint32_t* __restrict__ d_pos, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, float eps,
    const models::qwen::vision::DeviceRope* rope) {
  const std::uint32_t block = blockIdx.x;
  const std::size_t tid = threadIdx.x;
  const std::uint32_t half_rot = rotary_dim / 2;
  const std::uint32_t pos = *d_pos;
  __shared__ float s_norm[256];
  __shared__ float s_key[kQ8 ? 256 : 1];

  const std::uint32_t num_q_blocks = num_heads;
  if (block < num_q_blocks) {
    const std::size_t head_base = static_cast<std::size_t>(block) * head_dim;
    const float* x_head = q + head_base;

    float sum_sq = 0.0F;
    for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
      const float val = x_head[i];
      sum_sq += val * val;
    }
    s_norm[tid] = sum_sq;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        s_norm[tid] += s_norm[tid + stride];
      }
      __syncthreads();
    }
    const float inv_rms =
        rsqrtf(s_norm[0] / static_cast<float>(head_dim) + eps);
    __syncthreads();

    for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
      const float w = (q_weight != nullptr) ? q_weight[i] : 1.0F;
      s_norm[i] = (x_head[i] * inv_rms) * w;
    }
    __syncthreads();

    for (std::size_t p = tid; p < half_rot; p += blockDim.x) {
      const float q0 = s_norm[p];
      const float q1 = s_norm[p + half_rot];
      const float exponent =
          static_cast<float>(2 * p) / static_cast<float>(rotary_dim);
      const float inv_freq = 1.0F / powf(rope_theta, exponent);
      const float angle =
          models::qwen::vision::RopePosition(rope, pos, p) * inv_freq;
      const float cos_val = cosf(angle);
      const float sin_val = sinf(angle);
      q_out[head_base + p] = (q0 * cos_val) - (q1 * sin_val);
      q_out[head_base + p + half_rot] = (q0 * sin_val) + (q1 * cos_val);
    }
    for (std::size_t i = rotary_dim + tid; i < head_dim; i += blockDim.x) {
      q_out[head_base + i] = s_norm[i];
    }
    return;
  }

  const std::uint32_t kv_h = block - num_q_blocks;
  if (kv_h >= num_kv_heads) {
    return;
  }

  const std::size_t head_base = static_cast<std::size_t>(kv_h) * head_dim;
  const float* x_head = k + head_base;

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float val = x_head[i];
    sum_sq += val * val;
  }
  s_norm[tid] = sum_sq;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_norm[tid] += s_norm[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(s_norm[0] / static_cast<float>(head_dim) + eps);
  __syncthreads();

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float w = (k_weight != nullptr) ? k_weight[i] : 1.0F;
    s_norm[i] = (x_head[i] * inv_rms) * w;
  }
  __syncthreads();

  const std::size_t kv_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
            max_context +
        pos) *
       head_dim);
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t f16_offset =
      ((static_cast<std::size_t>(layer_idx) * max_context + pos) * kv_width) +
      head_base;

  for (std::size_t p = tid; p < half_rot; p += blockDim.x) {
    const float k0 = s_norm[p];
    const float k1 = s_norm[p + half_rot];
    const float exponent =
        static_cast<float>(2 * p) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0F / powf(rope_theta, exponent);
    const float angle =
        models::qwen::vision::RopePosition(rope, pos, p) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const float k0_rot = (k0 * cos_val) - (k1 * sin_val);
    const float k1_rot = (k0 * sin_val) + (k1 * cos_val);
    k_out[head_base + p] = k0_rot;
    k_out[head_base + p + half_rot] = k1_rot;
    if (k_cache != nullptr) {
      k_cache[kv_offset + p] = k0_rot;
      k_cache[kv_offset + p + half_rot] = k1_rot;
    }
    if constexpr (kQ8) {
      s_key[p] = k0_rot;
      s_key[p + half_rot] = k1_rot;
    } else if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset + p] = __float2half_rn(k0_rot);
      k_cache_f16[f16_offset + p + half_rot] = __float2half_rn(k1_rot);
    }
  }
  for (std::size_t i = rotary_dim + tid; i < head_dim; i += blockDim.x) {
    const float key = s_norm[i];
    k_out[head_base + i] = key;
    if (k_cache != nullptr) {
      k_cache[kv_offset + i] = key;
    }
    if constexpr (kQ8) {
      s_key[i] = key;
    } else if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset + i] = __float2half_rn(key);
    }
  }

  if constexpr (kQ8) {
    __syncthreads();
    const std::size_t row =
        kv_q8::RowOffset(layer_idx, pos, kv_h, max_context, num_kv_heads);
    const auto lane = static_cast<std::uint32_t>(tid % 32);
    const auto block_index = static_cast<std::uint32_t>(tid / 32);
    kv_q8::StoreBlock(s_key[tid], lane,
                      reinterpret_cast<std::uint8_t*>(k_cache_f16) + row,
                      block_index);
    kv_q8::StoreBlock(v[head_base + tid], lane,
                      reinterpret_cast<std::uint8_t*>(v_cache_f16) + row,
                      block_index);
    if (v_cache != nullptr) {
      v_cache[kv_offset + tid] = v[head_base + tid];
    }
    return;
  }

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float val = v[head_base + i];
    if (v_cache != nullptr) {
      v_cache[kv_offset + i] = val;
    }
    if (v_cache_f16 != nullptr) {
      v_cache_f16[f16_offset + i] = __float2half_rn(val);
    }
  }
}

}  // namespace

void LaunchFusedQKNormRoPEKvWrite(
    const float* q, const float* k, const float* v, const float* q_weight,
    const float* k_weight, float* q_out, float* k_out, float* k_cache,
    float* v_cache, void* k_cache_f16, void* v_cache_f16,
    std::uint32_t layer_idx, const std::uint32_t* d_pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, float eps, hipStream_t stream,
    const models::qwen::vision::DeviceRope* rope, bool kv_q8) {
  if (kv_q8) {
    if (head_dim != kv_q8::kHeadDim)
      throw std::invalid_argument("8-bit KV cache needs 256-wide heads");
    FusedQKNormRoPEKvWriteKernel<true>
        <<<num_heads + num_kv_heads, 256, 0, stream>>>(
            q, k, v, q_weight, k_weight, q_out, k_out, k_cache, v_cache,
            static_cast<__half*>(k_cache_f16),
            static_cast<__half*>(v_cache_f16), layer_idx, d_pos, max_context,
            num_heads, num_kv_heads, head_dim, rotary_dim, rope_theta, eps,
            rope);
    return;
  }
  FusedQKNormRoPEKvWriteKernel<false>
      <<<num_heads + num_kv_heads, 256, 0, stream>>>(
      q, k, v, q_weight, k_weight, q_out, k_out, k_cache, v_cache,
      static_cast<__half*>(k_cache_f16), static_cast<__half*>(v_cache_f16),
      layer_idx, d_pos, max_context, num_heads, num_kv_heads, head_dim,
      rotary_dim, rope_theta, eps, rope);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
