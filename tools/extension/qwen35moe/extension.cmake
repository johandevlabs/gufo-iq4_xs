# Spike: Strycast's Qwen3.6-35B-A3B / Ornith (qwen35moe) runner as an external
# Gufo extension. Everything below the public <gufo/ext/...> surface is a copy:
#   q35/family  Qwen family libraries (config, text, vision, kernels, hip_common)
#   q35/model   the Qwen runner
#   q35/fnext   Flash-Next's MMQ and routed-expert kernels

set(Q35 ${GUFO_EXTENSION_BUILD_ROOT}/q35)

gufo_extension_target(
  NAME strycast_q35_family
  SOURCES
    q35/family/config/qwen_config.cpp
    q35/family/text/tokenizer.cpp
    q35/family/text/chat_template.cpp
    q35/family/vision/image_cache.cpp
    q35/family/vision/prompt.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
    # The tokenizer and chat template use ICU, which a module build gets only
    # through this target (Gufo's SDK exports ICU::uc).
    ICU::uc
)

gufo_extension_target(
  NAME strycast_q35_kernels
  SOURCES
    q35/family/kernels/attention_batched.hip
    q35/family/kernels/attention_decode_graph.hip
    q35/family/kernels/attention_decode.hip
    q35/family/kernels/attention_decode_verify_wave64.hip
    q35/family/kernels/attention_tile.hip
    q35/family/kernels/attention_wmma.hip
    q35/family/kernels/attention_wmma_decode.hip
    q35/family/kernels/dflash_kernels.hip
    q35/family/kernels/embed.hip
    q35/family/kernels/fused.hip
    q35/family/kernels/gemv.hip
    q35/family/kernels/gemv_quant.hip
    q35/family/kernels/hipblas_gemm.hip
    q35/family/kernels/hipblaslt_gemm.hip
    q35/family/kernels/moe.hip
    q35/family/kernels/routed_f16.hip
    q35/family/kernels/norm.hip
    q35/family/kernels/prefill_embed.hip
    q35/family/kernels/prefill_fp16.hip
    q35/family/kernels/prefill_gemm.hip
    q35/family/kernels/prefill_norm.hip
    q35/family/kernels/prefill_quant_gemm.hip
    q35/family/kernels/prefill_quant_wave64.hip
    q35/family/kernels/prefill_residual.hip
    q35/family/kernels/prefill_rope.hip
    q35/family/kernels/prefill_ssm.hip
    q35/family/kernels/prefill_swiglu.hip
    q35/family/kernels/prefill_unpack.hip
    q35/family/kernels/qkv.hip
    q35/family/kernels/residual.hip
    q35/family/kernels/rope.hip
    q35/family/kernels/sample.hip
    q35/family/kernels/small_batch_quant16_wave64.hip
    q35/family/kernels/small_batch_wave64.hip
    q35/family/kernels/ssm_decode_recurrence.hip
    q35/family/kernels/ssm.hip
    q35/family/kernels/ssm_recurrence.hip
    q35/family/kernels/ssm_row_split.hip
    q35/family/kernels/swiglu.hip
    q35/family/kernels/unpack.hip
    q35/family/kernels/verify_gemm_27b.hip
    q35/family/vision/encoder.hip
    q35/family/vision/device_input.hip
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
    roc::hipblas
    roc::hipblaslt
    roc::rocblas
)

if(TARGET strycast_q35_kernels)
  set(K ${Q35}/family/kernels)
  set_source_files_properties(
    ${K}/attention_batched.hip ${K}/attention_decode_graph.hip ${K}/attention_decode.hip
    ${K}/attention_decode_verify_wave64.hip ${K}/attention_tile.hip ${K}/attention_wmma.hip
    ${K}/attention_wmma_decode.hip ${K}/dflash_kernels.hip ${K}/embed.hip ${K}/fused.hip
    ${K}/gemv.hip ${K}/gemv_quant.hip ${K}/hipblas_gemm.hip ${K}/hipblaslt_gemm.hip
    ${K}/moe.hip ${K}/routed_f16.hip ${K}/norm.hip ${K}/prefill_embed.hip ${K}/prefill_fp16.hip
    ${K}/prefill_gemm.hip ${K}/prefill_norm.hip ${K}/prefill_quant_gemm.hip
    ${K}/prefill_quant_wave64.hip ${K}/prefill_residual.hip ${K}/prefill_rope.hip
    ${K}/prefill_ssm.hip ${K}/prefill_swiglu.hip ${K}/prefill_unpack.hip ${K}/qkv.hip
    ${K}/residual.hip ${K}/rope.hip ${K}/sample.hip ${K}/small_batch_quant16_wave64.hip
    ${K}/small_batch_wave64.hip ${K}/ssm_decode_recurrence.hip ${K}/ssm.hip
    ${K}/ssm_recurrence.hip ${K}/ssm_row_split.hip ${K}/swiglu.hip ${K}/unpack.hip
    ${K}/verify_gemm_27b.hip ${Q35}/family/vision/encoder.hip ${Q35}/family/vision/device_input.hip
    PROPERTIES LANGUAGE HIP)
  # Per-source options as in Strycast's src/families/qwen/kernels/CMakeLists.txt.
  set_source_files_properties(
    ${K}/small_batch_wave64.hip ${K}/prefill_quant_wave64.hip ${K}/attention_decode_verify_wave64.hip
    PROPERTIES COMPILE_OPTIONS -mwavefrontsize64)
  set_source_files_properties(${K}/small_batch_quant16_wave64.hip PROPERTIES COMPILE_OPTIONS
    "-mwavefrontsize64;-gline-tables-only;-Xarch_device;-mllvm=-amdgpu-sched-strategy=iterative-ilp")
  set_property(SOURCE ${K}/prefill_quant_gemm.hip ${K}/prefill_quant_wave64.hip
    APPEND PROPERTY COMPILE_OPTIONS -fno-slp-vectorize)
  set_property(SOURCE ${K}/routed_f16.hip
    APPEND PROPERTY COMPILE_OPTIONS -O3 -ffast-math -fno-finite-math-only)

  # Flash-Next's MMQ library and routed-expert kernels, as in its extension.cmake.
  set(MMQ ${Q35}/fnext/kernels/rocm/mmq)
  set(MMQ_SOURCES ${MMQ}/qfn_ggml_stubs.hip.cpp ${MMQ}/qfn_mmq.hip.cpp ${MMQ}/mmid.hip.cpp
    ${MMQ}/mmvq.hip.cpp ${MMQ}/moe_wave64.hip.cpp ${MMQ}/quantize.hip.cpp)
  set(FNK_SOURCES ${Q35}/fnext/kernels/rocm/kernels.hip.cpp ${Q35}/fnext/kernels/rocm/w8a8_wave64.hip)
  target_sources(strycast_q35_kernels PRIVATE ${MMQ_SOURCES} ${FNK_SOURCES})
  set_source_files_properties(${MMQ_SOURCES} ${FNK_SOURCES} PROPERTIES LANGUAGE HIP)
  set_property(SOURCE ${MMQ_SOURCES} APPEND PROPERTY COMPILE_DEFINITIONS GGML_HIP_NO_VMM)
  set_property(SOURCE ${MMQ_SOURCES} APPEND PROPERTY COMPILE_OPTIONS
    -O3 -ffast-math -fno-finite-math-only -Wno-unused-value -Wno-unused-command-line-argument)
  set_property(SOURCE ${MMQ}/moe_wave64.hip.cpp ${Q35}/fnext/kernels/rocm/w8a8_wave64.hip
    APPEND PROPERTY COMPILE_OPTIONS -mwavefrontsize64)
  set_property(SOURCE ${FNK_SOURCES} APPEND PROPERTY COMPILE_DEFINITIONS QFN_ROCM_BUILD=1)
  set_property(SOURCE ${FNK_SOURCES} APPEND PROPERTY COMPILE_OPTIONS
    -O3 -ffast-math -fno-finite-math-only -Wno-unused-command-line-argument)
  # The runner includes qfn_mmq.h too.
  target_include_directories(strycast_q35_kernels SYSTEM PUBLIC ${MMQ})
  target_include_directories(strycast_q35_kernels SYSTEM PUBLIC ${ROCWMMA_INCLUDE_DIR})
  target_compile_definitions(strycast_q35_kernels PUBLIC ENGINE_ENABLE_HIP=1)
  target_link_libraries(strycast_q35_kernels PUBLIC strycast_q35_family)
endif()

gufo_extension_target(
  NAME strycast_q35
  SOURCES
    q35/model/continuation_layout.cpp
    q35/model/oracles.cpp
    q35/model/state.cpp
    q35/model/weights.cpp
    q35/model/ssm.cpp
    q35/model/forward.cpp
    q35/model/modules/attention.cpp
    q35/model/modules/embed.cpp
    q35/model/modules/ffn.cpp
    q35/model/modules/moe.cpp
    q35/model/modules/norm.cpp
    q35/model/modules/quant_gemm.cpp
    q35/model/modules/residual.cpp
    q35/model/modules/rope.cpp
    q35/model/modules/sample.cpp
    q35/model/modules/ssm.cpp
    q35/model/modules/unembed.cpp
    q35/model/generator.cpp
    q35/model/mtp_reference.cpp
    q35/model/dflash_weights.cpp
    q35/model/probe.cpp
    q35/model/continuation_identity.cpp
    q35/model/extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
    strycast_q35_family
)

# The model and family sources carry HIP overloads behind ENGINE_ENABLE_HIP,
# which the in-tree targets get from gufo_core. gufo_ext_hip, present only in
# HIP builds, supplies it here, with hipBLAS for the GEMM helpers they include.
# Compile-only: a loadable module resolves its services from the gufo executable
# and must not link the static library.
foreach(target strycast_q35_family strycast_q35)
  if(TARGET ${target} AND TARGET gufo_ext_hip)
    target_link_libraries(${target} PRIVATE $<COMPILE_ONLY:gufo_ext_hip> roc::hipblas)
  endif()
endforeach()

gufo_extension_target(
  NAME strycast_q35_hip
  SOURCES
    q35/model/continuation_adapter.cpp
    q35/model/hip/arena.cpp
    q35/model/hip/batched_decode.cpp
    q35/model/hip/decode.cpp
    q35/model/hip/decode_step.cpp
    q35/model/hip/executor.cpp
    q35/model/hip/model_loader.cpp
    q35/model/hip/prefill.cpp
    q35/model/hip/prefill_chunk.cpp
    q35/model/hip/prompt_hidden_buffer.cpp
    q35/model/hip/qwen_speculative_verifier.cpp
    q35/model/hip/ssm_replay.cpp
    q35/model/hip/mtp/model.cpp
    q35/model/hip/mtp/executor.cpp
    q35/model/hip/mtp/backend.cpp
    q35/model/hip/dflash/model.cpp
    q35/model/hip/dflash/executor.cpp
    q35/model/hip/dflash/batch.cpp
    q35/model/hip/dflash/backend.cpp
    q35/model/runner.cpp
    q35/model/drivers.cpp
    q35/model/provider.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_cache
    gufo_ext_hip
    strycast_q35
    strycast_q35_kernels
    roc::hipblas
)

gufo_extension_hip_sources(strycast_q35_hip
  SOURCES
    q35/model/hip/attention.hip
    q35/model/hip/batched_ssm.hip
)

install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/strycast_qwen35moe)
