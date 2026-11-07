#include "src/models/qwen/continuation_identity.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "gufo/ext/continuation.hpp"
#include "gufo/ext/speculative_verifier.hpp"
#include "src/families/qwen/hip_common/attention_policy.hpp"
#include "src/families/qwen/text/chat_template.hpp"

namespace gufo::models::qwen {

bool IsSha256Hex(std::string_view value) noexcept {
  return value.size() == 64 && std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::vector<std::uint8_t> QwenCompatibilityIdentity(
    std::string_view artifact_fingerprint,
    std::string_view draft_artifact_fingerprint, std::uint32_t max_context,
    const hip::QwenExecutionPolicy& execution_policy, bool speculative,
    const speculative::SpeculativeOptions& speculative_options,
    speculative::DFlashDraftPolicy draft_policy) {
  return QwenCompatibilityIdentity(
      artifact_fingerprint, draft_artifact_fingerprint, max_context,
      execution_policy,
      speculative ? QwenDraftKind::kDFlash : QwenDraftKind::kNone,
      speculative_options, draft_policy, false);
}

std::vector<std::uint8_t> QwenCompatibilityIdentity(
    std::string_view artifact_fingerprint,
    std::string_view draft_artifact_fingerprint, std::uint32_t max_context,
    const hip::QwenExecutionPolicy& execution_policy, QwenDraftKind draft_kind,
    const speculative::SpeculativeOptions& speculative_options,
    speculative::DFlashDraftPolicy draft_policy, bool ornith_template) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen disk cache requires an artifact fingerprint");
  }
  if (draft_kind == QwenDraftKind::kDFlash &&
      !IsSha256Hex(draft_artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen DFlash disk cache requires a draft artifact fingerprint");
  }
  if (draft_kind == QwenDraftKind::kMtp &&
      !IsSha256Hex(draft_artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen MTP disk cache requires a draft artifact fingerprint");
  }
  const std::string state_abi =
      QwenStateAbi(draft_kind, execution_policy.kv_cache_storage,
                   execution_policy.UsesBf16RecurrentState());
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=qwen3.8\n"
           << "artifact_id=" << ext::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=embedded-in-artifact\n"
           << "chat_template="
           << (ornith_template ? "ornith-compiled-v1"
                               : "qwen38-reasoning-compiled-v3")
           << '\n'
           << "chat_template_reference_sha256="
           << (ornith_template
                   ? tokenization::QwenChatTemplate::OrnithTemplateSha256()
                   : tokenization::QwenChatTemplate::OfficialTemplateSha256())
           << '\n'
           << "state_abi=" << state_abi << "-component-v2\n"
           << "payload_layout=qwen-component-v2\n"
           << "checkpoint_frontier=complete-prefix-v1\n"
           << "numerics=qwen-bf16-fp32-prefill-v1\n"
           << "rmsnorm=fused-square-sum-v1\n"
           << "prefill_attention=visible-causal-tail-v1\n"
           << "attention_split_min_context="
           << hip::detail::kSplitKDecodeAttentionMinContext << '\n'
           << "attention_split_count="
           << hip::detail::kSplitKDecodeAttentionMaxSplits << '\n'
           << "kv_storage="
           << (execution_policy.UsesFp16AttentionKv() ? "fp16"
               : execution_policy.UsesQ8AttentionKv() ? "q8_0"
                                                      : "fp32")
           << '\n'
           << "recurrent_storage="
           << (execution_policy.UsesBf16RecurrentState() ? "bf16" : "fp32")
           << '\n'
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "rope_window_policy=qwen-gguf-config-v1\n"
           << "adapters=none\n";
  if (draft_kind == QwenDraftKind::kDFlash) {
    identity
        << "draft_backend=dflash2-gfx1151-v1\n"
        << "draft_artifact_id=" << ext::kGgufIdentityScheme << ':'
        << draft_artifact_fingerprint << '\n'
        << "draft_state_layout=dflash-window-kv-and-frontier-v4\n"
        << "draft_policy=" << speculative::DFlashDraftPolicyName(draft_policy)
        << "-v3\n"
        << "draft_max_tokens=" << speculative_options.max_draft_tokens << '\n'
        << "draft_min_tokens=" << speculative_options.min_draft_tokens << '\n'
        << "draft_initial_tokens=" << speculative_options.initial_draft_tokens
        << '\n'
        << "draft_rolling_window=" << speculative_options.rolling_window << '\n'
        << std::setprecision(std::numeric_limits<float>::max_digits10)
        << "draft_target_acceptance="
        << speculative_options.target_acceptance_rate << '\n'
        << "draft_adaptive="
        << (speculative_options.enable_adaptive_draft_length ? "true" : "false")
        << '\n'
        << "draft_batched_verification="
        << (speculative_options.use_batched_verification ? "true" : "false")
        << '\n'
        << "draft_target_verification=decode-equivalent-v1\n";
  } else if (draft_kind == QwenDraftKind::kMtp) {
    // The state layout and proposal strategy are both part of the identity:
    // MTP keeps a full-context cache plus the last target hidden row, and its
    // proposal chain is greedy, so neither may cross-reuse DFlash2 entries.
    identity
        << "draft_backend=qwen-mtp-gfx1151-v1\n"
        << "draft_artifact_id=" << ext::kGgufIdentityScheme << ':'
        << draft_artifact_fingerprint << '\n'
        << "draft_state_layout=qwen-mtp-full-kv-and-hidden-row-v1\n"
        << "draft_propose=mtp-chain-greedy-v1\n"
        << "draft_max_tokens=" << speculative_options.max_draft_tokens << '\n'
        << "draft_min_tokens=" << speculative_options.min_draft_tokens << '\n'
        << "draft_initial_tokens=" << speculative_options.initial_draft_tokens
        << '\n'
        << "draft_rolling_window=" << speculative_options.rolling_window << '\n'
        << std::setprecision(std::numeric_limits<float>::max_digits10)
        << "draft_target_acceptance="
        << speculative_options.target_acceptance_rate << '\n'
        << "draft_adaptive="
        << (speculative_options.enable_adaptive_draft_length ? "true" : "false")
        << '\n'
        << "draft_batched_verification="
        << (speculative_options.use_batched_verification ? "true" : "false")
        << '\n'
        << "draft_target_verification=decode-equivalent-v1\n";
  }
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

std::vector<std::uint8_t> QwenProcessIdentity(const void* model,
                                              const void* draft,
                                              std::uint32_t max_context,
                                              const void* mtp) {
  std::ostringstream identity;
  identity << "schema=gufo-process-qwen-component-v2\n"
           << "model_instance=" << model << '\n'
           << "draft_instance=" << draft << '\n';
  if (mtp != nullptr)
    identity << "mtp_instance=" << mtp << '\n';
  identity << "context_tokens=" << max_context << '\n';
  const auto canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

}  // namespace gufo::models::qwen
