#include "src/models/qwen/runner.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/component_cache_resources.hpp"
#include "gufo/ext/experimental/admission_query.hpp"
#include "gufo/ext/gguf_identity.hpp"
#include "gufo/ext/host_budget.hpp"
#include "gufo/ext/speculative_verifier.hpp"
#include "src/families/qwen/config/qwen_config.hpp"
#include "src/families/qwen/control_tokens/control_tokens.hpp"
#include "src/families/qwen/hip_common/attention_policy.hpp"
#include "src/families/qwen/hip_common/execution_policy.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/families/qwen/vision/encoder.hpp"
#include "src/families/qwen/vision/image_cache.hpp"
#include "src/families/qwen/vision/prompt.hpp"
#include "src/models/qwen/continuation_adapter.hpp"
#include "src/models/qwen/continuation_identity.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/hip/qwen_speculative_verifier.hpp"
#include "src/models/qwen/probe.hpp"

namespace gufo::models::qwen {
namespace {

using namespace gufo::ext;

constexpr std::size_t kQwenMaxResumeTokens = 9;
// A step that keeps more rows than an ordinary verification (a wide copy)
// leaves the draft caught up and cannot rewind (FinishSpeculativeDecode).
static_assert(kQwenMaxResumeTokens ==
              hip::QwenDFlashGpuDraftBackend::kMaxPendingRows + 1);

[[nodiscard]] inline hip::ComponentCacheHooks HipComponentCacheHooks() {
  return {[] { return ext::HostSnapshotCeilingBytes(); },
          [](std::string_view) {}, [](std::string_view) {}};
}

template<typename State, typename Tokens>
bool EndsTurn(const State& state, const Tokens& tokens, int token) {
  return state.stop_at_eos() && tokens.IsStopToken(token);
}

/// Draft ceiling for a speculative step shared by `active` sessions. On an MoE
/// target each stacked verification row reads mostly different experts, so
/// batched verification costs grow with the rows and concurrent sessions
/// shorten their drafts. Measured on Qwen3.6-35B-A3B DFlash2 (gfx1151) under
/// sustained load, total tok/s for ceilings 1/2/3/4 at temperature 0.6: two
/// sessions 87.9/95.8/94.8/90.2, three 105.7/103.0/98.9/98.2, four
/// 117.1/113.4/109.6/110.1, against 79.0/95.7/106.0 without drafting. Dense
/// targets share their weights across the rows and keep the full draft: the
/// cap cost Qwen3.8-27B 6% at two sessions and 26% at four. MTP keeps one draft
/// token at two sessions and stops drafting at three: on the 35B, two greedy
/// sessions gave 86.9 total tok/s with one MTP draft token against 85.6 with
/// two, and one token gave 99.4/107.2 at three/four sessions against
/// 99.1/111.0 with none. Seeded sampled requests take the ceiling for the
/// configured sessions instead, see QwenTextRunner::DraftCap.
std::uint32_t ConcurrentDraftCap(std::size_t active, bool moe_target,
                                 bool mtp_draft) {
  if (active <= 1 || !moe_target)
    return std::numeric_limits<std::uint32_t>::max();
  if (active == 2)
    return mtp_draft ? 1 : 2;
  return mtp_draft ? 0 : 1;
}

struct QwenImageContext final : TextPromptContext {
  std::shared_ptr<const models::qwen::vision::Prompt> prompt;
};

tokenization::ChatTemplateOptions QwenChatOptions(
    const ChatRequest& request, std::uint32_t max_context,
    const tokenization::QwenTokenizer& tokenizer) {
  auto options = tokenizer.UsesOrnithChatTemplate()
                     ? tokenization::ResolveOrnithChatOptions(
                           request.reasoning, request.add_vision_id)
                     : tokenization::ResolveQwenChatOptions(
                           request.reasoning, request.add_vision_id);
  // gufo #285: the rendered prompt may be as long as the context can hold.
  options.max_output_bytes =
      tokenization::RenderedPromptBoundBytes(max_context);
  return options;
}

// The text before tokenization. An image renders as its one placeholder, not
// the visual tokens Prepare expands it into. Template errors are request
// errors, as in Prepare.
std::optional<std::string> RenderQwenPrompt(
    const ChatRequest& request, std::uint32_t max_context,
    const tokenization::QwenTokenizer& tokenizer) {
  std::string error;
  auto rendered = tokenization::QwenChatTemplate::Render(
      request.messages,
      request.tool_choice == ChatRequest::ToolChoice::kNone
          ? std::span<const tokenization::ChatTool>{}
          : std::span<const tokenization::ChatTool>{request.tools},
      QwenChatOptions(request, max_context, tokenizer), &error);
  if (!rendered)
    throw std::invalid_argument(error);
  return rendered;
}

std::optional<std::size_t> CountQwenPromptTokens(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache) {
  const auto options = QwenChatOptions(request, max_context, tokenizer);
  const auto tools =
      request.tool_choice == ChatRequest::ToolChoice::kNone
          ? std::span<const tokenization::ChatTool>{}
          : std::span<const tokenization::ChatTool>{request.tools};
  if (std::ranges::none_of(request.messages,
                           [](const auto& m) { return !m.images.empty(); })) {
    const auto tokens = tokenization::QwenChatTemplate::RenderAndTokenize(
        tokenizer, request.messages, tools, options);
    if (!tokens)
      return std::nullopt;
    return tokens->size();
  }
  return models::qwen::vision::Prepare(
             tokenizer, request.messages, tools, options,
             encoder ? encoder->identity() : std::string_view{},
             std::numeric_limits<std::int32_t>::max(),
             request.cache_prompt ? &image_cache : nullptr)
      .tokens.size();
}

TextPreparedPrompt PrepareQwenPrompt(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache) {
  const bool has_images = std::ranges::any_of(
      request.messages, [](const auto& m) { return !m.images.empty(); });
  const auto options = QwenChatOptions(request, max_context, tokenizer);
  auto prompt = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          tokenizer, request.messages,
          request.tool_choice == ChatRequest::ToolChoice::kNone
              ? std::span<const tokenization::ChatTool>{}
              : std::span<const tokenization::ChatTool>{request.tools},
          options,
          encoder && has_images ? encoder->identity() : std::string_view{},
          max_context, request.cache_prompt ? &image_cache : nullptr));
  const auto cache_prefix = prompt->stable_prefix_tokens;
  if (prompt->images.empty())
    return {std::move(prompt->tokens), {}, cache_prefix};
  if (!encoder)
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  auto context = std::make_shared<QwenImageContext>();
  context->cache_identity = prompt->cache_identity;
  for (const auto& image : prompt->images) {
    const auto identity = prompt->IdentityForPrefix(image.grid.offset);
    context->cache_prefixes.push_back(
        {image.grid.offset, {identity.begin(), identity.end()}});
  }
  context->prompt = prompt;
  return {prompt->tokens, std::move(context), cache_prefix};
}

std::shared_ptr<const models::qwen::vision::Prompt> QwenPrompt(
    const std::shared_ptr<const TextPromptContext>& context) {
  if (!context)
    return {};
  const auto* image = dynamic_cast<const QwenImageContext*>(context.get());
  if (!image)
    throw std::invalid_argument("invalid Qwen prompt context");
  return image->prompt;
}

inline std::string_view ChatRoleName(tokenization::ChatRole role) {
  switch (role) {
    case tokenization::ChatRole::kSystem:
      return "system";
    case tokenization::ChatRole::kDeveloper:
      return "developer";
    case tokenization::ChatRole::kAssistant:
      return "assistant";
    case tokenization::ChatRole::kTool:
      return "tool";
    case tokenization::ChatRole::kUser:
      return "user";
  }
  return "user";
}

// === STATE ===
class QwenTextRunnerState final : public TextRunnerState {
public:
  QwenTextRunnerState(std::shared_ptr<const hip::QwenGpuModel> model,
                      models::qwen::ContinuationAdapter& adapter,
                      cache::Slot& slot,
                      speculative::SpeculativeOptions options)
      : model_(std::move(model)),
        executor_(&adapter.GetExecutor(slot)),
        dflash_backend_(adapter.GetDraftBackend(slot)),
        mtp_backend_(adapter.GetMtpBackend(slot)),
        adapter_(&adapter),
        slot_(&slot),
        frontier_(adapter.GetFrontier(slot).token),
        frontier_logits_(adapter.GetFrontier(slot).logits),
        frontier_published_(adapter.GetFrontier(slot).published),
        resume_tokens_(adapter.GetFrontier(slot).pending) {
    resume_tokens_.reserve(kQwenMaxResumeTokens);
    rollback_tokens_.reserve(kQwenMaxResumeTokens);
    speculative::IDraftBackend* draft = dflash_backend_;
    if (draft == nullptr)
      draft = mtp_backend_;
    if (draft) {
      verifier_ =
          hip::CreateQwenGpuSpeculativeVerifier(*executor_, *draft, options);
      executor_->SetPromptHiddenCapture(true, draft->TargetHiddenLayerIds());
    }
  }

  void SetCancellationCheck(const CancellationCheck& check) override {
    executor_->SetCancellationCheck(check);
  }
  ~QwenTextRunnerState() override { executor_->SetCancellationCheck({}); }

  class PrefillScope {
  public:
    explicit PrefillScope(QwenTextRunnerState& state)
        : state_(state), pending_size_(state.resume_tokens_.size()) {
      std::ranges::copy(state.resume_tokens_, pending_.begin());
      state_.adapter_->BeginPrefill(*state_.slot_);
    }
    ~PrefillScope() { state_.adapter_->FinishPrefill(*state_.slot_); }
    void Rollback() {
      state_.adapter_->RollbackPrefill(*state_.slot_);
      state_.resume_tokens_.assign(pending_.begin(),
                                   pending_.begin() + pending_size_);
    }
    PrefillScope(const PrefillScope&) = delete;
    PrefillScope& operator=(const PrefillScope&) = delete;

  private:
    QwenTextRunnerState& state_;
    std::array<TextRunnerToken, kQwenMaxResumeTokens> pending_{};
    std::size_t pending_size_;
  };
  [[nodiscard]] PrefillScope BeginPrefill() { return PrefillScope(*this); }

  void Invalidate() noexcept override {
    (void)adapter_->Invalidate(*slot_);
    sequence_.clear();
    position_ = 0;
    frontier_.reset();
    frontier_logits_.clear();
    frontier_published_ = false;
    resume_tokens_.clear();
    rollback_position_.reset();
    rollback_tokens_.clear();
    rollback_logits_.clear();
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    try {
      auto usage = executor_->GetMemoryUsage();
      if (dflash_backend_ != nullptr) {
        const auto draft = dflash_backend_->GetMemoryUsage();
        usage.request_state_bytes += draft.request_state_bytes;
        usage.temporary_scratch_bytes += draft.temporary_scratch_bytes;
      } else if (mtp_backend_ != nullptr) {
        const auto draft = mtp_backend_->GetMemoryUsage();
        usage.request_state_bytes += draft.request_state_bytes;
        usage.temporary_scratch_bytes += draft.temporary_scratch_bytes;
      }
      return {
          .per_request_state_bytes = usage.request_state_bytes,
          .temporary_scratch_bytes = usage.temporary_scratch_bytes,
      };
    } catch (...) {
      return {};
    }
  }

  [[nodiscard]] bool HasDecodeFrontier() const {
    return adapter_->HasDecodeFrontier(*slot_);
  }

  [[nodiscard]] bool speculative() const noexcept {
    return verifier_ != nullptr;
  }

  /// First prompt position whose features the draft still needs before
  /// `horizon`, the next position the state is used at; zero keeps them all.
  [[nodiscard]] std::uint32_t KeepDraftContextFrom(
      std::size_t horizon) const noexcept {
    const std::size_t window =
        dflash_backend_ != nullptr ? dflash_backend_->TargetContextWindow() : 0;
    return window != 0 && horizon > window
               ? static_cast<std::uint32_t>(horizon - window)
               : 0U;
  }

  void set_draft_context_horizon(std::size_t position) noexcept {
    draft_context_horizon_ = position;
  }
  [[nodiscard]] std::size_t TakeDraftContextHorizon() noexcept {
    return std::exchange(draft_context_horizon_, 0);
  }

  void PrimeSpeculative(std::span<const TextRunnerToken> prompt,
                        bool decode_ready, std::uint32_t keep_context_from) {
    if (verifier_ == nullptr) {
      throw std::logic_error("Qwen state has no speculative verifier");
    }
    frontier_ = verifier_->Prime(prompt, decode_ready, keep_context_from);
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = verifier_->CopyLastTargetLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    sequence_.assign(prompt.begin(), prompt.end());
    position_ = prompt.size();
    frontier_published_ = false;
    resume_tokens_.clear();
    rollback_position_.reset();
  }

  void PreparePrefixReuse(std::span<const TextRunnerToken> prefix) {
    if (position_ != prefix.size() || !frontier_.has_value()) {
      throw std::logic_error(
          "Qwen retained prefix does not match its target state");
    }
    if (verifier_ != nullptr) {
      verifier_->BeginRequest();
      sequence_.assign(prefix.begin(), prefix.end());
      // Verification's frontier may be a sampled draw. A new request must
      // select from its distribution again, including when it switches to
      // greedy sampling or omits the previously published pending token.
      if (!resume_tokens_.empty() && !frontier_logits_.empty()) {
        sampling::SamplerState greedy;
        frontier_ = greedy.Sample(frontier_logits_);
      }
    }
    frontier_published_ = false;
    rollback_position_.reset();
  }

  bool ResumeGeneratedToken(TextRunnerToken token, bool decode_ready) {
    if (resume_tokens_.empty())
      return false;
    if (resume_tokens_.front() != token) {
      resume_tokens_.clear();
      return false;
    }
    resume_tokens_.erase(resume_tokens_.begin());
    // This token was already published by the previous request. Consume it
    // with decode arithmetic, exactly as AR did, before chunking the new
    // prompt suffix. Including it in a prefill chunk changes recurrent state.
    const auto position = static_cast<std::uint32_t>(position_);
    frontier_ = verifier_ != nullptr
                    ? verifier_->AdvanceCommittedToken(token, position)
                    : executor_->ForwardToken(token, position);
    if (verifier_ != nullptr)
      sequence_.push_back(token);
    ++position_;
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = executor_->CopyLastLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    return true;
  }

  void ExtendSpeculative(std::span<const TextRunnerToken> suffix,
                         bool decode_ready, std::uint32_t keep_context_from) {
    if (verifier_ == nullptr || !frontier_.has_value()) {
      throw std::logic_error("Qwen speculative prefix is not initialized");
    }
    if (sequence_.size() != position_) {
      throw std::logic_error(
          "Qwen speculative sequence does not match retained position");
    }
    frontier_ =
        verifier_->ExtendPrompt(suffix, static_cast<std::uint32_t>(position_),
                                decode_ready, keep_context_from);
    sequence_.insert(sequence_.end(), suffix.begin(), suffix.end());
    position_ += suffix.size();
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = verifier_->CopyLastTargetLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    frontier_published_ = false;
  }

  [[nodiscard]] TextRunnerToken SelectFrontier(
      sampling::SamplerState& sampler) const {
    if (!frontier_.has_value()) {
      throw std::logic_error("Qwen state has no next-token frontier");
    }
    if (sampler.config().can_use_unmodified_argmax() ||
        (verifier_ == nullptr && sampler.CanSelectArgmax(*frontier_))) {
      return *frontier_;
    }
    if (frontier_logits_.empty()) {
      return executor_->SampleLastLogits(sampler);
    }
    return executor_->SampleCachedLogits(frontier_logits_, sampler);
  }

  [[nodiscard]] bool PrepareSpeculativeDecode(std::size_t max_tokens,
                                              sampling::SamplerState& sampler,
                                              TextDecodeStep& result) {
    if (verifier_ == nullptr || !frontier_.has_value()) {
      throw std::logic_error("Qwen speculative state has no frontier");
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Qwen speculative decode budget must be at least one token");
    }
    resume_tokens_.clear();
    if (!frontier_published_) {
      if (!sampler.config().can_use_unmodified_argmax())
        frontier_ = SelectFrontier(sampler);
      const auto logprobs =
          sampler.config().logprobs
              ? sampling::CollectLogprobs(CopyFrontierLogits(), *frontier_,
                                          sampler.config().logprobs)
              : std::nullopt;
      if (!AppendSpeculativeSelection(*frontier_, sampler, result, logprobs))
        return false;
      frontier_published_ = true;
    }
    return result.selections.size() < max_tokens;
  }

  [[nodiscard]] speculative::SpeculativeVerifier::StepRequest
  VerificationRequest(std::size_t remaining, sampling::SamplerState& sampler) {
    rollback_position_ = position_;
    rollback_tokens_.assign(1, *frontier_);
    rollback_logits_ = std::move(frontier_logits_);
    // Verification keeps its own EOS handling: the tokenizer stop set already
    // covers EOS, so a sentinel id here would change nothing. An ignore-EOS
    // request instead publishes the stop token in AppendSpeculativeSelection
    // and continues, at the cost of a window clipped on every EOS crossing.
    return {*verifier_,
            sequence_,
            static_cast<std::uint32_t>(position_),
            *frontier_,
            model_->GetTokenizer().GetEosTokenId(),
            static_cast<std::uint32_t>(std::min<std::size_t>(
                remaining, std::numeric_limits<std::uint32_t>::max())),
            sampler};
  }

  void FinishSpeculativeDecode(
      speculative::SpeculativeVerifier::StepResult verification,
      sampling::SamplerState& sampler, TextDecodeStep& result) {
    result.draft_rounds = verification.draft_count > 0 ? 1 : 0;
    result.draft_tokens = verification.draft_count;
    result.draft_accepted_tokens = verification.accepted_count;
    result.lookup_tokens = verification.lookup_tokens;
    result.lookup_accepted_tokens = verification.lookup_accepted_tokens;
    result.execution_plan = {
        .kind = verification.physical_width > 1
                    ? TextExecutionPlanKind::kBatched
                    : TextExecutionPlanKind::kSerial,
        .physical_width = verification.physical_width,
    };
    std::size_t index = 0;
    for (const TextRunnerToken token : verification.emitted_tokens) {
      auto logprobs = verification.logprobs.empty()
                          ? std::nullopt
                          : std::optional(verification.logprobs.at(index++));
      // Each prediction consumes one input, including a terminal prediction.
      // EOS itself remains the unconsumed frontier and is not published.
      ++position_;
      if (!AppendSpeculativeSelection(token, sampler, result,
                                      std::move(logprobs)))
        break;
      rollback_tokens_.push_back(token);
    }
    frontier_ = verification.next_token;
    frontier_logits_ = std::move(verification.next_token_logits);
    frontier_published_ = !result.stop;
    set_pending_token(frontier_published_ ? frontier_ : std::nullopt);
    // A wide copy that kept more rows than the draft may leave pending has
    // already injected them (CatchUpPendingFeatures): its state is final, like
    // a target-only step's.
    if (verification.draft_count == 0 ||
        verification.emitted_tokens.size() >
            hip::QwenDFlashGpuDraftBackend::kMaxPendingRows)
      rollback_position_.reset();
  }

  void PrepareCancellation() {
    if (!rollback_position_)
      return;
    if (rollback_tokens_.empty() ||
        rollback_tokens_.size() > kQwenMaxResumeTokens ||
        rollback_logits_.empty())
      throw std::logic_error(
          "Qwen cancelled verification has no exact frontier");
    // Verification already saved the target state. Its target context is still
    // pending in the draft, so rewinding needs no new per-step copy.
    if (dflash_backend_ != nullptr) {
      dflash_backend_->DiscardPendingTargetContext(
          static_cast<std::uint32_t>(*rollback_position_));
    } else if (mtp_backend_ != nullptr) {
      mtp_backend_->DiscardPendingTargetContext(
          static_cast<std::uint32_t>(*rollback_position_));
    }
    executor_->RestoreState();
    executor_->FinishVerification();
    position_ = *rollback_position_;
    sequence_.resize(position_);
    frontier_logits_ = std::move(rollback_logits_);
    sampling::SamplerState greedy;
    frontier_ = greedy.Sample(frontier_logits_);
    frontier_published_ = false;
    resume_tokens_ = std::move(rollback_tokens_);
    rollback_position_.reset();
  }

  [[nodiscard]] TextDecodeStep DecodeSpeculative(
      std::size_t max_tokens, sampling::SamplerState& sampler,
      std::uint32_t max_draft_tokens) {
    TextDecodeStep result;
    sampling::SamplerState working_sampler = sampler;
    if (PrepareSpeculativeDecode(max_tokens, working_sampler, result)) {
      const auto request = VerificationRequest(
          max_tokens - result.selections.size(), working_sampler);
      auto verification = verifier_->VerifyStep(
          sequence_, request.position, request.current_token, request.eos_id,
          request.max_emitted_tokens, working_sampler, max_draft_tokens);
      FinishSpeculativeDecode(std::move(verification), working_sampler, result);
    }
    sampler.SetRngState(working_sampler.rng_state());
    return result;
  }

  [[nodiscard]] hip::QwenGpuExecutor& executor() const { return *executor_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }
  [[nodiscard]] const std::optional<TextRunnerToken>& frontier()
      const noexcept {
    return frontier_;
  }
  void set_frontier(TextRunnerToken frontier) noexcept {
    frontier_ = frontier;
    frontier_logits_.clear();
    frontier_published_ = false;
    resume_tokens_.clear();
  }
  [[nodiscard]] std::vector<float> CopyFrontierLogits() const {
    if (!frontier_logits_.empty()) {
      return frontier_logits_;
    }
    const auto logits = executor_->CopyLastLogits();
    return {logits.begin(), logits.end()};
  }
  void RestoreFrontierLogits(std::vector<float> logits) {
    frontier_logits_ = std::move(logits);
  }
  [[nodiscard]] const std::vector<TextRunnerToken>& pending_tokens()
      const noexcept {
    return resume_tokens_;
  }
  void set_pending_token(std::optional<TextRunnerToken> token) {
    resume_tokens_.clear();
    if (token)
      resume_tokens_.push_back(*token);
  }
  void RestorePendingTokens(std::span<const TextRunnerToken> tokens) {
    resume_tokens_.assign(tokens.begin(), tokens.end());
    rollback_position_.reset();
  }

private:
  bool AppendSpeculativeSelection(
      TextRunnerToken token, sampling::SamplerState& sampler,
      TextDecodeStep& result,
      std::optional<sampling::TokenLogprobs> logprobs = {}) {
    if (stop_at_eos() && model_->GetTokenizer().IsStopToken(token)) {
      result.terminal_logprobs = std::move(logprobs);
      result.stop = true;
      return false;
    }
    result.selections.push_back({
        .stop = false,
        .token = token,
        .piece = std::string(model_->GetTokenizer().DecodeToken(token)),
        .logprobs = std::move(logprobs),
    });
    sequence_.push_back(token);
    set_pending_token(token);
    sampler.Accept(token);
    return true;
  }

  std::shared_ptr<const hip::QwenGpuModel> model_;
  hip::QwenGpuExecutor* executor_{};
  std::unique_ptr<speculative::SpeculativeVerifier> verifier_;
  // Owned by the component slot; used to account the session's draft
  // allocations and to roll the draft back when a verification is cancelled.
  // At most one is set.
  hip::QwenDFlashGpuDraftBackend* dflash_backend_{nullptr};
  hip::QwenMtpGpuDraftBackend* mtp_backend_{nullptr};
  models::qwen::ContinuationAdapter* adapter_{};
  cache::Slot* slot_{};
  std::size_t draft_context_horizon_{0};
  std::vector<TextRunnerToken> sequence_;
  std::size_t position_{0};
  std::optional<TextRunnerToken>& frontier_;
  std::vector<float>& frontier_logits_;
  bool& frontier_published_;
  std::vector<TextRunnerToken>& resume_tokens_;
  std::optional<std::size_t> rollback_position_;
  std::vector<TextRunnerToken> rollback_tokens_;
  std::vector<float> rollback_logits_;
};

QwenTextRunnerState& RequireQwenState(TextRunnerState& state) {
  auto* qwen = dynamic_cast<QwenTextRunnerState*>(&state);
  if (qwen == nullptr) {
    throw std::logic_error("text runner state is not Qwen");
  }
  return *qwen;
}

const QwenTextRunnerState& RequireQwenState(const TextRunnerState& state) {
  const auto* qwen = dynamic_cast<const QwenTextRunnerState*>(&state);
  if (qwen == nullptr) {
    throw std::logic_error("text runner state is not Qwen");
  }
  return *qwen;
}

// Arms the session's layer hook from the host's yield for one prefill step.
class PrefillYieldScope {
public:
  PrefillYieldScope(hip::QwenGpuExecutor& executor,
                    const TextRunnerState& state)
      : executor_(executor) {
    if (state.prefill_yield_groups() > 1)
      executor_.SetPrefillYield(state.prefill_yield_groups(),
                                [&state] { state.YieldPrefill(); });
  }
  ~PrefillYieldScope() { executor_.SetPrefillYield(1, {}); }
  PrefillYieldScope(const PrefillYieldScope&) = delete;
  PrefillYieldScope& operator=(const PrefillYieldScope&) = delete;

private:
  hip::QwenGpuExecutor& executor_;
};

// Beside decoders, prompts keep the 4,096-row pass and split it into layer
// groups: one pass takes about 6.5 s, a decode step about 82 ms.
constexpr std::size_t kMaxPrefillLayerGroups = 16;

// Shared runner base for the HIP models. Device-loss probing is the runtime's
// `ext::DeviceProbe` (RFC step 6); the probe holds its buffer and private
// stream from construction, with the 5 s bound the scheduler expects.

// === RUNNER ===
class QwenTextRunner final : public HipTextModelRunner {
public:
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }
  QwenTextRunner(
      std::shared_ptr<const hip::QwenGpuModel> model, std::uint32_t max_context,
      std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model = nullptr,
      speculative::SpeculativeOptions speculative_options = {},
      speculative::DFlashDraftPolicy dflash_policy =
          speculative::DFlashDraftPolicy::kAdaptive,
      std::string artifact_fingerprint = {},
      std::string draft_artifact_fingerprint = {},
      std::shared_ptr<const hip::QwenMtpGpuModel> mtp_model = nullptr,
      std::size_t session_count = 1,
      hip::QwenKvCacheStorage kv_storage = hip::QwenKvCacheStorage::kFp16)
      : model_(std::move(model)),
        dflash_model_(std::move(dflash_model)),
        mtp_model_(std::move(mtp_model)),
        max_context_(max_context),
        speculative_options_(speculative_options),
        dflash_policy_(dflash_policy),
        session_count_(session_count),
        execution_policy_([kv_storage] {
          auto policy = hip::QwenExecutionPolicy::Production();
          policy.kv_cache_storage = kv_storage;
          return policy;
        }()) {
    if (!artifact_fingerprint.empty()) {
      component_identity_ = QwenCompatibilityIdentity(
          artifact_fingerprint, draft_artifact_fingerprint, max_context_,
          execution_policy_, DraftKind(), speculative_options_, dflash_policy_,
          model_->GetTokenizer().UsesOrnithChatTemplate());
    } else {
      component_identity_ = QwenProcessIdentity(
          model_.get(), dflash_model_.get(), max_context_, mtp_model_.get());
    }
  }

  [[nodiscard]] std::unique_ptr<ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t ram) const override {
    (void)ram;
    auto resources = hip::CreateHipComponentCacheResources(
        std::make_unique<models::qwen::ContinuationAdapter>(
            model_, max_context_, component_identity_, execution_policy_,
            dflash_model_,
            hip::QwenDFlashGpuDraftConfig{
                .max_context = max_context_,
                .max_draft_tokens = speculative_options_.max_draft_tokens,
                .policy = dflash_policy_,
                .max_copy_rows = copy_rows_},
            &ledger, mtp_model_,
            hip::QwenMtpGpuDraftConfig{
                .max_context = max_context_,
                .max_draft_tokens = speculative_options_.max_draft_tokens,
                .prime_chunk = speculative_options_.mtp_prime_chunk}),
        ledger, HipComponentCacheHooks());
    // Qwen's 256-row chunks are small. Preserve a bounded group before
    // waiting, instead of synchronizing every half-MiB range individually.
    resources->reassignment_batch_size = 16;
    const auto& adapter =
        static_cast<models::qwen::ContinuationAdapter&>(*resources->adapter);
    resources->pinned_bytes = [qwen = &adapter] {
      return qwen->PinnedCaptureBytes();
    };
    return resources;
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    // The verifier admits a wide lookup copy; the draft keeps neural blocks
    // and batched copies at --draft-tokens (QwenDFlashGpuDraftConfig).
    auto verifier_options = speculative_options_;
    if (dflash_model_ != nullptr &&
        copy_rows_ > verifier_options.max_draft_tokens + 1) {
      verifier_options.max_draft_tokens = copy_rows_ - 1;
      verifier_options.initial_draft_tokens = copy_rows_ - 1;
    }
    return std::make_unique<QwenTextRunnerState>(
        model_, dynamic_cast<models::qwen::ContinuationAdapter&>(adapter), slot,
        verifier_options);
  }
  /// Caps wide copy verification; call before any state is bound.
  void LimitCopyRows(std::uint32_t rows) noexcept { copy_rows_ = rows; }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    auto& qwen = RequireQwenState(state);
    qwen.set_position(position);
    if (!qwen.frontier())
      qwen.set_frontier(
          0);  // Partial prefill: the suffix computes its frontier.
  }
  [[nodiscard]] bool HasDecodeFrontier(
      const TextRunnerState& state) const override {
    return RequireQwenState(state).HasDecodeFrontier();
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    const bool speculative_enabled = DraftKind() != QwenDraftKind::kNone;
    return {
        .model_id = model_->GetConfig().model_name,
        .state_abi =
            QwenStateAbi(DraftKind(), execution_policy_.kv_cache_storage,
                         execution_policy_.UsesBf16RecurrentState()) +
            "-component-v2",
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .supports_vision = model_->VisionEncoder() != nullptr,
                .incremental_prefill = true,
                .final_token_advance_required = !speculative_enabled,
                .incremental_text_is_exact = true,
                .multi_token_decode = speculative_enabled,
                // Speculative sessions never take the single-token advance
                // path, so concurrent MTP sessions verify in a batch too.
                .batched_multi_token_decode =
                    speculative_enabled && MaximumDecodeBatchWidth() > 1,
                .batched_multi_token_decode_max_width =
                    speculative_enabled ? MaximumDecodeBatchWidth() : 0,
                .prefix_reuse = true,
                .partial_prefill_retention = true,
                .prefill_rollback = true,
            },
        .sampling_defaults = QwenSamplingDefaults(model_->GetConfig()),
        // Each session owns its executor, so a yield's decodes, batched
        // ones included, share no buffer with the pass they interrupt.
        .scheduling =
            TextSchedulingProfile{
                .pass_tokens = std::min<std::size_t>(
                    max_context_, hip::QwenGpuArena::kMaxPromptBatch),
                .max_layer_groups = kMaxPrefillLayerGroups,
                .max_yield_decode_width = MaximumDecodeBatchWidth(),
            },
        .prefill_step_tokens = std::min<std::size_t>(
            max_context_, hip::QwenGpuArena::kMaxPromptBatch),
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    auto usage = hip::QwenGpuExecutor::EstimateMemoryUsage(
        model_->GetConfig(), max_context_, execution_policy_);
    usage.request_state_bytes += hip::QwenGpuArena::RecurrentStateBytes(
        model_->GetConfig(), execution_policy_);
    std::size_t resident_weights = model_->GetResidentBytes();
    if (dflash_model_ != nullptr) {
      const auto draft = hip::QwenDFlashGpuExecutor::EstimateMemoryUsage(
          *dflash_model_, max_context_, MaximumDecodeBatchWidth());
      usage.request_state_bytes += draft.request_state_bytes;
      usage.temporary_scratch_bytes += draft.temporary_scratch_bytes;
      resident_weights += dflash_model_->GetPackedWeightBytes();
      // A wide copy grows the session's SSM replay ring to every one of its
      // rows (QwenGpuArena::EnsureSsmReplayCapacity).
      if (const auto ring = std::bit_ceil(std::size_t{copy_rows_});
          ring > hip::kSsmReplayCapacity)
        usage.temporary_scratch_bytes +=
            hip::QwenGpuArena::SsmReplayLogBytes(model_->GetConfig(), ring) -
            hip::QwenGpuArena::SsmReplayLogBytes(model_->GetConfig(),
                                                 hip::kSsmReplayCapacity);
    } else if (mtp_model_ != nullptr) {
      const auto draft = hip::QwenMtpGpuExecutor::EstimateMemoryUsage(
          *mtp_model_, max_context_, speculative_options_.mtp_prime_chunk);
      usage.request_state_bytes += draft.request_state_bytes;
      usage.temporary_scratch_bytes += draft.temporary_scratch_bytes;
      resident_weights += mtp_model_->GetPackedWeightBytes();
    }
    std::optional<std::size_t> capacity;
    // Snapshot buffers are device allocations, but on a unified-memory APU
    // they are carved from the same RAM as every host allocation, and HIP's
    // free figure does not see host pressure. Cap them by the host budget too.
    // They keep HIP's own free figure: only the state capacity check reads the
    // OS video-memory budget.
    std::size_t snapshot_capacity = HostSnapshotBudgetBytes();
    std::size_t snapshot_ceiling = HostSnapshotCeilingBytes();
    if (const auto memory = gufo::hip::QueryAdmissionMemory()) {
      capacity = memory->free_bytes;
      snapshot_capacity =
          std::min<std::size_t>(snapshot_capacity, memory->hip_free_bytes);
      snapshot_ceiling =
          std::min<std::size_t>(snapshot_ceiling, memory->hip_free_bytes);
      gufo::hip::LogStateCapacity(
          *memory, *capacity,
          usage.request_state_bytes + usage.temporary_scratch_bytes, 0);
    }
    return {
        .resident_weights_bytes = resident_weights,
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = usage.request_state_bytes,
        .temporary_scratch_bytes = usage.temporary_scratch_bytes,
        .retained_snapshot_capacity_bytes = snapshot_capacity,
        .retained_snapshot_ceiling_bytes = snapshot_ceiling,
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {
            .kind = TextExecutionPlanKind::kSerial,
            .physical_width = 1,
        },
    };
    // Report actual session widths, including C6. Speculation reserves up to
    // eight token rows per session in the coordinator's existing scratch.
    for (std::size_t width = 2; width <= MaximumDecodeBatchWidth(); ++width) {
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = width,
      });
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->GetTokenizer().GetVocabSize(), [this](std::uint32_t id) {
          const auto& tokenizer = model_->GetTokenizer();
          return sampling::ConstraintVocabulary::Piece{
              tokenizer.DecodeTokenCopy(id), tokenizer.IsStopToken(id)};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->GetTokenizer().GetVocabSize());
  }

  // llama.cpp's Qwen3-Coder format forces "\n</think>" and also ends
  // reasoning at a tool call.
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"<think>", "\n</think>", {"</think>", "<tool_call>"}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->GetTokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->GetTokenizer(), request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        QwenChatOptions(request, max_context_, model_->GetTokenizer()));
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    return PrepareQwenPrompt(request, model_->GetTokenizer(),
                             model_->VisionEncoder(), max_context_,
                             image_cache_);
  }

  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    return CountQwenPromptTokens(request, model_->GetTokenizer(),
                                 model_->VisionEncoder(), max_context_,
                                 image_cache_);
  }

  [[nodiscard]] std::optional<std::string> RenderPrompt(
      const ChatRequest& request) const override {
    return RenderQwenPrompt(request, max_context_, model_->GetTokenizer());
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    RequireQwenState(state).executor().ConfigureVision(QwenPrompt(context),
                                                       model_->VisionEncoder());
  }

  [[nodiscard]] ext::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return QwenChatOptions(request, max_context_, model_->GetTokenizer())
                   .enable_thinking
               ? ext::InitialOutputState::kReasoning
               : ext::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->GetTokenizer().Decode(tokens);
  }

  [[nodiscard]] std::optional<TextHiddenStateEncoding> HiddenStateEncoding()
      const override {
    return TextHiddenStateEncoding{
        .width = model_->GetConfig().hidden_size,
        .max_tokens = max_context_,
    };
  }

  // Runs on a private executor, never on a component-cache slot, so sessions,
  // prefix caches and snapshots are untouched. The capture is the last
  // decoder layer's residual output before output_norm (prefill_chunk.cpp).
  [[nodiscard]] TextHiddenStateStatus EncodeHiddenStates(
      const TextHiddenStateRequest& request, TextHiddenStates* states,
      std::string* error) const override {
    const auto finish = [&](TextHiddenStateStatus status, std::string message) {
      if (states != nullptr)
        *states = {};
      if (error != nullptr)
        *error = std::move(message);
      return status;
    };
    if (states == nullptr)
      throw std::invalid_argument("hidden-state encoding needs an output");
    const auto tokens = request.tokens;
    if (tokens.empty())
      return finish(TextHiddenStateStatus::kInvalidInput,
                    "hidden-state encoding needs at least one token");
    if (tokens.size() > max_context_)
      return finish(TextHiddenStateStatus::kInvalidInput,
                    "prompt exceeds the hidden-state encoding limit");
    const std::size_t vocabulary = model_->GetConfig().vocab_size;
    if (std::ranges::any_of(tokens, [vocabulary](TextRunnerToken token) {
          return token >= vocabulary;
        }))
      return finish(TextHiddenStateStatus::kInvalidInput,
                    "prompt token is outside the model vocabulary");

    std::lock_guard lock(hidden_state_mutex_);
    try {
      auto& executor = HiddenStateExecutor(tokens.size());
      // Leave the executor empty and unarmed however the call ends.
      struct Release {
        hip::QwenGpuExecutor& executor;
        ~Release() {
          executor.SetCancellationCheck({});
          try {
            executor.SetPromptHiddenCapture(false);
          } catch (...) {
          }
          executor.Reset();
        }
      } release{executor};
      executor.Reset();
      executor.SetCancellationCheck(request.is_cancelled);
      executor.SetPromptHiddenCapture(true);
      (void)executor.ForwardPromptBatch(tokens, 0, false);
      const std::size_t width = model_->GetConfig().hidden_size;
      const auto hidden = executor.GetPromptHiddenStates();
      if (hidden.size() != tokens.size() * width)
        throw std::logic_error("Qwen hidden-state capture size mismatch");
      states->rows = tokens.size();
      states->width = width;
      states->values.assign(hidden.begin(), hidden.end());
    } catch (const hip::QwenPrefillCancelled&) {
      return finish(TextHiddenStateStatus::kCancelled,
                    "hidden-state encoding cancelled");
    } catch (const std::exception& e) {
      // A failed device pass may leave the executor unusable; the next call
      // builds a fresh one.
      hidden_state_executor_.reset();
      return finish(TextHiddenStateStatus::kFailed, e.what());
    }
    if (error != nullptr)
      error->clear();
    return TextHiddenStateStatus::kOk;
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    auto& qwen = RequireQwenState(state);
    qwen.PreparePrefixReuse(prefix);
  }

  void SetDraftContextHorizon(TextRunnerState& state,
                              std::size_t position) const override {
    RequireQwenState(state).set_draft_context_horizon(position);
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& qwen = RequireQwenState(state);
    // Keep the entry recurrence pinned so cancellation can stop between
    // layers without discarding already completed prompt work.
    auto activity = qwen.BeginPrefill();
    const PrefillYieldScope yield(qwen.executor(), state);
    try {
      if (offset != qwen.position()) {
        throw std::logic_error(
            "Qwen prefill offset does not match retained state");
      }
      if (offset >= prompt.size()) {
        throw std::logic_error("Qwen prefill has no remaining input");
      }
      max_input_tokens = std::min<std::size_t>(
          max_input_tokens, qwen.executor().GetMaxPromptBatch());
      const std::uint32_t keep_context_from =
          qwen.KeepDraftContextFrom(qwen.TakeDraftContextHorizon());
      if (max_input_tokens != 0 && offset != 0 &&
          qwen.ResumeGeneratedToken(prompt[offset],
                                    offset + 1 == prompt.size())) {
        return {.consumed_tokens = 1,
                .decode_ready = offset + 1 == prompt.size()};
      }
      if (qwen.speculative()) {
        if (offset == 0) {
          const auto consumed = std::min(max_input_tokens, prompt.size());
          if (consumed == 0) {
            throw std::logic_error(
                "Qwen prefill requires a nonzero token budget");
          }
          qwen.PrimeSpeculative(prompt.first(consumed),
                                consumed == prompt.size(), keep_context_from);
          return {.consumed_tokens = consumed,
                  .decode_ready = consumed == prompt.size()};
        }
        const std::size_t consumed =
            std::min(max_input_tokens, prompt.size() - offset);
        if (consumed == 0) {
          throw std::logic_error(
              "Qwen DFlash retained prefix has no suffix to prefill");
        }
        qwen.ExtendSpeculative(prompt.subspan(offset, consumed),
                               offset + consumed == prompt.size(),
                               keep_context_from);
        return {
            .consumed_tokens = consumed,
            .decode_ready = offset + consumed == prompt.size(),
        };
      }

      const std::size_t consumed =
          std::min(max_input_tokens, prompt.size() - offset);
      const bool decode_ready = offset + consumed == prompt.size();
      const auto frontier = qwen.executor().ForwardPromptBatch(
          prompt.subspan(offset, consumed), static_cast<std::uint32_t>(offset),
          decode_ready);
      qwen.set_position(offset + consumed);
      if (decode_ready) {
        qwen.set_frontier(frontier);
      }
      return {
          .consumed_tokens = consumed,
          .decode_ready = decode_ready,
      };
    } catch (const hip::QwenPrefillCancelled&) {
      activity.Rollback();
      return {.interrupted = true};
    }
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& qwen = RequireQwenState(state);
    const TextRunnerToken token = qwen.SelectFrontier(sampler);
    auto logprobs =
        sampler.config().logprobs
            ? sampling::CollectLogprobs(qwen.CopyFrontierLogits(), token,
                                        sampler.config().logprobs)
            : std::nullopt;
    if (EndsTurn(qwen, model_->GetTokenizer(), token))
      return {.stop = true,
              .token = token,
              .piece = {},
              .logprobs = std::move(logprobs)};
    // Keep the selected token separate from the argmax frontier: sampled
    // cancellation and prompt snapshot restore must not replace that argmax.
    qwen.set_pending_token(token);
    return {
        .stop = false,
        .token = token,
        .piece = std::string(model_->GetTokenizer().DecodeToken(token)),
        .logprobs = std::move(logprobs),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& qwen = RequireQwenState(state);
    if (qwen.speculative()) {
      throw std::logic_error(
          "Qwen speculative decoding requires a multi-token decode step");
    }
    if (!qwen.frontier().has_value()) {
      throw std::logic_error("Qwen decode state has no retained frontier");
    }
    const auto frontier = qwen.executor().ForwardToken(
        token, static_cast<std::uint32_t>(qwen.position()));
    qwen.set_position(qwen.position() + 1);
    qwen.set_frontier(frontier);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    auto& qwen = RequireQwenState(state);
    if (!qwen.speculative()) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    return qwen.DecodeSpeculative(max_tokens, sampler, DraftCap(sampler, 1));
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (DraftKind() == QwenDraftKind::kNone || decodes.size() < 2 ||
        decodes.size() > MaximumDecodeBatchWidth()) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    std::vector<TextDecodeStep> steps(decodes.size());
    std::vector<QwenTextRunnerState*> states;
    std::vector<sampling::SamplerState> samplers;
    std::vector<speculative::SpeculativeVerifier::StepRequest> requests;
    std::vector<std::size_t> indices;
    states.reserve(decodes.size());
    samplers.reserve(decodes.size());
    requests.reserve(decodes.size());
    indices.reserve(decodes.size());
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      const auto& decode = decodes[index];
      auto& state = RequireQwenState(decode.state.get());
      states.push_back(&state);
      auto& sampler = samplers.emplace_back(decode.sampler.get());
      if (state.PrepareSpeculativeDecode(decode.max_tokens, sampler,
                                         steps[index])) {
        requests.push_back(state.VerificationRequest(
            decode.max_tokens - steps[index].selections.size(), sampler));
        indices.push_back(index);
      }
    }
    for (auto& request : requests)
      request.max_draft_tokens = DraftCap(request.sampler, requests.size());
    auto verified = speculative::SpeculativeVerifier::VerifyBatch(requests);
    for (std::size_t item = 0; item < indices.size(); ++item) {
      const std::size_t index = indices[item];
      states[index]->FinishSpeculativeDecode(std::move(verified[item]),
                                             samplers[index], steps[index]);
    }
    for (std::size_t index = 0; index < decodes.size(); ++index)
      decodes[index].sampler.get().SetRngState(samplers[index].rng_state());
    return steps;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (DraftKind() != QwenDraftKind::kNone) {
      throw std::logic_error(
          "Qwen speculative sessions do not support batch advance");
    }
    if (advances.size() < 2 || advances.size() > 8) {
      throw std::invalid_argument(
          "Qwen batched decode requires two to eight sessions");
    }

    std::vector<hip::QwenGpuBatchItem> items;
    std::vector<QwenTextRunnerState*> states;
    items.reserve(advances.size());
    states.reserve(advances.size());
    for (const auto& advance : advances) {
      auto& qwen = RequireQwenState(advance.state.get());
      if (!qwen.frontier().has_value()) {
        throw std::logic_error("Qwen batched state has no retained frontier");
      }
      states.push_back(&qwen);
      items.push_back({
          .executor = &qwen.executor(),
          .token_id = advance.token,
          .position = static_cast<std::uint32_t>(qwen.position()),
      });
    }

    const auto frontiers = hip::QwenGpuExecutor::ForwardTokenBatch(items);
    if (frontiers.size() != states.size()) {
      throw std::runtime_error(
          "Qwen batched decode returned an invalid frontier count");
    }
    for (std::size_t index = 0; index < states.size(); ++index) {
      states[index]->set_position(states[index]->position() + 1);
      states[index]->set_frontier(frontiers[index]);
    }
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireQwenState(state).position();
  }
  void PrepareCancellation(TextRunnerState& state) const override {
    RequireQwenState(state).PrepareCancellation();
  }

private:
  // Shares the loaded weights; only the attention/recurrent state is private.
  // Sized to the longest prompt encoded so far in 1,024-token steps, so a
  // runner with a long serving context does not reserve a second full-length
  // KV cache for short encoder prompts.
  [[nodiscard]] hip::QwenGpuExecutor& HiddenStateExecutor(
      std::size_t tokens) const {
    constexpr std::size_t kStep = 1024;
    if (hidden_state_executor_ == nullptr ||
        hidden_state_executor_->GetMaxContext() < tokens) {
      const auto context = static_cast<std::uint32_t>(std::min<std::size_t>(
          max_context_,
          std::max<std::size_t>(2, (tokens + kStep - 1) / kStep * kStep)));
      hidden_state_executor_.reset();
      hidden_state_executor_ = std::make_unique<hip::QwenGpuExecutor>(
          model_, context, execution_policy_);
    }
    return *hidden_state_executor_;
  }

  /// Draft ceiling for one request while `active` requests decode. As in
  /// Flash-Next (#703), greedy and unseeded sampled requests follow the
  /// requests decoding now. A seeded sampled request takes the ceiling for the
  /// configured sessions: its drafts consume its RNG, so it must draft the same
  /// at any load to replay the same text.
  [[nodiscard]] std::uint32_t DraftCap(const sampling::SamplerState& sampler,
                                       std::size_t active) const {
    const bool seeded =
        sampler.config().uses_random_sampling() && sampler.config().seed >= 0;
    return ConcurrentDraftCap(seeded ? session_count_ : active,
                              model_->GetConfig().IsMoE(),
                              DraftKind() == QwenDraftKind::kMtp);
  }

  [[nodiscard]] QwenDraftKind DraftKind() const noexcept {
    if (dflash_model_ != nullptr) {
      return QwenDraftKind::kDFlash;
    }
    return mtp_model_ != nullptr ? QwenDraftKind::kMtp : QwenDraftKind::kNone;
  }

  [[nodiscard]] std::size_t MaximumDecodeBatchWidth() const noexcept {
    const std::size_t rows_per_session =
        DraftKind() != QwenDraftKind::kNone ? 8 : 1;
    return std::clamp<std::size_t>(max_context_ / rows_per_session, 1, 8);
  }

  mutable models::qwen::vision::ImageCache image_cache_;
  std::shared_ptr<const hip::QwenGpuModel> model_;
  std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model_;
  std::shared_ptr<const hip::QwenMtpGpuModel> mtp_model_;
  std::uint32_t max_context_;
  speculative::SpeculativeOptions speculative_options_;
  speculative::DFlashDraftPolicy dflash_policy_{
      speculative::DFlashDraftPolicy::kAdaptive};
  std::size_t session_count_;
  hip::QwenExecutionPolicy execution_policy_;
  /// Rows a lookup copy of a request decoding alone may verify (DFlash).
  std::uint32_t copy_rows_{static_cast<std::uint32_t>(
      hip::QwenGpuExecutor::kMaxWideVerificationRows)};
  cache::Identity component_identity_;
  mutable std::mutex hidden_state_mutex_;
  mutable std::unique_ptr<hip::QwenGpuExecutor> hidden_state_executor_;
};

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateQwenTextRunner(
    std::shared_ptr<const hip::QwenGpuModel> model, std::uint32_t max_context,
    std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model,
    const speculative::SpeculativeOptions& speculative_options,
    speculative::DFlashDraftPolicy dflash_policy,
    std::string artifact_fingerprint, std::string draft_artifact_fingerprint) {
  return std::make_shared<QwenTextRunner>(
      std::move(model), max_context, std::move(dflash_model),
      speculative_options, dflash_policy, std::move(artifact_fingerprint),
      std::move(draft_artifact_fingerprint));
}

std::shared_ptr<ext::TextModelRunner> CreateQwenTextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  std::shared_ptr<const core::GgufReader> reader = options.reader;
  if (!reader) {
    std::string load_error;
    auto reader_owner =
        core::GgufReader::OpenFile(options.model_path, &load_error);
    if (!reader_owner) {
      if (error)
        *error = "Failed to open GGUF: " + load_error;
      return nullptr;
    }
    reader = std::shared_ptr<const core::GgufReader>(std::move(reader_owner));
  }
  // Declared in extension.json; the host delivers it only when given.
  const auto* kv_option =
      ext::FindOption<std::string>(options.provider_options, "--kv-cache");
  const std::string kv_cache = kv_option != nullptr ? *kv_option : "";
  if (!kv_cache.empty() && kv_cache != "f16" && kv_cache != "q8_0") {
    if (error)
      *error = "--kv-cache must be f16 or q8_0";
    return nullptr;
  }
  const auto kv_storage = kv_cache == "q8_0"
                              ? hip::QwenKvCacheStorage::kQ8
                              : hip::QwenKvCacheStorage::kFp16;
  std::shared_ptr<models::qwen::vision::Encoder> vision;
  try {
    // The Qwen3.8-27B and Ornith-1.5-122B (qwen35moe) trunks take the shared
    // vision tower at their own width.
    const std::uint32_t width =
        reader->GetMetadataUint64("qwen35.embedding_length") == 5120      ? 5120
        : reader->GetMetadataUint64("qwen35moe.embedding_length") == 3072 ? 3072
                                                                          : 0;
    if (width != 0) {
      vision = models::qwen::vision::Encoder::Open(
          options.model_path, options.vision_model_path, width);
    } else if (!options.vision_model_path.empty()) {
      if (error)
        *error =
            "image input supports Qwen3.8-27B, Ornith-1.5-122B and Flash-Next";
      return nullptr;
    }
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
  std::string load_error;
  auto model =
      hip::QwenGpuModel::CreateFromGguf(reader, &load_error, std::move(vision));
  if (!model) {
    if (error)
      *error = "Failed to create GPU model: " + load_error;
    return nullptr;
  }
  try {
    models::qwen::vision::PrepareForServing(model->VisionEncoder());
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
  std::uint32_t max_context = options.max_context;
  if (max_context == 0) {
    max_context = model->GetConfig().context_length;
  }
  if (max_context < 2 || max_context > model->GetConfig().context_length) {
    if (error)
      *error = "HTTP context exceeds the loaded Qwen model context";
    return nullptr;
  }
  if (kv_storage == hip::QwenKvCacheStorage::kQ8) {
    // The 8-bit kernels cover 256-wide heads, eight query heads per KV head
    // and a 64-wide rotary span: Qwen3.6-35B-A3B.
    const auto& config = model->GetConfig();
    if (config.head_dim != 256 || config.rotary_dim != 64 ||
        config.num_attention_heads != 8 * config.num_key_value_heads) {
      if (error)
        *error = "--kv-cache q8_0 supports Qwen3.6-35B-A3B only";
      return nullptr;
    }
  }
  if (options.speculative.backend == ext::TextSpeculativeBackend::kDSpark) {
    if (error)
      *error = "DSpark HTTP decoding requires a DeepSeek model";
    return nullptr;
  }
  if (options.speculative.backend == ext::TextSpeculativeBackend::kMtp &&
      options.speculative.min_draft_tokens != 1) {
    if (error)
      *error =
          "MTP drafts a fixed proposal chain; --min-draft-tokens must be 1";
    return nullptr;
  }
  if (options.speculative.backend == ext::TextSpeculativeBackend::kDFlash &&
      options.speculative.min_draft_tokens != 1) {
    if (error)
      *error =
          "DFlash2 requires --min-draft-tokens 1; bound blocks with "
          "--draft-tokens";
    return nullptr;
  }
  std::string model_fingerprint = options.disk_cache.model_artifact_fingerprint;
  if (!options.disk_cache.directory.empty() && model_fingerprint.empty()) {
    try {
      model_fingerprint = core::GgufIdentityHex(*reader);
    } catch (const std::exception& e) {
      if (error)
        *error =
            std::string("Qwen disk cache fingerprinting failed: ") + e.what();
      return nullptr;
    }
  }
  std::string draft_fingerprint =
      options.disk_cache.draft_model_artifact_fingerprint;
  std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model;
  std::shared_ptr<const hip::QwenMtpGpuModel> mtp_model;
  speculative::SpeculativeOptions speculative_options;
  speculative_options.mtp_prime_chunk = options.speculative.mtp_prime_chunk;
  if (options.speculative.backend == ext::TextSpeculativeBackend::kDFlash) {
    if (options.speculative.draft_model_path.empty()) {
      if (error)
        *error = "DFlash HTTP decoding requires --dflash-model";
      return nullptr;
    }
    std::string dflash_error;
    auto dflash_reader_owner = core::GgufReader::OpenFile(
        options.speculative.draft_model_path, &dflash_error);
    if (!dflash_reader_owner) {
      if (error)
        *error = "Failed to open DFlash GGUF: " + dflash_error;
      return nullptr;
    }
    std::shared_ptr<const core::GgufReader> dflash_reader(
        std::move(dflash_reader_owner));
    if (!options.disk_cache.directory.empty() && draft_fingerprint.empty()) {
      try {
        draft_fingerprint = core::GgufIdentityHex(*dflash_reader);
      } catch (const std::exception& e) {
        if (error)
          *error = std::string("DFlash disk cache fingerprinting failed: ") +
                   e.what();
        return nullptr;
      }
    }
    dflash_model = hip::QwenDFlashGpuModel::Create(std::move(dflash_reader),
                                                   model, &dflash_error);
    if (!dflash_model) {
      if (error)
        *error = "Failed to create DFlash model: " + dflash_error;
      return nullptr;
    }
    speculative_options.max_draft_tokens = options.speculative.max_draft_tokens;
    speculative_options.min_draft_tokens = options.speculative.min_draft_tokens;
    speculative_options.initial_draft_tokens =
        options.speculative.max_draft_tokens;
    speculative_options.use_batched_verification = true;
    speculative_options.retain_frontier_logits = true;
    speculative_options.enable_adaptive_draft_length = false;
  } else if (options.speculative.backend == ext::TextSpeculativeBackend::kMtp) {
    if (options.speculative.draft_model_path.empty()) {
      if (error)
        *error = "MTP HTTP decoding requires --mtp-model";
      return nullptr;
    }
    std::string mtp_error;
    auto mtp_reader_owner = core::GgufReader::OpenFile(
        options.speculative.draft_model_path, &mtp_error);
    if (!mtp_reader_owner) {
      if (error)
        *error = "Failed to open MTP GGUF: " + mtp_error;
      return nullptr;
    }
    std::shared_ptr<const core::GgufReader> mtp_reader(
        std::move(mtp_reader_owner));
    if (!options.disk_cache.directory.empty() && draft_fingerprint.empty()) {
      try {
        draft_fingerprint = core::GgufIdentityHex(*mtp_reader);
      } catch (const std::exception& e) {
        if (error)
          *error =
              std::string("MTP disk cache fingerprinting failed: ") + e.what();
        return nullptr;
      }
    }
    mtp_model =
        hip::QwenMtpGpuModel::Create(std::move(mtp_reader), model, &mtp_error);
    if (!mtp_model) {
      if (error)
        *error = "Failed to create MTP model: " + mtp_error;
      return nullptr;
    }
    // Mirror the CLI MTP arm exactly so serve and `gufo prompt --speculative
    // mtp` stay comparable: adaptive width on (unlike DFlash2, whose
    // controller owns the width), batched verification on.
    //
    // Frontier logits are retained for the serve state machine's exact
    // cancellation. The CLI leaves them off, but that cannot move the token
    // stream: the Qwen target reports device-resident sampling, so
    // `capture_logits` is already false and retention only adds a D2H copy.
    speculative_options.max_draft_tokens = options.speculative.max_draft_tokens;
    speculative_options.min_draft_tokens = options.speculative.min_draft_tokens;
    speculative_options.initial_draft_tokens =
        options.speculative.max_draft_tokens;
    speculative_options.use_batched_verification = true;
    speculative_options.retain_frontier_logits = true;
    speculative_options.enable_adaptive_draft_length = true;
  }
  auto dflash_policy =
      options.speculative.dflash_policy == ext::TextDFlashDraftPolicy::kFixed
          ? speculative::DFlashDraftPolicy::kFixed
          : speculative::DFlashDraftPolicy::kAdaptive;
  const bool speculative = dflash_model != nullptr;
  const std::size_t sessions = std::max<std::size_t>(options.session_count, 1);
  auto runner = std::make_shared<QwenTextRunner>(
      std::move(model), max_context, std::move(dflash_model),
      speculative_options, dflash_policy, model_fingerprint, draft_fingerprint,
      std::move(mtp_model), sessions, kv_storage);
  // Each session's wide copies need a larger SSM replay ring (ResourceClaim).
  // Narrow the width until the configured sessions fit, rather than refusing
  // a session count the ordinary verifier width fits.
  for (auto rows = static_cast<std::uint32_t>(
           hip::QwenGpuExecutor::kMaxWideVerificationRows);
       speculative; rows /= 2) {
    runner->LimitCopyRows(rows);
    const auto claim = runner->ResourceClaim();
    if (rows <= hip::kSsmReplayCapacity || !claim.state_capacity_bytes ||
        !claim.per_request_state_bytes || !claim.temporary_scratch_bytes ||
        sessions * (*claim.per_request_state_bytes +
                    *claim.temporary_scratch_bytes) <=
            *claim.state_capacity_bytes)
      break;
  }
  return runner;
}

}  // namespace gufo::models::qwen

#endif  // ENGINE_ENABLE_HIP
