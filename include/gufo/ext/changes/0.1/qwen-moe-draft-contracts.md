# Qwen MoE configuration and bounded draft contracts

Strycast's additions for Qwen3.6-35B-A3B and Ornith 1.5 (`qwen35moe`)
serving:

- `ModelConfig` (`src/core/model_config.hpp`): `expert_count`,
  `expert_used_count`, `expert_ff_length` and `expert_shared_ff_length`
  (zero for dense models), and `IsMoE()`.
- `IDraftBackend` (`src/core/speculative/draft_backend.hpp`): virtual
  `TargetContextWindow()` (default 0, every position) and
  `SkipTargetContext(position, count, prime)` (default `false`), for a draft
  that reads only a bounded window of recent target features.
- `SpeculativeVerifier` (`src/core/speculative/speculative_verifier.hpp`):
  - `Prime` and `ExtendPrompt` take a trailing `keep_context_from`
    (default 0): a prompt span that ends at or before it skips draft context.
  - `VerifyStep` takes a trailing `max_draft_tokens` bound, and
    `StepRequest::max_draft_tokens` bounds batched steps the same way. Both
    default to no bound.
  - `SpeculativeOptions::mtp_prime_chunk` sets the rows per batched MTP prompt
    prime; 0 primes serially.
- Text runner contracts (`src/ext/text.hpp`):
  - `TextRunnerDescriptor::prefill_step_tokens` is the largest prompt step a
    lone request runs; 0 means unknown.
  - `TextModelRunner::SetDraftContextHorizon(state, position)`, a no-op by
    default, tells the runner where the state is next used before a `Prefill`.
  - `TextSpeculativeConfig::mtp_prime_chunk`.

Every addition has a default that keeps the previous behavior. Existing
extensions keep compiling and behaving as before; overriding classes must be
rebuilt against the new declarations.
