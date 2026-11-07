# Text runner KV cache format option

Strycast's opt-in 8-bit attention KV cache for Qwen3.6-35B-A3B
(`gufo serve llm --kv-cache q8_0`):

- `TextRunnerOptions::kv_cache` (`src/ext/text.hpp`) names the attention KV
  cache format. Empty or `"f16"` keeps the default; `"q8_0"` selects the
  8-bit cache where a provider supports it. Providers that support only the
  default reject any other value.

The field defaults to empty, so existing extensions keep compiling and
behaving as before.
