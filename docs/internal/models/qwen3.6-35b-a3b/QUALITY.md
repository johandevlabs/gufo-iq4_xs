# Qwen3.6 35B-A3B quality

These are execution-consistency checks: the GPU executor against the CPU
reference forward pass, plus GPU prefill/decode self-consistency. Original-target
and GGUF-conversion parity are not qualified here.

## What the target test checks

`qwen35ba3b_target_test` (gated by `GUFO_QWEN35BA3B_MODEL`, skipped when unset)
requires the loaded config to report an MoE architecture, then runs:

| Check | Method | Asserted contract |
| --- | --- | --- |
| GPU/CPU greedy parity, position 0–11 | Two fixed text fixtures; the GPU re-prefills each prefix while the CPU reference steps the same tokens (teacher forcing). Compares the final-position logits. | GPU logits must be **finite** and the greedy (argmax) next token must **equal the CPU reference** at every position. `CompareLogits` is called with `atol = rtol = 5e-2`, but the elementwise `match` that drives is not asserted; per-position max-abs and RMSE are printed for reference only. |
| Prefill/decode self-consistency | Same full prompt: one `ForwardPromptBatch` prefill versus a one-token-at-a-time `ForwardToken` decode loop. | The two next-token IDs must be **exactly equal**. |
| Long prompt chunk | One 1100-token `ForwardPromptBatch` (the paired gate/up route; the down projection takes the pair's tile map only when its type has 64- and 128-row tiles) against the same prompt prefilled as 600 + 500 tokens. | Logits **finite** and the greedy token **equal** (`CompareLogits`, `atol = rtol = 5e-2`). UD-Q4_K_XL, Q4_K_M, UD-Q6_K_XL and UD-Q8_K_XL give identical logits (max abs 0). |
| Verification equals decode | Eight draft tokens decoded one at a time with `ForwardToken`, then restored from a snapshot and verified as one `ForwardVerificationChunk`. | Every row's logits **bit-identical** to the decoded token's. |
| Concurrent verification | Three sessions verify 8, 8 and 7 rows together through `ForwardVerificationBatch` (23 stacked rows, more than the 8-row gated vector kernel takes per launch); each session's own `ForwardVerificationChunk` is the reference. | Logits **finite** and **bit-identical** to the per-session reference for every row. |

Executor context is capped at 4096 tokens for every check. The parity fixtures
must tokenize to at least 12 / 16 tokens respectively, or the test fails rather
than skips.

## MoE operator test

`qwen35ba3b_moe_ops_test` is a synthetic, model-free check of the MoE GEMM
families (gate/up/down, routed + shared) against a CPU reference. It qualifies
the kernels in isolation; it does not cover the full model or the quantized
GGUF path. It also checks the grouped BF16 GEMM against the per-slot GEMV
for 1, 7, 100 and 512 tokens on both projection shapes, requiring every
slot written, relative error < 1e-2, and byte-identical output across two
runs. The op test also checks the dense mode of the BF16 WMMA GEMM against
the FP32-activation GEMV for 1, 100 and 512 rows (relative error < 1e-2,
every row written). It also checks the routed F16 expert GEMM (Q8_0, Q6_K,
Q5_K and Q4_K at 16- and 48-row tiles; Q8_0 and Q6_K also at the 64- and
128-row tiles the paired prefill sends the down projection over, which Q5_K
and Q4_K must reject) and the paired gate/up GEMM (all four types, 64- and
128-row tiles) against per-slot GEMV references, requiring every output
written and relative error below 1e-2 (routed) and 5e-3 (paired).
Finally, it launches every projection that single-token decode and batched
decode / verification take through different kernels the way each path does
and requires each batched row to be bit-identical to that row decoded alone:
the routed gate/up/SwiGLU and down projections for all four types at 2-64
rows (MMQ vector kernels for Q8_0/Q5_K/Q4_K gate/up and all four down
projections, the per-slot GEMV for Q6_K gate/up), the router, shared-expert gate and shared expert, the gated-delta-net
input projections and controls, its output projection and RMSNorm.

## MTP head

`qwen_mtp_gpu_test`, run with the 35B GGUF as both base and draft, compares
the GPU MTP layer with the CPU reference (hidden cosine > 0.999, selected
logits within 2.0) and checks reset determinism. Measured on UD-Q8_K_XL:
hidden cosine 0.999992 (RMSE 0.0108, max abs 0.0414), selected-logit max abs
error 0.0102; UD-Q6_K_XL gives the same figures. For `gufo serve` it also
checks the batched prompt prime and the batched append of verified rows
against the serial per-token path (logit cosine > 0.99999), chunk-size
invariance, and that the memory estimate matches the allocation.

## DFlash2 draft

`qwen_dflash_gpu_test`, run with this model as the target and the converted
Q8_0 DFlash2 draft (`GUFO_QWEN27B_MODEL`, `GUFO_QWEN27B_DFLASH_MODEL`),
passes its draft-state, snapshot and batched-proposal checks.

For this MoE target, prompt context of at least 128 rows is projected into
the draft's KV cache with BF16-rounded activations on the WMMA GEMM; dense
targets keep the exact FP32-activation route. This changes only the draft's
context, never the verified output distribution. On three 2.4K-3.7K-token
prompts decoded greedily for 768 tokens, outputs and draft/accepted counts
were identical before and after the change. The concurrent draft cap only
shortens proposals, so it does not change the verified output either.

## Scope of the target test for prefill changes

The parity fixtures are 12-16 tokens; only the long-chunk check reaches the
routes that large chunks take (the paired routed tile map), and only at 1,100
tokens. Changes there are also qualified by comparing greedy completions on 6K and 12K prompts against the
previous build in server mode; the October 1, 2026 prefill changes keep them
byte-identical on UD-Q6_K_XL and UD-Q8_K_XL, and both artifacts pass the target
test and the MoE operator test.

## BF16 router and GDN controls in prefill

Prefill chunks of at least 128 tokens round the F32 router and the GDN
alpha/beta projections to BF16 (see [Experiments](EXPERIMENTS.md)), so their
output is not byte-identical to the FP32 route. The change was qualified on
UD-Q6_K_XL, greedy, plain decoding, against the same source without it:

| Check | FP32 route | BF16 route |
| --- | ---: | ---: |
| 5-shot MMLU, 969 questions over 57 subjects, mean prompt 651 tokens | 741 (76.47%) | 739 (76.26%) |
| Aggregation over 40/120/300 order records, 90 questions, up to 5.5K tokens | 40 | 40 |
| Recall of one or two access codes among three, 54 prompts at 2K-12K tokens | 54 | 54 |
| Recall of three codes among sixteen similar names, 30 prompts at 6K-24K tokens | 90 of 90 | 90 of 90 |

On MMLU the two routes pick the same letter on 97.0% of questions; 13 are right
only with FP32 and 11 only with BF16 (exact McNemar p = 0.84). `--validate-prefill`
keeps the sequential reference's top-1 token at 512, 2048 and 4096 tokens, with
an RMSE of 0.087 / 0.564 / 1.85 against 0.086 / 0.523 / 2.10 for the FP32 route.
Both artifacts pass the target test. This bounds a loss to about one MMLU point;
it does not measure long generations, tool calling or speculative decoding.

## Benchmark output consistency

The October 25 and 27, 2026 [benchmarks](BENCHMARKS.md) hash every greedy
completion. On UD-Q4_K_XL, Q4_K_M, UD-Q6_K_XL and UD-Q8_K_XL:

- DFlash2 and MTP match AR at all eight single-user depths, mixed and
  repetitive (Q4_K_M has no MTP head, so DFlash2 only).
- Every multi-user request (pp2048 / tg128, C1-C8) matches the same prompt
  decoded alone with AR: AR from C2, DFlash2 and MTP from C1.

Before `7b1e23a8`, the F32 gated-delta-net controls of batched rows were
summed in a different order than single-token decode. UD-Q6_K_XL and
UD-Q8_K_XL then matched AR in none of the prose/mixed multi-user requests and
at only two to four of the eight single-user depths; production `8e509128`
showed the same multi-user differences.
These hashes show which runs are numerically interchangeable; they do not
measure accuracy.

## K-quant experts in decode

Since the October 27 tables, decode and verification send Q4_K/Q5_K gate/up
and Q4_K/Q5_K/Q6_K down experts to the MMQ vector kernels instead of the
per-slot GEMV, with the expert activations quantized to Q8_1 as on the Q8_0
route. This changes the arithmetic for UD-Q4_K_XL and Q4_K_M, so their greedy
text can differ from earlier builds; UD-Q6_K_XL and UD-Q8_K_XL take the same
routes as before. Speculative and concurrent decoding stay bit-exact against
plain decoding on all four files (target test, MoE operator test and the
benchmark hashes above). The change has not been qualified on accuracy
benchmarks.

## Reproduce

Build the `gpu-test` preset and run the binaries inside the ROCm container:

```sh
cmake --build --preset gpu-test --target qwen35ba3b_target_test qwen35ba3b_moe_ops_test qwen_mtp_gpu_test --parallel 16
export GUFO_QWEN35BA3B_MODEL=path/to/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
./build/gpu-test/bin/qwen35ba3b_target_test
./build/gpu-test/bin/qwen35ba3b_moe_ops_test
./build/gpu-test/bin/qwen_mtp_gpu_test "$GUFO_QWEN35BA3B_MODEL" "$GUFO_QWEN35BA3B_MODEL"
```

A missing-weight skip is not a pass.

## Meaning of exact

Matching the CPU reference's greedy token tests GPU/CPU execution consistency,
not original-model accuracy; a quantized GPU path and a BF16 CPU path can
disagree within the tolerance above. Keep the reference, weights and fixtures
fixed before interpreting any numerical difference.
