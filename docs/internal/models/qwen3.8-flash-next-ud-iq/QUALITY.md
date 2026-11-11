# Qwen3.8 Flash-Next UD-IQ quality

Two kinds of evidence: how much each file loses against UD-Q4_K_XL, measured
with llama.cpp and independent of Gufo, and whether Gufo executes the files
correctly (kernel tests, the CPU F32 reference, model tests and functional
suites). Original-model and GGUF-conversion parity are not qualified, as for
UD-Q4_K_XL.

## The files against UD-Q4_K_XL

Next-token distributions of llama.cpp ROCm `b11557` (`5c8854fd`) on the 4,995
positions of Strata's `data/quant_quality/v1` set (commit `c62d5e3a`), with
Strata's formulas: top-1 agreement, KL(reference‖file) over the reference's 64
most likely tokens, and perplexity of the text. The reference is UD-Q4_K_XL;
no Q8_0 or BF16 file was available. A deterministic rerun was byte-identical.

| File | Top-1 | Mean KL | KL p99 | Perplexity | llama.cpp GPU weights |
| --- | ---: | ---: | ---: | ---: | ---: |
| UD-Q4_K_XL | reference | reference | reference | 4.025 | 78.1 GiB |
| UD-IQ4_XS | 89.43% | 0.096 | 1.31 | 3.906 | 60.1 GiB |
| UD-IQ3_XXS | 88.31% | 0.123 | 1.71 | 3.959 | 50.6 GiB |
| UD-IQ4_XS, three shards (`38bb39ee`) | 88.75% | 0.119 | 1.62 | 4.114 | 59.8 GiB |
| UD-IQ3_XXS, three shards (`38bb39ee`) | 85.51% | 0.186 | 2.58 | 4.106 | 50.2 GiB |

Per segment (top-1 / mean KL):

| File | Code | Docs | English | Multilingual | Agent |
| --- | ---: | ---: | ---: | ---: | ---: |
| UD-IQ4_XS | 95.50% / 0.057 | 81.18% / 0.132 | 88.69% / 0.055 | 93.89% / 0.034 | 87.89% / 0.203 |
| UD-IQ3_XXS | 92.99% / 0.082 | 79.38% / 0.176 | 88.39% / 0.066 | 92.79% / 0.046 | 87.99% / 0.245 |

Perplexity below the reference's does not mean either file beats UD-Q4_K_XL;
the reference is itself quantized. Both UD 3.1 files agree with it more
closely than the three-shard UD-IQ4_XS did; the three-shard UD-IQ3_XXS, measured
on November 2 with the same tool and reference, is the furthest from it.

## Gufo against the CPU F32 reference

`qwen38_flash_next_gpu_probe --reference` runs the GPU executor and the CPU F32
reference forward (independent decoders in `src/core/quant`) on the same
24-token prompt, at `e9d4e5cd`:

| File | Top-1 | Mean KL | Max KL |
| --- | ---: | ---: | ---: |
| UD-IQ4_XS | 22/24 | 0.0037 | 0.0217 |
| UD-IQ3_XXS | 23/24 | 0.0074 | 0.0503 |
| UD-IQ4_XS, three shards | 24/24 | 0.0188 | 0.1008 |
| UD-IQ3_XXS, three shards | 23/24 | 0.0298 | 0.2376 |

The UD-IQ4_XS differences are near ties (KL 0.0014 and 0.0123 at those
positions). The three-shard UD-IQ3_XXS picks " a" where the reference picks
" the" (KL 0.129). The reference rounds weights the same way as the GPU, so this
checks execution, not the quantization.

## Kernels

| Test | Checks |
| --- | --- |
| `qwen38_flash_next_iq_ops_test` | IQ2_S/IQ3_XXS/Q2_0/Q5_0 vector kernels (single and grouped rows, gated pairs, wave64 batches), Q6_K and IQ4_XS embedding rows, against the CPU decoders. |
| `qwen38_flash_next_routed_wmma_ops_test` | F16 WMMA expert GEMMs for every gate/up and down format against an F64 reference; 336 route comparisons of the paired 128-token gate/up and the 128-token down projection against the 16- and 48-row tiles, bit for bit, with random-sign scales for Q8_0/IQ4_NL/Q2_0/Q5_0 down. |
| `qwen38_flash_next_projection_ops_test` | Q6_K dense F16 GEMM at 640/10,240/16,384/13,312 rows, 2,560×6,144 and 201-token passes, against F64; fused Q6_K SSM and attention projections equal the separate ones across pass widths. |
| `qwen38_flash_next_probe_test` | The probe claims the new expert formats and still declines Q6_K experts. |
| `ggml_dequant_test` | Q2_0 dequantization hash equals ggml's `to_float`. |

## Model tests

On both files with the shared MTP predictor and the vision projector, at
`e9d4e5cd`:

- `qwen38_flash_next_session_test`: graph capture and replay, mixed batches,
  prefill chunk boundaries, MTP cache replay, rollback, image snapshots,
  AR/MTP logits and RNG at C2/4/6/8, lookup and draft vocabulary pass.
  The lookup check ("lookup changed tokens, RNG or proposals with batch
  peers") failed once in 12 runs on UD-IQ3_XXS and passed in the other 11;
  unmodified Gufo 0.33.0 passed 6 of 6 on UD-Q4_K_XL. Unresolved.
- `qwen38_flash_next_snapshot_test` and
  `qwen38_flash_next_continuation_adapter_test` (AR and MTP, including images
  and image restore) pass.

## Functional suites

`tests/functional/run.py --record-baseline` with the release binary at
`198892f1`, `--sessions 4`, the vision projector and the `qwen38` sampling
preset: `state-edges`, `tool-agent`, `long-context`, `tools`, `batch`,
`image-inputs` and `cache` pass on both files with `--speculative mtp`
(loaded mode `mtp`) and without it (`off`). `e9d4e5cd` changes only Q6_K
prefill decode, bit-exact. These record correctness only, not timings.
Once, on UD-IQ3_XXS, the greedy model answered `legacy-tool-cancel` with a
well-formed tool call instead of text; later runs passed.

On the older three-shard files, at `e9d4e5cd` with the same options, the
seven suites pass on UD-IQ3_XXS with MTP (3,772 drafted, 3,234 accepted) and
without it, and on UD-IQ4_XS with MTP (3,760 drafted, 3,102 accepted).

## UD-Q4_K_XL unchanged

At `e9d4e5cd`, against Gufo 0.33.0 (`77dff828`), UD-Q4_K_XL prefill logits of
a 1,552-token prompt and 358 matched decode steps are byte-identical
(`qwen38_flash_next_gpu_probe --dump`).

## Reproduce

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target \
  qwen38_flash_next_iq_ops_test qwen38_flash_next_routed_wmma_ops_test \
  qwen38_flash_next_projection_ops_test qwen38_flash_next_probe_test \
  qwen38_flash_next_model_tests qwen38_flash_next_gpu_probe ggml_dequant_test
nix develop -c ctest --preset gpu-full \
  -R 'qwen38_flash_next\.(iq_ops|routed_wmma_ops|projection_ops|probe)$|^ggml_dequant_test$' \
  --output-on-failure
T=/path/to/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf  # mmproj beside it
MTP=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
B=build/gpu-test/tests/models/qwen38_flash_next
$B/qwen38_flash_next_gpu_probe --model "$T" --reference --prompt "..."
$B/qwen38_flash_next_session_test --model "$T" --mtp-model "$MTP"
$B/qwen38_flash_next_snapshot_test --model "$T" --mtp-model "$MTP"
$B/qwen38_flash_next_continuation_adapter_test --model "$T" --mtp-model "$MTP"
```

## Benchmark method

`tools/bench/model-bench.py --model qwen3.8-flash-next-ud-iq` with
`artifacts/bench.json`: the UD-Q4_K_XL card's tables and workloads, with one
variant per file, one warmed sample per point, one server at a time and
nothing else on the GPU. The machine is a 96 GB GMKtec Evo-X2 (Ubuntu 24.04,
`powersave` governor, EPP `balance_performance`); its GPU reaches 90–100 °C
under sustained load. Gufo is a `nix build` of `e9d4e5cd`. Cold files for
the loading row use `POSIX_FADV_DONTNEED` verified with `mincore`, without
root; runtime libraries may be warm.

The llama.cpp reference is Gufo's pinned recipes (`.#llama-cpp-reference` at
`68d9053a`, `.#llama-cpp-mtp-reference` at `6fcaa16f`) with one local change:
`.devops/nix/llama-cpp-reference.nix` passes the gfx1151 `rocmPackages` to
`llama-cpp.override`. Without it the recipe builds hipBLASLt for every GPU
target, which did not fit in this machine's memory. Its server context is
35,456 through 32K, 68,224 at 64K and 133,760 at 128K, so its KV cache fits
beside the weights; Gufo uses 133,760 throughout.

Every Gufo multiple-user completion, AR and MTP at C1–C8 on both files,
matches its Gufo AR C1 reference. Single-user AR and MTP completions match at
depth 0; deeper rows compare prompts whose cached prefix differs by one token
(for example 4,095 and 4,096), as in the UD-Q4_K_XL card. The llama.cpp MTP
server's completions differ from its own AR completions at matched token
counts, and its AR completions change with the number of users; as for
UD-Q4_K_XL, its speeds are not qualified equivalence measurements.

### Memory fit

One Gufo `e9d4e5cd` session, one image request, then a synthetic prompt
filling the context and 128 output tokens; memory sampled every second with
`free` and `mem_info_gtt_used`:

| File, setup | Context | Prompt tokens | Prefill (tok/s) | Decode (tok/s) | GTT after fill | Lowest available |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UD-IQ4_XS, MTP, images | 262,144 | 258,189 | 1,523.9 | 45.59 | 74.1 GiB | 11.6 GiB |
| UD-IQ3_XXS, MTP, images | 262,144 | 258,189 | 1,470.7 | 44.15 | 64.8 GiB | 17.9 GiB |
| UD-IQ3_XXS, MTP, images | 196,608 | 193,306 | 1,515.1 | 45.01 | 63.1 GiB | 20.5 GiB |
| UD-IQ3_XXS, MTP, images | 131,072 | 128,330 | 1,571.2 | 43.35 | 61.4 GiB | 23.0 GiB |
| UD-IQ3_XXS, MTP, images | 65,536 | 63,318 | 1,643.6 | 42.13 | 59.7 GiB | 26.3 GiB |
| UD-IQ3_XXS, MTP, images | 32,768 | 30,918 | 1,678.7 | 43.92 | 58.9 GiB | 28.4 GiB |
| UD-IQ3_XXS, AR | 131,072 | 128,330 | 1,644.6 | 33.85 | 57.3 GiB | 27.7 GiB |
| UD-IQ4_XS, MTP | 32,768 | 30,918 | 1,753.6 | 39.48 | 67.2 GiB | 20.4 GiB |

Swap did not grow, and both images were described as red and blue.
Available memory includes Gufo's automatic RAM snapshot cache (half of free
RAM, at most 32 GiB). The UD-IQ4_XS 262K prompt reused 4,096 cached tokens.
For a 64 GB machine, a separate process held 32 GiB of GTT with `hipMalloc`,
leaving 60.5 GiB available; UD-IQ3_XXS AR at context 8,192 was killed by the
kernel OOM killer while loading. The test stopped there, since the OOM killer
also ended unrelated user processes.

### Host memory during the tables

During the llama.cpp UD-IQ4_XS steps the host had 0.5–4 GiB available, 4–8
GiB in swap and sustained major page faults; during its UD-IQ3_XXS steps and
all Gufo steps, swap stayed below 0.2 GiB. Sampled every 10 s with `free`
and `/proc/vmstat`.
