# Qwen3.8 Flash-Next UD-IQ quality

Two kinds of evidence: how much each file loses against UD-Q4_K_XL, measured
with llama.cpp and independent of Strycast, and whether Strycast executes the
files correctly (kernel tests, the CPU F32 reference, model tests and functional
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

## Strycast against the CPU F32 reference

`qwen38_flash_next_gpu_probe --reference` runs the GPU executor and the CPU F32
reference forward (independent decoders in `src/core/quant`) on the same
24-token prompt, at `f1d18b42`:

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

With the shared MTP predictor and the vision projector, at `f1d18b42`:

- `qwen38_flash_next_session_test` passes on UD-IQ3_XXS and UD-IQ4_XS: graph
  capture and replay, mixed batches, prefill chunk boundaries, MTP cache
  replay, rollback, image snapshots, AR/MTP logits and RNG at C2/4/6/8,
  lookup and the draft vocabulary pass. On the fork, before this port, the
  lookup check failed once in 12 runs on UD-IQ3_XXS; it passed here.
- `qwen38_flash_next_snapshot_test`,
  `qwen38_flash_next_continuation_adapter_test` (AR and MTP, including images
  and image restore), `--in-pass-only` checkpoints and
  `qwen38_flash_next_layered_prefill_test` pass on UD-IQ3_XXS.
- On the three-shard UD-IQ3_XXS, `session_test` passes everything up to the
  draft vocabulary pass, including the batch and lookup exactness checks. It
  stops there: the shared MTP head's drafts of rare Thai tokens are accepted
  8 of 22 times, below the check's half, against 17 of 22 for Dynamic 3.1
  UD-IQ3_XXS on the same prompt. Wrong table IDs would reject all of them, so
  this is the file's agreement with the MTP head, as with the Thai check on
  the [three-shard UD-IQ4_XS](../qwen3.8-flash-next-iq4-xs/QUALITY.md#model-tests).

## Functional suites

`tests/functional/run.py --record-baseline` with the release binary at
`f1d18b42`, `--sessions 4`, the vision projector and the `qwen38` sampling
preset: `state-edges`, `tool-agent`, `long-context`, `tools`, `batch`,
`image-inputs` and `cache` (20 cases) pass on UD-IQ3_XXS with
`--speculative mtp` (loaded mode `mtp`) and without it (`off`), and with MTP
on UD-IQ4_XS and on the three-shard UD-IQ3_XXS. These record correctness
only, not timings.

## Existing files unchanged

At `f1d18b42`, against Strycast staging `ca0547c3`:

- UD-Q4_K_XL logits are byte-identical for all rows of a 1,522-token
  prefill, the same prompt in 250-token chunks, and 65 matched decode steps
  (`qwen38_flash_next_gpu_probe --logit-rows`/`--decode-file --dump`).
- Greedy HTTP completions of four prompts (up to 1,545 prompt tokens, 128
  output tokens) are identical on the three-shard UD-IQ4_XS with MTP and on
  Qwen3.6-35B-A3B UD-Q5_K_M, whose MoE runner shares the routed kernels.
- `tools/ci/check-kernel-resources.py` against the staging binary: 2,938
  kernels, none over budget.

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
variant per file, one warmed sample per point (three for the single-user AR
Strycast cells, each with its own prompt), one server at a time and nothing
else on the GPU. The machine is a 96 GB GMKtec Evo-X2 (Ubuntu 24.04,
`powersave` governor, EPP `balance_performance`); its GPU reached 104 °C
under sustained load. Strycast is a `nix build` of `f1d18b42`, measured on
November 14, 2026. Cold files for the loading row use `POSIX_FADV_DONTNEED`
verified with `mincore`, without root; runtime libraries may be warm.

The llama.cpp reference comes from Strycast's pinned recipes
(`.#llama-cpp-reference` at `68d9053a`, `.#llama-cpp-mtp-reference` at
`6fcaa16f`) with the one change described in the
[UD-IQ4_XS card](../qwen3.8-flash-next-iq4-xs/QUALITY.md#benchmark-method).
Its server context is 35,456 through 32K, 68,224 at 64K and 133,760 at 128K,
so its KV cache fits beside the weights; Strycast uses 133,760 throughout.
The reference cells were measured on November 11, 2026 on this machine with
the same files, driver and `bench.json`, driven from Gufo 0.33.0 with this
IQ code; the reference recipes are the same in Strycast.

Every Strycast multiple-user completion, AR and MTP at C1–C8 on both files,
matches its AR C1 reference. Single-user AR text (the first repetition, whose
prompt MTP shares) matches MTP at depth 0; deeper rows compare prompts whose
cached prefix differs by one token (for example 4,095 and 4,096), as in the
UD-Q4_K_XL card. The llama.cpp MTP server's completions differ from its own
AR completions at matched token counts, and its AR completions change with
the number of users; as for UD-Q4_K_XL, its speeds are not qualified
equivalence measurements.

### Memory fit

One Strycast `f1d18b42` session, one image request, then a synthetic prompt
filling the context and 128 output tokens; memory sampled every second with
`free` and `mem_info_gtt_used`:

| File, setup | Context | Prompt tokens | Prefill (tok/s) | Decode (tok/s) | GTT after fill | Lowest available |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UD-IQ4_XS, MTP, images | 262,144 | 258,189 | 1,537.1 | 44.88 | 74.1 GiB | 11.9 GiB |
| UD-IQ3_XXS, MTP, images | 262,144 | 258,189 | 1,479.4 | 44.33 | 64.8 GiB | 17.8 GiB |
| UD-IQ3_XXS, MTP, images | 196,608 | 193,306 | 1,530.4 | 45.03 | 63.1 GiB | 20.4 GiB |
| UD-IQ3_XXS, MTP, images | 131,072 | 128,330 | 1,586.9 | 43.08 | 61.4 GiB | 22.9 GiB |
| UD-IQ3_XXS, MTP, images | 65,536 | 63,318 | 1,651.3 | 42.18 | 59.7 GiB | 26.5 GiB |
| UD-IQ3_XXS, MTP, images | 32,768 | 30,918 | 1,682.9 | 44.02 | 58.9 GiB | 28.4 GiB |
| UD-IQ3_XXS, AR | 131,072 | 128,330 | 1,641.6 | 33.44 | 57.3 GiB | 27.7 GiB |

Swap did not grow, and every image was described as red and blue.
Available memory includes the automatic RAM snapshot cache (half of free
RAM, at most 32 GiB). A 64 GB machine was not simulated on this branch. On
the fork, with a separate process holding 32 GiB of GTT (60.5 GiB
available), UD-IQ3_XXS AR at context 8,192 was killed by the OOM killer
while loading.

### Host memory during the tables

During the llama.cpp UD-IQ4_XS steps the host had 0.5–4 GiB available, 4–8
GiB in swap and sustained major page faults; during its UD-IQ3_XXS steps
swap stayed below 0.2 GiB, and during all Strycast steps below 0.1 GiB.
Sampled every 10 s with `free` and `/proc/vmstat`.
