# Qwen3.8 Flash-Next GSQ-RCO quality

Two kinds of evidence: how much the file loses against UD-Q4_K_XL, measured
with llama.cpp and independent of Gufo, and whether Gufo executes the file
correctly (kernel tests, the CPU F32 reference, model tests and functional
suites). Original-model and GGUF-conversion parity are not qualified, as for
UD-Q4_K_XL.

## The file against UD-Q4_K_XL

Teacher-forced next-token distributions from llama.cpp ROCm `b11557`
(`5c8854fd`) on the 4,995 positions of Strata's `data/quant_quality/v1` set
(commit `c62d5e3a`): the top 64 log-probabilities per position from
`llama-server`, compared with Strata's formulas (top-1 agreement,
KL(reference‖file) over the reference's 64 most likely tokens, both
renormalized, a token missing from the file's list taking its lowest listed
log-probability). The reference is UD-Q4_K_XL. This pipeline reimplements
Strata's `compare` step. UD 3.1 UD-IQ3_XXS, rerun with it on the same day,
gives 88.37% / 0.123 / p99 1.83 against its
[published](../qwen3.8-flash-next-ud-iq/QUALITY.md#the-files-against-ud-q4_k_xl)
88.31% / 0.123 / 1.71; compare the two files within one pipeline.
Perplexity was not measured.

| File | Top-1 | Mean KL | KL p99 | llama.cpp GPU weights |
| --- | ---: | ---: | ---: | ---: |
| GSQ-RCO IQ3_XXS | 84.48% | 0.201 | 2.46 | 43.5 GiB |
| UD 3.1 UD-IQ3_XXS | 88.37% | 0.123 | 1.83 | 50.6 GiB |

Per segment (top-1 / mean KL):

| File | Code | Docs | English | Multilingual | Agent |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSQ-RCO IQ3_XXS | 90.89% / 0.152 | 73.37% / 0.285 | 83.28% / 0.139 | 90.69% / 0.068 | 84.18% / 0.358 |
| UD 3.1 UD-IQ3_XXS | 93.49% / 0.086 | 78.78% / 0.165 | 89.29% / 0.063 | 92.79% / 0.043 | 87.49% / 0.257 |

GSQ-RCO is further from UD-Q4_K_XL than UD-IQ3_XXS in every segment, most
in agent text and documentation. Summary:
[`artifacts/quant-quality.json`](artifacts/quant-quality.json).

## Gufo against the CPU F32 reference

`qwen38_flash_next_gpu_probe --reference` runs the GPU executor and the CPU F32
reference forward (independent decoders in `src/core/quant`) on the same
24-token prompt, at `a7d2edd2`: top-1 22/24, mean KL 0.0081, max KL 0.059.
32 greedy decode steps after a 56-token prompt, compared with the reference
teacher-forced on the same tokens over its top 64 logits: top-1 32/32, mean KL
0.0043 (`ee2c7ead`; the later commits are bit-identical). The
reference rounds weights the same way as the GPU, so this checks execution,
not the quantization.

## Kernels

| Test | Checks |
| --- | --- |
| `qwen38_flash_next_iq_ops_test` | Dense GEMVs and identity-routed F16 GEMMs for Q4_K, Q5_K, IQ4_XS, IQ3_S (K 2,560 and 6,144), IQ4_NL and Q2_0 (K 640) against the CPU decoders; 48- and 128-token tiles equal and chunk-invariant; IQ3_S embedding rows; BF16 GEMMs chunk-invariant at 1,100 tokens for both HC shapes; the fused HC mixer and the combine's inject partials byte for byte against the separate kernels. |
| `qwen38_flash_next_routed_wmma_ops_test` | IQ2_XXS and IQ2_XS gate/up routes against an F64 reference and against each other across tile widths. |
| `qwen38_flash_next_projection_ops_test` | Split BF16 decode GEMVs at 320×10,240 and 10,240×320. |
| `qwen38_flash_next_probe_test` | The probe claims IQ2_XXS and IQ2_XS experts. |

## Model tests

With the shared MTP predictor and the vision projector, at `a7d2edd2`:

- `qwen38_flash_next_session_test`: graph capture and replay, mixed batches,
  prefill chunk boundaries, MTP cache replay, rollback, image snapshots,
  AR/MTP logits and RNG at C2/4/6/8, lookup and draft vocabulary pass, with
  the Thai check below skipped.
- `qwen38_flash_next_snapshot_test` and
  `qwen38_flash_next_continuation_adapter_test` (AR and MTP, including images
  and image restore) pass.

`session_test` requires a repeated Thai continuation to keep at least a third
rare tokens. GSQ-RCO keeps repeating a common token instead, and the CPU F32
reference does the same (token 271 for 60% of the continuation, against 220
for 33% on UD 3.1 UD-IQ3_XXS). This is the file's behavior, so that check is
skipped in a local build; the test's other checks pass.

## Functional suites

`tests/functional/run.py --record-baseline`, release binary at `a7d2edd2`,
`--sessions 4`, the vision projector and the `qwen38` sampling preset:
`state-edges`, `tool-agent`, `long-context`, `tools`, `batch`,
`image-inputs` and `cache` pass with `--speculative mtp` (loaded mode
`mtp`) and without it (`off`). These record correctness only, not timings.

### Tool-agent sampling

`tool-agent`'s sampled peers case (seed 41, temperature 0.7) sometimes reads
a file instead of editing it. Over 30 seeds of that request at `4744f2c2`:

| Engine, file | Edits |
| --- | ---: |
| Gufo, GSQ-RCO, MTP, before `4744f2c2` | 18/30 |
| Gufo, GSQ-RCO, MTP | 17/30 |
| Gufo, GSQ-RCO, AR | 11/30 |
| Gufo, UD 3.1 UD-IQ3_XXS, MTP | 30/30 |
| llama.cpp `b11557`, GSQ-RCO | 30/30 |

The decision is one token after `<tool_call>\n<function`, `=` (then the edit
tool) against `=read`:

| Engine, file | `=` | `=read` |
| --- | ---: | ---: |
| Gufo, GSQ-RCO | 0.58 | 0.42 |
| CPU F32 reference, GSQ-RCO | 0.685 | 0.315 |
| llama.cpp, GSQ-RCO | 0.92 | 0.077 |
| Gufo, UD 3.1 UD-IQ3_XXS | 0.97 | 0.03 |
| llama.cpp, UD 3.1 UD-IQ3_XXS | 0.89 | 0.10 |

On GSQ-RCO the choice is close to even in Gufo and in Gufo's own F32
reference, so sampling takes either branch; llama.cpp's different arithmetic
favors the edit. On plain text the engines agree closely (decode KL 0.003
Gufo against llama.cpp, 0.005 against the F32 reference). This is recorded
as a limitation of the file at this point, not a kernel failure.

## Other files

UD 3.1 UD-IQ3_XXS `session_test` passed at `4f504c6c`. Its greedy output was
byte-identical before and after `ee2c7ead` (AR and MTP) and `bbfc1e82` (AR),
the two tuning changes whose kernels the UD files also run.

At `a7d2edd2`, against Gufo 0.33.0 (`77dff828`), UD-Q4_K_XL logits are
byte-identical for all rows of a 1,522-token prefill, the same prompt in
250-token chunks, and 65 matched decode steps
(`qwen38_flash_next_gpu_probe --logit-rows`/`--decode-file --dump`).

## Reproduce

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target \
  qwen38_flash_next_iq_ops_test qwen38_flash_next_routed_wmma_ops_test \
  qwen38_flash_next_projection_ops_test qwen38_flash_next_probe_test \
  qwen38_flash_next_model_tests qwen38_flash_next_gpu_probe
nix develop -c ctest --preset gpu-full \
  -R 'qwen38_flash_next\.(iq_ops|routed_wmma_ops|projection_ops|probe)$' \
  --output-on-failure
T=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf  # mmproj beside it
MTP=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
B=build/gpu-test/tests/models/qwen38_flash_next
$B/qwen38_flash_next_gpu_probe --model "$T" --reference --prompt "..."
$B/qwen38_flash_next_session_test --model "$T" --mtp-model "$MTP"
$B/qwen38_flash_next_snapshot_test --model "$T" --mtp-model "$MTP"
$B/qwen38_flash_next_continuation_adapter_test --model "$T" --mtp-model "$MTP"
```

## Benchmark method

`tools/bench/model-bench.py --model qwen3.8-flash-next-gsq-rco` with
`artifacts/bench.json`: the UD-IQ card's tables and workloads with one
variant, one warmed sample per point, one server at a time and nothing else
on the GPU. The machine is a 96 GB GMKtec Evo-X2 (Ubuntu 24.04,
`powersave` governor, EPP `balance_performance`); its GPU reached 38–102 °C.
Gufo is a `nix build` of `a7d2edd2`. Cold files for the loading row use
`POSIX_FADV_DONTNEED` verified with `mincore`, without root; runtime
libraries may be warm.

The llama.cpp reference is the same build as for the
[UD-IQ card](../qwen3.8-flash-next-ud-iq/QUALITY.md#benchmark-method)
(`68d9053a` AR, `6fcaa16f` MTP). Its server context is 35,456 through 32K,
68,224 at 64K and 133,760 at 128K; Gufo uses 133,760 throughout. Host swap
stayed at 0.1 GiB for both engines; available memory fell to 8.6 GiB during
llama.cpp's MTP runs. Sampled every 10 s with `free` and `/proc/vmstat`.

Every Gufo multiple-user completion, AR and MTP at C1–C8, matches its Gufo AR
C1 reference. Single-user AR and MTP completions match at depth 0 and 64K;
the other depths compare prompts whose cached prefix differs by one token, as
in the UD-Q4_K_XL card. The llama.cpp AR and mixed-text MTP completions
change with the number of users; as for UD-Q4_K_XL, its speeds are not
qualified equivalence measurements.

The same Gufo binary on UD 3.1 UD-IQ3_XXS gave pp 1,791.6 and tg 36.0 tok/s
at depth 0 ([`artifacts/ud31-control/`](artifacts/ud31-control/); published
at `e9d4e5cd`: 1,821 and 34.4).

### Memory fit

One Gufo `a7d2edd2` session, one image request, then a synthetic prompt
filling the context and 128 output tokens; memory sampled every second with
`free` and `mem_info_gtt_used`:

| Setup | Context | Prompt tokens | Prefill (tok/s) | Decode (tok/s) | GTT after fill | Lowest available |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP, images | 262,144 | 258,189 | 1,302.6 | 45.04 | 57.7 GiB | 24.7 GiB |
| MTP, images | 196,608 | 193,306 | 1,342.9 | 45.52 | 56.1 GiB | 27.6 GiB |
| MTP, images | 131,072 | 128,330 | 1,384.8 | 44.75 | 54.4 GiB | 30.1 GiB |
| MTP, images | 65,536 | 63,318 | 1,435.1 | 44.75 | 52.7 GiB | 33.5 GiB |
| MTP, images | 32,768 | 30,918 | 1,464.8 | 46.93 | 51.9 GiB | 35.3 GiB |
| AR | 131,072 | 128,330 | 1,428.7 | 33.49 | 50.3 GiB | 34.6 GiB |

Swap did not grow, and every image was described as red and blue. At each
context this is 7.0–7.1 GiB less GTT than UD 3.1 UD-IQ3_XXS
([its fit](../qwen3.8-flash-next-ud-iq/QUALITY.md#memory-fit)). Available
memory includes Gufo's automatic RAM snapshot cache (half of free RAM, at
most 32 GiB). A 64 GB machine was not simulated.
