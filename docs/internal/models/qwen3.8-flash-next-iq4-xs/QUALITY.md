# Qwen3.8 Flash-Next UD-IQ4_XS quality

Two kinds of evidence: how much the UD-IQ4_XS file loses against UD-Q4_K_XL
(measured with llama.cpp, independent of Strycast), and whether Strycast
executes the file correctly (kernel tests, the CPU F32 reference, the model
tests and the functional suites). Original-model and GGUF-conversion parity
are not qualified here, as for UD-Q4_K_XL.

## The file against UD-Q4_K_XL

Next-token distributions of llama.cpp ROCm `b11557` on the 4,995 positions of
Strata's `data/quant_quality/v1` set (commit `c62d5e3a`), with Strata's
formulas, against UD-Q4_K_XL (no Q8_0 file was available). A deterministic
rerun was byte-identical.

| File | Top-1 agreement | Mean KL | KL p99 | Perplexity |
| --- | ---: | ---: | ---: | ---: |
| UD-Q4_K_XL | reference | reference | reference | 4.025 |
| UD-IQ4_XS | 88.75% | 0.119 | 1.62 | 4.114 |

Strata's own table against Q8_0 gives the same picture: UD-Q4_K_XL 93.09% /
0.056 / 4.068 and UD-IQ4_XS 88.91% / 0.101 / 4.120 (top-1 / KL / perplexity;
Q8_0 4.052).

## Strycast against llama.cpp on the same file

16 greedy prompts (code, maths, French, Japanese, JSON, one of about 11K
tokens), 192 tokens each, measured by the Strycast maintainer. Neither engine
is bit-identical to the other, so the yardstick is their agreement on
UD-Q4_K_XL:

| Pair | Common prefix (words) |
| --- | ---: |
| Strycast vs llama.cpp, UD-Q4_K_XL | 292 / 1196 |
| Strycast vs llama.cpp, UD-IQ4_XS | 288 / 1199 |

The texts are coherent throughout and the arithmetic answers are right on
both engines.

## Strycast against the CPU F32 reference

`qwen38_flash_next_gpu_probe --reference` runs the GPU executor and the CPU
F32 reference forward (with independent IQ3_S/IQ4_XS/IQ4_NL/Q6_K decoders in
`src/core/quant`) on the same 24-token prompt. The GPU argmax equals the
reference at all 24 positions; mean KL 0.019, maximum 0.101.

## Kernels

| Test | Checks |
| --- | --- |
| `qwen38_flash_next_iq_ops_test` | IQ3_S/IQ4_XS/IQ4_NL vector kernels (single and grouped rows, gated pairs, wave64 batches) and the Q6_K head GEMV against the CPU decoders; the load-time layout rules. |
| `qwen38_flash_next_routed_wmma_ops_test` | F16 WMMA expert GEMMs for IQ3_S/IQ4_XS gate/up and IQ4_NL down against an F64 reference; the paired 128-token gate/up and the 128-token down projection against the separate 16- and 48-row tiles, bit for bit, over model shapes (`CheckPassRoutes`: IQ3_S and IQ4_XS gate/up with IQ4_NL and Q8_0 down), including short and ragged passes. |
| `qwen38_flash_next_probe_test` | The probe claims IQ3_S/IQ4_XS/IQ4_NL experts only in layouts the executor runs, and still declines other formats. |

IQ blocks are decoded to signed bytes and take Q8_0's F16 arithmetic in every
routed GEMM, so a prompt row's expert outputs do not depend on the pass that
computes it. For IQ3_S and IQ4_XS these kernels round the combined scale
(`d` times the sub-block scale) to F16, as for Q4_K, while the decode vector
kernels keep it in F32, so prefill and decode do not compute exactly the same
products; IQ4_NL's scale is `d` itself and stays exact. The CPU F32 reference comparison above,
whose prompt is prefilled through these kernels, shows the effect is small:
the same top-1 at all 24 positions, mean KL 0.019.

## Model tests

On UD-IQ4_XS with the shared MTP predictor, at Strycast `274e0ba0`:

- `qwen38_flash_next_session_test`: graph capture and replay, mixed batches,
  prefill chunk boundaries, prefetch hints, MTP cache replay, policy
  snapshots, rollback, image snapshots, AR/MTP logits and RNG at C2/4/6/8
  (`batch C2..C8 AR_MTP_logits_RNG_exact=1`) and lookup all pass. The test
  then stops at the draft-vocabulary check ("the repeated Thai continuation
  has too few rare tokens"), which clean `staging` also fails on UD-Q4_K_XL.
- `qwen38_flash_next_snapshot_test`: passes.
- `qwen38_flash_next_continuation_adapter_test`: passes in AR and MTP
  (boundaries, earlier edits, pending and failed requests, images and image
  restore, device exhaustion), and so does `qwen38_flash_next.in_pass_checkpoints`.

## Functional suites

`tests/functional/run.py --record-baseline` with the release binary,
`--sessions 4`, the vision projector and the shared MTP predictor:
`state-edges`, `tool-agent`, `long-context`, `tools`, `batch`, `image-inputs`
and `cache` pass with `--speculative mtp` (loaded mode `mtp`) and without it
(loaded mode `off`). These record correctness only, not timings.

## UD-Q4_K_XL unchanged

Against `staging` `8c946372`, UD-Q4_K_XL prefill logits of a 1,552-token
prompt and 358 matched decode steps are byte-identical
(`qwen38_flash_next_gpu_probe --dump`).

## Benchmark output consistency

Multi-user AR and MTP completions match the Strycast AR C1 reference at every
concurrency. Single-user AR text (the first repetition, whose prompt MTP
shares) matches MTP at depth 0 and differs deeper with identical token
counts, the same pattern as the published UD-Q4_K_XL artifacts. llama.cpp
MTP matches its own AR on pp2048 C1, fresh and prepared;
deeper, its AR and MTP text differ, so only its depth-0 MTP row is qualified.

## Reproduce

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target \
  qwen38_flash_next_iq_ops_test qwen38_flash_next_routed_wmma_ops_test \
  qwen38_flash_next_probe_test qwen38_flash_next_model_tests qwen38_flash_next_gpu_probe
nix develop -c ctest --preset gpu-full \
  -R 'qwen38_flash_next\.(iq_ops|routed_wmma_ops|probe)$' --output-on-failure
T=/path/to/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
MTP=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
B=build/gpu-test/tests/models/qwen38_flash_next
$B/qwen38_flash_next_gpu_probe --model "$T" --reference --prompt "..."
$B/qwen38_flash_next_session_test --model "$T" --mtp-model "$MTP"
$B/qwen38_flash_next_snapshot_test --model "$T" --mtp-model "$MTP"
```

## Benchmark method

`tools/bench/model-bench.py --model qwen3.8-flash-next-iq4-xs` with
`artifacts/bench.json`, the same tables and workloads as the UD-Q4_K_XL card,
one sample per point (three for the single-user AR Strycast cells, each
with its own prompt), on a 96 GB GMKtec Evo-X2 (Ubuntu 24.04, `powersave`
governor, EPP `balance_performance`). Its GPU reached 97-100 °C under
sustained load with about 2.5% clock loss. Cold files for the loading row use
`POSIX_FADV_DONTNEED` verified with `mincore`, without root.

The llama.cpp reference comes from Strycast's pinned recipes
(`.#llama-cpp-reference` at `68d9053a`, `.#llama-cpp-mtp-reference` at
`6fcaa16f`) with one change: `.devops/nix/llama-cpp-reference.nix` passes
the gfx1151 `rocmPackages` to `llama-cpp.override`. Without it the recipe
builds the all-target hipBLASLt, which did not fit on this machine. The
reference server context was 35456 through 32K, 68224 at 64K and 133760 at
128K. The reference cells were measured on November 2, 2026, driven from
Strycast `a53ec202` with the same benchmark driver. The reference recipes are
unchanged at `ca0547c3`, where the Strycast cells were measured on November
14; the driver changed only its documentation paths.

Against the previous cells (Strycast `274e0ba0`, on Gufo 0.31), three
alternating rounds of both `nix build` binaries on this machine gave
identical completions; `ca0547c3` decodes AR 3% faster, prefills the same
within 1%, and is slower with MTP for several users: mixed text −11% at
C4–C8, repetitive text −5% to −7% at C1–C8
([`artifacts/v033-ab.json`](artifacts/v033-ab.json)). The draft counts are
the same, so the time per verification step grew with the Gufo 0.33 base.
