# Qwen3.8 Flash-Next UD-IQ4_XS (fork work in progress)

Goal of this fork: run Unsloth's **UD-IQ4_XS** Flash-Next target at Gufo speed on
smaller Strix Halo machines (96 GB unified memory, e.g. GMKtec Evo-X2). Upstream
Gufo supports only **UD-Q4_K_XL**. Smaller quants (for example IQ3_XXS) are a
possible follow-up, not a goal.

Status: **Phase 0 (baseline) complete; no engine code changed yet.** Branch
`feat/qwen-flash-iq4-xs`, based on upstream `main` `2b4daed`. If this works, the
intent is an upstream pull request; quality of the IQ4_XS file itself is already
judged acceptable from day-to-day use with llama.cpp.

See also: [model guide](README.md) · [benchmarks](BENCHMARKS.md) ·
[quality](QUALITY.md) · [experiments](EXPERIMENTS.md).

## What the IQ4_XS file contains

Both quants come from `unsloth/Qwen3.8-Flash-Next-GGUF` at the revision the model
guide already pins, `38bb39ee97821de2c9009abb7e93950eec396e66`. They have the same
1,224 tensors with the same shapes. Attention, SSM, shared experts,
hyper-connections and embeddings are identical; only the routed experts and the
output head differ:

| Tensor | UD-Q4_K_XL (supported) | UD-IQ4_XS | Size |
| --- | --- | --- | --- |
| `ffn_gate_exps`, `ffn_up_exps` | Q4_K (Q5_K on blk.2) | **IQ3_S** (IQ4_XS on blk.2) | 42.4 → 32.4 GiB |
| `ffn_down_exps` | Q5_1 (Q8_0 on 5 layers) | **IQ4_NL** (same 5 layers Q8_0) | 25.2 → 18.9 GiB |
| `output.weight` | Q8_0 | **Q6_K** | 0.63 → 0.49 GiB |
| everything else | identical | identical | 35.4 GiB |
| **total file** | | | 103.7 → 87.2 GiB |

So the work is **IQ3_S** and **IQ4_NL** routed experts plus a **Q6_K** logits
head. True IQ4_XS appears on one layer only. Gate/up experts drop from 4.5 to
3.44 bits per weight.

The 26.8 GiB `per_layer_token_embd` table is read from disk with direct I/O and
never uploaded (`NgramTable`, `ngram.cpp`). GPU-resident weights are therefore:

| Target | GPU-resident weights |
| --- | ---: |
| UD-Q4_K_XL | 76.9 GiB |
| UD-IQ4_XS | 60.4 GiB |

Measured Gufo Q4_K_XL peak below is 80.8 GiB; the same workload on IQ4_XS
should land near 64 GiB, leaving room on a 96 GB machine for long context, MTP
and several sessions.

### Model identities (SHA-256, verified locally)

UD-IQ4_XS (identical on `main` `766911a6`; last file commit `2c41bd2a`):

| Shard | SHA-256 |
| --- | --- |
| 00001-of-00003 | `5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4` |
| 00002-of-00003 | `577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7` |
| 00003-of-00003 | `d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833` |

UD-Q4_K_XL shards match [model-identities.json](artifacts/model-identities.json).

## Phase 0 baseline

October 12, 2026, on a 128 GB Strix Halo **laptop** (Ryzen AI MAX+ 395,
Radeon 8060S `gfx1151`, mains power, ACPI platform profile `performance`).
Gufo `main` `2b4daed` built with `cmake --preset release` (GCC 15.3, ROCm 7.2.4).
llama.cpp `b11557` (`5c8854fd8`), ROCm (`GPU_TARGETS=gfx1151`) and Vulkan
(RADV, Mesa 26.2.4) builds. `llama-bench -ngl 999 -fa on -lm dio -p 2048 -n 128
-d 0,16384 -r 2`; `gufo bench -p 2048 -n 128 -d 0,16384 -r 2`. Two repetitions,
mean ± deviation as printed by each tool.

| Engine | Target | pp2048 | tg128 | pp2048 @ d16K | tg128 @ d16K |
| --- | --- | ---: | ---: | ---: | ---: |
| Gufo `main` | Q4_K_XL | **1469.61** ± 4.79 | 26.32 ± 1.24 | **1239.06** ± 4.50 | 26.71 ± 0.35 |
| llama.cpp ROCm | Q4_K_XL | 504.15 ± 3.81 | 20.04 ± 0.01 | 391.86 ± 3.62 | 18.76 ± 0.02 |
| llama.cpp ROCm | IQ4_XS | 488.81 ± 2.74 | 20.64 ± 0.01 | 378.63 ± 3.91 | 19.41 ± 0.05 |
| llama.cpp Vulkan | IQ4_XS | 489.23 ± 8.14 | **27.28** ± 0.06 | 357.72 ± 0.73 | **26.03** ± 0.05 |

Gufo Q4_K_XL: model load 12.8 s; peak GPU memory 80.8 GiB (VRAM + GTT in use,
sampled every 2 s, includes the desktop session).

Observations:

- This laptop reaches 86% of the published desktop Gufo prefill (1,702.93) and
  94% of decode (27.98). It sustains about 60 W under load (80 W bursts), so
  absolute numbers are machine-specific. Compare branches only within one
  session; take headline numbers on the 96 GB desktop.
- llama.cpp is no slower on IQ4_XS than on Q4_K_XL, so the IQ formats are not
  inherently slow on this GPU.
- **llama.cpp Vulkan already decodes IQ4_XS at about Gufo's Q4_K_XL rate.**
  Gufo's clear lead is prefill (about 3×), plus MTP, concurrency and caching.
  Decode must at least match Vulkan; reading about 0.5 GB less per token than
  Q4_K_XL suggests Gufo IQ4_XS AR decode could exceed it slightly if the IQ3_S
  kernels stay bandwidth-bound.
- Prefill is where most of the kernel work and the risk sit: the WMMA routed
  expert GEMM needs IQ3_S and IQ4_NL dequantization to keep the 3× lead.

## Plan

Q4_K_XL code paths must stay byte-identical: new formats are added as new cases,
and existing Flash-Next operator tests plus logit dumps on Q4_K_XL guard them.
Every new format must follow the existing exactness contracts, in particular that
MTP verification uses the same quantization and reduction as single-token decode.

### Phase 1: correct, not yet fast

1. Accept IQ3_S, IQ4_XS and IQ4_NL routed experts and a Q6_K head in
   `weights.cpp`; add the missing CPU dequantization cases in `cpu_ops.cpp`
   (the decoders already exist in `src/core/quant/ggml_dequant.cpp`).
2. Native Q6_K logits head in `Dense`/`DenseBatch`; lift the Q6_K rejection in
   `device_model.cpp`. The vendored `vec_dot_q6_K_q8_1` and MMQ loaders exist.
3. Decode and medium-batch routed experts: add the formats to the MMVQ
   dispatch (`mmvq.hip.cpp`), the vector-projection checks and the tiled MMQ
   entry points (`qfn_mmq.hip.cpp`). The vendored llama.cpp code already has
   their dot products, tile loaders and lookup tables.
4. Prefill falls back to the tiled MMQ path (`wmma_experts` false).
5. Tests: extend `routed_wmma_ops_test` with IQ3_S/IQ4_NL/IQ4_XS generators
   against a floating-point reference; add a Q6_K projection check.
6. Gate: the model loads; GPU logits match Gufo's CPU reference; greedy output
   and top-1 agreement are close to llama.cpp on the same file.

At this point, open an upstream issue to confirm interest and the evidence they
expect before the larger kernel work.

### Phase 2: speed

1. `RoutedF16GEMMKernel`: IQ4_NL (down; 32-element blocks, a 16-entry lookup)
   and IQ3_S (gate/up; 256-element superblocks with a 512-entry grid in LDS,
   sign bits and 4-bit scales), single and paired gate/up. Keep the grid table a
   template parameter so IQ3_XXS could reuse it.
2. Tune IQ3_S decode, the grouped MTP path and the wave64 batched path
   (`moe_wave64`), which is Q4_K-only today.
3. Profile with `tools/prof`, following the optimize-kernel skill; compare against
   Gufo Q4_K_XL and llama.cpp IQ4_XS (ROCm and Vulkan).

### Phase 3: qualification

Functional suites (`state-edges`, `tool-agent`, `long-context`, `cache`; AR and
MTP) on IQ4_XS; MTP acceptance versus Q4_K_XL; IQ4_XS BENCHMARKS/QUALITY entries
measured on the 96 GB desktop; no Q4_K_XL regression.

## Development setup notes

- **Laptop power profile.** On `balanced` the GPU stayed at 600 MHz and 15 W
  under load, and every engine ran about 3× slower. Select
  `performance` before measuring and record clocks/power with each run:
  `echo performance | sudo tee /sys/firmware/acpi/platform_profile`.
- **Compiler.** GCC 16's `<format>` uses `[[__gnu__::__noinline__]]`, which
  ROCm's `hip/amd_detail/host_defines.h` macro `__noinline__` breaks. Build with
  GCC 15 (upstream qualifies 15.3): `CC=gcc-15 CXX=g++-15 cmake --preset release`.
- **Dependencies (Arch).** `rocwmma` headers are needed besides the packages in
  the [build instructions](../../../README.md#without-nix).
- **llama.cpp loading.** `llama-bench` replaced `-mmp 0` with `--load-mode`; use
  `-lm dio`. It keeps `per_layer_token_embd` on the CPU and the rest (61.2 GiB)
  on the GPU, with two graph splits.
- **Nix** is not installed on the laptop. Upstream evidence for a pull request
  should come from their pinned `nix build`.

## Open questions

- Gufo IQ4_XS peak memory and admitted context on the 96 GB desktop, including
  its GTT limit and host needs (OS, RAM prompt cache).
- Which llama.cpp backend the 96 GB desktop uses today, for its own baseline.
- MTP acceptance with the IQ3_S trunk.
