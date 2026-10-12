# Qwen3.8 Flash-Next UD-IQ4_XS (fork work in progress)

Goal of this fork: run Unsloth's **UD-IQ4_XS** Flash-Next target at Gufo speed on
smaller Strix Halo machines (96 GB unified memory, e.g. GMKtec Evo-X2). Upstream
Gufo supports only **UD-Q4_K_XL**. Smaller quants (for example IQ3_XXS) are a
possible follow-up, not a goal.

Status: **runs on a 96 GB Strix Halo (GMKtec Evo-X2) at the model's full
262K context with MTP and vision, from a published container image; on that
machine it matches upstream's published Q4_K_XL prefill and is 6-16× faster
to prefill and up to 2.6× faster to decode than llama.cpp on the same file.**
Branch `feat/qwen-flash-iq4-xs`, based on upstream `main` `2b4daed`. If this works, the
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

UD-Q4_K_XL shards and the shared Q8_0 MTP sidecar
(`5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6`) match
[model-identities.json](artifacts/model-identities.json). The vision projector
`mmproj-BF16.gguf` at the same revision:
`2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355`.

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

## Phase 1 results

October 12, 2026, same laptop and settings as the baseline (GCC 15.3, ROCm 7.2.4,
`performance` profile). Release build of the branch.

| Engine | Target | pp2048 | tg128 | pp2048 @ d16K | tg128 @ d16K |
| --- | --- | ---: | ---: | ---: | ---: |
| Gufo branch | IQ4_XS | **909.70** ± 53.81 | **27.55** ± 1.17 | **823.74** ± 0.80 | **26.43** ± 0.59 |
| Gufo `main` | Q4_K_XL | 1469.61 ± 4.79 | 26.32 ± 1.24 | 1239.06 ± 4.50 | 26.71 ± 0.35 |
| llama.cpp Vulkan | IQ4_XS | 489.23 ± 8.14 | 27.28 ± 0.06 | 357.72 ± 0.73 | 26.03 ± 0.05 |

IQ4_XS: model load 9.8–10.6 s; peak GPU memory 63.7–64.2 GiB (prompt and
benchmark runs), against 80.8 GiB for Q4_K_XL.

What changed:

- Loader and CPU reference accept IQ3_S, IQ4_XS and IQ4_NL routed experts.
- Native Q6_K logits head: a one-wave-per-row MMVQ kernel for 1–8 rows, applied
  eight rows at a time for wider heads, so decode, verification and batched
  heads share each row's arithmetic. Q6_K stays rejected for embeddings and
  experts.
- Decode (fused gate/up for 1–8 tokens), medium batches (paired vector up to 32
  rows) and tiled MMQ (prefill and wide batches) for the three expert formats,
  from the vendored llama.cpp code.
- Q4_K_XL routing is unchanged: new formats only add cases.

Checks:

- `qwen38_flash_next_iq_ops_test` (new): every format within 0.3–0.5% of an
  F64 dot product over the CPU dequantizers (vector, paired and MMQ paths);
  batched rows, paired gate/up, fused gate/up at widths 1–8 and the Q6_K head
  at widths 1–8 are bit-identical to single-row decode; inactive experts write
  zeros. `routed_wmma_ops` and `projection_ops` still pass.
- **Q4_K_XL regression:** `gpu_probe` logits from the branch and from `main`
  are byte-identical (SHA-256) for a 779-token prefill and 382 token-by-token
  decode steps.
- GPU prefill versus the F32 CPU oracle of the same file, 26-token prompt:
  top-1 agreement 25/26 for IQ4_XS and for Q4_K_XL (each one flip at a near
  tie). Mean/median KL 0.0268/0.0063 for IQ4_XS, 0.0177/0.0021 for Q4_K_XL.
  The likely difference: IQ4_XS prefill currently uses MMQ with 8-bit
  activations, Q4_K_XL the WMMA path with F16 activations. Recheck after
  Phase 2.
- Greedy output on a short prompt tracks llama.cpp (ROCm) on the same file:
  identical for about 30 tokens, then a near-tie wording change with the same
  structure.

## Phase 2 results

October 12, 2026, same laptop, `performance` profile, release builds, every
comparison against Q4_K_XL measured in the same session.

### Changes

- **WMMA prefill** (`RoutedF16GEMMKernel`): IQ3_S, IQ4_XS and IQ4_NL are
  decoded to signed bytes when a stage is written to LDS (IQ3_S grid in LDS)
  and then reuse the Q8_0 matrix path unchanged. Paired gate/up (SwiGLU in the
  epilogue) runs for IQ as for Q4_K.
- **Batched decode**: IQ3_S/IQ4_XS gate/up rows of a request batch are grouped
  by expert (the wave64 kernel, now templated on the format); IQ3_S fragments
  are decoded once and reused for every grouped row. IQ4_NL down slots take
  the expert-ordered, 4/8-row tiles Q5_1 uses.
- Q4_K_XL code paths are untouched: new formats only add cases or template
  instances.

### Speed

| Single user (`gufo bench`, tok/s) | IQ4_XS | Q4_K_XL |
| --- | ---: | ---: |
| pp2048 | 1441.74 ± 5.03 | 1488.09 ± 9.90 |
| tg128 | 27.81 ± 0.69 | 27.64 ± 0.47 |
| pp2048 @ d16K | 1273.34 ± 0.87 | 1329.53 ± 0.41 |
| tg128 @ d16K | 26.98 ± 0.90 | 26.73 ± 0.27 |

Single user over HTTP (`tools/bench/model-bench.py`, pp2048 / tg128, greedy,
thinking off, one sample per point; IQ4_XS / Q4_K_XL):

| Mode | Depth | pp (tok/s) | tg (tok/s) | MTP acceptance (%) |
| --- | ---: | ---: | ---: | ---: |
| AR | 0 | 1406.65 / 1410.63 | 28.94 / 28.16 | — |
| AR | 16,384 | 1194.41 / 1209.22 | 28.75 / 28.00 | — |
| MTP mixed | 0 | 1365.65 / 1356.38 | 36.59 / 33.40 | 58.1 / 50.0 |
| MTP mixed | 16,384 | 1163.45 / 1188.02 | 39.43 / 36.03 | 56.6 / 58.1 |
| MTP repetitive | 0 | 1378.42 / 1397.95 | 69.65 / 62.39 | 89.0 / 88.6 |
| MTP repetitive | 16,384 | 1171.55 / 1204.67 | 50.94 / 44.56 | 74.2 / 74.3 |

Multiple users, HTTP benchmark driver (`tools/bench/model-bench.py`, pp2048 /
tg128, context 4096 per user, one sample per point; sum of per-request decode
rates, tok/s, IQ4_XS / Q4_K_XL):

| Users | AR | MTP mixed | MTP repetitive |
| ---: | ---: | ---: | ---: |
| 1 | 29.07 / 28.12 | 36.49 / 33.02 | 67.17 / 61.31 |
| 2 | 51.23 / 49.94 | 62.74 / 55.15 | 84.69 / 94.02 |
| 4 | 83.77 / 81.89 | 82.06 / 81.63 | 129.54 / 131.67 |
| 6 | 103.16 / 102.28 | 98.01 / 95.15 | 143.79 / 137.60 |
| 8 | 113.90 / 112.11 | 111.32 / 110.72 | 140.55 / 152.68 |

MTP rates also depend on draft acceptance, which differs between the two
trunks; repetitive-text rows scatter in both directions.

Kernel times at real shapes (2048-token prefill, top-10 of 512 experts;
decode: 1-64 rows), release build, throwaway harness over the production entry
points:

| Kernel | Q4_K_XL | IQ4_XS |
| --- | ---: | ---: |
| prefill gate/up, paired (ms/layer) | 7.04 (Q4_K) | 8.85 (IQ3_S) |
| prefill down (ms/layer) | 4.30 (Q5_1) | 4.11 (IQ4_NL) |
| decode gate/up, 1 row (µs) | 55-58 | 88-100 |
| decode gate/up, 8 rows, same experts (µs) | 191-204 | 202 (was 752) |
| decode gate/up, 64 rows, random experts (µs) | 3329 | 3333 (was 4623) |
| decode down, 64 rows, random experts (µs) | 2107 | 1654 (was 2578) |

### Quality and exactness

- `gpu_probe --reference`, 26-token prompt against the F32 CPU oracle of the
  same file: IQ4_XS 26/26 top-1, mean/median/max KL 0.0153/0.0035/0.079
  (Phase 1 MMQ fallback: 25/26, 0.0268/0.0063/0.151; Q4_K_XL: 25/26,
  0.0177/0.0021/0.289). The Phase 1 gap was the 8-bit activations of the
  fallback.
- `routed_wmma_ops`: IQ3_S/IQ4_XS/IQ4_NL F16 route within 3e-4 to 4e-4 of F64
  relative to scale (Q4_K: 3.6e-4) on the 16/48/64-row tiles, ragged rows and
  tiny batches; paired gate/up equals the separate projections value for value
  and replays bit-exactly.
- `iq_ops`: batched gate/up widths 2-33 and down widths 2-32 are bit-identical
  to single-row decode.
- `session_test` (full suite, with MTP and the vision projector) passes on
  IQ4_XS: C2/C4/C6/C8 AR/MTP logits and RNG exact, sampled acceptance and
  residuals, 25 sampling strategies, MTP cache replay, EOS, cancellation,
  rollback, snapshots and image attachment. `--batch-only` also passes on
  Q4_K_XL after the shared-kernel changes.
- Q4_K_XL `gpu_probe` logits remain byte-identical to `main` (779-token prefill
  and 382 decode steps) after every Phase 2 commit.

### Rejected

- Staging IQ codes as 4-bit table indices in LDS (half the code footprint):
  slower, gate/up 8.85 → 9.86 ms/layer and down 4.11 → 4.63 ms/layer, because
  each row is then decoded per K block in the WMMA loop (by both half-waves)
  instead of once while staging.
- Batched IQ gate/up as separate projections plus SwiGLU across requests:
  exact, but no faster than per-request loops; replaced by the grouped kernel.

## Phase 3 results (laptop)

October 12, 2026. Functional suites (`tests/functional/run.py --record-baseline`,
sampling preset `qwen38`, `--sessions 4`) with `state-edges`, `tool-agent`,
`long-context` and `cache`, release build of `71bea96`:

| Mode | Result | Loaded mode | Drafts proposed / accepted |
| --- | --- | --- | --- |
| `--speculative off` (sidecar path given) | 11/11 passed | off | 0 / 0 |
| `--speculative mtp` | 11/11 passed | mtp | 1019 / 897 |

`main` cannot load the IQ4_XS target, so there is no main/PR timing comparison
for it; the runs record a baseline for later branch changes. The suites need the
dev shell's Python SDK versions (`openai` 2.41.1, `httpx` 0.28.1); `openai` 3.x
replaced `httpx` and breaks `tests/functional/metrics.py`.

### Container image

`ghcr.io/johandevlabs/gufo-toolboxes/gufo-runtime:edge` (also tagged
`sha-<gufo revision>`) is built by the
[gufo-toolboxes fork](https://github.com/johandevlabs/gufo-toolboxes) with
upstream's Nix recipe, its `gufo-engine` input pointed at this branch and
re-locked to the branch head on every build. Rebuild by running its
`Build & Publish Gufo OCI Images` workflow after pushing here.

Laptop smoke test of the first image (Gufo `71bea96`, 7.76 GiB): `gufo diagnose`
passes inside the container (ROCm 7.2.3 from the image); `gufo serve` with the
IQ4_XS target and MTP sidecar loads in 24 s at 32K context (66 GiB GPU memory
reported by the loader) and answers a greedy chat request correctly at
50.7 tok/s decode (133/187 drafts accepted).

```sh
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  --ulimit memlock=-1 -p 8080:8080 -v /path/to/models:/models:ro \
  ghcr.io/johandevlabs/gufo-toolboxes/gufo-runtime:edge \
  gufo serve --host 0.0.0.0 --port 8080 llm \
  --model /models/Qwen3.8-Flash-Next-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --speculative mtp \
  --mtp-model /models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
```

With rootless Podman, use `--userns=keep-id:uid=1000,gid=1000
--group-add keep-groups` instead of the two `--group-add` options, as in the
[main README](../../../README.md#quickstart).

## Phase 3 results (96 GB Evo-X2)

October 12, 2026, GMKtec Evo-X2 (Ryzen AI MAX+ 395, 96 GB LPDDR5X, GTT limit
96 GiB, kernel 7.0), sustained 119-120 W at about 2.53 GHz under load. Gufo ran
from `ghcr.io/johandevlabs/gufo-toolboxes/gufo-runtime:edge` (digest
`sha256:95350e5dbb8f275a92affab12b8398148f6bb6b724e865b4a5667660b153c25c`,
Gufo `71bea96`). Model files verified against the hashes above.

**Memory.** `gufo serve` with the IQ4_XS target, the shared Q8_0 MTP sidecar
and the BF16 vision projector at the native 262,144-token context loads in
11.1 s and reports 78.0 GiB GPU memory in use (the configured capacity is
reserved at admission) and 18.8 GiB host memory available. Q4_K_XL needs
16.4 GiB more for its weights alone.

**`gufo bench`** (pp2048 / tg128, two repetitions): pp2048 1701.90 ± 0.14,
tg128 30.20 ± 0.06, pp2048 @ d16K 1577.90 ± 3.37, tg128 @ d16K 29.15 ± 0.02
tok/s, against upstream's published Q4_K_XL 1702.93 / 27.98 tok/s at d0 on a
128 GB machine.

**Against llama.cpp on the same machine and file**, through the same HTTP driver
(`tools/bench/model-bench.py`, pp2048 / tg128, greedy, thinking off, one sample
per point; depth is the cached prefix). AR reference: llama.cpp `aa94f2086`
(October 10, 2026) Vulkan build with the driver's flags (`-ngl 999 -fa on
--cache-reuse 0 --jinja --reasoning off --cache-ram 0`, default batch sizes).
MTP reference: Unsloth's llama.cpp fork `586b15ef8` (Vulkan) with the
non-shared `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` sidecar, because neither
llama.cpp build loads the shared sidecar Gufo uses (`token_embd.weight` not
found) and the main build does not load the non-shared one either
(`output_hc_norm.weight` not found). Gain is Gufo relative to llama.cpp.

| Mode | Depth | Gufo pp | llama.cpp pp | Gain | Gufo tg | llama.cpp tg | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AR | 0 | 1628.43 | 271.21 | 6.0× | 29.78 | 24.25 | +23% |
| AR | 16,384 | 1486.34 | 230.95 | 6.4× | 29.69 | 22.22 | +34% |
| AR | 65,536 | 1411.92 | 155.13 | 9.1× | 29.25 | 14.67 | +99% |
| AR | 131,072 | 1345.58 | 102.92 | 13.1× | 28.64 | 10.99 | 2.6× |
| MTP mixed | 0 | 1541.52 | 278.72 | 5.5× | 38.48 | 33.85 | +14% |
| MTP mixed | 16,384 | 1405.39 | 215.31 | 6.5× | 42.39 | 26.27 | +61% |
| MTP mixed | 65,536 | 1365.64 | 133.09 | 10.3× | 35.69 | 16.88 | 2.1× |
| MTP mixed | 131,072 | 1313.49 | 86.67 | 15.2× | 39.24 | 15.02 | 2.6× |
| MTP repetitive | 0 | 1547.33 | 267.11 | 5.8× | 74.87 | 48.13 | +56% |
| MTP repetitive | 16,384 | 1393.88 | 211.34 | 6.6× | 55.01 | 42.03 | +31% |
| MTP repetitive | 65,536 | 1361.02 | 133.28 | 10.2× | 55.80 | 28.36 | +97% |
| MTP repetitive | 131,072 | 1355.97 | 86.78 | 15.6× | 51.82 | 22.33 | 2.3× |

| Users | AR Gufo | AR llama.cpp | MTP mixed Gufo | MTP mixed llama.cpp | MTP repetitive Gufo | MTP repetitive llama.cpp |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29.70 | 23.91 | 38.49 | 33.95 | 75.29 | 48.80 |
| 2 | 53.41 | 38.89 | 68.07 | 39.77 | 96.82 | 50.60 |
| 4 | 89.93 | 53.00 | 91.84 | 30.10 | 152.73 | 32.13 |
| 6 | 114.48 | 62.34 | 113.13 | 37.39 | 166.42 | 33.35 |
| 8 | 129.27 | 64.05 | 125.46 | 42.69 | 168.90 | 37.01 |

Multi-user rows sum the individual request decode rates (context 4096 per
user, all sessions prefilled first). llama.cpp's MTP throughput falls at C4 and
beyond in this configuration.

**Tuned llama.cpp.** A second llama.cpp run used the batch sizes from the
author's own launch scripts (`-b 8192 -ub 2048`, both builds) and an MTP draft
length of 3 (`--spec-draft-n-max 3`); the q8_0 KV cache from those scripts was
left out because it changes results. Larger batches raise llama.cpp's short-prompt
prefill (271 → 351 tok/s) but lower it at 64K/128K, and the shorter drafts help
MTP decode at depth. At 128K the Unsloth fork lost the Vulkan device in both MTP
workloads (`vk::Queue::submit: ErrorDeviceLost`, HTTP 500, server terminated);
the default-flag run completed those cells.

| Mode | Depth | Gufo pp | llama.cpp pp (default / tuned) | Gufo tg | llama.cpp tg (default / tuned) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR | 0 | 1628.43 | 271.21 / 351.12 | 29.78 | 24.25 / 24.13 |
| AR | 16,384 | 1486.34 | 230.95 / 244.10 | 29.69 | 22.22 / 21.61 |
| AR | 65,536 | 1411.92 | 155.13 / 119.20 | 29.25 | 14.67 / 14.43 |
| AR | 131,072 | 1345.58 | 102.92 / 90.61 | 28.64 | 10.99 / 10.78 |
| MTP mixed | 0 | 1541.52 | 278.72 / 267.71 | 38.48 | 33.85 / 33.00 |
| MTP mixed | 16,384 | 1405.39 | 215.31 / 224.95 | 42.39 | 26.27 / 27.90 |
| MTP mixed | 65,536 | 1365.64 | 133.09 / 113.20 | 35.69 | 16.88 / 21.15 |
| MTP mixed | 131,072 | 1313.49 | 86.67 / device lost | 39.24 | 15.02 / device lost |
| MTP repetitive | 0 | 1547.33 | 267.11 / 279.16 | 74.87 | 48.13 / 48.22 |
| MTP repetitive | 16,384 | 1393.88 | 211.34 / 224.01 | 55.01 | 42.03 / 43.14 |
| MTP repetitive | 65,536 | 1361.02 | 133.28 / 113.78 | 55.80 | 28.36 / 29.89 |
| MTP repetitive | 131,072 | 1355.97 | 86.78 / device lost | 51.82 | 22.33 / device lost |

| Users | AR: Gufo / llama.cpp default / tuned | MTP mixed: Gufo / default / tuned | MTP repetitive: Gufo / default / tuned |
| ---: | ---: | ---: | ---: |
| 1 | 29.70 / 23.91 / 23.66 | 38.49 / 33.95 / 32.75 | 75.29 / 48.80 / 50.02 |
| 2 | 53.41 / 38.89 / 39.31 | 68.07 / 39.77 / 43.12 | 96.82 / 50.60 / 49.24 |
| 4 | 89.93 / 53.00 / 55.42 | 91.84 / 30.10 / 28.78 | 152.73 / 32.13 / 32.41 |
| 6 | 114.48 / 62.34 / 64.27 | 113.13 / 37.39 / 36.66 | 166.42 / 33.35 / 45.92 |
| 8 | 129.27 / 64.05 / 67.41 | 125.46 / 42.69 / 45.25 | 168.90 / 37.01 / 43.89 |

## Plan

Q4_K_XL code paths must stay byte-identical: new formats are added as new cases,
and existing Flash-Next operator tests plus logit dumps on Q4_K_XL guard them.
Every new format must follow the existing exactness contracts, in particular that
MTP verification uses the same quantization and reduction as single-token decode.

### Phase 1: correct, not yet fast (done)

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

### Phase 2: speed (done)

1. `RoutedF16GEMMKernel`: IQ4_NL (down; 32-element blocks, a 16-entry lookup)
   and IQ3_S (gate/up; 256-element superblocks with a 512-entry grid in LDS,
   sign bits and 4-bit scales), single and paired gate/up. Keep the grid table a
   template parameter so IQ3_XXS could reuse it.
2. Tune IQ3_S decode, the grouped MTP path and the wave64 batched path
   (`moe_wave64`), which is Q4_K-only today.
3. Profile with `tools/prof`, following the optimize-kernel skill; compare against
   Gufo Q4_K_XL and llama.cpp IQ4_XS (ROCm and Vulkan).

### Phase 3: qualification (laptop and Evo-X2 done)

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
- **Test builds.** `gpu-test` builds `gufo-kernel-bench`, which needs
  `rocprofiler-sdk` (not packaged on Arch); configure with
  `-DGUFO_BUILD_TOOLS=OFF` to build the operator tests and probes without it.
- **Memory.** Do not load a model while a parallel HIP build runs: weights and
  compilers share the same RAM, and the combination exhausted it.
- **Formatting.** The system `clang-format` 23 flags untouched files that the
  pinned version accepts; format only changed files.
- **Profiling.** `rocprofv3` (`rocprofiler-sdk`) is not packaged on Arch, so
  `tools/prof/prof.py` cannot run here; kernel times above come from hipEvent
  harnesses over the production entry points.
- **Rounding.** In this toolchain `__fmul_rn` is a plain multiply (the OCML
  rounded variants are not enabled), so it does not stop fast-math from fusing
  a product into a caller's sum. Use an empty `asm volatile("" : "+v"(x))` to
  materialize a value when two paths must round identically.
- **Nix** is not installed on the laptop. Upstream evidence for a pull request
  should come from their pinned `nix build`.

## Open questions

- Gufo IQ4_XS peak memory and admitted context on the 96 GB desktop, including
  its GTT limit and host needs (OS, RAM prompt cache).
- Which llama.cpp backend the 96 GB desktop uses today, for its own baseline.
- MTP acceptance with the IQ3_S trunk.
