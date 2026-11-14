# Qwen3.8 Flash-Next UD-IQ3_XXS and UD-IQ4_XS (Dynamic 3.1)

Unsloth's Dynamic 3.1 **UD-IQ3_XXS** and **UD-IQ4_XS** quantizations of
Qwen3.8 Flash-Next, for Strix Halo machines with 96 GB of memory, and the
older three-shard UD-IQ3_XXS they replaced. Everything else (architecture,
MTP, images, YaRN, serving options) is as in the
[Flash-Next guide](../qwen3.8-flash-next/README.md). The older three-shard
UD-IQ4_XS has [its own card](../qwen3.8-flash-next-iq4-xs/README.md).

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Which file

| File | Download | GPU weights | Agreement with UD-Q4_K_XL |
| --- | ---: | ---: | --- |
| UD-Q4_K_XL | 111.3 GB | 78.1 GiB | reference |
| UD-IQ4_XS | 93.7 GB | 60.1 GiB | 89.4% of next tokens, mean KL 0.096 |
| UD-IQ3_XXS | 83.6 GB | 50.6 GiB | 88.3% of next tokens, mean KL 0.123 |
| UD-IQ3_XXS, three shards | 82.0 GB | 50.2 GiB | 85.5% of next tokens, mean KL 0.186 |

GPU weights are the model buffer llama.cpp allocates for each file. Agreement
is llama.cpp's next-token choice on 4,995 positions of code, documentation,
English, multilingual and agent text ([Quality](QUALITY.md)). The two
Dynamic 3.1 files are close; UD-IQ3_XXS uses about 9 GiB less GPU memory.
About 29 GB of each download is the per-layer token embedding, which is read
from disk as for UD-Q4_K_XL and never placed on the GPU.

## Files

`unsloth/Qwen3.8-Flash-Next-GGUF`, revision `d96201194f10` (Dynamic 3.1), the
same revision as UD-Q4_K_XL; the older UD-IQ3_XXS is at `38bb39ee9782`.
SHA-256 per shard: [`artifacts/model-identities.json`](artifacts/model-identities.json).

| File | Shards | Gate/up experts (layers) | Down experts (layers) | `ssm_out` | Embedding |
| --- | ---: | --- | --- | --- | --- |
| UD-IQ4_XS | 4 | IQ3_XXS 28, IQ4_XS 17, IQ2_S 3 | Q5_0 33, IQ4_NL 13, Q2_0 2 | Q8_0 | IQ4_XS |
| UD-IQ3_XXS | 4 | IQ2_S 26, IQ3_XXS 21, IQ4_XS 1 | IQ4_NL 45, Q2_0 3 | Q6_K | Q6_K |
| UD-IQ3_XXS, three shards | 3 | IQ2_S 47, IQ3_S 1 | IQ4_NL 48 | Q6_K | Q6_K |

In the Dynamic 3.1 files the other dense projections (`attn_qkv`,
`attn_gate`, QSA `attn_q/k/v/output`, shared-expert gate/up) and the output
head are Q6_K, except layer 2's `attn_qkv`, `attn_gate` and shared gate/up
in Q8_0; the older UD-IQ3_XXS has a Q6_K trunk too. `ffn_down_shexp`, HC and
PLE projections stay Q8_0 and the indexer BF16. Gate and up share one format
in every layer. The MTP predictor and the vision projector are the same files
as for UD-Q4_K_XL.

## Load and run

```sh
REV=d96201194f103c906994f283f504209dc8d2a10c
nix develop -c hf download unsloth/Qwen3.8-Flash-Next-GGUF --revision "$REV" \
  --include "UD-IQ3_XXS/*" --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --include "mmproj-BF16.gguf" --local-dir models/qwen3.8-flash-next
nix build
MODEL=models/qwen3.8-flash-next/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
MTP=models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

Use `UD-IQ4_XS` in place of `UD-IQ3_XXS` for the larger file. On a 96 GB
Evo-X2, both run one session at the native 262,144-token context with MTP and
images; see [memory fit](QUALITY.md#memory-fit).

## New formats

| Format | ggml type | Block | Notes |
| --- | ---: | --- | --- |
| IQ2_S | 22 | 256 values, 82 bytes | 1024-entry grid, sign bytes, a 4-bit scale per 16 values |
| IQ3_XXS | 18 | 256 values, 98 bytes | 256-entry grid, packed signs and scale in eight 32-bit words |
| Q2_0 | 42 | 64 values, 18 bytes | F16 `d`, 2-bit codes, value `(q − 1)·d`; about half of real `d` are negative |
| Q5_0 | 6 | 32 values, 22 bytes | as in ggml |
| Q6_K | 14 | 256 values, 210 bytes | dense projections, head and embedding |

`src/core/quant` holds independent CPU decoders for all of them; Q2_0 is new
in core (`gguf_reader` type 42, `DequantizeQ2_0`, hash fixed against ggml's
`to_float`). `cpu_ops` falls back to the core decoder for every format it has
no dedicated path for, so the F32 reference forward covers every file.

## Kernels

- **Routed prefill** keeps the F16 WMMA expert GEMMs (`RoutedF16GEMMKernel`,
  `RoutedPairF16Kernel`, `RoutedDownF16Kernel`). IQ2_S, IQ3_XXS, Q2_0 and Q5_0
  blocks are decoded to signed bytes plus per-16 F16 scale/bias, then take
  Q8_0's F16 arithmetic, so a row's result is the same in the paired
  128-token route and the separate 16- and 48-row tiles. IQ grids are staged
  in LDS (`iq_tables.hpp`). The down kernel's scale word is opaque to the
  compiler (`asm volatile`): with a visible zero bias, fast math rewrote the
  packed FMA and negative Q2_0/IQ4_NL scales rounded differently between the
  wide and narrow routes.
- **Decode and verification** (up to eight rows) use the vendored `vec_dot`
  kernels (`mmvq`), with `vec_dot_q2_0_q8_1` taken from llama.cpp `b11557`.
- **Dense Q6_K prefill** runs `DenseF16GEMMKernel<…, kQ6>` (BM 256, BK 2, one
  row per thread). It loads each half superblock once per two K stages and
  decodes codes with a direct 0x6400 magic (`−1056` bias) and compile-time
  shifts. Stacked projections (`ssm_in` 16,384 rows, `attn_qkv` 13,312 rows)
  use the fused SSM and attention launches with the Q6_K decoder
  (`StackedF16Route`). The fused SSM projection runs its last, partial token
  tile as a full one: the missing tokens read zero activations, so their
  accumulators stay +0, and the variant that skips them spilled registers.
  The route is chosen by `prefill_phase`, not by row count, so decode,
  verification and batched MTP keep the GEMV and stay exact.
- **Dense Q6_K decode** and the Q6_K head use the GEMV in eight-row chunks.
- **Shared experts** with Q6_K gate/up and Q8_0 down run on the side stream
  and skip the F32 copy, as Q8_0 ones do (`OverlappableSharedExpert`).
- **Embedding** decodes Q6_K and IQ4_XS rows directly (`EmbeddingType`).

The load-time rules of the IQ4_XS card still apply: an IQ, Q2_0 or Q5_0
expert layer must take the WMMA route, and such experts in the MTP block are
refused. The probe claims the new expert formats and still declines Q6_K
experts.

## Tests

`qwen38_flash_next_iq_ops_test`, `_routed_wmma_ops_test` (336 route
comparisons), `_projection_ops_test` (Q6_K dense, stacked SSM and attention),
`_probe_test` and `ggml_dequant_test` cover the formats. See
[Quality](QUALITY.md) for model-level evidence.
