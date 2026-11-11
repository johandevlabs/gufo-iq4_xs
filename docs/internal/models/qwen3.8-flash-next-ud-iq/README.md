# Qwen3.8 Flash-Next UD-IQ implementation notes

Developer detail behind the [public model page](../../../models/qwen3.8-flash-next-ud-iq.md).
[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

Fork-only support for Unsloth's smaller Qwen3.8 Flash-Next GGUFs. Everything
outside the routed experts, the dense projections and the embedding is the
UD-Q4_K_XL implementation described in
[Flash-Next's notes](../qwen3.8-flash-next/README.md); UD-Q4_K_XL logits stay
byte-identical to Gufo 0.33.0.

## Files

`unsloth/Qwen3.8-Flash-Next-GGUF`, revision `d96201194f10` (Unsloth Dynamic
3.1), the same revision as UD-Q4_K_XL. The files are also identical at
`084c631ac221`, where they were downloaded. SHA-256 per shard:
[`artifacts/model-identities.json`](artifacts/model-identities.json).

| File | Shards | Size | Gate/up experts (layers) | Down experts (layers) | `ssm_out` | Embedding |
| --- | ---: | ---: | --- | --- | --- | --- |
| UD-IQ4_XS | 4 | 93.7 GB | IQ3_XXS 28, IQ4_XS 17, IQ2_S 3 | Q5_0 33, IQ4_NL 13, Q2_0 2 | Q8_0 | IQ4_XS |
| UD-IQ3_XXS | 4 | 83.6 GB | IQ2_S 26, IQ3_XXS 21, IQ4_XS 1 | IQ4_NL 45, Q2_0 3 | Q6_K | Q6_K |

In both files the other dense projections (`attn_qkv`, `attn_gate`, QSA
`attn_q/k/v/output`, shared-expert gate/up) and the output head are Q6_K,
except layer 2's `attn_qkv`, `attn_gate` and shared gate/up in Q8_0.
`ffn_down_shexp`, HC and PLE projections stay Q8_0 and the indexer BF16.
Gate and up share one format in every layer. Shard 4 is the 28.8 GB IQ4_NL per-layer token embedding, the same
bytes in both files; it is read from disk as for UD-Q4_K_XL.

The earlier three-shard UD-IQ4_XS (revision `38bb39ee9782`: IQ3_S gate/up,
IQ4_NL down, Q8_0 dense, Q6_K head) still loads.

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
no dedicated path for, so the F32 reference forward covers both files.

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
  (`StackedF16Route`). The route is chosen by `prefill_phase`, not by row
  count, so decode, verification and batched MTP keep the GEMV and stay exact.
- **Dense Q6_K decode** and the Q6_K head use the GEMV in eight-row chunks.
- **Shared experts** with Q6_K gate/up and Q8_0 down run on the side stream
  and skip the F32 copy, as Q8_0 ones do (`OverlappableSharedExpert`).
- **Embedding** decodes Q6_K and IQ4_XS rows directly (`EmbeddingType`).

Load-time rules are unchanged: an IQ layer must take the WMMA route, and IQ
experts in the MTP block are refused. The probe claims the new expert formats
and still declines Q6_K experts.

## Tests

`qwen38_flash_next_iq_ops_test`, `_routed_wmma_ops_test` (336 route
comparisons), `_projection_ops_test` (Q6_K dense, stacked SSM and attention),
`_probe_test` and `ggml_dequant_test` cover the formats. See
[Quality](QUALITY.md) for model-level evidence.
