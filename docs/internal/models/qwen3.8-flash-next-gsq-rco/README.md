# Qwen3.8 Flash-Next GSQ-RCO implementation notes

Developer detail behind the [public model page](../../../models/qwen3.8-flash-next-gsq-rco.md).
[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

Fork-only support for ISTA-DASLab's GSQ-RCO IQ3_XXS GGUF of Qwen3.8
Flash-Next. It builds on the [UD-IQ support](../qwen3.8-flash-next-ud-iq/README.md),
which brought the IQ2_S, IQ3_XXS, Q2_0 and Q6_K kernels, and on the
UD-Q4_K_XL implementation in [Flash-Next's notes](../qwen3.8-flash-next/README.md).
Most GSQ-RCO kernels run only for formats or BF16 tensors the other files
do not have; [Quality](QUALITY.md#other-files) lists what was checked on
UD 3.1.

## File

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision `ed59f92082b1`,
`IQ3_XXS`: two shards, 75.8 GB. Shard 2 (28.8 GB) holds only the IQ4_NL
per-layer token embedding, read from disk as for UD-Q4_K_XL. The file has no
MTP layer; Gufo uses the shared `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`
sidecar. SHA-256 per shard: [`artifacts/model-identities.json`](artifacts/model-identities.json).

Formats by tensor role (48 layers: 36 linear attention, 12 QSA):

| Role | Formats (layers) |
| --- | --- |
| Routed gate = up | IQ3_S 13, IQ2_XS 10, IQ2_S 10, IQ2_XXS 9, IQ3_XXS 6 |
| Routed down | Q2_0 30, IQ4_NL 18 |
| Shared gate | IQ4_XS 21, IQ3_S 13, Q4_K 12, Q5_K 1, Q6_K 1 |
| Shared up | IQ3_S 27, IQ4_XS 10, Q4_K 7, Q6_K 3, Q5_K 1 |
| Shared down | IQ4_NL 40, Q2_0 8 |
| `attn_qkv` (linear-attention input) | Q6_K 14, IQ4_XS 11, Q4_K 8, Q5_K 3 |
| `attn_gate` | IQ4_XS 11, IQ3_S 10, Q4_K 7, Q6_K 5, Q5_K 3 |
| `ssm_out` | Q6_K 19, IQ4_XS 6, Q4_K 4, Q5_K 4, IQ3_S 3 |
| QSA `attn_q/k/v/output` | Q6_K, IQ4_XS, Q4_K, Q5_K, IQ3_S |
| Router, shared-expert gate, HC down/up/inject, PLE key/value, `ssm_alpha/beta`, indexer | BF16 |
| `ple_conv1d` | F16 |
| Head / embedding | Q5_K / IQ3_S |

Gate and up share one format in every layer, so the paired expert kernels
apply. Every dense projection mixes formats across layers; each layer binds
its own.

## Loader

- `weights.cpp` admits the dense formats (`kDenseFormats`), IQ2_XXS
  (256 values, 66 bytes) and IQ2_XS (256 values, 74 bytes) experts, BF16
  routers and shared-expert gates, F16 `ple_conv1d` and an IQ3_S embedding.
- `device_model.cpp` widens the BF16 router and shared gate on the host to
  F32, then stacks them as F16 like UD-Q4_K_XL's F32 tensors. HC inject is
  widened to F32 (`CopyF32`). F16 `ple_conv1d` is widened. `Upload` refuses an
  unsupported dense, expert or embedding format up front.
- The probe claims 14 expert formats, including IQ2_XXS and IQ2_XS.
- The CPU F32 reference decodes `ple_conv1d` through `cpu::DequantizeRow`.

## Kernels

- **Dense projections in decode and verification** use
  `mul_mat_vec_q_dense` with an exact K-quant dot
  (`vec_dot_kquant_dense`, explicit `__fmaf_rn`). Q4_K, Q5_K, Q6_K, IQ4_XS,
  IQ3_S, IQ4_NL and Q2_0 dispatch through `get_dense_vec_dot`.
- **Dense projections in prefill** run the routed F16 expert GEMM with an
  identity route (one expert, every row; `DenseRoutedF16`), 128-token tiles
  (`kDenseTileRows`). Each block is decoded to F16 once per tile.
- **IQ2_XXS and IQ2_XS experts** decode grids from LDS (`kIq2xxsGrid`,
  `kIq2xsGrid`) in the paired routed F16 GEMMs, like IQ2_S.
- **IQ decode signs** (`vecdotq.hpp`): `ksigns8` and `iq_signed4` apply a
  grid row's eight signs with two 32-bit operations instead of
  `__vcmpne4`/`__vsub4`, which gfx1151 emulates per byte. Bit-identical
  results; the gated expert GEMVs at one token run 1.35–1.8 times as fast.
  The IQ3_S dense GEMV's final multiply-add contracts differently, so
  GSQ-RCO decode logits move at ulp level.
- **BF16 projections** (HC, PLE, `ssm_alpha/beta`, indexer) always take
  `DenseBf16Gemm` (`OwnBf16`), never hipBLASLt, whose BF16 kernels rotate
  each tile's K loop by its position in the batch. Tile plans per shape:
  `m ≥ 2048, k ≤ 1024`: 256×128 with eight-row-tile groups from 1,024
  tokens, otherwise 256×128 with eight row waves; `m ≤ 512, k ≥ 4096` from
  1,024 tokens: 256×128 with eight row waves; else 64×128. Every plan gives
  each output the same ordered K16 products.
- **BF16 decode GEMVs** with `k ≥ 8192` (HC down) split each row over four
  waves and sum the partials in order through LDS (`SplitSmallGemm`).
- **HC combine in prefill** always takes the Vec4 row kernel, whose norm
  reduction differs from the scalar one: a chunk boundary must not change a
  row. Before a BF16 mixer it also writes the norm as BF16 rows for the down
  projection, and that mixer's inject partials with the separate epilogue's
  expressions.
- **HC mixer in prefill** (`HcMixBf16Gemm`): the BF16 up projection mixes
  its gates with the F32 norm in the GEMM epilogue
  (`DenseF16GEMMKernel<…, kHcMix, …, kBf16>`) instead of writing 10,240
  gates per token. Byte for byte `DenseBf16Gemm` plus `HcMixEpilogueVec4`.
- The layer hook (prefill yield) does not park `x_half`; after it the BF16
  down projection narrows the parked F32 norm again.

## Tests

`qwen38_flash_next_iq_ops_test` (dense GEMVs and identity-routed GEMMs for
every dense format, IQ3_S embedding, BF16 chunk invariance at 1,100 tokens,
fused HC mixer and combine inject partials byte for byte),
`_routed_wmma_ops_test` (IQ2_XXS/IQ2_XS routes), `_projection_ops_test`
(split BF16 GEMVs at 320×10,240 and 10,240×320) and `_probe_test` cover the
code. See [Quality](QUALITY.md) for model-level evidence.
