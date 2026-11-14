# Qwen3.8 Flash-Next UD-IQ experiments

These changes were tuned on johandevlabs/gufo-iq4_xs (Gufo 0.33.0 with the
same IQ code) before the port to Strycast. Prefill numbers are `gufo bench`
pp2048/tg128 at depth 0 on a 96 GB GMKtec Evo-X2, one warmed sample, `nix
build` binaries. A/B pairs ran back to back with cool-downs; the laptop
(128 GB, USB-C power) was used for correctness only after its noise reversed
one decision.

## Q6_K trunk prefill (November 10, 2026)

Retained: dense Q6_K projections in prefill decode to F16 in
`DenseF16GEMMKernel` instead of using integer W8A8 WMMA. The F16 route is
faster and closer to the CPU F32 reference (mean KL on the 24-token probe:
UD-IQ3_XXS 0.026 → 0.0074, UD-IQ4_XS 0.0057 → 0.0037). The W8A8 route also
broke batched-MTP C4 exactness when chosen by row count; the F16 route is
selected by `prefill_phase`. The W8A8 route was deleted.

## Q6_K dense GEMM tuning (November 10, 2026)

Retained, each bit-exact against the previous kernel:

| Change | UD-IQ3_XXS pp | UD-IQ4_XS pp |
| --- | ---: | ---: |
| F16 Q6_K trunk (start) | 1,673 | 1,738 |
| Half superblock loaded once per two K stages; stacked Q6_K `ssm_in` in the fused SSM projection | 1,804 | 1,838 |
| Q6_K shared experts on the side stream; stacked Q6_K `attn_qkv` in the fused attention projection | 1,871 | 1,903 |
| Direct magic code decode, compile-time shifts | 1,907 | 1,930 |

The side-stream row was first rejected from a laptop A/B (1,208.7 vs
1,212.2 tok/s) taken while charging from a 60 W supply; an Evo profile showed
265 ms of lost overlap and the Evo A/B confirmed the gain. Decode rose about
3% with the same change. The earlier three-shard UD-IQ4_XS stayed at
1,930 throughout.

## Port to Strycast (November 14, 2026)

Strycast's kernel resource check flagged the fused Q6_K SSM projection
(`DenseF16GEMMKernel<256, 128, 2, 8, 1, 2, …, kQ6>`): 256 VGPRs, 7 spilled,
32 bytes of scratch. Its partial-tile variant, which skips 16-token subtiles
past the batch, needed the extra registers. The kernel now runs the last tile
as a full one: the missing tokens read zero activations and their
accumulators stay +0, so every output is unchanged (UD-IQ3_XXS prefill
logits byte-identical for 1,522 tokens in one pass and in 250- and 77-token
chunks). The kernel uses no scratch; the full-tile loop is unchanged.

Rejected on the way, each still spilling: loading the half superblock's high
bits, scales and `d` after the WMMAs (26 spilled VGPRs), loading activation
chunks in LDS order and 32-bit Q6_K row offsets (6–9 spilled VGPRs).

## Rejected

- **Reuse the producer's Q8_1 rows for Q6_K decode GEMVs:** no tg change;
  decode is graph-captured and the quantization is not on the critical path.
- **Q8_0 output head:** a hybrid with UD-Q4_K_XL's Q8_0 head had shown +3–5%
  prefill on another branch. In this fork both heads measure the same
  (1,938/1,936/1,935 vs 1,936/1,934/1,932 tok/s), so the Q6_K head stays.
- **Closing the last 1–2% between UD-IQ3_XXS and UD-IQ4_XS prefill:** left
  as is; the files differ in nearly every expert format.
