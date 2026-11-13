# Qwen3.8 Flash-Next GSQ-RCO experiments

Evo rows are `tools/bench/model-bench.py` pp2048/tg128 at depth 0 on a 96 GB
GMKtec Evo-X2, one warmed sample, `nix build` binaries. Laptop rows (128 GB
Strix Halo, `performance` profile, USB-C power) are `gufo bench -p 2048`
A/B pairs from the same build directory, three repetitions, two alternating
rounds; laptop GPU timestamps read zero, so kernel times there are host-timed
batches of 20 launches.

## Tuning (November 12–13, 2026)

Every change below is retained and gives the same output bits as the
previous kernel, except where noted.

| Commit | Change | Measured on | Effect |
| --- | --- | --- | --- |
| `1895c16b` | `DenseBf16Gemm` tile plans per HC shape | laptop profile | up 1.50 → 0.83 ms, down 1.42 → 0.84 ms at 2,048 tokens; prefill GPU time −6.8% |
| `4744f2c2` | HC down decode GEMV split over four waves | laptop profile | 52 → 34 µs per call; decode GPU time −5.6% |
| — | (Evo after both) | Evo | pp 1,425.6, tg 32.4 tok/s |
| `a0894a24` | Dense prefill tiles 48 → 128 tokens | Evo, with the next row | |
| `9abc10e7` | Prefill combine writes the BF16 norm for the HC down projection | Evo | pp 1,476.7, tg 32.5 tok/s |
| `ee2c7ead` | IQ grid signs with word operations | laptop | gated expert GEMV 1.35–1.8 times as fast; tg 30.1 → 31.5 tok/s. GSQ-RCO logits move at ulp level (IQ3_S dense GEMV contraction) |
| `bbfc1e82` | HC mixer in the BF16 up projection's epilogue; combine writes the inject partials | laptop | pp 1,285 → 1,349 tok/s |
| `a7d2edd2` | (final; adds a layer-hook fix) | Evo | pp 1,568.7, tg 34.1 tok/s; 8-user AR 96.7 → 113.3 tok/s |

`4744f2c2` changes the sampled path enough to flip one functional sampled
check (see [Quality](QUALITY.md#tool-agent-sampling)); a 30-seed A/B showed
no change in the miss rate, so the split was kept.

## Rejected

- **hipBLASLt for BF16 projections:** its BF16 kernels start each tile's K
  loop at a block rotated by the tile's position in the batch, so a row's
  result depended on the prefill chunk width; chunked prefill failed
  `session_test`. Every BF16 shape now uses `DenseBf16Gemm`.
- **Scalar HC combine for prefill tails under 16 rows:** its norm reduction
  order differs from the Vec4 kernel's; a chunk boundary changed rows.
- **`HcInjectPart`'s pinned FMA order for the F32 inject partials:** differs
  from `HcMixEpilogueVec4`'s compiled order by up to 1.2e-6 relative. The
  combine repeats the epilogue's expressions instead, and
  `qwen38_flash_next_iq_ops_test` checks the partials byte for byte.
