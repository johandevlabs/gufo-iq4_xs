# Qwen3.8 Flash-Next UD-IQ4_XS benchmarks

GMKtec Evo-X2, AMD Strix Halo `gfx1151` with 96 GB unified memory (about 92
GiB usable), Ubuntu 24.04, `powersave` governor. Unsloth `UD-IQ4_XS` target
at revision `38bb39ee9782` with the shared-Q8_0 MTP predictor; adaptive
drafting. HTTP, greedy, thinking off, one sample per point; single-user AR
Strycast cells are the mean ± sd of three.
Strycast `ca0547c3` (November 14, 2026); llama.cpp `68d9053a` AR and `6fcaa16f`
MTP (November 2), each from Strycast's pinned recipe with one fix (see
[Quality](QUALITY.md#benchmark-method)). Positive gain favors Strycast.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

llama.cpp cells are host-limited on this machine: AR used up to 5.7 GiB of
swap at 128K, and MTP used swap at every depth and filled the 8 GiB swap at
64K and 128K. A machine with more memory may differ most there.

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
| Flash-Next IQ4_XS AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1814.71 ± 8.03 | 542.70 | +234.4% | 31.95 ± 0.02 | 22.46 | +42.3% |
| 4,096 | 1739.81 ± 4.31 | 502.28 | +246.4% | 31.90 ± 0.12 | 21.69 | +47.1% |
| 8,192 | 1709.04 ± 6.91 | 472.54 | +261.7% | 31.92 ± 0.10 | 21.04 | +51.7% |
| 12,288 | 1690.85 ± 10.17 | 437.07 | +286.9% | 31.82 ± 0.16 | 20.33 | +56.5% |
| 16,384 | 1678.40 ± 16.91 | 411.06 | +308.3% | 31.78 ± 0.15 | 19.65 | +61.7% |
| 32,768 | 1608.41 ± 17.27 | 335.69 | +379.1% | 31.85 ± 0.02 | 17.66 | +80.4% |
| 65,536 | 1530.64 ± 21.30 | 245.56 | +523.3% | 31.65 ± 0.07 | 12.68 | +149.6% |
| 131,072 | 1433.89 ± 13.86 | 158.31 | +805.7% | 31.45 ± 0.03 | 8.77 | +258.6% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Strycast's predictor catch-up. llama.cpp's MTP text matches
its AR text only at depth 0.

<!-- bench:single-mtp -->
| Flash-Next IQ4_XS MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1764.00 | 490.56 | +259.6% | 42.59 | 30.27 | +40.7% | 153.55 | 44.00 | +249.0% |
| 4,096 | 1692.03 | 454.03 | +272.7% | 51.19 | 31.82 | +60.9% | 152.05 | 42.21 | +260.2% |
| 8,192 | 1662.88 | 428.20 | +288.3% | 48.11 | 29.93 | +60.7% | 151.52 | 42.30 | +258.2% |
| 12,288 | 1617.62 | 403.16 | +301.2% | 47.02 | 27.47 | +71.2% | 142.83 | 40.36 | +253.9% |
| 16,384 | 1594.76 | 380.28 | +319.4% | 49.88 | 29.72 | +67.8% | 146.38 | 40.41 | +262.2% |
| 32,768 | 1561.35 | 312.20 | +400.1% | 46.82 | 26.26 | +78.3% | 147.20 | 37.47 | +292.8% |
| 65,536 | 1496.40 | 224.84 | +565.5% | 47.57 | 18.19 | +161.5% | 143.52 | 29.33 | +389.3% |
| 131,072 | 1418.65 | 147.68 | +860.6% | 48.93 | 15.81 | +209.5% | 142.54 | 21.67 | +557.8% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.

<!-- bench:multi-ar -->
| Flash-Next IQ4_XS AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 31.85 | 23.10 | +37.9% |
| 2 | 56.43 | 36.48 | +54.7% |
| 4 | 94.68 | 56.78 | +66.7% |
| 6 | 121.43 | 61.84 | +96.4% |
| 8 | 137.83 | 73.69 | +87.0% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates.

<!-- bench:multi-mtp -->
| Flash-Next IQ4_XS MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42.31 | 31.03 | +36.4% | 153.08 | 44.07 | +247.4% |
| 2 | 71.59 | 40.60 | +76.3% | 173.35 | 50.89 | +240.6% |
| 4 | 96.40 | 45.97 | +109.7% | 206.31 | 49.86 | +313.8% |
| 6 | 110.65 | 53.66 | +106.2% | 220.59 | 51.15 | +331.3% |
| 8 | 126.73 | 65.76 | +92.7% | 239.00 | 57.53 | +315.4% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

## Loading time

C1, context capacity 262144. Cold target and sidecar files to HTTP
readiness. Strycast loads with MTP; llama.cpp with MTP was OOM-killed while
loading at this capacity, so its cell is AR, without the sidecar.

<!-- bench:loading -->
| Flash-Next IQ4_XS<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| IQ4_XS | 12.05 | 32.15 | +166.8% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP. llama.cpp also
held about 28.6 GiB of anonymous host memory, which this counter does not
include.

<!-- bench:memory -->
| Flash-Next IQ4_XS AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 75.30 | 71.76 | -4.7% |
| 16K prefix, pp4096 + tg128 | 77.03 | 72.09 | -6.4% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)
