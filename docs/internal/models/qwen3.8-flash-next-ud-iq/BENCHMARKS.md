# Qwen3.8 Flash-Next UD-IQ benchmarks

AMD Strix Halo `gfx1151` in a 96 GB GMKtec Evo-X2 (about 92 GiB usable).
Unsloth Dynamic 3.1 `UD-IQ4_XS` and `UD-IQ3_XXS` targets with shared-Q8_0
MTP; adaptive drafting. HTTP, greedy, thinking off, one sample per point;
single-user AR Strycast cells are the mean ± sd of three. Strycast
`f1d18b42` (November 14, 2026); llama.cpp `b11069` AR / `6fcaa16f` MTP
(November 11), each from Strycast's pinned recipe with one fix (see
[Quality](QUALITY.md#benchmark-method)). Positive gain favors Strycast.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

llama.cpp keeps the 27.5 GiB per-layer embedding in host memory beside its
GPU weights. With UD-IQ4_XS that left 0.5–4 GiB available and 4–8 GiB in swap
during its runs, so its UD-IQ4_XS cells show this 96 GB machine's limit, not
the engine's unconstrained speed. Its UD-IQ3_XXS runs and all Strycast runs kept
swap below 0.2 GiB. The llama.cpp MTP server ran out of GPU memory with
UD-IQ4_XS at the 128K depth (`N/A`); its 8-user mixed-text cell ran out of
memory once while loading and is a retry.

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar-iq4 -->
| Flash-Next UD 3.1 IQ4_XS AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1806.51 ± 25.43 | 523.06 | +245.4% | 33.44 ± 0.10 | 23.65 | +41.4% |
| 4,096 | 1739.43 ± 2.05 | 487.78 | +256.6% | 33.36 ± 0.15 | 22.82 | +46.2% |
| 8,192 | 1705.49 ± 14.34 | 456.45 | +273.6% | 33.32 ± 0.08 | 22.01 | +51.4% |
| 12,288 | 1679.96 ± 12.71 | 426.27 | +294.1% | 33.31 ± 0.15 | 21.25 | +56.8% |
| 16,384 | 1667.99 ± 21.40 | 400.90 | +316.1% | 33.36 ± 0.12 | 20.56 | +62.3% |
| 32,768 | 1605.49 ± 17.12 | 328.48 | +388.8% | 33.34 ± 0.06 | 18.38 | +81.4% |
| 65,536 | 1521.63 ± 27.53 | 241.25 | +530.7% | 33.22 ± 0.07 | 13.29 | +150.0% |
| 131,072 | 1440.52 ± 7.04 | 155.45 | +826.7% | 32.91 ± 0.10 | 8.82 | +273.1% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-iq4.svg)

---

<!-- bench:single-ar-iq3 -->
| Flash-Next UD 3.1 IQ3_XXS AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1793.54 ± 26.65 | 500.44 | +258.4% | 34.39 ± 0.01 | 23.76 | +44.7% |
| 4,096 | 1744.07 ± 3.48 | 467.71 | +272.9% | 34.36 ± 0.14 | 23.04 | +49.1% |
| 8,192 | 1710.31 ± 25.45 | 440.84 | +288.0% | 34.23 ± 0.19 | 22.26 | +53.8% |
| 12,288 | 1671.83 ± 20.25 | 411.58 | +306.2% | 34.26 ± 0.11 | 21.41 | +60.0% |
| 16,384 | 1658.30 ± 37.24 | 388.68 | +326.6% | 34.22 ± 0.13 | 20.73 | +65.1% |
| 32,768 | 1595.18 ± 28.18 | 319.26 | +399.6% | 34.21 ± 0.05 | 18.57 | +84.2% |
| 65,536 | 1509.31 ± 22.31 | 237.82 | +534.6% | 34.15 ± 0.06 | 13.67 | +149.8% |
| 131,072 | 1425.33 ± 14.79 | 158.01 | +802.1% | 33.79 ± 0.11 | 9.47 | +256.8% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-iq3.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp-iq4 -->
| Flash-Next UD 3.1 IQ4_XS MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1730.03 | 488.40 | +254.2% | 42.21 | 32.30 | +30.7% | 106.52 | 45.47 | +134.3% |
| 4,096 | 1616.12 | 452.39 | +257.2% | 49.12 | 35.36 | +38.9% | 104.02 | 44.41 | +134.2% |
| 8,192 | 1617.49 | 426.78 | +279.0% | 51.14 | 30.24 | +69.1% | 108.53 | 43.59 | +149.0% |
| 12,288 | 1589.15 | 401.15 | +296.1% | 51.42 | 28.91 | +77.9% | 105.75 | 41.66 | +153.8% |
| 16,384 | 1572.15 | 377.02 | +317.0% | 49.26 | 30.75 | +60.2% | 104.75 | 41.46 | +152.7% |
| 32,768 | 1526.13 | 309.07 | +393.8% | 48.32 | 27.34 | +76.7% | 105.22 | 39.10 | +169.1% |
| 65,536 | 1471.75 | 223.56 | +558.3% | 47.61 | 19.87 | +139.6% | 105.86 | 29.23 | +262.2% |
| 131,072 | 1384.06 | N/A | N/A | 48.41 | N/A | N/A | 107.46 | N/A | N/A |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-iq4.svg)

---

<!-- bench:single-mtp-iq3 -->
| Flash-Next UD 3.1 IQ3_XXS MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1716.56 | 477.54 | +259.5% | 49.43 | 32.18 | +53.6% | 100.94 | 46.79 | +115.7% |
| 4,096 | 1624.90 | 437.10 | +271.7% | 47.98 | 33.04 | +45.2% | 105.05 | 43.89 | +139.3% |
| 8,192 | 1600.25 | 411.91 | +288.5% | 45.87 | 30.57 | +50.0% | 105.77 | 44.35 | +138.5% |
| 12,288 | 1577.23 | 388.08 | +306.4% | 46.42 | 29.19 | +59.0% | 100.96 | 42.88 | +135.4% |
| 16,384 | 1553.26 | 364.21 | +326.5% | 50.14 | 27.17 | +84.5% | 100.06 | 41.78 | +139.5% |
| 32,768 | 1541.83 | 300.76 | +412.6% | 48.87 | 28.76 | +69.9% | 97.91 | 38.52 | +154.2% |
| 65,536 | 1434.77 | 222.00 | +546.3% | 45.85 | 19.43 | +136.0% | 100.78 | 30.12 | +234.6% |
| 131,072 | 1395.10 | 146.93 | +849.5% | 46.99 | 14.94 | +214.5% | 99.45 | 21.65 | +359.4% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-iq3.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.

<!-- bench:multi-ar-iq4 -->
| Flash-Next UD 3.1 IQ4_XS AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 33.43 | 24.35 | +37.3% |
| 2 | 56.02 | 39.53 | +41.7% |
| 4 | 84.48 | 58.43 | +44.6% |
| 6 | 99.49 | 65.82 | +51.2% |
| 8 | 106.14 | 69.24 | +53.3% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-iq4.svg)

---

<!-- bench:multi-ar-iq3 -->
| Flash-Next UD 3.1 IQ3_XXS AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 34.30 | 24.97 | +37.4% |
| 2 | 57.41 | 39.49 | +45.4% |
| 4 | 85.50 | 57.54 | +48.6% |
| 6 | 98.46 | 64.37 | +53.0% |
| 8 | 103.54 | 67.25 | +54.0% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-iq3.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates.

<!-- bench:multi-mtp-iq4 -->
| Flash-Next UD 3.1 IQ4_XS MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42.03 | 32.43 | +29.6% | 107.27 | 46.62 | +130.1% |
| 2 | 71.70 | 45.20 | +58.6% | 115.74 | 53.02 | +118.3% |
| 4 | 85.73 | 50.22 | +70.7% | 120.97 | 49.73 | +143.3% |
| 6 | 88.19 | 57.51 | +53.3% | 123.92 | 52.08 | +137.9% |
| 8 | 91.84 | 63.03 | +45.7% | 126.00 | 57.54 | +119.0% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-iq4.svg)

---

<!-- bench:multi-mtp-iq3 -->
| Flash-Next UD 3.1 IQ3_XXS MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 48.98 | 33.12 | +47.9% | 101.97 | 47.93 | +112.7% |
| 2 | 73.28 | 50.15 | +46.1% | 110.61 | 53.21 | +107.9% |
| 4 | 84.65 | 50.72 | +66.9% | 111.91 | 51.31 | +118.1% |
| 6 | 86.87 | 56.67 | +53.3% | 113.76 | 53.29 | +113.5% |
| 8 | 89.71 | 64.33 | +39.5% | 114.55 | 69.27 | +65.4% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-iq3.svg)

## Loading time

C1, context capacity 262144. Cold target/sidecar files to HTTP readiness;
Gufo loads MTP, llama.cpp AR only (its UD-IQ4_XS MTP server already ran out
of memory at capacity 133,760).

<!-- bench:loading -->
| Flash-Next UD 3.1<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| IQ3_XXS | 10.10 | 24.84 | +145.9% |
| IQ4_XS | 11.78 | 34.51 | +193.0% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP. llama.cpp's
figure excludes the 27.5 GiB per-layer embedding it holds in host memory;
Gufo reads that tensor through the reclaimable file cache.

<!-- bench:memory-iq4 -->
| Flash-Next UD 3.1 IQ4_XS AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 75.48 | 71.88 | -4.8% |
| 16K prefix, pp4096 + tg128 | 77.21 | 72.21 | -6.5% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-iq4.svg)

---

<!-- bench:memory-iq3 -->
| Flash-Next UD 3.1 IQ3_XXS AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 66.21 | 62.52 | -5.6% |
| 16K prefix, pp4096 + tg128 | 67.96 | 62.85 | -7.5% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-iq3.svg)

## Older files

Unsloth's three-shard UD-IQ3_XXS from revision `38bb39ee`, which Dynamic 3.1
replaced. Strycast only, the same binary, host and settings as the tables
above, November 14, 2026; artifacts in
[`artifacts/older-files/`](artifacts/older-files/), whose `iq3` variant is
this file. Depth is the cached prefix in tokens; 8-user rows are total
decode. The three-shard UD-IQ4_XS has
[its own benchmarks](../qwen3.8-flash-next-iq4-xs/BENCHMARKS.md).

| Flash-Next three-shard UD-IQ3_XXS, Strycast | tok/s |
| --- | ---: |
| AR pp, depth 0 | 1800.12 |
| AR pp, depth 131,072 | 1435.64 |
| AR tg, depth 0 | 34.17 |
| AR tg, depth 131,072 | 33.55 |
| MTP tg mixed, depth 0 | 48.80 |
| MTP tg mixed, depth 131,072 | 43.98 |
| MTP tg repetitive, depth 0 | 104.56 |
| MTP tg repetitive, depth 131,072 | 91.93 |
| 8 users AR | 104.12 |
| 8 users MTP mixed | 89.91 |
| 8 users MTP repetitive | 115.27 |

Single-user AR and MTP completions match at depth 0, and every multi-user
completion matches its AR C1 reference.
