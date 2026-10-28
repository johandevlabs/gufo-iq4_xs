# Qwen3.6 35B-A3B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Gufo versus llama.cpp
`b11069` (`68d9053a`), both compiled with the pinned ROCm 7.2.3 toolchain.
Targets are the Unsloth UD-Q4_K_XL / UD-Q6_K_XL / UD-Q8_K_XL files and the
lmstudio-community Q4_K_M; DFlash2 uses the aminya Q8_0 draft and MTP uses each
Unsloth target's own head (Q4_K_M has none) ([files and
revisions](README.md)). HTTP, greedy, thinking off.

The Q4 (UD-Q4_K_XL) and Q4_K_M rows were measured on October 27, 2026 with the
build that decodes K-quant experts with the MMQ vector kernels. The Q6 and Q8
rows were measured on October 25 with `7b1e23a8`; those targets decode through
the same kernels in both builds ([builds](artifacts/model-identities.json)). Loading was re-measured for every
target on October 27.

Gufo drafts up to 7 DFlash2 tokens with its adaptive controller and 2 MTP
tokens; llama.cpp uses its default of 3 for both modes. Its repetitive MTP
rows therefore accept up to 2.88 tokens per step against Gufo's 1.91.

Every Gufo DFlash2 and MTP completion, single-user at every depth and
multi-user at every level, matches Gufo AR on every target, and every
multi-user AR completion matches the same prompt decoded alone (see
[Quality](QUALITY.md#benchmark-output-consistency)).

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar-q4 -->
| Qwen35B-A3B Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3266.95 | 1406.89 | +132.2% | 61.37 | 52.88 | +16.1% |
| 4,096 | 2955.61 | 1292.78 | +128.6% | 59.73 | 51.45 | +16.1% |
| 8,192 | 2801.98 | 1198.97 | +133.7% | 58.52 | 50.44 | +16.0% |
| 12,288 | 2597.12 | 1116.84 | +132.5% | 57.15 | 49.38 | +15.7% |
| 16,384 | 2430.27 | 1049.84 | +131.5% | 56.14 | 48.43 | +15.9% |
| 32,768 | 1950.53 | 846.06 | +130.5% | 51.92 | 45.13 | +15.0% |
| 65,536 | 1403.08 | 611.04 | +129.6% | 45.17 | 39.44 | +14.5% |
| 131,072 | 902.10 | 395.50 | +128.1% | 35.88 | 31.57 | +13.7% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q4.svg)

---

<!-- bench:single-ar-q6 -->
| Qwen35B-A3B Q6 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3089.57 | 1244.48 | +148.3% | 55.05 | 47.99 | +14.7% |
| 4,096 | 2852.53 | 1175.64 | +142.6% | 53.78 | 47.49 | +13.2% |
| 8,192 | 2660.41 | 1100.87 | +141.7% | 52.85 | 46.87 | +12.8% |
| 12,288 | 2462.15 | 1034.19 | +138.1% | 51.69 | 45.94 | +12.5% |
| 16,384 | 2310.36 | 976.90 | +136.5% | 50.70 | 45.13 | +12.3% |
| 32,768 | 1873.21 | 797.00 | +135.0% | 47.41 | 42.17 | +12.4% |
| 65,536 | 1356.96 | 583.86 | +132.4% | 41.70 | 37.11 | +12.4% |
| 131,072 | 885.31 | 385.23 | +129.8% | 33.68 | 30.13 | +11.8% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q6.svg)

---

<!-- bench:single-ar-q8 -->
| Qwen35B-A3B Q8 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2846.65 | 1285.34 | +121.5% | 52.19 | 45.77 | +14.0% |
| 4,096 | 2655.22 | 1156.16 | +129.7% | 50.95 | 45.04 | +13.1% |
| 8,192 | 2473.93 | 1096.24 | +125.7% | 50.16 | 44.23 | +13.4% |
| 12,288 | 2328.92 | 1002.01 | +132.4% | 49.12 | 44.06 | +11.5% |
| 16,384 | 2198.86 | 950.59 | +131.3% | 48.38 | 43.32 | +11.7% |
| 32,768 | 1787.75 | 774.25 | +130.9% | 45.22 | 40.57 | +11.5% |
| 65,536 | 1309.72 | 574.61 | +127.9% | 39.93 | 35.80 | +11.5% |
| 131,072 | 864.14 | 377.60 | +128.9% | 32.54 | 29.32 | +11.0% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q8.svg)

---

<!-- bench:single-ar-q4km -->
| Qwen35B-A3B Q4_K_M AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3103.31 | 1344.48 | +130.8% | 70.71 | 62.61 | +12.9% |
| 4,096 | 2883.51 | 1239.47 | +132.6% | 68.48 | 61.34 | +11.6% |
| 8,192 | 2642.62 | 1152.16 | +129.4% | 66.89 | 60.29 | +10.9% |
| 12,288 | 2484.12 | 1084.03 | +129.2% | 65.09 | 58.67 | +10.9% |
| 16,384 | 2331.04 | 1020.81 | +128.4% | 63.64 | 57.42 | +10.8% |
| 32,768 | 1888.73 | 827.87 | +128.1% | 58.32 | 52.52 | +11.0% |
| 65,536 | 1362.72 | 599.07 | +127.5% | 50.01 | 45.13 | +10.8% |
| 131,072 | 888.27 | 393.85 | +125.5% | 38.90 | 35.15 | +10.7% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q4km.svg)

## Single user, DFlash2

pp is the highest measured rate per engine and depth across mixed/repetitive
text.

<!-- bench:single-dflash2-q4 -->
| Qwen35B-A3B Q4 DFlash2<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2989.93 | 1307.78 | +128.6% | 74.53 | 71.68 | +4.0% | 213.95 | 110.69 | +93.3% |
| 4,096 | 2730.72 | 1194.36 | +128.6% | 77.31 | 66.59 | +16.1% | 199.82 | 105.10 | +90.1% |
| 8,192 | 2541.22 | 1113.55 | +128.2% | 71.81 | 70.38 | +2.0% | 197.49 | 106.50 | +85.4% |
| 12,288 | 2396.08 | 1041.80 | +130.0% | 72.41 | 75.74 | -4.4% | 195.36 | 101.82 | +91.9% |
| 16,384 | 2249.92 | 981.11 | +129.3% | 71.78 | 66.16 | +8.5% | 167.60 | 102.91 | +62.9% |
| 32,768 | 1824.76 | 808.49 | +125.7% | 63.53 | 58.36 | +8.9% | 173.41 | 98.85 | +75.4% |
| 65,536 | 1339.82 | 589.76 | +127.2% | 58.30 | 58.65 | -0.6% | 143.77 | 79.62 | +80.6% |
| 131,072 | 864.84 | 383.37 | +125.6% | 54.58 | 46.40 | +17.6% | 118.85 | 59.31 | +100.4% |
<!-- /bench -->

![Single user, DFlash2](artifacts/charts/single-dflash2-q4.svg)

---

<!-- bench:single-dflash2-q6 -->
| Qwen35B-A3B Q6 DFlash2<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2875.90 | 1188.48 | +142.0% | 68.01 | 66.03 | +3.0% | 181.17 | 98.96 | +83.1% |
| 4,096 | 2610.46 | 1088.76 | +139.8% | 65.04 | 62.49 | +4.1% | 169.59 | 93.12 | +82.1% |
| 8,192 | 2418.80 | 1024.31 | +136.1% | 65.49 | 55.28 | +18.5% | 167.95 | 94.95 | +76.9% |
| 12,288 | 2276.96 | 963.31 | +136.4% | 63.09 | 61.68 | +2.3% | 165.27 | 91.62 | +80.4% |
| 16,384 | 2160.89 | 907.46 | +138.1% | 64.90 | 59.35 | +9.4% | 144.98 | 90.74 | +59.8% |
| 32,768 | 1765.37 | 760.81 | +132.0% | 60.27 | 53.51 | +12.6% | 149.15 | 90.82 | +64.2% |
| 65,536 | 1292.72 | 563.69 | +129.3% | 53.53 | 47.62 | +12.4% | 126.05 | 73.87 | +70.6% |
| 131,072 | 848.84 | 373.09 | +127.5% | 49.62 | 43.47 | +14.1% | 107.45 | 55.46 | +93.7% |
<!-- /bench -->

![Single user, DFlash2](artifacts/charts/single-dflash2-q6.svg)

---

<!-- bench:single-dflash2-q8 -->
| Qwen35B-A3B Q8 DFlash2<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2658.02 | 1212.36 | +119.2% | 65.44 | 63.24 | +3.5% | 174.06 | 90.83 | +91.6% |
| 4,096 | 2431.69 | 1110.30 | +119.0% | 62.85 | 61.97 | +1.4% | 162.00 | 86.48 | +87.3% |
| 8,192 | 2287.95 | 1024.27 | +123.4% | 65.68 | 56.42 | +16.4% | 161.97 | 86.62 | +87.0% |
| 12,288 | 2160.80 | 965.17 | +123.9% | 62.29 | 63.03 | -1.2% | 159.29 | 83.94 | +89.8% |
| 16,384 | 2041.08 | 908.96 | +124.6% | 65.56 | 58.73 | +11.6% | 147.61 | 85.59 | +72.5% |
| 32,768 | 1680.84 | 755.08 | +122.6% | 55.01 | 52.81 | +4.2% | 143.98 | 83.36 | +72.7% |
| 65,536 | 1266.57 | 561.23 | +125.7% | 52.43 | 47.03 | +11.5% | 121.36 | 68.52 | +77.1% |
| 131,072 | 828.15 | 371.68 | +122.8% | 50.99 | 41.63 | +22.5% | 104.32 | 52.15 | +100.0% |
<!-- /bench -->

![Single user, DFlash2](artifacts/charts/single-dflash2-q8.svg)

---

<!-- bench:single-dflash2-q4km -->
| Qwen35B-A3B Q4_K_M DFlash2<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2887.32 | 1266.45 | +128.0% | 81.11 | 82.15 | -1.3% | 228.82 | 120.43 | +90.0% |
| 4,096 | 2620.12 | 1167.48 | +124.4% | 78.56 | 66.06 | +18.9% | 199.55 | 108.15 | +84.5% |
| 8,192 | 2445.77 | 1080.28 | +126.4% | 82.16 | 66.09 | +24.3% | 213.04 | 108.83 | +95.8% |
| 12,288 | 2312.21 | 1013.29 | +128.2% | 78.57 | 73.40 | +7.0% | 208.88 | 102.92 | +103.0% |
| 16,384 | 2180.35 | 958.40 | +127.5% | 80.35 | 68.65 | +17.0% | 183.46 | 105.18 | +74.4% |
| 32,768 | 1780.91 | 793.40 | +124.5% | 74.24 | 65.11 | +14.0% | 178.15 | 101.86 | +74.9% |
| 65,536 | 1320.99 | 581.57 | +127.1% | 64.98 | 61.47 | +5.7% | 151.02 | 78.58 | +92.2% |
| 131,072 | 856.19 | 381.70 | +124.3% | 62.99 | 47.25 | +33.3% | 125.82 | 65.09 | +93.3% |
<!-- /bench -->

![Single user, DFlash2](artifacts/charts/single-dflash2-q4km.svg)

## Single user, MTP

The GGUF's own MTP head on both engines. pp is the highest measured rate per
engine and depth across mixed/repetitive text.

<!-- bench:single-mtp-q4 -->
| Qwen35B-A3B Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3234.43 | 1353.27 | +139.0% | 83.13 | 71.58 | +16.1% | 106.03 | 106.80 | -0.7% |
| 4,096 | 2895.62 | 1215.60 | +138.2% | 77.81 | 70.78 | +9.9% | 103.30 | 103.84 | -0.5% |
| 8,192 | 2688.18 | 1108.04 | +142.6% | 78.71 | 71.83 | +9.6% | 99.98 | 101.56 | -1.6% |
| 12,288 | 2509.25 | 1032.26 | +143.1% | 74.47 | 67.19 | +10.8% | 97.17 | 99.91 | -2.7% |
| 16,384 | 2353.25 | 966.42 | +143.5% | 74.41 | 69.09 | +7.7% | 95.45 | 98.58 | -3.2% |
| 32,768 | 1895.66 | 780.33 | +142.9% | 69.20 | 56.30 | +22.9% | 86.94 | 91.41 | -4.9% |
| 65,536 | 1377.90 | 558.58 | +146.7% | 55.34 | 51.16 | +8.2% | 73.74 | 77.66 | -5.0% |
| 131,072 | 890.76 | 358.22 | +148.7% | 44.80 | 42.46 | +5.5% | 55.15 | 62.78 | -12.2% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q4.svg)

---

<!-- bench:single-mtp-q6 -->
| Qwen35B-A3B Q6 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3050.12 | 1229.36 | +148.1% | 67.37 | 55.92 | +20.5% | 94.98 | 95.74 | -0.8% |
| 4,096 | 2756.32 | 1103.94 | +149.7% | 69.28 | 64.00 | +8.3% | 92.94 | 92.96 | -0.0% |
| 8,192 | 2532.56 | 1018.04 | +148.8% | 71.39 | 64.82 | +10.1% | 89.91 | 90.98 | -1.2% |
| 12,288 | 2397.55 | 958.12 | +150.2% | 68.19 | 57.52 | +18.6% | 87.38 | 88.10 | -0.8% |
| 16,384 | 2258.32 | 895.61 | +152.2% | 69.74 | 63.87 | +9.2% | 86.32 | 88.42 | -2.4% |
| 32,768 | 1825.95 | 736.49 | +147.9% | 62.25 | 56.45 | +10.3% | 79.09 | 82.56 | -4.2% |
| 65,536 | 1345.28 | 536.13 | +150.9% | 50.30 | 46.05 | +9.2% | 68.18 | 72.56 | -6.0% |
| 131,072 | 866.30 | 346.69 | +149.9% | 41.91 | 36.02 | +16.4% | 51.93 | 57.91 | -10.3% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q6.svg)

---

<!-- bench:single-mtp-q8 -->
| Qwen35B-A3B Q8 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2821.27 | 1254.59 | +124.9% | 63.81 | 59.00 | +8.2% | 88.58 | 88.43 | +0.2% |
| 4,096 | 2541.91 | 1138.95 | +123.2% | 68.37 | 60.48 | +13.0% | 86.88 | 86.43 | +0.5% |
| 8,192 | 2405.36 | 1020.49 | +135.7% | 70.69 | 59.78 | +18.3% | 84.03 | 85.00 | -1.1% |
| 12,288 | 2261.74 | 956.13 | +136.6% | 65.56 | 53.75 | +22.0% | 82.01 | 83.53 | -1.8% |
| 16,384 | 2133.22 | 894.66 | +138.4% | 67.95 | 58.97 | +15.2% | 81.17 | 82.95 | -2.1% |
| 32,768 | 1736.77 | 731.51 | +137.4% | 58.77 | 54.15 | +8.5% | 74.55 | 77.29 | -3.5% |
| 65,536 | 1298.06 | 528.60 | +145.6% | 51.03 | 45.34 | +12.5% | 64.75 | 67.10 | -3.5% |
| 131,072 | 846.52 | 346.35 | +144.4% | 38.98 | 36.09 | +8.0% | 49.86 | 54.62 | -8.7% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q8.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
Every session is prefilled before timed decoding; throughput sums individual
request decode rates.

<!-- bench:multi-ar-q4 -->
| Qwen35B-A3B Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 61.40 | 52.81 | +16.3% |
| 2 | 104.64 | 85.42 | +22.5% |
| 4 | 165.44 | 136.37 | +21.3% |
| 6 | 207.32 | 150.53 | +37.7% |
| 8 | 230.91 | 170.00 | +35.8% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q4.svg)

---

<!-- bench:multi-ar-q6 -->
| Qwen35B-A3B Q6 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 55.09 | 48.24 | +14.2% |
| 2 | 92.45 | 81.70 | +13.2% |
| 4 | 139.84 | 124.43 | +12.4% |
| 6 | 171.66 | 138.28 | +24.1% |
| 8 | 192.01 | 148.36 | +29.4% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q6.svg)

---

<!-- bench:multi-ar-q8 -->
| Qwen35B-A3B Q8 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 52.17 | 46.52 | +12.1% |
| 2 | 89.57 | 71.36 | +25.5% |
| 4 | 144.71 | 106.06 | +36.4% |
| 6 | 184.79 | 125.10 | +47.7% |
| 8 | 212.93 | 142.36 | +49.6% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q8.svg)

---

<!-- bench:multi-ar-q4km -->
| Qwen35B-A3B Q4_K_M AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 70.68 | 62.69 | +12.7% |
| 2 | 124.31 | 104.16 | +19.3% |
| 4 | 192.25 | 147.29 | +30.5% |
| 6 | 231.22 | 150.19 | +54.0% |
| 8 | 244.19 | 166.23 | +46.9% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q4km.svg)

## Multiple users, DFlash2

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. Every session is prefilled before timed decoding.

<!-- bench:multi-dflash2-q4 -->
| Qwen35B-A3B Q4 DFlash2<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 74.18 | 71.29 | +4.1% | 211.77 | 109.64 | +93.2% |
| 2 | 122.23 | 97.96 | +24.8% | 199.89 | 151.40 | +32.0% |
| 4 | 154.26 | 130.69 | +18.0% | 226.11 | 163.99 | +37.9% |
| 6 | 162.46 | 152.11 | +6.8% | 233.06 | 199.36 | +16.9% |
| 8 | 177.65 | 166.10 | +7.0% | 253.43 | 227.83 | +11.2% |
<!-- /bench -->

![Multiple users, DFlash2](artifacts/charts/multi-dflash2-q4.svg)

---

<!-- bench:multi-dflash2-q6 -->
| Qwen35B-A3B Q6 DFlash2<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 67.95 | 67.80 | +0.2% | 180.15 | 99.51 | +81.0% |
| 2 | 106.67 | 87.30 | +22.2% | 169.67 | 133.80 | +26.8% |
| 4 | 134.59 | 123.51 | +9.0% | 190.58 | 156.25 | +22.0% |
| 6 | 137.69 | 148.31 | -7.2% | 189.52 | 186.66 | +1.5% |
| 8 | 147.67 | 158.16 | -6.6% | 208.98 | 207.36 | +0.8% |
<!-- /bench -->

![Multiple users, DFlash2](artifacts/charts/multi-dflash2-q6.svg)

---

<!-- bench:multi-dflash2-q8 -->
| Qwen35B-A3B Q8 DFlash2<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 65.34 | 62.75 | +4.1% | 173.04 | 89.99 | +92.3% |
| 2 | 102.47 | 81.50 | +25.7% | 168.26 | 142.68 | +17.9% |
| 4 | 142.53 | 113.51 | +25.6% | 185.62 | 147.69 | +25.7% |
| 6 | 149.45 | 126.99 | +17.7% | 207.86 | 173.99 | +19.5% |
| 8 | 166.72 | 147.45 | +13.1% | 233.92 | 189.30 | +23.6% |
<!-- /bench -->

![Multiple users, DFlash2](artifacts/charts/multi-dflash2-q8.svg)

---

<!-- bench:multi-dflash2-q4km -->
| Qwen35B-A3B Q4_K_M DFlash2<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 80.98 | 81.52 | -0.7% | 224.57 | 116.07 | +93.5% |
| 2 | 127.86 | 100.15 | +27.7% | 230.73 | 127.96 | +80.3% |
| 4 | 161.82 | 134.69 | +20.1% | 228.61 | 181.30 | +26.1% |
| 6 | 167.56 | 165.46 | +1.3% | 242.12 | 216.98 | +11.6% |
| 8 | 182.52 | 166.60 | +9.6% | 265.88 | 231.36 | +14.9% |
<!-- /bench -->

![Multiple users, DFlash2](artifacts/charts/multi-dflash2-q4km.svg)

## Multiple users, MTP

Same prompts and method as the DFlash2 table.

<!-- bench:multi-mtp-q4 -->
| Qwen35B-A3B Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 82.45 | 71.01 | +16.1% | 104.69 | 106.69 | -1.9% |
| 2 | 117.87 | 96.44 | +22.2% | 134.73 | 116.20 | +15.9% |
| 4 | 159.19 | 130.00 | +22.5% | 151.91 | 125.09 | +21.4% |
| 6 | 197.82 | 151.92 | +30.2% | 189.94 | 134.24 | +41.5% |
| 8 | 219.99 | 171.86 | +28.0% | 220.12 | 146.10 | +50.7% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q4.svg)

---

<!-- bench:multi-mtp-q6 -->
| Qwen35B-A3B Q6 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 67.09 | 59.63 | +12.5% | 93.51 | 95.81 | -2.4% |
| 2 | 97.41 | 92.55 | +5.3% | 114.06 | 100.17 | +13.9% |
| 4 | 136.33 | 110.24 | +23.7% | 131.61 | 128.68 | +2.3% |
| 6 | 164.73 | 136.14 | +21.0% | 160.17 | 113.82 | +40.7% |
| 8 | 183.09 | 143.40 | +27.7% | 182.51 | 134.93 | +35.3% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q6.svg)

---

<!-- bench:multi-mtp-q8 -->
| Qwen35B-A3B Q8 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 62.97 | 59.64 | +5.6% | 87.71 | 88.41 | -0.8% |
| 2 | 99.82 | 81.48 | +22.5% | 113.60 | 91.26 | +24.5% |
| 4 | 140.54 | 107.25 | +31.0% | 140.03 | 107.06 | +30.8% |
| 6 | 177.13 | 129.17 | +37.1% | 167.30 | 109.75 | +52.4% |
| 8 | 202.45 | 144.96 | +39.7% | 191.80 | 126.52 | +51.6% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q8.svg)

## Loading time

C1, context capacity 133760, DFlash2. Cold target/draft files to HTTP readiness.

<!-- bench:loading -->
| Qwen35B-A3B<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 9.92 | 9.62 | -3.0% |
| Q6 | 13.97 | 13.23 | -5.3% |
| Q8 | 16.51 | 15.37 | -6.9% |
| Q4_K_M | 9.65 | 9.53 | -1.2% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133760, AR. Peak memory reported by HIP. llama-server
allocates its KV cache up front, so its peak does not grow with the prefix.

<!-- bench:memory-q4 -->
| Qwen35B-A3B Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 31.59 | 26.64 | -15.7% |
| 16K prefix, pp4096 + tg128 | 32.59 | 26.81 | -17.7% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q4.svg)

---

<!-- bench:memory-q6 -->
| Qwen35B-A3B Q6 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 40.68 | 35.50 | -12.7% |
| 16K prefix, pp4096 + tg128 | 41.68 | 35.68 | -14.4% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q6.svg)

---

<!-- bench:memory-q8 -->
| Qwen35B-A3B Q8 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 46.72 | 41.46 | -11.3% |
| 16K prefix, pp4096 + tg128 | 47.72 | 41.64 | -12.7% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q8.svg)

---

<!-- bench:memory-q4km -->
| Qwen35B-A3B Q4_K_M AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 30.02 | 25.59 | -14.8% |
| 16K prefix, pp4096 + tg128 | 31.02 | 25.77 | -16.9% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q4km.svg)
