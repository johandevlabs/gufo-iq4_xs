# Qwen3.6 35B-A3B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Gufo `b43f7b4d` versus
llama.cpp `b11069` (`68d9053a`), both compiled with the pinned ROCm 7.2.3
toolchain. Targets are the Unsloth UD-Q4_K_XL / UD-Q6_K_XL / UD-Q8_K_XL files;
DFlash2 uses the aminya Q8_0 draft and MTP uses each target's own head
([files and revisions](README.md)). HTTP, greedy, thinking off. Every row
was measured on October 25, 2026.

Gufo drafts up to 7 DFlash2 tokens with its adaptive controller and 2 MTP
tokens; llama.cpp uses its default of 3 for both modes. Its repetitive MTP
rows therefore accept up to 2.88 tokens per step against Gufo's 1.91.

Single-user speculative rows reuse the cached prefix one token earlier than the
AR row at most depths, so their completions are not compared with AR. At
depth 0, Gufo DFlash2 and MTP give the same completion, which matches Gufo AR
on Q4 and differs from it on Q6 and Q8.

The multi-user tables check every completion against Gufo's isolated C1 AR
output. Q4 passes at every level. On Q6 and Q8, batched decode and
speculative verification take a different MoE route than one-token decode,
and outputs diverge after near-tied tokens (see
[README](README.md#speculative-decoding-dflash2)); production `8e509128` shows
the same differences. Those Gufo cells were measured with the check downgraded
to a warning, and each artifact records its exact-match count. On the prose
and mixed prompts no Q6 or Q8 request matches (AR from C2, DFlash2 and MTP from C1); every
repetitive request matches.

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar-q4 -->
| Qwen35B-A3B Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3168.55 | 1406.89 | +125.2% | 56.00 | 52.88 | +5.9% |
| 4,096 | 2930.98 | 1292.78 | +126.7% | 54.55 | 51.45 | +6.0% |
| 8,192 | 2733.91 | 1198.97 | +128.0% | 53.59 | 50.44 | +6.2% |
| 12,288 | 2532.36 | 1116.84 | +126.7% | 52.34 | 49.38 | +6.0% |
| 16,384 | 2373.80 | 1049.84 | +126.1% | 51.49 | 48.43 | +6.3% |
| 32,768 | 1914.89 | 846.06 | +126.3% | 47.89 | 45.13 | +6.1% |
| 65,536 | 1382.03 | 611.04 | +126.2% | 42.14 | 39.44 | +6.8% |
| 131,072 | 894.22 | 395.50 | +126.1% | 33.99 | 31.57 | +7.7% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q4.svg)

---

<!-- bench:single-ar-q6 -->
| Qwen35B-A3B Q6 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3079.27 | 1244.48 | +147.4% | 55.07 | 47.99 | +14.8% |
| 4,096 | 2855.26 | 1175.64 | +142.9% | 53.78 | 47.49 | +13.2% |
| 8,192 | 2671.92 | 1100.87 | +142.7% | 52.88 | 46.87 | +12.8% |
| 12,288 | 2477.88 | 1034.19 | +139.6% | 51.66 | 45.94 | +12.5% |
| 16,384 | 2330.05 | 976.90 | +138.5% | 50.79 | 45.13 | +12.5% |
| 32,768 | 1875.94 | 797.00 | +135.4% | 47.39 | 42.17 | +12.4% |
| 65,536 | 1353.40 | 583.86 | +131.8% | 41.66 | 37.11 | +12.3% |
| 131,072 | 883.25 | 385.23 | +129.3% | 33.67 | 30.13 | +11.7% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q6.svg)

---

<!-- bench:single-ar-q8 -->
| Qwen35B-A3B Q8 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2831.96 | 1285.34 | +120.3% | 52.07 | 45.77 | +13.8% |
| 4,096 | 2654.24 | 1156.16 | +129.6% | 50.97 | 45.04 | +13.2% |
| 8,192 | 2484.09 | 1096.24 | +126.6% | 50.25 | 44.23 | +13.6% |
| 12,288 | 2336.13 | 1002.01 | +133.1% | 49.20 | 44.06 | +11.7% |
| 16,384 | 2194.73 | 950.59 | +130.9% | 48.43 | 43.32 | +11.8% |
| 32,768 | 1792.66 | 774.25 | +131.5% | 45.23 | 40.57 | +11.5% |
| 65,536 | 1320.39 | 574.61 | +129.8% | 40.01 | 35.80 | +11.8% |
| 131,072 | 864.89 | 377.60 | +129.0% | 32.63 | 29.32 | +11.3% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q8.svg)

## Single user, DFlash2

pp is the highest measured rate per engine and depth across mixed/repetitive
text.

<!-- bench:single-dflash2-q4 -->
| Qwen35B-A3B Q4 DFlash2<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2958.43 | 1307.78 | +126.2% | 69.29 | 71.68 | -3.3% | 166.24 | 110.69 | +50.2% |
| 4,096 | 2606.85 | 1194.36 | +118.3% | 65.51 | 66.59 | -1.6% | 156.32 | 105.10 | +48.7% |
| 8,192 | 2492.52 | 1113.55 | +123.8% | 59.37 | 70.38 | -15.6% | 153.67 | 106.50 | +44.3% |
| 12,288 | 2348.00 | 1041.80 | +125.4% | 67.71 | 75.74 | -10.6% | 151.66 | 101.82 | +48.9% |
| 16,384 | 2210.53 | 981.11 | +125.3% | 60.40 | 66.16 | -8.7% | 133.58 | 102.91 | +29.8% |
| 32,768 | 1798.94 | 808.49 | +122.5% | 58.83 | 58.36 | +0.8% | 138.10 | 98.85 | +39.7% |
| 65,536 | 1316.81 | 589.76 | +123.3% | 53.22 | 58.65 | -9.3% | 118.64 | 79.62 | +49.0% |
| 131,072 | 861.17 | 383.37 | +124.6% | 49.69 | 46.40 | +7.1% | 101.87 | 59.31 | +71.8% |
<!-- /bench -->

![Single user, DFlash2](artifacts/charts/single-dflash2-q4.svg)

---

<!-- bench:single-dflash2-q6 -->
| Qwen35B-A3B Q6 DFlash2<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2871.23 | 1188.48 | +141.6% | 64.22 | 66.03 | -2.7% | 181.73 | 98.96 | +83.6% |
| 4,096 | 2607.55 | 1088.76 | +139.5% | 64.71 | 62.49 | +3.6% | 169.81 | 93.12 | +82.4% |
| 8,192 | 2425.85 | 1024.31 | +136.8% | 71.44 | 55.28 | +29.2% | 167.89 | 94.95 | +76.8% |
| 12,288 | 2274.59 | 963.31 | +136.1% | 60.12 | 61.68 | -2.5% | 165.37 | 91.62 | +80.5% |
| 16,384 | 2156.47 | 907.46 | +137.6% | 63.74 | 59.35 | +7.4% | 152.92 | 90.74 | +68.5% |
| 32,768 | 1750.11 | 760.81 | +130.0% | 58.93 | 53.51 | +10.1% | 148.64 | 90.82 | +63.7% |
| 65,536 | 1302.61 | 563.69 | +131.1% | 54.20 | 47.62 | +13.8% | 126.17 | 73.87 | +70.8% |
| 131,072 | 849.40 | 373.09 | +127.7% | 49.67 | 43.47 | +14.3% | 107.56 | 55.46 | +93.9% |
<!-- /bench -->

![Single user, DFlash2](artifacts/charts/single-dflash2-q6.svg)

---

<!-- bench:single-dflash2-q8 -->
| Qwen35B-A3B Q8 DFlash2<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2669.52 | 1212.36 | +120.2% | 68.07 | 63.24 | +7.6% | 174.21 | 90.83 | +91.8% |
| 4,096 | 2425.44 | 1110.30 | +118.4% | 61.12 | 61.97 | -1.4% | 162.30 | 86.48 | +87.7% |
| 8,192 | 2282.56 | 1024.27 | +122.8% | 57.29 | 56.42 | +1.5% | 162.03 | 86.62 | +87.1% |
| 12,288 | 2152.52 | 965.17 | +123.0% | 58.80 | 63.03 | -6.7% | 159.82 | 83.94 | +90.4% |
| 16,384 | 2047.09 | 908.96 | +125.2% | 61.69 | 58.73 | +5.0% | 147.37 | 85.59 | +72.2% |
| 32,768 | 1680.87 | 755.08 | +122.6% | 62.82 | 52.81 | +19.0% | 144.50 | 83.36 | +73.3% |
| 65,536 | 1262.59 | 561.23 | +125.0% | 51.15 | 47.03 | +8.8% | 123.31 | 68.52 | +80.0% |
| 131,072 | 833.43 | 371.68 | +124.2% | 46.72 | 41.63 | +12.2% | 104.67 | 52.15 | +100.7% |
<!-- /bench -->

![Single user, DFlash2](artifacts/charts/single-dflash2-q8.svg)

## Single user, MTP

The GGUF's own MTP head on both engines. pp is the highest measured rate per
engine and depth across mixed/repetitive text.

<!-- bench:single-mtp-q4 -->
| Qwen35B-A3B Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3160.15 | 1353.27 | +133.5% | 70.52 | 71.58 | -1.5% | 93.33 | 106.80 | -12.6% |
| 4,096 | 2863.82 | 1215.60 | +135.6% | 73.33 | 70.78 | +3.6% | 91.20 | 103.84 | -12.2% |
| 8,192 | 2602.31 | 1108.04 | +134.9% | 69.96 | 71.83 | -2.6% | 88.23 | 101.56 | -13.1% |
| 12,288 | 2468.44 | 1032.26 | +139.1% | 65.05 | 67.19 | -3.2% | 86.08 | 99.91 | -13.8% |
| 16,384 | 2317.83 | 966.42 | +139.8% | 68.43 | 69.09 | -1.0% | 84.28 | 98.58 | -14.5% |
| 32,768 | 1871.82 | 780.33 | +139.9% | 61.29 | 56.30 | +8.9% | 77.88 | 91.41 | -14.8% |
| 65,536 | 1352.83 | 558.58 | +142.2% | 51.91 | 51.16 | +1.5% | 67.27 | 77.66 | -13.4% |
| 131,072 | 877.59 | 358.22 | +145.0% | 41.56 | 42.46 | -2.1% | 51.32 | 62.78 | -18.3% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q4.svg)

---

<!-- bench:single-mtp-q6 -->
| Qwen35B-A3B Q6 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3057.46 | 1229.36 | +148.7% | 66.95 | 55.92 | +19.7% | 95.08 | 95.74 | -0.7% |
| 4,096 | 2780.64 | 1103.94 | +151.9% | 73.56 | 64.00 | +14.9% | 93.22 | 92.96 | +0.3% |
| 8,192 | 2543.43 | 1018.04 | +149.8% | 71.40 | 64.82 | +10.2% | 90.38 | 90.98 | -0.7% |
| 12,288 | 2401.30 | 958.12 | +150.6% | 68.64 | 57.52 | +19.3% | 87.52 | 88.10 | -0.7% |
| 16,384 | 2259.23 | 895.61 | +152.3% | 70.06 | 63.87 | +9.7% | 86.54 | 88.42 | -2.1% |
| 32,768 | 1832.64 | 736.49 | +148.8% | 58.61 | 56.45 | +3.8% | 79.38 | 82.56 | -3.9% |
| 65,536 | 1332.32 | 536.13 | +148.5% | 53.64 | 46.05 | +16.5% | 68.16 | 72.56 | -6.1% |
| 131,072 | 864.78 | 346.69 | +149.4% | 40.64 | 36.02 | +12.8% | 52.16 | 57.91 | -9.9% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q6.svg)

---

<!-- bench:single-mtp-q8 -->
| Qwen35B-A3B Q8 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2804.10 | 1254.59 | +123.5% | 65.07 | 59.00 | +10.3% | 88.74 | 88.43 | +0.4% |
| 4,096 | 2580.20 | 1138.95 | +126.5% | 61.39 | 60.48 | +1.5% | 87.02 | 86.43 | +0.7% |
| 8,192 | 2428.65 | 1020.49 | +138.0% | 62.19 | 59.78 | +4.0% | 84.46 | 85.00 | -0.6% |
| 12,288 | 2276.44 | 956.13 | +138.1% | 61.71 | 53.75 | +14.8% | 82.21 | 83.53 | -1.6% |
| 16,384 | 2146.99 | 894.66 | +140.0% | 65.88 | 58.97 | +11.7% | 81.13 | 82.95 | -2.2% |
| 32,768 | 1751.61 | 731.51 | +139.5% | 59.83 | 54.15 | +10.5% | 74.56 | 77.29 | -3.5% |
| 65,536 | 1298.99 | 528.60 | +145.7% | 50.30 | 45.34 | +10.9% | 64.88 | 67.10 | -3.3% |
| 131,072 | 847.94 | 346.35 | +144.8% | 38.76 | 36.09 | +7.4% | 49.94 | 54.62 | -8.6% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp-q8.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
Every session is prefilled before timed decoding; throughput sums individual
request decode rates.

<!-- bench:multi-ar-q4 -->
| Qwen35B-A3B Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 55.85 | 52.81 | +5.8% |
| 2 | 88.68 | 85.42 | +3.8% |
| 4 | 128.38 | 136.37 | -5.9% |
| 6 | 150.63 | 150.53 | +0.1% |
| 8 | 166.01 | 170.00 | -2.3% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q4.svg)

---

<!-- bench:multi-ar-q6 -->
| Qwen35B-A3B Q6 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 55.04 | 48.24 | +14.1% |
| 2 | 92.73 | 81.70 | +13.5% |
| 4 | 140.02 | 124.43 | +12.5% |
| 6 | 171.34 | 138.28 | +23.9% |
| 8 | 191.16 | 148.36 | +28.8% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q6.svg)

---

<!-- bench:multi-ar-q8 -->
| Qwen35B-A3B Q8 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 52.09 | 46.52 | +12.0% |
| 2 | 89.01 | 71.36 | +24.7% |
| 4 | 143.36 | 106.06 | +35.2% |
| 6 | 185.93 | 125.10 | +48.6% |
| 8 | 213.97 | 142.36 | +50.3% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q8.svg)

## Multiple users, DFlash2

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. Every session is prefilled before timed decoding.

<!-- bench:multi-dflash2-q4 -->
| Qwen35B-A3B Q4 DFlash2<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 69.28 | 71.29 | -2.8% | 164.75 | 109.64 | +50.3% |
| 2 | 98.18 | 97.96 | +0.2% | 156.91 | 151.40 | +3.6% |
| 4 | 127.57 | 130.69 | -2.4% | 164.79 | 163.99 | +0.5% |
| 6 | 129.66 | 152.11 | -14.8% | 165.28 | 199.36 | -17.1% |
| 8 | 139.11 | 166.10 | -16.2% | 176.19 | 227.83 | -22.7% |
<!-- /bench -->

![Multiple users, DFlash2](artifacts/charts/multi-dflash2-q4.svg)

---

<!-- bench:multi-dflash2-q6 -->
| Qwen35B-A3B Q6 DFlash2<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64.05 | 67.80 | -5.5% | 180.89 | 99.51 | +81.8% |
| 2 | 94.28 | 87.30 | +8.0% | 170.24 | 133.80 | +27.2% |
| 4 | 131.35 | 123.51 | +6.3% | 187.41 | 156.25 | +19.9% |
| 6 | 137.98 | 148.31 | -7.0% | 193.09 | 186.66 | +3.4% |
| 8 | 150.49 | 158.16 | -4.8% | 211.12 | 207.36 | +1.8% |
<!-- /bench -->

![Multiple users, DFlash2](artifacts/charts/multi-dflash2-q6.svg)

---

<!-- bench:multi-dflash2-q8 -->
| Qwen35B-A3B Q8 DFlash2<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 67.85 | 62.75 | +8.1% | 172.55 | 89.99 | +91.7% |
| 2 | 110.75 | 81.50 | +35.9% | 171.62 | 142.68 | +20.3% |
| 4 | 145.61 | 113.51 | +28.3% | 186.03 | 147.69 | +26.0% |
| 6 | 154.06 | 126.99 | +21.3% | 207.14 | 173.99 | +19.1% |
| 8 | 170.28 | 147.45 | +15.5% | 224.75 | 189.30 | +18.7% |
<!-- /bench -->

![Multiple users, DFlash2](artifacts/charts/multi-dflash2-q8.svg)

## Multiple users, MTP

Same prompts and method as the DFlash2 table.

<!-- bench:multi-mtp-q4 -->
| Qwen35B-A3B Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 70.37 | 71.01 | -0.9% | 92.96 | 106.69 | -12.9% |
| 2 | 96.82 | 96.44 | +0.4% | 112.01 | 116.20 | -3.6% |
| 4 | 125.29 | 130.00 | -3.6% | 125.27 | 125.09 | +0.1% |
| 6 | 147.59 | 151.92 | -2.9% | 147.03 | 134.24 | +9.5% |
| 8 | 159.48 | 171.86 | -7.2% | 159.37 | 146.10 | +9.1% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q4.svg)

---

<!-- bench:multi-mtp-q6 -->
| Qwen35B-A3B Q6 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 66.49 | 59.63 | +11.5% | 93.71 | 95.81 | -2.2% |
| 2 | 98.69 | 92.55 | +6.6% | 116.86 | 100.17 | +16.7% |
| 4 | 136.44 | 110.24 | +23.8% | 131.75 | 128.68 | +2.4% |
| 6 | 164.73 | 136.14 | +21.0% | 160.45 | 113.82 | +41.0% |
| 8 | 183.53 | 143.40 | +28.0% | 178.98 | 134.93 | +32.6% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q6.svg)

---

<!-- bench:multi-mtp-q8 -->
| Qwen35B-A3B Q8 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64.80 | 59.64 | +8.7% | 87.57 | 88.41 | -1.0% |
| 2 | 102.68 | 81.48 | +26.0% | 113.97 | 91.26 | +24.9% |
| 4 | 140.61 | 107.25 | +31.1% | 129.88 | 107.06 | +21.3% |
| 6 | 178.06 | 129.17 | +37.8% | 166.49 | 109.75 | +51.7% |
| 8 | 203.32 | 144.96 | +40.3% | 202.63 | 126.52 | +60.2% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp-q8.svg)

## Loading time

C1, context capacity 133760, DFlash2. Cold target/draft files to HTTP readiness.

<!-- bench:loading -->
| Qwen35B-A3B<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 9.92 | 9.57 | -3.5% |
| Q6 | 13.86 | 13.16 | -5.1% |
| Q8 | 16.56 | 15.34 | -7.4% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133760, AR. Peak memory reported by HIP. llama-server
allocates its KV cache up front, so its peak does not grow with the prefix.

<!-- bench:memory-q4 -->
| Qwen35B-A3B Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 31.34 | 26.64 | -15.0% |
| 16K prefix, pp4096 + tg128 | 32.34 | 26.81 | -17.1% |
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
