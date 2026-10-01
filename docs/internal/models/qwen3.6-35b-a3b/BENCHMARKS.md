# Qwen3.6 35B-A3B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory (126976 MiB reported). Gufo
`bec9787` plus the uncommitted Qwen3.6-35B-A3B MoE work. Built on the host,
run inside the ROCm 7.2.3 container (the host's ROCm 7.1 cannot run Gufo).
Measured September 26, 2026, with the production inference services and the
OOM cron stopped so the benchmark owned the GPU.

Prefill and generation are 3 repetitions; the 27B regression check is 2.
The llama.cpp comparison uses the ROCm fork `llama.cpp-flashnext-hip`
b10672 (`llama-bench -ngl 99 -fa on -r 3`), measured the same day.

pp512 rows with a large ± include one slow warm-up repetition and are not
representative: an interleaved A/B of the same Q8_K_XL build measured
1294.77 ± 13.30 t/s at pp512, and llama.cpp's Q6_K_XL pp512 (809.83 ± 223.14)
is affected the same way. Treat pp512 gains as indicative only.

## Qwen3.6 35B-A3B, UD-Q8_K_XL

| Prefill prompt (tokens) | pp (t/s) |
| ---: | ---: |
| 512 | 1188.74 ± 465.09 |
| 1024 | 1793.14 ± 1.92 |
| 2048 | 1963.22 ± 10.27 |
| 4096 | 2029.74 ± 7.10 |
| 8192 | 1981.57 ± 2.17 |
| 16384 | 1875.30 ± 1.17 |

| Workload | t/s |
| --- | ---: |
| tg128 | 51.43 ± 0.02 |

| Prompt (tokens) | gufo (t/s) | llama.cpp (t/s) | gain |
| ---: | ---: | ---: | ---: |
| 512 | 1188.74 (see note) | 1070.72 | +11.0% |
| 1024 | 1793.14 | 1061.29 | +69.0% |
| 2048 | 1963.22 | 1032.33 | +90.2% |
| 4096 | 2029.74 | 1003.12 | +102.3% |
| 8192 | 1981.57 | 960.41 | +106.3% |
| 16384 | 1875.30 | 881.82 | +112.7% |
| tg128 | 51.43 | n/a | n/a |

Cold mapped-weight load of the 36.65 GiB artifact to readiness is roughly
5–11 s depending on page-cache warmth. The single-shot `hipHostRegister`
(no 4 GiB chunking) loads the model cleanly with the production services
stopped; see [Experiments](EXPERIMENTS.md).

## Qwen3.6 35B-A3B, UD-Q6_K_XL

| Prefill prompt (tokens) | pp (t/s) |
| ---: | ---: |
| 512 | 1319.28 ± 10.56 |
| 1024 | 1651.77 ± 32.85 |
| 2048 | 1891.08 ± 2.35 |
| 4096 | 2000.77 ± 6.76 |
| 8192 | 1953.46 ± 5.87 |
| 16384 | 1852.64 ± 0.53 |

| Workload | t/s |
| --- | ---: |
| tg128 | 53.95 ± 0.04 |

| Prompt (tokens) | gufo (t/s) | llama.cpp (t/s) | gain |
| ---: | ---: | ---: | ---: |
| 512 | 1319.28 | 809.83 (see note) | +62.9% |
| 1024 | 1651.77 | 942.11 | +75.3% |
| 2048 | 1891.08 | 930.46 | +103.2% |
| 4096 | 2000.77 | 920.41 | +117.4% |
| 8192 | 1953.46 | 867.42 | +125.2% |
| 16384 | 1852.64 | 800.98 | +131.3% |
| tg128 | 53.95 | 48.83 | +10.5% |

## Prefill after the October 1, 2026 kernel changes

Routed down projection on the mixed pair map, one-wave GDN post-norm and the
fused residual/RMSNorm/quantize pass (see [Experiments](EXPERIMENTS.md)),
measured October 1, 2026 as an interleaved A/B against the build of the tree
they were developed on: `gufo bench --n-prompt 1024,2048,4096,8192 --n-gen 0
--repetitions 3`, three rounds for UD-Q6_K_XL and two for UD-Q8_K_XL, mean of
the rounds, production inference services and the OOM cron stopped. The tables
above predate the routed F16 expert GEMM and are not comparable with these.

| Prompt (tokens) | UD-Q6_K_XL before (t/s) | after (t/s) | gain | UD-Q8_K_XL before (t/s) | after (t/s) | gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 2268.9 | 2439.8 | +7.5% | 2065.1 | 2186.6 | +5.9% |
| 2048 | 2581.6 | 2715.7 | +5.2% | 2305.8 | 2438.3 | +5.7% |
| 4096 | 2695.6 | 2860.8 | +6.1% | 2396.4 | 2530.6 | +5.6% |
| 8192 | 2603.0 | 2746.6 | +5.5% | 2321.5 | 2448.9 | +5.5% |

Server mode, one session, greedy with thinking off: a 12,262-token prompt
prefills at 2475 → 2617 t/s on UD-Q6_K_XL and 2204 → 2343 t/s on UD-Q8_K_XL,
a 6,113-token prompt at 2572 → 2708 and 2318 → 2422 t/s. The 128-token
completions are byte-identical before and after on both artifacts.

## Qwen3.8 27B regression (UD-Q8_K_XL)

Confirms the shared attention/GDN path is unchanged by the MoE work.

| Prefill prompt (tokens) | t/s |
| ---: | ---: |
| 2048 | 268.88 ± 0.57 |
| 8192 | 265.06 ± 0.53 |

## Reproduce

```sh
cmake --build --preset release --parallel 16
# stop the production services + OOM cron first (see the handoff)
for QUANT in Q8_K_XL Q6_K_XL; do
  /home/sami/gufo/run-in-container.sh bench \
    --model /home/sami/models-mtp/Qwen3.6-35B-A3B-MTP-UD-$QUANT.gguf \
    --n-prompt 512,1024,2048,4096,8192,16384 --n-gen 128 --repetitions 3
done
/home/sami/gufo/run-in-container.sh bench \
  --model /home/sami/models-gufo/qwen38-27b/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --n-prompt 2048,8192 --n-gen 0 --repetitions 2
```

Greedy, thinking off. [Quality and measurement scope](QUALITY.md).
