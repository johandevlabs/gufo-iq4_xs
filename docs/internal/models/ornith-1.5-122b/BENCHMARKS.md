# Ornith 1.5 122B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Gufo (Strycast `staging`
plus this model's commits) versus llama.cpp `68d9053a`, the revision Strycast
pins, on `Ornith-1.5-122B-Q4_K_M.gguf` ([files and revisions](README.md)).
Both were compiled with host ROCm 7.2.2, not the pinned 7.2.3, and llama.cpp
with the same `--amdgpu-unroll-threshold-local=600` HIP flag as the pinned
recipe. HTTP through `tools/bench/model-bench.py`, greedy, thinking off.
Measured October 28, 2026, inside one GPU admission hold so no other model
shared the GPU.

Gufo drafts 2 MTP tokens from the GGUF's own MTP layer. llama.cpp's
`draft-mtp` loads the whole 72 GiB GGUF a second time as its draft model,
which does not fit in 128 GB, so its MTP cells are N/A.

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens. Depth
131,072 is not measured yet.

<!-- bench:single-ar-q4km -->
| Ornith 122B Q4_K_M AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1014.29 | 368.06 | +175.6% | 28.14 | 26.70 | +5.4% |
| 4,096 | 949.21 | 344.41 | +175.6% | 27.38 | 25.92 | +5.6% |
| 8,192 | 886.06 | 327.77 | +170.3% | 26.63 | 25.18 | +5.8% |
| 16,384 | 790.15 | 298.75 | +164.5% | 25.12 | 23.75 | +5.8% |
| 32,768 | 648.42 | 252.70 | +156.6% | 22.72 | 21.48 | +5.8% |
| 65,536 | 466.88 | 193.55 | +141.2% | 18.89 | 18.00 | +4.9% |
<!-- /bench -->

## Single user, MTP

Same workload as above with 2 MTP draft tokens. The mixed rows decode prose,
the repetitive rows copy text. MTP stops paying off with depth: at 32,768
mixed decode is 3% ahead of AR and repetitive is behind it, and at 65,536 both
are slower than AR (18.28 and 18.36 against 18.89 tok/s). llama.cpp's cells
are N/A because its `draft-mtp` loads the whole GGUF again as the draft model,
which does not fit.

<!-- bench:single-mtp-q4km -->
| Ornith 122B Q4_K_M MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 987.93 | N/A | N/A | 31.37 | N/A | N/A | 32.20 | N/A | N/A |
| 4,096 | 898.76 | N/A | N/A | 29.77 | N/A | N/A | 33.27 | N/A | N/A |
| 8,192 | 830.84 | N/A | N/A | 30.98 | N/A | N/A | 31.20 | N/A | N/A |
| 16,384 | 740.91 | N/A | N/A | 28.31 | N/A | N/A | 25.74 | N/A | N/A |
| 32,768 | 624.95 | N/A | N/A | 23.32 | N/A | N/A | 21.78 | N/A | N/A |
| 65,536 | 463.00 | N/A | N/A | 18.28 | N/A | N/A | 18.36 | N/A | N/A |
<!-- /bench -->

## Loading time

<!-- bench:loading -->
| Ornith 122B<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4_K_M | 19.48 | 19.85 | +1.9% |
<!-- /bench -->

## Memory occupation

Gufo holds 6 to 8 GiB more than llama.cpp at these context settings.

<!-- bench:memory-q4km -->
| Ornith 122B Q4_K_M AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 77.48 | 71.15 | -8.2% |
| 16K prefix, pp4096 + tg128 | 79.47 | 71.15 | -10.5% |
<!-- /bench -->

## Earlier measurements

The sections below were measured October 21 to 23, 2026 on the gufo-org
#299 base, before the rebase onto Strycast, and are kept for the MTP and
weight-placement results they record.

### MTP draft depth

Fresh server per arm, six prompts of 256 tokens at temperature 0, thinking
off, 4 GiB RAM cache.

| Draft tokens | Median 256-token decode | Accepted |
| ---: | ---: | ---: |
| 2 | 8364 ms (30.6 t/s) | 1218 / 2162 (56.3%) |
| 3 | 8629 ms (29.7 t/s) | 1286 / 2614 (49.2%) |

Replies were identical across all arms. MTP-off tg128 on the same build was
about 27.3 t/s.

### Sampled requests with MTP

Six prompts of 256 tokens at temperature 0.6, top-p 0.95, top-k 20, fixed
seeds, thinking off; the 12-token first prompt is excluded. Median decode rate
after the first token.

| Build | Draft tokens | Decode (t/s) | Drafts accepted |
| --- | ---: | ---: | ---: |
| Without sampled proposals (no drafts) | 2 | 26.7 | none |
| With sampled proposals | 2 | 31.4 | 637 / 1102 (58%) |
| MTP off | | 27.8 | |

### First token after load

Three fresh loads per arm, eight chat requests each (six distinct prompts of
16 to 19 tokens, two exact repeats), MTP depth 2.

| Weights | Load | Uncached first token, median / p90 / max | Over 1 s |
| --- | ---: | ---: | ---: |
| Registered host copy | 29 to 35 s | 282 / 633 / 1077 ms | 2 of 18 |
| Device memory | 21 to 23 s | 285 / 356 / 497 ms | 0 of 18 |

With the registered copy, each slow request coincided with kernel compaction
migrating pages (up to 2.4 million pages in one 6 s stall). In those windows
one routed-expert GEMM that normally takes about 1 ms ran for 2 to 7 s. The
device-memory runs had 1.1 million pages migrated during their uncached
requests and no stall. Cache hits answered in 3 to 5 ms in both arms.

## Not measured

- Depth 131,072, single user.
- Concurrency above one session.
