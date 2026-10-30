# Nightly Strix Halo run

Each night at 01:00 a Strix Halo host tests the head of `staging`, so that a
promotion to `main` is ready to merge in the morning. The host's launcher
(outside this repository) fetches the portable Linux bundles CI built for
`staging` and for `main`, waits for a turn in the host's bench queue, and runs
[`run.sh`](run.sh) from a checkout of the commit:

1. **GPU tests.** It builds the test tree with the pinned ROCm 10.1 toolchain
   and runs every CTest test except MiniMax H3 and DeepSeek V4, whose weights
   are not on the host, and `qwen_mtp_reference_test`, whose fixtures come
   from another 27B checkpoint. `qwen35ba3b_target_test` also runs once for
   each 35B quantization. Tests whose model or reference files are not
   configured exit with 77 and are listed as skipped. Kernel scratch is checked
   on the CI binary: against Gufo's allowance file, or, for kernels already
   over it, against `main`.
2. **Gufo's functional suite.** The full tier on Gufo's own models
   (Flash-Next, 27B, Embedding Gemma 2, Qwen3 ASR and the three TTS
   variants), using the `staging` bundle, with server phases timed against
   the `main` bundle.
3. **Benchmarks.** Every Gufo table of the Strycast models, measured with the
   `staging` bundle and compared with the committed artifacts by
   [`compare_bench.py`](compare_bench.py). A night fails when a value drops by
   more than 3% (single-user AR), 8% (speculative and multi-user tables), 15%
   or 2 s (loading) or 0.5 GiB (memory), or when a speculative completion
   differs from AR. A single-user or multi-user table that regressed is
   measured once more and the better value counts, since concurrent tables
   move up to 10% between two runs of one binary.

Phases still pending at 05:45 are skipped, which fails the night, so the host
is back in production by 06:00.

The launcher then reports to GitHub:

- the `nightly/athena` status on the `staging` commit, required to merge into
  `main`;
- the report as a comment on the `staging` → `main` promotion PR, which it
  opens when `staging` is ahead, with the `gpu-checked` label when the night
  passed;
- after a passing night, the bundle as the rolling
  [`nightly`](https://github.com/slimsami/strycast/releases/tag/nightly)
  pre-release. When that tree reaches `main`, the release workflow attaches
  the same bundle to the release.

A night is skipped when `staging` has not moved since the last run.

## Host settings

`run.sh --config FILE` reads these variables. For a manual run, `--phases`
picks some of `gpu_tests`, `functional` and `benchmarks`.

| Variable | Meaning |
| --- | --- |
| `TOOLCHAIN` | Directory with `configure-rocm10.sh`, `run-rocm10.sh`, `rocm10-link.sh` and `clang-offload-bundler` |
| `ROCM_PATH` | The ROCm 10.1 CLR the test binaries run against |
| `BENCH_PYTHON` | Python for `model-bench.py`, run under `run-rocm10.sh` |
| `Q35_Q4`, `Q35_Q6`, `Q35_Q8`, `Q35_Q4KM`, `Q35_DRAFT` | Qwen3.6 35B-A3B files and its DFlash2 draft, as in its guide |
| `ORNITH_GGUF` | Ornith 1.5 122B Q4_K_M |
| `QWEN27_GGUF`, `QWEN27_DFLASH` | Qwen3.8 27B UD-Q4_K_XL and its DFlash2 draft |
| `EMBEDDING_ROOT`, `ASR_ROOT` | Embedding Gemma 2 and Qwen3 ASR snapshots |
| `FUNCTIONAL_ASSETS` | JSON mapping each `tests/functional/devlab.json` asset to its directory |
