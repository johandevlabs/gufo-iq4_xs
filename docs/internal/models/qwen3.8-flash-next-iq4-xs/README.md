# Qwen3.8 Flash-Next UD-IQ4_XS

Unsloth's **UD-IQ4_XS** quantization of Qwen3.8 Flash-Next, for Strix Halo
machines with 96 GB of memory. Gufo itself supports UD-Q4_K_XL, whose 76.9
GiB of GPU-resident weights leave too little room on a 96 GB machine; this
file holds 60.4 GiB. Everything else (architecture, MTP, images, YaRN,
serving options) is as in the [Flash-Next guide](../qwen3.8-flash-next/README.md).

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md)

## Files

`unsloth/Qwen3.8-Flash-Next-GGUF`, revision
[`38bb39ee9782`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS),
three shards (87.2 GiB). The SHA-256 of each file is in
[`artifacts/model-identities.json`](artifacts/model-identities.json).

The pin matters: on November 4, 2026 Unsloth replaced `UD-IQ4_XS` with a
Dynamic 3.1 build (four shards) that uses other formats almost everywhere
(IQ3_XXS/IQ2_S experts, Q5_0 and Q2_0 down projections, Q6_K attention and
shared experts, an IQ4_XS embedding). That file is not supported; Gufo
declines it when probing the model.

| Tensors | Format |
| --- | --- |
| Routed gate/up experts | IQ3_S (layer 2: IQ4_XS) |
| Routed down experts | IQ4_NL (five layers: Q8_0) |
| Output head | Q6_K |
| Attention, shared experts, embedding | Q8_0, as in UD-Q4_K_XL |
| Per-layer token embedding (26.8 GiB) | IQ4_NL, read from disk as in UD-Q4_K_XL |

The shared Q8_0 MTP predictor and the BF16 vision projector are the same files
as for UD-Q4_K_XL.

## Load and run

```sh
REV=38bb39ee97821de2c9009abb7e93950eec396e66
nix develop -c hf download unsloth/Qwen3.8-Flash-Next-GGUF --revision "$REV" \
  --include "UD-IQ4_XS/*" --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --include "mmproj-BF16.gguf" --local-dir models/qwen3.8-flash-next-iq4-xs
nix build
MODEL=models/qwen3.8-flash-next-iq4-xs/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
MTP=models/qwen3.8-flash-next-iq4-xs/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

On a 96 GB Evo-X2 (about 92 GiB usable), one session at the native 262,144
context with MTP and the vision projector loads; see the memory table in
[Benchmarks](BENCHMARKS.md).

## Kernels

The routed experts run on the same kernels as UD-Q4_K_XL, with IQ decoders:

- **Prefill** takes the F16 WMMA expert GEMMs. IQ3_S/IQ4_XS gate/up and
  IQ4_NL down blocks are decoded to signed bytes and run Q8_0's F16
  arithmetic, so a row's result is the same in the paired 128-token route and
  the separate 16- and 48-row tiles (`routed_wmma_ops_test`).
- **Decode and verification** (up to eight rows) take the vector kernels,
  which read the IQ blocks with llama.cpp's `vec_dot` formulas.
- The Q6_K head uses a Q6_K GEMV, eight rows per launch with single-row
  arithmetic, so decode, verification and batched heads agree.

IQ formats have no tiled MMQ route, so a model is refused at load when an IQ
layer cannot take the WMMA route (gate and up formats differ, or the down
projection is IQ3_S/IQ4_XS), or when the MTP block holds IQ experts: its
catch-up runs 9-32 rows outside prefill, which would need the tiled route.
