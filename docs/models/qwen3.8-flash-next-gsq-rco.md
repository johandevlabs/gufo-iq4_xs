# Qwen3.8 Flash-Next GSQ-RCO IQ3_XXS

ISTA-DASLab's GSQ-RCO IQ3_XXS quantization of Qwen3.8 Flash-Next, the
smallest Flash-Next file Gufo runs. It works with the same options, modes and
features as the [UD-Q4_K_XL model](qwen3.8-flash-next.md).

## Which file

| File | Download | GPU weights | Agreement with UD-Q4_K_XL |
| --- | ---: | ---: | --- |
| UD-Q4_K_XL | 111.3 GB | 78.1 GiB | reference |
| [UD 3.1 UD-IQ3_XXS](qwen3.8-flash-next-ud-iq.md) | 83.6 GB | 50.6 GiB | 88.4% of next tokens, mean KL 0.123 |
| GSQ-RCO IQ3_XXS | 75.8 GB | 43.5 GiB | 84.5% of next tokens, mean KL 0.201 |

GPU weights are the model buffer llama.cpp allocates for each file. Agreement
is llama.cpp's next-token choice on 4,995 positions of code, documentation,
English, multilingual and agent text, both files measured the same way.
GSQ-RCO takes about 7 GiB less GPU memory than UD-IQ3_XXS and agrees less
closely with UD-Q4_K_XL, most in agent text and documentation. About 29 GB of
the download is a per-layer embedding that Gufo reads from disk and never
places on the GPU.

With sampling, tool choices can be less reliable: in one tested
tool-calling request GSQ-RCO picked the intended tool for 17 of 30 seeds,
UD-IQ3_XXS for 30 of 30 ([details](../internal/models/qwen3.8-flash-next-gsq-rco/QUALITY.md#tool-agent-sampling)).

## Memory

One session runs at the native 262,144-token context with MTP and images on a
96 GB machine. With about 258,000 tokens in use, 24.7 GiB stayed available,
without swapping. GPU memory with MTP and images, with the context filled:

| Context | 32,768 | 65,536 | 131,072 | 196,608 | 262,144 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU memory | 51.9 GiB | 52.7 GiB | 54.4 GiB | 56.1 GiB | 57.7 GiB |

That is about 7 GiB less than UD 3.1 UD-IQ3_XXS at every context. Without
MTP and images, a 131,072-token context used 50.3 GiB. Whether it runs on a
64 GB machine has not been tested yet.

## Files

| File | Needed for |
| --- | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf` | Text |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` from Unsloth | MTP |
| `mmproj-BF16.gguf` from Unsloth | Images |

```sh
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
  --revision ed59f92082b1e93c0e96d60a8b11aab089b52f09 \
  --include "*IQ3_XXS*" \
  --local-dir models/qwen3.8-flash-next-gsq-rco
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision d96201194f103c906994f283f504209dc8d2a10c \
  --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --include "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
```

The GSQ-RCO repository has no MTP head; the shared one from Unsloth works
with it.

## Run

Pass the first shard; Gufo finds the other.

```sh
MODEL=models/qwen3.8-flash-next-gsq-rco/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
MTP=models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

Decoding modes, thinking, long context and images work as described on the
[UD-Q4_K_XL page](qwen3.8-flash-next.md#decoding-modes).

## Performance

As of November 13, 2026, on a 96 GB GMKtec Evo-X2, one user unless stated:

| Workload | GSQ-RCO IQ3_XXS |
| --- | ---: |
| Prefill | 1,569 tokens/s |
| Prefill after 128K cached tokens | 1,280 tokens/s |
| Decode, AR | 34.1 tokens/s |
| Decode, MTP, mixed text | 41–51 tokens/s |
| Decode, MTP, repetitive text | up to 121 tokens/s |
| Decode, 8 users, MTP, mixed text | 106.2 tokens/s in total |
| Cold start | 9.8 s |

Both engines ran without swapping. Gufo prefills 3.0 to 8.0 times as fast as
llama.cpp and decodes 1.4 to 3.6 times as fast in AR, growing with the cached
context; with MTP it decodes 1.7 to 3.2 times as fast on mixed text.

The [full benchmarks](../internal/models/qwen3.8-flash-next-gsq-rco/BENCHMARKS.md)
compare with llama.cpp. The
[quality report](../internal/models/qwen3.8-flash-next-gsq-rco/QUALITY.md)
covers the file against UD-Q4_K_XL and Gufo against its reference.
