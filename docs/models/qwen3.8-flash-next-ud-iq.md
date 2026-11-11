# Qwen3.8 Flash-Next UD-IQ4_XS and UD-IQ3_XXS

Unsloth's smaller quantizations of Qwen3.8 Flash-Next, for 96 GB Strix Halo
machines. They run with the same options,
modes and features as the [UD-Q4_K_XL model](qwen3.8-flash-next.md).

## Which file

| File | Download | GPU weights | Agreement with UD-Q4_K_XL |
| --- | ---: | ---: | --- |
| UD-Q4_K_XL | 111.3 GB | 78.1 GiB | reference |
| UD-IQ4_XS | 93.7 GB | 60.1 GiB | 89.4% of next tokens, mean KL 0.096 |
| UD-IQ3_XXS | 83.6 GB | 50.6 GiB | 88.3% of next tokens, mean KL 0.123 |

GPU weights are the model buffer llama.cpp allocates for each file. Agreement
is llama.cpp's next-token choice on 4,995 positions of code, documentation,
English, multilingual and agent text. The two files are close in quality;
in Gufo, UD-IQ3_XXS uses about 9 GiB less GPU memory. About
29 GB of each download is a per-layer embedding that Gufo reads from disk and
never places on the GPU.

## Memory

Both files run one session at the native 262,144-token context with MTP and
images on a 96 GB machine. With about 258,000 tokens in use, UD-IQ4_XS left
11.6 GiB available and UD-IQ3_XXS 17.9 GiB, without swapping. GPU memory for
UD-IQ3_XXS with MTP and images, with the context filled:

| Context | 32,768 | 65,536 | 131,072 | 196,608 | 262,144 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU memory | 58.9 GiB | 59.7 GiB | 61.4 GiB | 63.1 GiB | 64.8 GiB |

Neither file fits a 64 GB machine. With 32 GiB of the 96 GB machine held
back, leaving 60.5 GiB available, UD-IQ3_XXS ran out of memory while loading
with an 8,192-token context, without MTP or images.

## Files

| File | Needed for |
| --- | --- |
| `UD-IQ4_XS/*` or `UD-IQ3_XXS/*` (four shards) | Text |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | MTP |
| `mmproj-BF16.gguf` | Images |

```sh
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision d96201194f103c906994f283f504209dc8d2a10c \
  --include "UD-IQ4_XS/*" \
  --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --include "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
```

Replace `UD-IQ4_XS` with `UD-IQ3_XXS` for the smaller file. The MTP head and
the vision projector are the same files as for UD-Q4_K_XL. The three-shard
UD-IQ4_XS from revision `38bb39ee9782` also runs.

## Run

Pass the first shard; Gufo finds the others.

```sh
MODEL=models/qwen3.8-flash-next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
MTP=models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

Decoding modes, thinking, long context and images work as described on the
[UD-Q4_K_XL page](qwen3.8-flash-next.md#decoding-modes).

## Performance

As of November 11, 2026, on a 96 GB GMKtec Evo-X2, one user unless stated:

| Workload | UD-IQ4_XS | UD-IQ3_XXS |
| --- | ---: | ---: |
| Prefill | 1,847 tokens/s | 1,821 tokens/s |
| Prefill after 128K cached tokens | 1,445 tokens/s | 1,446 tokens/s |
| Decode, AR | 33.4 tokens/s | 34.4 tokens/s |
| Decode, MTP, mixed text | 42–51 tokens/s | 46–50 tokens/s |
| Decode, MTP, repetitive text | up to 108 tokens/s | up to 106 tokens/s |
| Decode, 8 users, MTP, mixed text | 91.9 tokens/s in total | 90.2 tokens/s in total |
| Cold start | 11.6 s | 10.1 s |

With UD-IQ3_XXS, which both engines ran without swapping, Gufo prefills 3.6
to 9.2 times as fast as llama.cpp and decodes 1.4 to 3.6 times as fast in AR,
growing with the cached context.

The [full benchmarks](../internal/models/qwen3.8-flash-next-ud-iq/BENCHMARKS.md)
compare with llama.cpp. The
[quality report](../internal/models/qwen3.8-flash-next-ud-iq/QUALITY.md)
covers both files against UD-Q4_K_XL and Gufo against its reference.
