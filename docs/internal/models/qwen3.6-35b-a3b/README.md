# Qwen3.6 35B-A3B

Sparse Mixture-of-Experts text model (`qwen35moe`) on gfx1151. 256 routed
experts with 8 active per token, plus one always-on shared expert. Layers
alternate gated-delta-net (GDN) recurrence and full attention in a 3:1 ratio;
the attention layers use 16 query / 2 key-value heads at `head_dim` 256.
Three Unsloth GGUFs from
[`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/tree/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d)
and the Q4_K_M from
[`lmstudio-community/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-GGUF/tree/68a34855558af61cbef0324d31f411be8a506b08)
are supported; request state and KV/SSM caches need memory on top of the
weights:

| File | Weights | Routed experts |
| --- | ---: | --- |
| [`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf) | 21.3 GiB | gate/up Q4_K (layer 39 Q5_K); down Q5_K (layers 34, 38, 39 Q6_K) |
| [`Qwen3.6-35B-A3B-Q4_K_M.gguf`](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-GGUF/blob/68a34855558af61cbef0324d31f411be8a506b08/Qwen3.6-35B-A3B-Q4_K_M.gguf) (lmstudio-community) | 19.7 GiB | gate/up Q4_K; down Q6_K on 20 layers, Q4_K on the other 20 |
| [`Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf) | 30.4 GiB | gate/up Q6_K (layer 39 Q8_0); down Q8_0 |
| [`Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf) | 36.4 GiB | gate/up Q8_0 (layer 39 BF16); down Q8_0 (layers 34, 38, 39 BF16) |

UD-Q4_K_XL prefills fastest (3,267 tok/s at pp2048, versus 3,103 for Q4_K_M,
3,090 for UD-Q6_K_XL and 2,847 for UD-Q8_K_XL); Q4_K_M decodes fastest (70.7
tok/s, versus 61.4 for UD-Q4_K_XL) and UD-Q8_K_XL slowest; see
[Benchmarks](BENCHMARKS.md). Q4_K_M is the smallest file and suits machines
with 64 GB of memory. Every Unsloth file carries an MTP head (`blk.40.nextn.*`)
that Gufo uses as a speculative draft; the Q4_K_M has none, so it runs AR or
DFlash2 only. A separate DFlash2 draft is published as a GGUF or can be
converted (below).

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix build
MODEL=path/to/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf
./result/bin/gufo serve llm --model "$MODEL" --sessions 1 --context 32768
./result/bin/gufo prompt --model "$MODEL" --prompt "Explain virtual memory." --temperature 0
```

Text-only model; there is no matching `mmproj` projector in scope. In prefill,
each routed-expert projection picks its route by format: Q8_0, Q6_K, Q5_K and
Q4_K take Flash-Next's routed F16 WMMA GEMM, BF16 the grouped WMMA GEMM, and
anything else the per-slot GEMV. Chunks of 1024+ tokens run gate and up as one
paired launch when the down projection takes the routed GEMM too; the down
projection then reuses the pair's 64/128-row tile map for Q8_0 and Q6_K, while
Q5_K and Q4_K, which the kernel decodes only in 16- and 48-row tiles, keep the
regular map. Decode and speculative
verification run the vendored llama.cpp-derived MMQ vector kernels: gate/up
for Q8_0, Q5_K and Q4_K, down for Q8_0, Q6_K, Q5_K and Q4_K, with the
expert activations quantized to Q8_1. Other formats, such as UD-Q6_K_XL's
Q6_K gate/up, take the per-slot GEMV. The kernels' device context is bound at
model load when the config reports an MoE architecture.

| Mode | Selection | Behavior |
| --- | --- | --- |
| AR | No speculative option | Target-only generation. |
| MTP | `--speculative mtp --mtp-model PATH` (`gufo prompt`, `gufo bench`, `gufo serve`) | Drafts with the GGUF's own MTP head; pass the same file as the target. `--draft-tokens N` caps the draft, `--min-draft-tokens N` sets its floor. In `gufo serve`, `--mtp-prime-chunk N` sets the rows per batched prompt-prime slice (default 1024; 0 restores the per-token prime). |
| DFlash2 | `--speculative dflash2 --dflash-model PATH` (`gufo prompt`, `gufo bench`, `gufo serve`) | Block-diffusion draft converted to GGUF (below); it taps 8 target layers. `--draft-tokens N` caps proposals, `--draft-policy fixed` fixes them. |

### Speculative decoding (MTP)

```sh
./result/bin/gufo prompt --model "$MODEL" --speculative mtp \
  --mtp-model "$MODEL" --draft-tokens 2 --min-draft-tokens 2 \
  --prompt "Explain virtual memory." --temperature 0
```

In a September 26, 2026 sweep n=2 was fastest on UD-Q6_K_XL and UD-Q8_K_XL;
an earlier UD-Q8_K_XL run favoured n=1. With n=2, the October 25 benchmark
decodes UD-Q6_K_XL at 67.37 tok/s on mixed text and 94.98 on repetitive text
at pp2048, versus 55.05 without MTP. Acceptance depends on the generated text,
so re-check n on your own workload.

As in Qwen3.8 Flash-Next, each MTP draft token scores only the first 98,304
output rows plus a per-request table of 1,024 output rows (about 2–4 MB per
session, depending on the head's quant): the special tokens and the request's
own rarer tokens. A request that keeps bringing new rare tokens drafts over the
full head. Only proposals change; the target still verifies over the full
vocabulary.

### Speculative decoding (DFlash2)

A Q8_0 conversion of
[`incoai/Qwen3.6-35B-A3B-DFlash2`](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2)
is published as
[`aminya/Qwen3.6-35B-A3B-DFlash2-GGUF`](https://huggingface.co/aminya/Qwen3.6-35B-A3B-DFlash2-GGUF)
(`Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf`, revision
[`1e26e349`](https://huggingface.co/aminya/Qwen3.6-35B-A3B-DFlash2-GGUF/blob/1e26e3499d865b9d2fc55e019d65b3f0e8ab6846/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf));
[Benchmarks](BENCHMARKS.md) use that file. To convert the checkpoint yourself,
use a llama.cpp checkout whose converter registers `DFlash2DraftModel` (tested
at `b81c99b47`); the converter takes the tokenizer from the target checkpoint:

```sh
nix develop -c hf download incoai/Qwen3.6-35B-A3B-DFlash2 --local-dir draft
nix develop -c hf download Qwen/Qwen3.6-35B-A3B config.json tokenizer.json \
  tokenizer_config.json vocab.json merges.txt --local-dir target
python3 llama.cpp/convert_hf_to_gguf.py draft --target-model-dir target \
  --outtype q8_0 --outfile Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf
./result/bin/gufo prompt --model "$MODEL" --speculative dflash2 \
  --dflash-model Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf --draft-tokens 7 \
  --prompt "Explain virtual memory." --temperature 0
```

Run the converter with numpy 2. With numpy 1.26 on Python 3.14, Q8_0
conversion drops the sign of every quant in tensors with rows of 4,096 values
or more (`attn_output`, `ffn_down`, `fc`). The draft still loads, but accepts
no drafts.

With such a local conversion on UD-Q6_K_XL, 512 greedy tokens with thinking off, a prose explanation ran at
73.5 tok/s with `--draft-tokens 3 --draft-policy fixed` and a code rewrite at
89.7 tok/s with 7, versus 52 tok/s without a draft; the best MTP setting
reached 63.6 and 68.7. The adaptive policy (66.6 / 88.8) does not find the
shorter draft on prose, so pick the length for the workload. `gufo bench`
understates DFlash2 because its synthetic prompt drafts poorly.

Greedy MTP and DFlash2 output is bit-exact against AR, as on the Qwen3.8 27B:
verifying a batch of tokens, and batched decode of concurrent requests,
reproduce single-token decode arithmetic row for row. This model's
gated-delta-net alpha/beta controls are F32, so batched rows compute them with
the decode kernel rather than the row GEMV, whose sum order differs; before
that, the one-ulp drift flipped near-tied expert choices on UD-Q6_K_XL and
UD-Q8_K_XL. The target test asserts the identity (see [Quality](QUALITY.md)).

### Serving

```sh
./result/bin/gufo serve --sessions 4 llm --model "$MODEL" --context 262144 \
  --speculative dflash2 --dflash-model Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf \
  --draft-tokens 7 --draft-policy adaptive \
  --prefill-chunk 2048 --prefill-decode-rounds 8
```

For MTP, replace the draft arguments with
`--speculative mtp --mtp-model "$MODEL" --draft-tokens 2`.

Concurrent requests share one batched verification step. Each extra row reads
mostly different experts, so a step costs close to one decode step per
stacked row; the server therefore shortens DFlash2 drafts as more requests are
active (2 draft tokens at 2 requests, 1 at 3 or more) and a lone request keeps
the full draft. MTP takes the same batched step, with the sessions' drafts
proposed in one batch, and keeps 1 draft token at 2 requests and none at 3 or
more, where one draft token no longer pays for its row; there it decodes within
3% of the plain rate, the cost of keeping each session's draft context current.

Prefill is not batched across sessions: requests take turns, and while other
sessions decode, a prompt is prefilled in `--prefill-chunk` token slices.
`--prefill-decode-rounds N` gives the decoding sessions N rounds between two
slices, so a larger chunk does not lengthen their stalls in proportion. Each
session holds its own KV/SSM state: at context 262144, one session used
38.7 GiB of device memory in total and four 59.7 GiB. On this hardware DFlash2
decodes much faster than MTP on repetitive text, the two are close on mixed
text, and MTP prefills slightly faster; see
[Benchmarks](BENCHMARKS.md#multiple-users-dflash2).

Thinking/template controls and HTTP sampling defaults are described in
[the server guide](../../SERVER.md#reasoning-controls).

## Scope

- Quantized MoE expert GEMM (Q8_0 / Q6_K / Q5_K / Q4_K) and the fused
  gate/up/down expert path; dense attention and GDN layers are shared with the other Qwen runners.
- No vision, no audio. Speculative decoding is MTP (the GGUF's own head) or a
  converted DFlash2 draft; both run in `gufo prompt`, `gufo bench` and
  `gufo serve`.
- The prefill attention kernel is compiled only for head shapes listed in
  `detail/attention_policy.hpp`; unsupported shapes fall back to the baseline
  path (see [Experiments](EXPERIMENTS.md)).
