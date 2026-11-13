# Gufo: The ***Open Source*** Strix Halo AI engine

> [!NOTE]
> This is a community fork of [gufo-org/gufo](https://github.com/gufo-org/gufo),
> not the official project. It adds Qwen3.8 Flash-Next in Unsloth's smaller
> **UD-IQ4_XS** and **UD-IQ3_XXS** quantizations, both the current UD 3.1 files
> and the older ones, and ISTA-DASLab's **GSQ-RCO IQ3_XXS**, so the model runs
> on **96 GB** Strix Halo machines with MTP, images and the full 262K context. Everything else is Gufo 0.33.0; the
> Gufo README follows the fork section.

## This fork

Gufo supports Flash-Next as UD-Q4_K_XL, whose 78 GiB of GPU weights leave too
little room on a 96 GB machine. This fork runs smaller files instead.

New in November 2026:

- Based on Gufo 0.33.0.
- Unsloth's UD 3.1 files, which replaced UD-IQ4_XS on November 4, run with
  their new expert formats (IQ2_S, IQ3_XXS, Q2_0, Q5_0), Q6_K trunk and IQ4_XS
  embedding.
- UD 3.1 UD-IQ3_XXS agrees with UD-Q4_K_XL about as closely as the older
  UD-IQ4_XS did, with about 9 GiB less GPU memory.
- The older three-shard UD-IQ4_XS and UD-IQ3_XXS still run and pass the same
  functional suites.
- ISTA-DASLab's GSQ-RCO IQ3_XXS runs with its IQ2_XXS/IQ2_XS experts,
  mixed K-quant and IQ dense layers and BF16 hyper-connections. It uses about
  7 GiB less GPU memory than UD 3.1 UD-IQ3_XXS and agrees less closely with
  UD-Q4_K_XL.
- The [container image](https://github.com/johandevlabs/gufo-toolboxes) is
  built from this version.

### Files

| File | Revision | Download | Agreement with UD-Q4_K_XL | GPU weights |
| --- | --- | ---: | --- | ---: |
| UD 3.1 UD-IQ4_XS (4 shards) | `d9620119` | 93.7 GB | 89.4% of next tokens, mean KL 0.096 | 60.1 GiB |
| UD 3.1 UD-IQ3_XXS (4 shards) | `d9620119` | 83.6 GB | 88.3% of next tokens, mean KL 0.123 | 50.6 GiB |
| Older UD-IQ4_XS (3 shards) | `38bb39ee` | 93.7 GB | 88.8% of next tokens, mean KL 0.119 | 59.8 GiB |
| Older UD-IQ3_XXS (3 shards) | `38bb39ee` | 82.0 GB | 85.5% of next tokens, mean KL 0.186 | 50.2 GiB |
| GSQ-RCO IQ3_XXS (2 shards) | `ed59f920` | 75.8 GB | 84.5% of next tokens, mean KL 0.201 | 43.5 GiB |

Agreement is llama.cpp's next-token choice on 4,995 positions of code,
documentation, English, multilingual and agent text; GPU weights are the
model buffer llama.cpp allocates. For a new download, take UD 3.1 UD-IQ3_XXS,
or UD 3.1 UD-IQ4_XS for slightly higher agreement. With MTP and images at the
full 262,144-token context, the UD 3.1 files used 64.8 GiB (UD-IQ3_XXS) and
74.1 GiB (UD-IQ4_XS) of GPU memory, GSQ-RCO IQ3_XXS 57.7 GiB. Neither UD 3.1
file fits a 64 GB machine, and the older UD-IQ3_XXS needs only about 1.4 GiB
less than UD 3.1 UD-IQ3_XXS; GSQ-RCO has not been tested on 64 GB yet. The
Unsloth revisions are of `unsloth/Qwen3.8-Flash-Next-GGUF`, the GSQ-RCO one
of `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`.

### Performance

As of November 11, 2026, on a 96 GB GMKtec Evo-X2, tokens/s, one sample per
point. Prefill is a 2,048-token prompt; decode is 128 tokens.

| Workload | UD 3.1 UD-IQ4_XS | UD 3.1 UD-IQ3_XXS | Older UD-IQ4_XS | Older UD-IQ3_XXS | llama.cpp, UD 3.1 UD-IQ3_XXS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill | 1,847 | 1,821 | 1,808 | 1,836 | 500 |
| Prefill after 128K cached tokens | 1,445 | 1,446 | 1,472 | 1,493 | 158 |
| Decode, AR | 33.4 | 34.4 | 31.9 | 34.2 | 23.8 |
| Decode, AR, after 128K | 32.9 | 33.7 | 31.5 | 33.6 | 9.5 |
| Decode, MTP, mixed text | 42.2 | 49.5 | 42.5 | 48.9 | — |
| Decode, MTP, mixed text, after 128K | 48.4 | 46.9 | 48.0 | 43.9 | — |
| Decode, MTP, repetitive text | 106.2 | 102.0 | 152.4 | 106.8 | — |
| Decode, MTP, repetitive text, after 128K | 107.4 | 99.2 | 134.6 | 92.5 | — |
| 8 users, AR, in total | 105.8 | 103.1 | 137.0 | 104.1 | 67.3 |
| 8 users, MTP, mixed text, in total | 91.9 | 90.2 | 126.8 | 89.0 | — |

Gufo prefills 3.6 to 9.2 times as fast as llama.cpp and decodes 1.4 to 3.6
times as fast in AR, the gap growing with the cached context. The older
UD-IQ4_XS decodes fastest when many tokens go through the model at once
(repetitive text with MTP, 8 users). llama.cpp is the ROCm build Gufo benchmarks against;
its MTP output differs from its own AR output, so its MTP speeds are not
compared here.

GSQ-RCO IQ3_XXS as of November 13, 2026, on the same machine and in the same
units:

| Workload | Gufo | llama.cpp |
| --- | ---: | ---: |
| Prefill | 1,569 | 527 |
| Prefill after 128K cached tokens | 1,280 | 160 |
| Decode, AR | 34.1 | 23.8 |
| Decode, AR, after 128K | 33.4 | 9.3 |
| Decode, MTP, mixed text | 46.6 | — |
| Decode, MTP, mixed text, after 128K | 49.6 | — |
| Decode, MTP, repetitive text | 120.7 | — |
| Decode, MTP, repetitive text, after 128K | 116.1 | — |
| 8 users, AR, in total | 113.3 | 62.8 |
| 8 users, MTP, mixed text, in total | 106.2 | — |

The [model guide](docs/models/qwen3.8-flash-next-ud-iq.md) has the download,
run and memory details, with links to the
[full benchmarks](docs/internal/models/qwen3.8-flash-next-ud-iq/BENCHMARKS.md)
and the [quality report](docs/internal/models/qwen3.8-flash-next-ud-iq/QUALITY.md);
GSQ-RCO has [its own guide](docs/models/qwen3.8-flash-next-gsq-rco.md).
UD-Q4_K_XL behaves exactly as in Gufo 0.33.0, with byte-identical logits.

### Run

Download a file with its MTP head and vision projector:

```sh
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision d96201194f103c906994f283f504209dc8d2a10c \
  --include "UD-IQ3_XXS/*" \
  --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --include "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
```

Replace `UD-IQ3_XXS` with `UD-IQ4_XS` for the larger file. The older files
are at revision `38bb39ee9782`. GSQ-RCO IQ3_XXS comes from its own repository;
see [its guide](docs/models/qwen3.8-flash-next-gsq-rco.md#files).

Serve it with the container built from this fork:

```sh
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" \
  --ulimit memlock=-1 -p 8080:8080 -v "$PWD/models":/models:ro \
  ghcr.io/johandevlabs/gufo-toolboxes/gufo-runtime:edge \
  gufo serve --host 0.0.0.0 --port 8080 llm \
  --model /models/qwen3.8-flash-next/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf \
  --speculative mtp \
  --mtp-model /models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --mmproj /models/qwen3.8-flash-next/mmproj-BF16.gguf
```

For an older file, pass its first shard, `...-00001-of-00003.gguf`. To build
from source, clone this repository and follow
[install](docs/install.md#build-from-source).

---

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

<p align="center">
  <a href="https://github.com/gufo-org/gufo/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/gufo-org/gufo?label=release&amp;logo=github&amp;style=flat&amp;labelColor=141714&amp;color=2f5d3a"></a>
  <a href="https://gufo.blog/docs"><img alt="Documentation" src="https://img.shields.io/badge/docs-gufo.blog-2f5d3a?logo=readthedocs&amp;logoColor=white&amp;style=flat&amp;labelColor=141714&amp;color=2f5d3a"></a>
  <a href="docs/extension-api.md#versions"><img alt="Extension API series" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgufo-org%2Fgufo%2Fmain%2Finclude%2Fgufo%2Fext%2Fapi-series.json&amp;query=%24.minor&amp;prefix=0.&amp;label=extension%20API&amp;logo=cplusplus&amp;style=flat&amp;labelColor=141714&amp;color=2f5d3a"></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/github/license/gufo-org/gufo?logo=opensourceinitiative&amp;logoColor=white&amp;style=flat&amp;labelColor=141714&amp;color=2f5d3a"></a>
</p>

Gufo is an open-source inference engine for one machine: AMD Strix Halo, the
Ryzen AI MAX+ 395 with its Radeon 8060S (`gfx1151`) and up to 128 GiB of
unified memory. It serves text, embedding, speech, image and video models over
OpenAI-compatible APIs, from a single binary with kernels tuned for that chip.
It runs on Linux, where releases are built and tested, and natively on
Windows.

Gufo supports one model per use case, the best that fits, and checks each one
against the original before optimizing it. Its cache is built for coding agents
that resend long conversations, and extensions let you bring your own model.

## Models

| Model | What it does | Weights license |
| --- | --- | --- |
| [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next.md) | Text and images | [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) |
| [Qwen3.8 27B](docs/models/qwen3.8-27b.md) | Text and images | Apache 2.0 |
| [DeepSeek V4 Flash](docs/models/deepseek-v4-flash.md) | Text | MIT |
| [EmbeddingGemma 2](docs/models/embeddinggemma-2.md) | Embeddings for text, images, audio and video | Apache 2.0 |
| [Qwen3-ASR 1.7B](docs/models/qwen3-asr.md) | Speech to text | Apache 2.0 |
| [Qwen3-TTS 1.7B](docs/models/qwen3-tts.md) | Text to speech and voice cloning | Apache 2.0 |
| [Qwen-Image-2.1](docs/models/qwen-image-2.1.md) | Image generation and editing | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), non-commercial |
| [Qwen-Image-2.1-Turbo](docs/models/qwen-image-2.1.md#qwen-image-21-turbo) | Eight-step image generation and editing | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE), non-commercial |
| [MiniMax H3](docs/models/minimax-h3.md) | Video with sound | [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) |

Each model page lists the files to download, the modes and the measured speed.
Gufo's own code is MIT; the weights keep their publishers' terms.

Extensions add more models without changing Gufo. With `gufo` set up as in the
[quickstart](#quickstart), install a reviewed one from the
[extension registry](https://github.com/gufo-org/gufo-extension):

```sh
gufo extension install lfm3
```

[Extensions](docs/extensions.md) explains what that runs and how to trust it;
the [catalogue](https://gufo.blog/extensions) lists the official and community
extensions.

## Quickstart

Download Qwen3.8 Flash-Next and its MTP head:

```sh
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision d96201194f103c906994f283f504209dc8d2a10c \
  --include "UD-Q4_K_XL/*" \
  --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --local-dir models/flash-next
```

Define `gufo` to run in the container, with the GPU and the models folder:

```sh
gufo() {
  podman run --rm -it --network host \
    --userns=keep-id:uid=1000,gid=1000 --group-add keep-groups \
    --device /dev/kfd --device /dev/dri --ulimit memlock=-1 \
    -v ./models:/models:ro \
    -v gufo-extensions:/home/gufo/.local/share/gufo/extensions \
    ghcr.io/gufo-org/toolboxes/gufo-runtime:latest gufo "$@"
}
```

Serve the model, then ask it something from another terminal:

```sh
gufo serve llm --served-model-name flash-next \
  --model /models/flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --speculative mtp --mtp-model /models/flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "flash-next", "messages": [{"role": "user", "content": "Say something"}]}'
```

Any OpenAI client works with the base URL `http://localhost:8080/v1`: Open
WebUI, the OpenAI SDKs, Codex and other agents. If the GPU isn't found, see
[troubleshooting](docs/troubleshooting.md#the-gpu-isnt-found). For the Debian
package, Nix, a source build or Windows, see [install](docs/install.md).

## Documentation

The documentation is on [gufo.blog/docs](https://gufo.blog/docs), one version
per release, and in [`docs/`](docs/README.md) for the development version.

- **Get started:** [install](docs/install.md), [models](docs/models/README.md),
  [command line](docs/cli.md), [server API](docs/server.md).
- **Run it well:** [cache](docs/cache.md),
  [troubleshooting](docs/troubleshooting.md), [benchmarks](docs/benchmarks.md),
  [versions](docs/versions.md).
- **Extend it:** [extensions](docs/extensions.md),
  [writing an extension](docs/writing-extensions.md),
  [extension API](docs/extension-api.md).

The [changelog](CHANGELOG.md) and [releases](https://github.com/gufo-org/gufo/releases)
list what changed.

## Principles

- **One machine.** Every kernel is tuned for Strix Halo with 128 GiB. Smaller
  memory configurations work for the models that fit.
- **The best model per use case.** Fewer models means less code to maintain and
  more time to optimize and test each one. Each model keeps its own kernels, so
  a change to one can't break another.
- **Quality before speed.** Every model has a quality report that compares it
  with an independent reference; an optimization that changes the output isn't
  one.
- **Real workloads.** Concurrent requests, cancellation and long conversations
  are first-class, not afterthoughts.
- **Small production footprint.** Few runtime dependencies; development and
  reference tools stay out of the package.

## Contributing

Contributions are welcome: Gufo needs the Strix Halo community to keep
improving. Start with [development](docs/internal/development.md) and
[testing](docs/internal/testing.md); [AGENTS.md](AGENTS.md) holds the rules
that people and coding agents follow. To add a model without changing Gufo,
[write an extension](docs/writing-extensions.md).

Community projects built on Gufo include
[Gufo RDMA for Dual Strix Halo](https://github.com/neuhaus/gufo), which runs
models across two machines and isn't merged yet.

## License

Gufo's original code is [MIT licensed](LICENSE). Its design builds on other
open-source engines, listed in [reference projects](docs/reference-projects.md). Adapted code and dependencies
keep their own notices in [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`, installed
under `share/licenses/gufo`. Model weights aren't bundled and keep their
publishers' terms.
