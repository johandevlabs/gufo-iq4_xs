# Strycast: Strix Halo outcasts for Gufo

<p align="center">
  <img src="assets/strycast-logo.png" alt="Strycast logo" width="180">
</p>

Strycast ("Strix outcast") is a community fork of [Gufo](https://github.com/gufo-org/gufo),
the open-source inference engine for AMD Strix Halo. Gufo supports only the best
model for each size on a 128 GiB Strix Halo. Strycast takes models that aren't
Gufo's first choice there but are worth running anyway:

- **Smaller memory.** Models that fit a 64 GiB Strix Halo or a Ryzen AI laptop
  (gfx1150) where Gufo's flagship models don't.
- **A specific use case.** For example, a low-latency local assistant (Home
  Assistant voice control) that leaves room on the same box for speech-to-text,
  text-to-speech, embeddings and reranking.

Strycast tracks Gufo releases and is rebased on each one, or on a newer Gufo
`main` commit when it needs work that is not released yet (currently Gufo
v0.33.0). Everything Gufo supports still works here; the models below
are the additions. Strycast is not affiliated with gufo-org: report problems
with these models here, not upstream.

| Model | Inference modes | Hugging Face weights | Benchmarks | Quality |
| --- | --- | --- | --- | --- |
| [Qwen3.6 35B-A3B](docs/internal/models/qwen3.6-35b-a3b/README.md) | Q4/Q4_K_M/Q6/Q8, AR, DFlash2, MTP | Unsloth [Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf) / [Q6_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf) / [Q8_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf) · lmstudio [Q4_K_M](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-GGUF/blob/68a34855558af61cbef0324d31f411be8a506b08/Qwen3.6-35B-A3B-Q4_K_M.gguf) · [DFlash2 Q8_0](https://huggingface.co/aminya/Qwen3.6-35B-A3B-DFlash2-GGUF/blob/1e26e3499d865b9d2fc55e019d65b3f0e8ab6846/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf) | Q4: **3,400.57 tok/s pp**; up to **218.56 tok/s tg** single user and **269.74 aggregated tok/s** on 8 concurrent requests with DFlash2 · [Benchmarks](docs/internal/models/qwen3.6-35b-a3b/BENCHMARKS.md) | [Quality](docs/internal/models/qwen3.6-35b-a3b/QUALITY.md) |
| [Ornith 1.5 122B](docs/internal/models/ornith-1.5-122b/README.md) (Qwen3.5-122B-A10B fine-tune) | Q4_K_M, images, AR, MTP | [Q4_K_M](https://huggingface.co/ornith-ai/Ornith-1.5-122B-GGUF/blob/8d5893d767c449a4f18dd08f4bb6d0d6d4426175/Ornith-1.5-122B-Q4_K_M.gguf) · [mmproj BF16](https://huggingface.co/ornith-ai/Ornith-1.5-122B-GGUF/blob/8d5893d767c449a4f18dd08f4bb6d0d6d4426175/mmproj-Ornith-1.5-122B-BF16.gguf) | **1,079.22 tok/s pp** (AR); up to **36.39 tok/s tg** single user with MTP; multiple users not measured yet · [Benchmarks](docs/internal/models/ornith-1.5-122b/BENCHMARKS.md) | [Quality](docs/internal/models/ornith-1.5-122b/QUALITY.md) |
| [Qwen3.8 Flash-Next UD-IQ4_XS](docs/internal/models/qwen3.8-flash-next-iq4-xs/README.md) (for 96 GB) | IQ4_XS, images, AR, MTP | Unsloth [UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS) · [MTP Q8_0](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf) · [mmproj BF16](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/mmproj-BF16.gguf) | 96 GB Evo-X2: **1,814.71 tok/s pp** (AR); up to **153.55 tok/s tg** single user and **239.00 aggregated tok/s** on 8 concurrent requests with MTP · [Benchmarks](docs/internal/models/qwen3.8-flash-next-iq4-xs/BENCHMARKS.md) | [Quality](docs/internal/models/qwen3.8-flash-next-iq4-xs/QUALITY.md) |

Same measurement rules as Gufo's [benchmarks](docs/benchmarks.md).
Qwen3.6 35B-A3B was [gufo-org/gufo#299](https://github.com/gufo-org/gufo/pull/299);
Ornith 1.5 122B was contributed by [@lucasmcoleman](https://github.com/lucasmcoleman)
in [#12](https://github.com/slimsami/strycast/pull/12), formerly
[gufo-org/gufo#735](https://github.com/gufo-org/gufo/pull/735).

### Credits

- [@lucasmcoleman](https://github.com/lucasmcoleman) brought
  up Ornith 1.5 122B: its attention, MMVQ decode and chat-template support, and
  Gufo's device-memory weights change ([gufo-org/gufo#708](https://github.com/gufo-org/gufo/pull/708)),
  which every Qwen model here now uses.
- [@D3rPapts](https://github.com/D3rPapts) wrote the MMQ dispatch for K-quant
  35B experts in decode ([#7](https://github.com/slimsami/strycast/pull/7)),
  which put UD-Q4_K_XL ahead of llama.cpp in every multi-user row.

### Related forks

Other community forks of Gufo for Strix Halo. They are not part of Strycast;
report problems with them in their own repositories.

- [D3rPapts/gufo_windows](https://github.com/D3rPapts/gufo_windows): a native
  Windows 11 build for 64 GiB Strix Halo machines, with one router port for
  text, image (Qwen-Image 2.1 with true CFG and multi-reference edits,
  MOSS-Image), video (MiniMax H3, Wan 2.2, Wan 2.1 camera moves) and audio
  (Qwen3-TTS, ACE-Step), live previews, a ComfyUI-compatible API, SSD expert
  streaming and UD-Q3_K_XL for Qwen3.8 Flash-Next.
- [franzmoca/gufo](https://github.com/franzmoca/gufo): DeepSeek-V4.1-Flash
  (763B) on a single 128 GB Strix Halo, at IQ1_M in GPU memory or IQ2_XXS with
  experts streamed from SSD, and Gemma 4 26B-A4B and 31B, with MTP and image
  input.
- [neuhaus/gufo](https://github.com/neuhaus/gufo): Gufo RDMA for two Strix
  Halo machines linked by InfiniBand, RoCE v2 or USB4, with tensor parallelism,
  so Qwen3.8 Flash-Next runs at Q8 with longer contexts.

### Contributing a model

PRs for other models are welcome; open them against `staging`. Approved PRs
are squash-merged there, and `staging` is promoted to `main` once its CPU and
GPU tests pass, so `main` only holds tested work. Strycast keeps Gufo's quality
bar:

- Greedy speculative and batched decoding must stay bit-exact against plain
  (AR) decoding, with a test that asserts it.
- Each model gets a guide under `docs/internal/models` with its files pinned to
  a revision, plus `BENCHMARKS.md` and `QUALITY.md`.
- Changes must leave the outputs of Gufo's own models byte-identical.
- Build with the ROCm 10.1 toolchain Gufo pins; other compilers can break the
  bit-exact checks.

A model PR can be opened at any stage, so the work is visible and can be
reviewed as it goes. It is merged only once Strycast runs the model faster than
llama.cpp runs the same GGUF on the same machine; otherwise there is little
reason to run it on Strycast. The model's `BENCHMARKS.md` carries that
comparison, and its guide lists any gaps that remain.

### Rebasing onto Gufo

Each Gufo release is brought in by rebasing, not merging, so Strycast stays a
short series of commits on top of a Gufo tag, or of a Gufo `main` commit
between releases. The rulesets block force-pushes
to `staging` and `main`, so a rebase goes through the
[Check a Gufo rebase](.github/workflows/rebase-upstream.yml) workflow and an
admin:

1. Rebase `staging` onto the new tag on a branch named `rebase/<tag>` (or onto
   a `main` commit on `rebase/main-<commit>`), update "currently Gufo" above
   (`v0.33.0`, or `main` and the commit's first eight hex digits), and open a
   PR against `staging`.
2. Run the GPU checks for every Strycast model and for Gufo's own models, put
   the results in the PR, and add the `gpu-checked` label.
3. Run the workflow with the PR number. It checks that the branch sits exactly
   on the tag or `main` commit, that it is newer than the current base, that
   no Strycast commit was dropped and that the required checks passed, then
   prints the commands that tag the old head `archive/staging-<old base>` and
   replace `staging`.
4. An admin sets the ruleset's bypass for the repository admin role to
   "always", runs those commands and sets it back to "pull requests only".
5. Promote with a `staging` → `main` PR as usual, labelled `gpu-checked`, and
   repeat steps 3 and 4 with that PR to replace `main`.

After a rebase, move an open branch onto the new `staging` with
`git rebase --onto origin/staging archive/staging-<old base> <branch>`.

### Nightly run and releases

Each night a Strix Halo machine runs the GPU tests, Gufo's functional suite
and the model benchmarks on the head of `staging`
([details](tools/nightly/README.md)). It reports to the `staging` → `main`
promotion PR, opening it when needed, sets a `nightly/athena` status on it,
and labels it `gpu-checked` after a passing night, so it can be merged in the
morning. The nightly run is being set up, and the status will become a
required check for merging into `main`.

Every push to `main`, whether a promotion or a rebase, is released once its
checks pass, by the [Strycast release](.github/workflows/release.yml)
workflow. It tags the commit `v<Gufo version>-strycast.<n>` (for example
`v0.26.0-strycast.1`), takes the release notes from the promotion PR, and
attaches the portable Linux bundle the nightly run tested for that tree (or,
for a rebase pushed straight to `main`, the bundle CI built). The bundle
carries its own ROCm runtime, so a Strix Halo machine needs only the amdgpu
kernel driver. The same bundle is published as an OCI image,
`ghcr.io/slimsami/strycast:<version>` and `:latest`, which runs with the
Quickstart's `podman run` options below. The latest bundle that passed a night
is also published as the
[`nightly`](https://github.com/slimsami/strycast/releases/tag/nightly)
pre-release.

The rest of this README is Gufo's.

## About Gufo

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
