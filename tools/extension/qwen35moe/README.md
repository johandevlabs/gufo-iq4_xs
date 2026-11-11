# qwen35moe Gufo extension

[slimsami/strycast-qwen35moe-gufo-extension](https://github.com/slimsami/strycast-qwen35moe-gufo-extension)
packages Strycast's Qwen3.6-35B-A3B and Ornith 1.5 runner (`qwen35moe`) as an
external Gufo extension, so stock Gufo can load it with
`gufo extension install`. Strycast is the source: that repository holds the
output of `generate.sh`, published by `.github/workflows/extension.yml` after
each Strycast release and tagged with the release's tag.

```sh
tools/extension/qwen35moe/generate.sh /tmp/qwen35moe-extension [ref]
```

1. `make_spike.py` copies `src/families/qwen` (`q35/family`), `src/models/qwen`
   (`q35/model`) and the Flash-Next kernels the 35B uses (`q35/fnext`), and
   rewrites private includes to Gufo's public `<gufo/ext/...>` headers with
   `ext-public-map.txt`. `extension.json` and `extension.cmake` are copied as
   they are.
2. `rename_ns.py` moves the copy from `gufo::` to `strycast::`, and the vendored
   MMQ library's C-linkage symbols to a `q35_` prefix, so it never shares a
   name with Gufo's own Qwen libraries.
3. The copy is committed as "generated" and `ports/*.patch` are applied. They
   adapt Strycast's core API additions to the public extension API of the Gufo
   revision in `gufo-revision`:

| Patch | Why |
| --- | --- |
| 0001 | Gufo's `ModelConfig` has no routed-expert fields; the extension derives its own. |
| 0002 | No hook for the DFlash2 context skip or the prefill-step checkpoint grid; both are dropped. |
| 0003 | Build fixes against Gufo main (a private test helper). |

`--kv-cache`, the fixed MTP prime chunk and the concurrent draft cap in the
draft backends need no port: Strycast itself takes them through Gufo's public
API since its rebase onto Gufo v0.33.0.

When a change to the copied code or a new Gufo revision breaks a patch, the
workflow fails. To update the patches, generate the tree, fix the failing
commit in it, and export the ports again:

```sh
git -C /tmp/qwen35moe-extension format-patch HEAD~3 -o tools/extension/qwen35moe/ports
```

Check a new Gufo revision on Strix Halo before raising `gufo-revision`: build
it with `-DGUFO_EXTENSION_DIRS=<generated tree>` and compare greedy outputs and
speed with Strycast's own build.
