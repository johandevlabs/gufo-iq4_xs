#!/usr/bin/env python3
"""Copy the qwen35moe extension tree out of a Strycast revision.

Copies src/families/qwen, src/models/qwen and Flash-Next's ROCm kernel
directory into an external-extension layout, rewrites private includes to
the extension's own paths or the public <gufo/ext/...> headers, and writes the
manifest and CMake recipe. Usage: make_spike.py <strycast checkout> <ref> <out dir>
"""
import io, json, re, shutil, subprocess, sys, tarfile
from pathlib import Path

repo, ref, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
here = Path(__file__).resolve().parent
NAME = "strycast_qwen35moe"

dirs = {
    "src/families/qwen/": "q35/family/",
    "src/models/qwen/": "q35/model/",
    "src/models/qwen38_flash_next/": "q35/fnext/",
}
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
tar = subprocess.run(["git", "-C", str(repo), "archive", ref, *[d.rstrip("/") for d in dirs]],
                     check=True, capture_output=True).stdout
public = {}
for line in (here / "ext-public-map.txt").read_text().splitlines():
    src, pub = line.split()
    public.setdefault(src, pub)
# Reached through the public image header (src/ext/image.hpp includes it).
public["src/core/image.hpp"] = "gufo/ext/image.hpp"

skip = re.compile(r"(^|/)(CMakeLists\.txt|extension\.(json|cmake)|library\.json|functional/|README\.md)")
# Flash-Next sources other than its kernels are copied for their headers only.
with tarfile.open(fileobj=io.BytesIO(tar)) as t:
    for m in t.getmembers():
        if not m.isfile():
            continue
        for old, new in dirs.items():
            if m.name.startswith(old):
                rel = m.name[len(old):]
                if skip.search(rel):
                    break
                dst = out / new / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(t.extractfile(m).read())
                break

missing = {}
def rewrite(match):
    path = match.group(1)
    for old, new in dirs.items():
        if path.startswith(old):
            return f'#include "{new}{path[len(old):]}"'
    if path in public:
        return f"#include <{public[path]}>"
    missing.setdefault(path, 0)
    missing[path] += 1
    return match.group(0)

inc = re.compile(r'#include ["<](src/[^">]+)[">]')
for f in out.rglob("*"):
    if f.is_file() and f.suffix in {".cpp", ".hpp", ".h", ".hip", ".cuh", ".inc", ".def"}:
        text = f.read_text()
        # Unused in the copied drivers; the comparator is a private test library.
        text = text.replace('#include "src/testing/compare/logit_comparator.hpp"\n', "")
        new = inc.sub(rewrite, text)
        new = new.replace("gufo/ext/generated/qwen_constants.hpp",
                          f"gufo/ext/generated/{NAME}_constants.hpp")
        new = new.replace("extension_qwen::kProvider_qwen27_q4",
                          f"extension_{NAME}::kProvider_qwen35moe")
        if new != text:
            f.write_text(new)

# Registration: one provider for qwen35moe only.
for f, pairs in {
    "q35/model/extension.cpp": [("void qwen_register(", f"void {NAME}_register("),
                                ('.id = "qwen27-q4"', '.id = "qwen35moe"')],
    "q35/model/provider.cpp": [("void qwen_provider_register(", f"void {NAME}_provider_register("),
                               ('.id = "qwen27-q4"', '.id = "qwen35moe"')],
}.items():
    p = out / f
    text = p.read_text()
    for a, b in pairs:
        assert a in text, (f, a)
        text = text.replace(a, b)
    p.write_text(text)

shutil.copy(here / "extension.json", out / "extension.json")
shutil.copy(here / "extension.cmake", out / "extension.cmake")
shutil.copy(repo / "LICENSE", out / "LICENSE") if (repo / "LICENSE").exists() else None
print("private includes left:")
for path, n in sorted(missing.items(), key=lambda x: -x[1]):
    print(f"  {n:4d} {path}")
