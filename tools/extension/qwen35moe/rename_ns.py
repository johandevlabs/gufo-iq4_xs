#!/usr/bin/env python3
"""Move a generated spike tree out of Gufo's namespaces (see make_spike.py).

Usage: rename_ns.py <tree> <core-namespaces.txt>
- gufo:: / namespace gufo become strycast:: / namespace strycast;
- q35/ns.hpp makes every strycast namespace that mirrors a Gufo one see
  Gufo's names, and forward-declares the core types the copy only names;
- the vendored MMQ library's C-linkage and global symbols get a q35_ prefix.
"""
import re, sys
from pathlib import Path

tree, ns_file = Path(sys.argv[1]), Path(sys.argv[2])
core_ns = [l.strip() for l in ns_file.read_text().splitlines() if l.strip()]

forward = {
    "core": ["class GgufReader;"],
    "speculative": ["struct SpeculativeOptions;"],
}
# Forward-declaration-only blocks for those types; the shim declares them.
fwd_block = re.compile(
    r"namespace gufo::(core|speculative) \{\n(?:class GgufReader;|struct SpeculativeOptions;)\n\}[^\n]*\n\n?")
globals_ = re.compile(r"\b(qfn_mmq\w*|ggml_hip_get_device|ggml_hip_set_device|ggml_time_us)\b")

def rename_globals(text):
    out = []
    for line in text.split("\n"):
        if not line.lstrip().startswith("#include"):
            line = globals_.sub(lambda m: "q35_" + m.group(1)[len("qfn_"):] if m.group(1).startswith("qfn_")
                                else "q35_" + m.group(1), line)
        out.append(line)
    return "\n".join(out)

exts = {".cpp", ".hpp", ".h", ".hip", ".cuh", ".inc"}
changed = 0
for f in sorted(tree.rglob("*")):
    if not f.is_file() or f.suffix not in exts:
        continue
    text = f.read_text()
    new = fwd_block.sub("", text)
    new = re.sub(r"\bnamespace gufo\b", "namespace strycast", new)
    new = re.sub(r"(?<![\w:])(::)?gufo::", lambda m: (m.group(1) or "") + "strycast::", new)
    new = rename_globals(new)
    if "strycast" in new and f.name != "ns.hpp":
        inc = '#include "q35/ns.hpp"\n'
        if f.suffix in {".hpp", ".h", ".cuh", ".inc"} and "#pragma once\n" in new:
            new = new.replace("#pragma once\n", "#pragma once\n\n" + inc, 1)
        else:
            new = inc + new
    if new != text:
        f.write_text(new)
        changed += 1

lines = [
    "#pragma once",
    "",
    "// Strycast's copy of the Qwen code lives in namespace strycast, so it links",
    "// beside Gufo's own Qwen extensions without duplicate symbols. Each strycast",
    "// namespace that mirrors a Gufo one also sees Gufo's names, so core types",
    "// such as hip::HipBuffer or ext::TextModelRunner still name Gufo's.",
    "",
]
for ns in core_ns:
    short = ns[len("gufo::"):] if ns != "gufo" else ""
    decls = " ".join(forward.get(short, []))
    lines.append(f"namespace {ns} {{{(' ' + decls + ' ') if decls else ''}}}")
lines.append("")
for ns in core_ns:
    mirror = "strycast" + ns[len("gufo"):]
    lines.append(f"namespace {mirror} {{ using namespace ::{ns}; }}")
(tree / "q35" / "ns.hpp").write_text("\n".join(lines) + "\n")
left = [str(p) for p in tree.rglob("*") if p.is_file() and p.suffix in exts and re.search(r"(?<![\w/])gufo::", p.read_text())]
print(f"files changed: {changed}; files still naming gufo:: {len(left)}")
for p in left[:20]:
    print("  ", p)
