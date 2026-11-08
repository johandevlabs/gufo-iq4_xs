#!/usr/bin/env bash
# Generate the qwen35moe Gufo extension from a Strycast revision.
# 1. make_spike.py copies the Qwen family, the Qwen model and runner and the
#    Flash-Next kernels the 35B uses, and rewrites private includes to Gufo's
#    public <gufo/ext/...> headers.
# 2. rename_ns.py moves the copy from gufo:: to strycast::.
# 3. The copy is committed as "generated", then ports/*.patch adapt it to the
#    public extension API of the Gufo revision in gufo-revision.
# Usage: tools/extension/qwen35moe/generate.sh <out dir> [ref, default HEAD]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
out=$1; ref=${2:-HEAD}
[ ! -e "$out" ] || { echo "$out exists" >&2; exit 1; }
python3 "$here/make_spike.py" "$repo" "$ref" "$out"
python3 "$here/rename_ns.py" "$out" "$here/core-namespaces.txt"
git -C "$out" init -q
git -C "$out" add -A
git -C "$out" -c user.name=generate -c user.email=generate@localhost \
  commit -q -m "generated from Strycast $(git -C "$repo" rev-parse --short=8 "$ref")"
git -C "$out" -c user.name=generate -c user.email=generate@localhost \
  am -q -3 --committer-date-is-author-date "$here"/ports/*.patch
git -C "$out" log --oneline
