#!/usr/bin/env bash
# Publish a Strycast release for a commit on main: tag v<Gufo version>-strycast.<n>
# and attach the portable Linux bundle the nightly run tested for its tree, or
# else the one "Repository checks" built for the commit. Run by
# .github/workflows/release.yml from a checkout of the commit.
#
# Environment: GH_TOKEN, REPO (owner/name), SHA (full commit on main), and
# optionally RUN_ID (the "Repository checks" push run that built SHA) and
# PUBLISH (false prepares the release in ./release without publishing). Writes
# the release tag to GITHUB_OUTPUT, for the image job.
set -euo pipefail

summary=${GITHUB_STEP_SUMMARY:-/dev/stdout}
outputs=${GITHUB_OUTPUT:-/dev/null}

fail() {
  echo "::error::$*"
  exit 1
}

[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || fail "SHA must be a full commit hash, got '$SHA'."
[ "$(git rev-parse HEAD)" = "$SHA" ] || fail "The checkout is not $SHA."
git fetch --quiet --no-tags origin '+refs/heads/main:refs/remotes/origin/main' '+refs/tags/v*:refs/tags/v*'
git merge-base --is-ancestor "$SHA" origin/main || fail "$SHA is not on main."

existing=$(git tag --points-at "$SHA" --list 'v*-strycast.*')
if [ -n "$existing" ]; then
  echo "$SHA is already released as $existing." | tee -a "$summary"
  echo "tag=$(head -1 <<<"$existing")" >> "$outputs"
  exit 0
fi

gufo=$(tr -d '[:space:]' < version.txt)
[[ "$gufo" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "version.txt holds '$gufo', not a Gufo version."
last=$(git tag --list "v$gufo-strycast.*" | sed "s/^v$gufo-strycast\.//" | sort -n | tail -1)
tag="v$gufo-strycast.$(( ${last:-0} + 1 ))"

name="strycast-$tag-linux-x86_64"
rm -rf release
mkdir -p release
work=$(mktemp -d)
# Prefer the bundle the nightly Strix Halo run tested. The `nightly`
# pre-release holds the last staging commit that passed, and a promotion gives
# main that commit's tree.
git fetch --quiet --no-tags origin '+refs/tags/nightly:refs/tags/nightly' 2>/dev/null || true
tested=$(git rev-parse -q --verify 'refs/tags/nightly^{commit}' || true)
if [ -n "$tested" ] && [ "$(git rev-parse "$tested^{tree}")" = "$(git rev-parse "$SHA^{tree}")" ]; then
  gh release download nightly --repo "$REPO" --dir "$work" \
    --pattern strycast-nightly-linux-x86_64.tar.gz --pattern SHA256SUMS
  (cd "$work" && sha256sum -c SHA256SUMS)
  tar -xzf "$work/strycast-nightly-linux-x86_64.tar.gz" -C "$work"
  mv "$work/strycast-nightly-linux-x86_64" "release/$name"
  revision=$tested
  origin="the one the nightly Strix Halo run tested on \`staging\` commit \`${tested:0:8}\`, which has
this commit's tree (its report is on the promotion PR)"
else
  # No nightly bundle for this tree, as after a rebase pushed straight to main:
  # use the main push's own CI build.
  if [ -z "${RUN_ID:-}" ]; then
    RUN_ID=$(gh api "repos/$REPO/actions/workflows/checks.yml/runs?head_sha=$SHA&event=push&status=success" \
      --jq '.workflow_runs[0].id // empty')
    [ -n "$RUN_ID" ] || fail "No successful \"Repository checks\" push run built $SHA; run it on main first."
  fi
  gh run download "$RUN_ID" --repo "$REPO" --dir "$work" \
    --name "linux-$SHA" --name "linux-$SHA-runtime" ||
    fail "Run $RUN_ID has no Linux bundle for $SHA (artifacts expire after 3 days); re-run \"Repository checks\" on main."
  mkdir -p "release/$name"
  tar -xzf "$work/linux-$SHA/linux-artifacts.tar.gz" -C "$work"
  tar -xzf "$work/linux-$SHA-runtime/linux-runtime.tar.gz" -C "release/$name"
  mv "$work/linux-artifacts/candidate" "release/$name/bin"
  revision=$SHA
  origin="the package \`Repository checks\` built for this commit
([run](https://github.com/$REPO/actions/runs/$RUN_ID)); no nightly run tested this tree"
fi
python3 tools/ci/linux_artifact.py verify --source "release/$name/bin" --revision "$revision"
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
  -I 'gzip -6 -n' -cf "release/$name.tar.gz" -C release "$name"
rm -rf "release/$name" "$work"
(cd release && sha256sum "$name.tar.gz" > SHA256SUMS)

# Notes: the promotion PR's description, which lists what reached main.
pr=$(gh api "repos/$REPO/commits/$SHA/pulls" \
  --jq '[.[] | select(.base.ref == "main" and .merged_at != null)][0] // empty')
{
  echo "Strycast on Gufo v$gufo, commit \`${SHA:0:8}\`."
  echo
  if [ -n "$pr" ]; then
    jq -r '.body // "" | gsub("\n*🤖 Generated with \\[Claude Code\\][^\n]*"; "") | sub("\\s+$"; "")' <<<"$pr"
    echo
    echo "Promoted in #$(jq -r .number <<<"$pr")."
  else
    echo "No promotion PR: \`main\` was replaced directly, as in a rebase onto a new Gufo release."
  fi
  cat <<EOF

## Linux bundle

\`$name.tar.gz\` runs on Strix Halo (\`gfx1151\`) with only the amdgpu kernel
driver installed: the ROCm runtime, libraries and FFmpeg come bundled. Your
user needs access to \`/dev/kfd\` and \`/dev/dri\`.

\`\`\`sh
sha256sum -c SHA256SUMS
tar -xzf $name.tar.gz
./$name/bin/gufo diagnose
./$name/bin/gufo serve llm --model /path/to/model.gguf
\`\`\`

The bundle is $origin. It is built with the toolchain the Nix flake pins.

## Container image

The same bundle as an OCI image, \`ghcr.io/${REPO,,}:${tag#v}\` (also
\`latest\`). It runs like Gufo's \`gufo-runtime\` image:

\`\`\`sh
podman run --rm --userns=keep-id:uid=1000,gid=1000 \\
  --device /dev/kfd --device /dev/dri --group-add keep-groups \\
  --ulimit memlock=-1 -p 8080:8080 -v ./models:/models:ro \\
  ghcr.io/${REPO,,}:${tag#v} \\
  gufo serve --host 0.0.0.0 --port 8080 llm --model /models/model.gguf
\`\`\`
EOF
} > release/notes.md

if [ "${PUBLISH:-true}" != true ]; then
  echo "Prepared $tag in ./release without publishing." | tee -a "$summary"
  exit 0
fi
gh release create "$tag" --repo "$REPO" --target "$SHA" --title "Strycast $tag" \
  --notes-file release/notes.md "release/$name.tar.gz" release/SHA256SUMS
echo "tag=$tag" >> "$outputs"
echo "Released [$tag](https://github.com/$REPO/releases/tag/$tag) from \`$SHA\`." | tee -a "$summary"
