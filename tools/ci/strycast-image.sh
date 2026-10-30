#!/usr/bin/env bash
# Publish the OCI image of a Strycast release: its Linux bundle on the base in
# tools/ci/Containerfile, tagged <version> (v0.26.0-strycast.1 gives
# 0.26.0-strycast.1) and latest. Run by .github/workflows/release.yml after the
# release exists.
#
# Environment: GH_TOKEN, REPO (owner/name), TAG (the release tag), ACTOR (the
# registry user), and optionally PUBLISH (false builds and checks the image
# without pushing it).
set -euo pipefail

summary=${GITHUB_STEP_SUMMARY:-/dev/stdout}

fail() {
  echo "::error::$*"
  exit 1
}

[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+-strycast\.[0-9]+$ ]] || fail "TAG must be a Strycast release tag, got '$TAG'."
image="ghcr.io/${REPO,,}"
version=${TAG#v}

if [ "${PUBLISH:-true}" = true ]; then
  docker login ghcr.io --username "$ACTOR" --password-stdin <<<"$GH_TOKEN"
  # Release tags are immutable, and so is their image.
  if docker manifest inspect "$image:$version" >/dev/null 2>&1; then
    echo "$image:$version already exists." | tee -a "$summary"
    exit 0
  fi
fi

work=$(mktemp -d)
name="strycast-$TAG-linux-x86_64"
gh release download "$TAG" --repo "$REPO" --dir "$work" --pattern "$name.tar.gz" --pattern SHA256SUMS
(cd "$work" && sha256sum -c SHA256SUMS)
mkdir "$work/context"
tar -xzf "$work/$name.tar.gz" -C "$work/context"
mv "$work/context/$name" "$work/context/strycast"
expected=$(jq -r .version "$work/context/strycast/bin/manifest.json")

docker build --file tools/ci/Containerfile --tag "$image:$version" \
  --label org.opencontainers.image.source="https://github.com/$REPO" \
  --label org.opencontainers.image.version="$version" \
  --label org.opencontainers.image.revision="$(jq -r .revision "$work/context/strycast/bin/manifest.json")" \
  --label org.opencontainers.image.description="Strycast $TAG for AMD Strix Halo (gfx1151)" \
  --label org.opencontainers.image.licenses=MIT \
  "$work/context"
rm -rf "$work"

actual=$(docker run --rm "$image:$version" gufo --version)
[ "$actual" = "$expected" ] || fail "The image reports '$actual', the bundle '$expected'."

if [ "${PUBLISH:-true}" != true ]; then
  echo "Built $image:$version without publishing." | tee -a "$summary"
  exit 0
fi
docker push "$image:$version"
docker tag "$image:$version" "$image:latest"
docker push "$image:latest"
echo "Published \`$image:$version\` and \`latest\`." | tee -a "$summary"
