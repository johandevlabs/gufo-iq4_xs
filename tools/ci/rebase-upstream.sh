#!/usr/bin/env bash
# Check a prepared rebase of staging or main onto a newer Gufo release or
# Gufo main commit, and print the commands that replace the branch. Run by
# .github/workflows/rebase-upstream.yml. The rulesets block force-pushes, so an
# admin lifts that for the push itself: see "Rebasing onto Gufo" in README.md.
#
# Environment: GH_TOKEN, REPO (owner/name), PR (pull request number).
set -euo pipefail

UPSTREAM=https://github.com/gufo-org/gufo.git
LABEL=gpu-checked
summary=${GITHUB_STEP_SUMMARY:-/dev/stdout}

fail() {
  echo "::error::$*"
  exit 1
}

pr_json=$(gh api "repos/$REPO/pulls/$PR")
field() { jq -r "$1" <<<"$pr_json"; }
state=$(field .state)
base=$(field .base.ref)
head=$(field .head.ref)
head_repo=$(field .head.repo.full_name)
sha=$(field .head.sha)

[ "$state" = open ] || fail "PR #$PR is $state."
[ "$head_repo" = "$REPO" ] || fail "PR #$PR comes from $head_repo, not $REPO."
case "$base:$head" in
  staging:rebase/* | main:staging) ;;
  *) fail "PR #$PR is $head -> $base; expected rebase/<base> -> staging or staging -> main." ;;
esac
jq -e --arg label "$LABEL" '.labels | map(.name) | index($label)' <<<"$pr_json" >/dev/null ||
  fail "PR #$PR has no '$LABEL' label. Add it once the GPU checks in the PR description pass."

git fetch --quiet --no-tags "$UPSTREAM" '+refs/tags/v*:refs/tags/upstream/v*' '+refs/heads/main:refs/remotes/upstream/main'
git fetch --quiet --no-tags origin "+refs/heads/$base:refs/remotes/origin/$base" "+refs/heads/$head:refs/remotes/origin/$head"
[ "$(git rev-parse "origin/$head")" = "$sha" ] || fail "$head moved while the workflow started; run it again."
old=$(git rev-parse "origin/$base")

# The Gufo commit a branch is based on: where it leaves Gufo's main. That is
# a release tag (upstream/v*) when it sits exactly on one, otherwise the main
# commit itself. `name` is how the README states it, `slug` names the archive.
gufo_base() {
  local commit tag
  commit=$(git merge-base "$1" upstream/main)
  if tag=$(git describe --tags --exact-match --match 'upstream/v*' "$commit" 2>/dev/null); then
    echo "$tag"
  else
    echo "$commit"
  fi
}
gufo_name() { case $1 in upstream/v*) echo "${1#upstream/}" ;; *) echo "main ${1:0:8}" ;; esac; }
gufo_slug() { gufo_name "$1" | tr ' ' '-'; }
old_ref=$(gufo_base "$old")
new_ref=$(gufo_base "$sha")
old_tag=$(gufo_name "$old_ref")
new_tag=$(gufo_name "$new_ref")
if [ "$(git rev-parse "$new_ref^{commit}")" = "$(git rev-parse "$old_ref^{commit}")" ] ||
   ! git merge-base --is-ancestor "$old_ref" "$new_ref"; then
  fail "$head is based on Gufo $new_tag, which is not newer than $base's $old_tag."
fi

# Exactly on the base: every commit past it must be Strycast's, not upstream's.
own=$(git rev-list --count "$sha" --not "$new_ref" upstream/main)
past_tag=$(git rev-list --count "$new_ref..$sha")
[ "$own" = "$past_tag" ] ||
  fail "$head carries $((past_tag - own)) upstream commits past $new_tag; rebase onto it itself."

if [ "$base" = main ]; then
  git fetch --quiet --no-tags origin '+refs/heads/staging:refs/remotes/origin/staging'
  [ "$(git rev-parse origin/staging)" = "$sha" ] || fail "staging moved; run it again."
fi

git show "$sha:README.md" | tr '\n' ' ' | grep -q "currently Gufo $new_tag" ||
  fail "README.md does not say 'currently Gufo $new_tag'."

# Every Strycast commit must survive the rebase; new ones are allowed.
subjects() { git log --no-merges --format=%s "$1..$2" | sort -u; }
dropped=$(comm -23 <(subjects "$old_ref" "$old") <(subjects "$new_ref" "$sha"))
[ -z "$dropped" ] || fail "commits missing after the rebase: $(tr '\n' ';' <<<"$dropped")"

# The checks the target branch requires must have passed on this commit. A
# rebase PR conflicts with its target, so they come from push runs on the
# rebase/<tag> branch (staging -> main reuses the same commit's results).
required=$(gh api "repos/$REPO/rules/branches/$base" \
  --jq '.[] | select(.type == "required_status_checks") | .parameters.required_status_checks[].context')
while IFS= read -r check; do
  [ -n "$check" ] || continue
  # A pull_request-only check, and a rebase PR conflicts with its target, so
  # it never runs. The source rules above check the same thing.
  [ "$check" = "Promotion comes from staging" ] && continue
  passed=$(gh api "repos/$REPO/commits/$sha/check-runs" -X GET -f check_name="$check" \
    --jq '[.check_runs[] | select(.conclusion == "success")] | length')
  [ "$passed" -gt 0 ] || fail "required check '$check' has not passed on $sha."
done <<<"$required"

archive="archive/$base-$(gufo_slug "$old_ref")"
if git ls-remote --exit-code --tags origin "refs/tags/$archive" >/dev/null; then
  archive="$archive-$(date -u +%Y%m%d%H%M)"
fi

{
  echo "## $base: Gufo $old_tag -> $new_tag"
  echo
  echo "- Pull request: #$PR ($head, \`${sha:0:8}\`)"
  echo "- Previous head \`${old:0:8}\`, to archive as \`$archive\`"
  echo "- Strycast commits: $(git rev-list --count --no-merges "$old_ref..$old") -> $own"
  echo
  echo '<details><summary>git range-diff</summary>'
  echo
  echo '```'
  # sed reads the whole diff: head would stop early and, under pipefail, the
  # SIGPIPE of a longer range-diff would fail the script.
  git range-diff --no-color "$old_ref..$old" "$new_ref..$sha" | sed -n '1,3000p'
  echo '```'
  echo '</details>'
} >>"$summary"

{
  echo
  echo "Checks passed. To replace $base:"
  echo
  echo '```sh'
  echo "git fetch origin $old $sha"
  echo "git push origin $old:refs/tags/$archive"
  echo "git push --force-with-lease=refs/heads/$base:$old origin $sha:refs/heads/$base"
  echo '```'
} | tee -a "$summary"
