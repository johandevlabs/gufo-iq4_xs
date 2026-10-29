#!/usr/bin/env bash
# Replace staging or main with a prepared rebase onto a newer Gufo release.
# Run by .github/workflows/rebase-upstream.yml, the only actor the branch
# rulesets let rewrite those branches. See "Rebasing onto Gufo" in README.md.
#
# Environment: GH_TOKEN, REPO (owner/name), PR (pull request number), ACTOR.
set -euo pipefail

UPSTREAM=https://github.com/gufo-org/gufo.git
LABEL=gpu-checked
summary=${GITHUB_STEP_SUMMARY:-/dev/stdout}

fail() {
  echo "::error::$*"
  exit 1
}

# workflow_dispatch already needs write access; rewriting history needs admin.
permission=$(gh api "repos/$REPO/collaborators/$ACTOR/permission" --jq .permission)
[ "$permission" = admin ] || fail "$ACTOR has $permission access; only admins can replace protected branches."

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
  *) fail "PR #$PR is $head -> $base; expected rebase/<tag> -> staging or staging -> main." ;;
esac
jq -e --arg label "$LABEL" '.labels | map(.name) | index($label)' <<<"$pr_json" >/dev/null ||
  fail "PR #$PR has no '$LABEL' label. Add it once the GPU checks in the PR description pass."

git fetch --quiet --no-tags "$UPSTREAM" '+refs/tags/v*:refs/tags/upstream/v*' '+refs/heads/main:refs/remotes/upstream/main'
git fetch --quiet --no-tags origin "+refs/heads/$base:refs/remotes/origin/$base" "+refs/heads/$head:refs/remotes/origin/$head"
[ "$(git rev-parse "origin/$head")" = "$sha" ] || fail "$head moved while the workflow started; run it again."
old=$(git rev-parse "origin/$base")

# The Gufo release a branch is based on: the closest upstream tag.
gufo_base() { git describe --tags --abbrev=0 --match 'upstream/v*' "$1" | sed 's|^upstream/||'; }
old_tag=$(gufo_base "$old")
new_tag=$(gufo_base "$sha")
newest=$(printf '%s\n%s\n' "$old_tag" "$new_tag" | sort -V | tail -1)
if [ "$new_tag" = "$old_tag" ] || [ "$newest" != "$new_tag" ]; then
  fail "$head is based on Gufo $new_tag, which is not newer than $base's $old_tag."
fi

# Exactly on the tag: every commit past it must be Strycast's, not upstream's.
own=$(git rev-list --count "$sha" --not "upstream/$new_tag" upstream/main)
past_tag=$(git rev-list --count "upstream/$new_tag..$sha")
[ "$own" = "$past_tag" ] ||
  fail "$head carries $((past_tag - own)) upstream commits past $new_tag; rebase onto the tag itself."

if [ "$base" = main ]; then
  git fetch --quiet --no-tags origin '+refs/heads/staging:refs/remotes/origin/staging'
  [ "$(git rev-parse origin/staging)" = "$sha" ] || fail "staging moved; run it again."
fi

git show "$sha:README.md" | tr '\n' ' ' | grep -q "currently Gufo $new_tag" ||
  fail "README.md does not say 'currently Gufo $new_tag'."

# Every Strycast commit must survive the rebase; new ones are allowed.
subjects() { git log --no-merges --format=%s "upstream/$1..$2" | sort -u; }
dropped=$(comm -23 <(subjects "$old_tag" "$old") <(subjects "$new_tag" "$sha"))
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

archive="archive/$base-$old_tag"
if git ls-remote --exit-code --tags origin "refs/tags/$archive" >/dev/null; then
  archive="$archive-$(date -u +%Y%m%d%H%M)"
fi

{
  echo "## $base: Gufo $old_tag -> $new_tag"
  echo
  echo "- Pull request: #$PR ($head, \`${sha:0:8}\`)"
  echo "- Previous head \`${old:0:8}\` archived as \`$archive\`"
  echo "- Strycast commits: $(git rev-list --count --no-merges "upstream/$old_tag..$old") -> $own"
  echo
  echo '<details><summary>git range-diff</summary>'
  echo
  echo '```'
  git range-diff --no-color "upstream/$old_tag..$old" "upstream/$new_tag..$sha" | head -n 3000
  echo '```'
  echo '</details>'
} >>"$summary"

git push origin "$old:refs/tags/$archive"
git push --force-with-lease="refs/heads/$base:$old" origin "$sha:refs/heads/$base"
echo "$base now points at ${sha:0:8} (Gufo $new_tag); the old head is $archive."
