#!/usr/bin/env bash
# Nightly Strix Halo run of a staging commit: GPU tests, Gufo's functional suite
# and the Strycast model benchmarks. Run on the GPU host by its nightly launcher
# inside a bench-queue job, from a checkout of the commit under test.
#
# Usage: run.sh --config FILE --head-bundle DIR --base-bundle DIR --out DIR [--deadline HH:MM]
#               [--phases "gpu_tests functional benchmarks"]
#   --config       host settings (model paths, toolchain); see tools/nightly/README.md
#   --head-bundle  portable Linux bundle CI built for this commit (bin/, runtime/)
#   --base-bundle  the same for main (or a commit with main's tree), the timing
#                  baseline of the functional suite
#   --out          results: result.json, report.md, logs/
#   --deadline     local time after which phases still pending are skipped
#   --phases       the phases to run, for a manual run; a night runs all three
#
# result.json holds {"state": "success" | "failure", "description": ...}; the
# launcher turns it into the commit status and the promotion PR comment.
set -uo pipefail

deadline=
phases=(gpu_tests functional benchmarks)
while [ $# -gt 0 ]; do
  case $1 in
    --config) config=$2 ;;
    --head-bundle) head_bundle=$2 ;;
    --base-bundle) base_bundle=$2 ;;
    --out) out=$2 ;;
    --deadline) deadline=$2 ;;
    --phases) read -ra phases <<< "$2" ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift 2
done
# Where the run's logs are published, for the reports' links.
RUN_URL=${RUN_URL:-https://github.com/slimsami/strycast/releases/tag/nightly}
# shellcheck source=/dev/null
source "$config"
export ROCM_PATH HIP_PLATFORM=amd
src=$(git rev-parse --show-toplevel)
sha=$(git rev-parse HEAD)
mkdir -p "$out/logs"
out=$(cd "$out" && pwd)
report=$out/report.md
head_gufo=$head_bundle/bin/gufo
base_gufo=$base_bundle/bin/gufo
base_revision=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["revision"])' \
  "$base_bundle/bin/manifest.json")
failures=()
skipped=()

log() { echo "== $(date +%H:%M:%S) $*"; }
fail() { failures+=("$1"); }
progress() { if command -v llq >/dev/null; then llq progress "$@" >/dev/null 2>&1; fi; }
# Phases still pending at the deadline are skipped so production is back in time.
late() {
  [ -n "$deadline" ] || return 1
  [ "$(date +%H:%M)" \> "$deadline" ] || return 1
  skipped+=("$1 (deadline $deadline)")
  log "skipping $1: past $deadline"
}

{
  echo "## Nightly Strix Halo run"
  echo
  echo "Commit \`$sha\` on $(hostname), $(date -u '+%Y-%m-%d %H:%M UTC')."
  echo "Binary: the CI bundle for this commit ($("$head_gufo" --version 2>&1 | tail -1))."
  echo
} > "$report"

# Python for the tests that CTest starts and for the functional client, kept in
# the build directory between nights; the requirements include numpy, and
# zstandard reads the compressed code objects before Python 3.14.
venv=$src/build/nightly-venv
python3 -m venv "$venv" &&
  "$venv/bin/pip" -q install -r "$src/tests/functional/requirements.txt" zstandard > "$out/logs/pip.log" 2>&1
venv_rc=$?

# 1. GPU tests: every test in the tree except the model families Strycast does
# not run nightly (MiniMax H3 and DeepSeek V4 need weights that are not here),
# qwen_mtp_reference_test, whose fixtures come from a checkpoint other than the
# published 27B quantizations, and kernel_resources_test, checked below on the
# CI binaries.
gpu_tests() {
  local tree=$src/build/nightly-gpu
  log "build GPU tests"
  # Link the test binaries against the pinned runtime so that tests which start
  # them from Python find it too.
  local flags
  flags=$(bash "$TOOLCHAIN/rocm10-link.sh")
  if [ $venv_rc != 0 ]; then
    fail "test Python setup"
    return
  fi
  if ! BUILD_TESTING=ON GUFO_BUILD_TOOLS=ON bash "$TOOLCHAIN/configure-rocm10.sh" "$src" nightly-gpu \
       > "$out/logs/configure.log" 2>&1 ||
     ! cmake -S "$src" -B "$tree" -DCMAKE_EXE_LINKER_FLAGS="$flags" -DCMAKE_SHARED_LINKER_FLAGS="$flags" \
         -DPython3_EXECUTABLE="$venv/bin/python" \
       >> "$out/logs/configure.log" 2>&1 ||
     ! cmake --build "$tree" -j"$(nproc)" > "$out/logs/build.log" 2>&1; then
    fail "GPU test build"
    { echo "### GPU tests"; echo; echo "The test tree did not build:"; echo
      echo '```'; grep -E "error:|FAILED:" "$out/logs/build.log" | sort -u | head -20; echo '```'; echo; } >> "$report"
    return
  fi
  log "run GPU tests"
  (
    cd "$tree" || exit 1
    export GUFO_QWEN27B_MODEL=$QWEN27_GGUF GUFO_QWEN27B_DFLASH_MODEL=$QWEN27_DFLASH
    export GUFO_QWEN27B_MTP_MODEL=$QWEN27_GGUF GUFO_QWEN_GGUF=$QWEN27_GGUF
    export GUFO_QWEN35BA3B_MODEL=$Q35_Q6 EMBEDDING_GEMMA2_MODEL_ROOT=$EMBEDDING_ROOT
    export QWEN3_ASR_MODEL_ROOT=$ASR_ROOT
    ctest -E '^(minimax_h3_|ds4\.|qwen_mtp_reference_test$|kernel_resources_test$)' --timeout 1800 --output-junit "$out/ctest.xml" \
      > "$out/logs/ctest.log" 2>&1
  )
  local rc=$?
  # The 35B target test runs once per quantization; ctest covered UD-Q6_K_XL.
  local target35=$tree/tests/models/qwen35ba3b/qwen35ba3b_target_test
  local extra=()
  for q in Q35_Q4 Q35_Q8 Q35_Q4KM; do
    (cd "$src" && GUFO_QWEN35BA3B_MODEL=${!q} timeout 1800 "$target35") \
      > "$out/logs/qwen35ba3b_target_test.$q.log" 2>&1
    local code=$?
    extra+=("| qwen35ba3b_target_test ($(basename "${!q}")) | $([ $code = 0 ] && echo passed || echo "failed ($code)") |")
    [ $code = 0 ] || rc=1
  done
  # Kernel scratch, checked on the CI binaries rather than the test tree. Gufo's
  # allowance file predates some kernels' clang 24 scratch, so a kernel over it
  # passes when it is no worse than on main.
  local check=("$venv/bin/python" "$src/tools/ci/check-kernel-resources.py"
               --bundler "$TOOLCHAIN/clang-offload-bundler")
  local elf=bin/libexec/gufo summary
  if "${check[@]}" --baseline "$src/tools/ci/kernel-resources.json" "$head_bundle/$elf" \
       > "$out/logs/kernel-resources.log" 2>&1; then
    extra+=("| kernel scratch against the allowance file | passed |")
  else
    summary=$(tail -1 "$out/logs/kernel-resources.log")
    if "${check[@]}" "$head_bundle/$elf" --compare "$base_bundle/$elf" \
         > "$out/logs/kernel-resources-main.log" 2>&1; then
      extra+=("| kernel scratch | passed: no kernel worse than main ($summary against the allowance file, as on main) |")
    else
      extra+=("| kernel scratch | failed: $(tail -1 "$out/logs/kernel-resources-main.log") against main |")
      rc=1
    fi
  fi
  python3 "$src/tools/nightly/ctest_report.py" "$out/ctest.xml" "${extra[@]}" >> "$report"
  [ $rc = 0 ] || fail "GPU tests"
}

# 2. Gufo's functional suite on its own models, full tier, timed against main.
functional() {
  log "functional suite"
  if [ $venv_rc != 0 ]; then
    fail "functional suite setup"
    return
  fi
  # The planner picks models from changed paths; these two select every model.
  # The revisions are recorded for the report, as for a pull request.
  if ! python3 "$src/tests/functional/devlab_plan.py" nightly/full src/models/qwen/engine.cpp --tier full \
       --output "$out/functional-plan.json" > "$out/logs/functional-plan.log" 2>&1 ||
     ! python3 -c 'import json, sys
path, base, head = sys.argv[1:]
plan = json.load(open(path))
plan.update(base=base, head=head)
open(path, "w").write(json.dumps(plan, indent=2) + "\n")' "$out/functional-plan.json" "$base_revision" "$sha"; then
    fail "functional plan"
    return
  fi
  "$venv/bin/python" "$src/tests/functional/devlab_run.py" --plan "$out/functional-plan.json" \
    --assets "$FUNCTIONAL_ASSETS" --binary "$head_gufo" --base-binary "$base_gufo" \
    --output "$out/functional-results" --report "$out/functional-report" --logs "$out/logs/functional" \
    > "$out/logs/functional.log" 2>&1
  local rc=$?
  cp "$out/functional-results/summary.json" "$out/functional-report/" 2>/dev/null
  python3 "$src/tests/functional/report.py" --plan "$out/functional-plan.json" \
    --results "$out/functional-report" --authorize success \
    --smoke "$([ $rc = 0 ] && echo success || echo failure)" --run-url "$RUN_URL" \
    --comment "$out/functional.md" --status "$out/functional.env" > "$out/logs/functional-report.log" 2>&1
  { echo "### Gufo functional suite"; echo; grep -v '^<!-- devlab-functional -->$' "$out/functional.md"; echo; } \
    >> "$report"
  grep -q '^state=success$' "$out/functional.env" || rc=1
  [ $rc = 0 ] || fail "functional suite"
}

# 3. Benchmarks: every Gufo table of the Strycast models, compared with the
# committed artifacts.
benchmarks() {
  log "benchmarks"
  # The memory tables read hipMemGetInfo from the CLR the bundle runs on.
  local py="env GUFO_HIP_LIBRARY=$ROCM_PATH/lib/libamdhip64.so $TOOLCHAIN/run-rocm10.sh $BENCH_PYTHON"
  local art=$out/bench
  local c35=$src/docs/models/qwen3.6-35b-a3b/artifacts corn=$src/docs/models/ornith-1.5-122b/artifacts
  mkdir -p "$art/35b" "$art/ornith"
  # Speculative concurrency tables check their outputs against AR C1 hashes,
  # recorded first with a one-user AR variant of the DFlash2 table.
  python3 - "$c35/bench.json" "$out/bench-arhash.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
table = dict(config['tables']['multi-dflash2'], modes=['ar'], concurrency=[1],
             cache_prompt=False, prefill_first=False)
config['tables'] = {'multi-dflash2': table}
json.dump(config, open(sys.argv[2], 'w'), indent=1)
PY
  local -A gguf=([q4]=$Q35_Q4 [q6]=$Q35_Q6 [q8]=$Q35_Q8 [q4km]=$Q35_Q4KM)
  mb35() {
    $py tools/bench/model-bench.py --model qwen3.6-35b-a3b --gufo "$head_gufo" --config "$2" \
      --artifacts-dir "$art/35b" --log-dir "$out/logs/bench" --gguf "$1=${gguf[$1]}" --draft "$Q35_DRAFT" \
      run --target gufo --table "$3" >> "$out/logs/bench.log" 2>&1 || fail "benchmark $3"
  }
  mbo() {
    $py tools/bench/model-bench.py --model ornith-1.5-122b --gufo "$head_gufo" --config "$1" \
      --artifacts-dir "$art/ornith" --log-dir "$out/logs/bench" --gguf "q4km=$ORNITH_GGUF" \
      run --target gufo --table "$2" >> "$out/logs/bench.log" 2>&1 || fail "benchmark Ornith $2"
  }
  cd "$src" || return
  for v in q4 q6 q8 q4km; do
    late "benchmarks from 35B $v" && return
    log "35B $v"
    mb35 $v "$c35/bench.json" single-ar-$v
    mb35 $v "$c35/bench.json" single-dflash2-$v
    [ $v = q4km ] || mb35 $v "$c35/bench-mtp.json" single-mtp-$v
    mb35 $v "$out/bench-arhash.json" multi-dflash2-$v
    mb35 $v "$c35/bench.json" multi-ar-$v
    mb35 $v "$c35/bench.json" multi-dflash2-$v
    [ $v = q4km ] || mb35 $v "$c35/bench-mtp.json" multi-mtp-$v
    mb35 $v "$c35/bench.json" memory-$v
  done
  late "35B loading and Ornith benchmarks" && return
  $py tools/bench/model-bench.py --model qwen3.6-35b-a3b --gufo "$head_gufo" --config "$c35/bench.json" \
    --artifacts-dir "$art/35b" --log-dir "$out/logs/bench" --gguf "q4=$Q35_Q4" --gguf "q6=$Q35_Q6" \
    --gguf "q8=$Q35_Q8" --gguf "q4km=$Q35_Q4KM" --draft "$Q35_DRAFT" \
    run --target gufo --table loading >> "$out/logs/bench.log" 2>&1 || fail "benchmark loading"
  log "Ornith"
  mbo "$corn/bench.json" single-ar-q4km
  mbo "$corn/bench-mtp.json" single-mtp-q4km
  mbo "$corn/bench.json" memory-q4km
  mbo "$corn/bench.json" loading
  local groups=("Qwen3.6 35B-A3B" "$c35" "$art/35b" "Ornith 1.5 122B" "$corn" "$art/ornith")
  # A table that regressed is measured once more, and the better value counts:
  # concurrent tables move up to 10% between two runs of one binary.
  python3 "$src/tools/nightly/compare_bench.py" --set-aside "$out/bench-retry.txt" "${groups[@]}" > /dev/null
  local index config table
  while read -r index config table; do
    late "the second measurement of $table" && break
    log "measure $table again"
    if [ "$index" = 0 ]; then
      mb35 "${table##*-}" "$c35/$config" "$table"
    else
      mbo "$corn/$config" "$table"
    fi
  done < "$out/bench-retry.txt"
  python3 "$src/tools/nightly/compare_bench.py" "${groups[@]}" >> "$report" || fail "benchmark regressions"
}

for i in "${!phases[@]}"; do
  phase=${phases[$i]}
  late "$phase" && continue
  cd "$src" || exit 1
  $phase
  progress "$((i + 1))/${#phases[@]}" "$phase"
done

if [ ${#skipped[@]} -gt 0 ]; then
  { echo "### Not run"; echo; printf -- '- %s\n' "${skipped[@]}"; echo; } >> "$report"
  fail "phases skipped at the deadline"
fi
if [ ${#failures[@]} -eq 0 ]; then
  state=success description="GPU tests, functional suite and benchmarks passed"
else
  joined=$(printf '%s, ' "${failures[@]}")
  state=failure description="Failed: ${joined%, }"
fi
python3 - "$out/result.json" "$state" "$description" "$sha" <<'PY'
import json, sys
path, state, description, sha = sys.argv[1:]
json.dump({'state': state, 'description': description[:140], 'sha': sha}, open(path, 'w'), indent=1)
PY
log "$state: $description"
[ "$state" = success ]
