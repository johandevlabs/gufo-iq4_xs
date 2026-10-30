#!/usr/bin/env python3
"""Compare a night's model-bench artifacts with the committed ones.

Usage: compare_bench.py [--set-aside FILE] LABEL COMMITTED_DIR NEW_DIR [LABEL COMMITTED_DIR NEW_DIR ...]

Prints the nightly report's benchmark section and exits 1 on a regression past
the tolerance or on a speculative output that differs from AR.

With --set-aside, the artifacts of every single-user or multi-user table with a
regression move to NEW_DIR/first and FILE lists the tables to measure again, one
`INDEX CONFIG TABLE` line each (INDEX counts the LABEL groups from 0). The next
comparison keeps the better of the two measurements of each value.
"""

import json
import re
import shutil
import sys
from pathlib import Path

# Relative tolerance per metric. AR repeats within about 1%; speculative tables
# (prefill included, which also injects the draft context) moved up to 6%
# between two runs of one binary.
AR_TOLERANCE = 0.03
SPECULATIVE_TOLERANCE = 0.08
LOADING_TOLERANCE = (0.15, 2.0)  # relative, or absolute seconds, whichever is larger
MEMORY_TOLERANCE_GIB = 0.5


def load(path):
    return json.loads(path.read_text())


def speculative(name):
    return 'dflash2' in name or 'mtp' in name


def row_metrics(name, rows):
    """(row, metric) -> (value, higher_is_better, tolerance check) for row-shaped tables."""
    for key, row in rows.items():
        if not isinstance(row, dict):
            continue
        for metric in ('pp', 'tg'):
            if isinstance(row.get(metric), (int, float)):
                spec = speculative(name)
                yield key, metric, row[metric], True, SPECULATIVE_TOLERANCE if spec else AR_TOLERANCE
        if isinstance(row.get('gib'), (int, float)):
            yield key, 'GiB', row['gib'], False, None
        if isinstance(row.get('ready_s'), (int, float)):
            yield key, 'ready s', row['ready_s'], False, None


def multi_metrics(results):
    for key, result in results.items():
        value = result.get('aggregate', {}).get('output_tokens_per_second', {}).get('median')
        if isinstance(value, (int, float)):
            yield f'{key} users', 'tok/s', value, True, SPECULATIVE_TOLERANCE


def table_of(name):
    """(config file, table id) of the model-bench run that wrote this artifact."""
    single = re.fullmatch(r'single-(ar|dflash2|mtp)(?:-repetition)?-(\w+)-gufo\.json', name)
    multi = re.fullmatch(r'multi-\w+-(\w+)-gufo-(ar|dflash2|mtp)\.json', name)
    if single:
        mode, variant = single.groups()
    elif multi:
        variant, mode = multi.groups()
    else:
        return None
    return ('bench-mtp.json' if mode == 'mtp' else 'bench.json',
            f'{name.split("-")[0]}-{mode}-{variant}')


def best(first, second):
    """Per value, the better of two measurements of one table."""
    merged = dict(second)
    for key, (value, higher, tolerance) in first.items():
        if key in merged:
            other = merged[key][0]
            better = value > other if higher else value < other
            if better:
                merged[key] = (value, higher, tolerance)
    return merged


def metrics(name, data):
    if 'rows' in data:
        return {(row, metric): rest for row, metric, *rest in row_metrics(name, data['rows'])}
    return {(row, metric): rest for row, metric, *rest in multi_metrics(data.get('results', {}))}


def regressed(metric, old, new, higher, tolerance):
    if metric == 'GiB':
        return new - old > MEMORY_TOLERANCE_GIB
    if metric == 'ready s':
        relative, absolute = LOADING_TOLERANCE
        return new - old > max(old * relative, absolute)
    change = (new - old) / old if old else 0
    return change < -tolerance if higher else change > tolerance


def completion_hashes(row):
    return [count.get('completion_sha256') for count in row.get('counts', [])]


def exactness(new_dir):
    """Speculative completions that differ from AR, from single and multi tables."""
    problems, checked = [], 0
    for path in sorted(new_dir.glob('single-*-gufo.json')):
        name = path.name
        if not speculative(name) or 'repetition' in name:
            continue
        mode = 'dflash2' if 'dflash2' in name else 'mtp'
        ar_path = new_dir / name.replace(f'single-{mode}-', 'single-ar-')
        if not ar_path.exists():
            continue
        spec_rows, ar_rows = load(path)['rows'], load(ar_path)['rows']
        for depth, row in spec_rows.items():
            if depth in ar_rows:
                checked += 1
                if completion_hashes(row) != completion_hashes(ar_rows[depth]):
                    problems.append(f'{name} depth {depth}')
    for path in sorted(new_dir.glob('multi-*-gufo-*.json')):
        for key, result in load(path).get('results', {}).items():
            for case, data in result.get('cases', {}).items():
                if 'matchesC1' in data:
                    checked += 1
                    if not data['matchesC1']:
                        problems.append(f'{path.name} {key} users {case}')
    return checked, problems


def main(args):
    set_aside = None
    if args[:1] == ['--set-aside']:
        set_aside, args = Path(args[1]), args[2:]
    if not args or len(args) % 3:
        sys.exit(__doc__)
    failed, retry = False, []
    print('### Benchmarks\n')
    print(f'Against the committed artifacts. Tolerance: {AR_TOLERANCE:.0%} for single-user AR, '
          f'{SPECULATIVE_TOLERANCE:.0%} for speculative and multi-user tables, '
          f'{LOADING_TOLERANCE[0]:.0%} or {LOADING_TOLERANCE[1]:.0f} s for loading, '
          f'{MEMORY_TOLERANCE_GIB} GiB for memory.\n')
    groups = zip(args[::3], map(Path, args[1::3]), map(Path, args[2::3]))
    for index, (label, committed, new) in enumerate(groups):
        regressions, improvements, missing, compared, measured_twice = [], [], [], 0, []
        regressed_tables = set()
        for path in sorted(new.glob('*-gufo*.json')):
            reference = committed / path.name
            if not reference.exists():
                missing.append(path.name)
                continue
            before, after = metrics(path.name, load(reference)), metrics(path.name, load(path))
            first = new / 'first' / path.name
            if first.exists():
                after = best(metrics(path.name, load(first)), after)
                measured_twice.append(path.name)
            for key, (value, higher, tolerance) in after.items():
                if key not in before:
                    continue
                compared += 1
                old = before[key][0]
                change = (value - old) / old * 100 if old else 0
                line = f'| {path.stem} | {key[0]} | {key[1]} | {old:g} | {value:g} | {change:+.1f}% |'
                if regressed(key[1], old, value, higher, tolerance):
                    regressions.append(line)
                    if table_of(path.name):
                        regressed_tables.add(table_of(path.name))
                elif regressed(key[1], value, old, higher, tolerance):
                    improvements.append(line)
        checked, problems = exactness(new)
        expected = {p.name for p in committed.glob('*-gufo*.json')}
        not_run = sorted(expected - {p.name for p in new.glob('*-gufo*.json')})
        verdict = 'regressed' if regressions or problems or not_run else 'within tolerance'
        print(f'#### {label}: {verdict}\n')
        print(f'{compared} values compared; {checked} speculative completions checked against AR, '
              f'{len(problems)} differ.\n')
        if measured_twice:
            print('Measured twice after a first regression, keeping the better value: '
                  + ', '.join(f'`{name}`' for name in measured_twice) + '\n')
        header = '| Table | Row | Metric | Committed | Tonight | Change |\n| --- | --- | --- | ---: | ---: | ---: |'
        if regressions:
            print('Regressions:\n\n' + header + '\n' + '\n'.join(regressions) + '\n')
        if improvements:
            print('Improvements past the tolerance:\n\n' + header + '\n' + '\n'.join(improvements) + '\n')
        if problems:
            print('Speculative output differs from AR: ' + ', '.join(f'`{p}`' for p in problems) + '\n')
        if not_run:
            print('Tables not measured: ' + ', '.join(f'`{name}`' for name in not_run) + '\n')
        if missing:
            print('No committed artifact to compare: ' + ', '.join(f'`{name}`' for name in missing) + '\n')
        failed |= bool(regressions or problems or not_run)
        # AR tables first: the speculative multi-user tables check against their C1 outputs.
        for config, table in sorted(regressed_tables, key=lambda t: (t[1].split('-')[1] != 'ar', t)):
            if not set_aside:
                break
            retry.append(f'{index} {config} {table}')
            (new / 'first').mkdir(exist_ok=True)
            for path in new.glob('*-gufo*.json'):
                if table_of(path.name) == (config, table):
                    shutil.move(path, new / 'first' / path.name)
    if set_aside:
        set_aside.write_text(''.join(line + '\n' for line in retry))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
