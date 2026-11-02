#!/usr/bin/env python3
"""Time again the functional models whose server phases came out slower.

Usage: retime_functional.py --plan FILE --report DIR --assets FILE --binary GUFO
                            --base-binary GUFO --logs DIR

A paired timing run compares two repeats per side, and some startup phases
are bimodal: Flash-Next's vision sidecar took 0.6-1.2 s or 2-3.3 s on either
binary, so both repeats of one side can land slow by chance. For every model
whose phases file in DIR has the `slower` verdict, the first file moves to
DIR/first and the same timing script runs again with REPEATS paired repeats;
that measurement is the one reported, and it stays `slower` only when every
repeat is. Prints the models timed again, one per line.
"""

import argparse
import fnmatch
import json
from pathlib import Path
import shutil
import subprocess
import sys

FUNCTIONAL = Path(__file__).resolve().parents[2] / 'tests' / 'functional'
sys.path.insert(0, str(FUNCTIONAL))

from report import timing_state  # noqa: E402

REPEATS = 6


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--plan', '--report', '--assets', '--binary', '--base-binary', '--logs'):
        parser.add_argument(name, required=True, type=Path)
    args = parser.parse_args()
    plan = load(args.plan)
    for model in plan.get('models', []):
        key = model['id']
        destination = args.report / f'{key}.phases.json'
        if timing_state(load(destination)) != 'slower':
            continue
        (args.report / 'first').mkdir(exist_ok=True)
        shutil.move(destination, args.report / 'first' / destination.name)
        # The same command and phase selection as devlab_run.py.
        script = 'phases.py' if model['category'] == 'llm' else 'service_phases.py'
        command = [sys.executable, str(FUNCTIONAL / script), '--model-id', key, '--assets', str(args.assets),
                   '--base-binary', str(args.base_binary), '--head-binary', str(args.binary),
                   '--base-revision', plan['base'], '--head-revision', plan['head'],
                   '--repeats', str(REPEATS), '--output', str(destination),
                   '--logs', str(args.logs / f'{key}-again')]
        with (args.logs / f'{key}.phases-again.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3 * 2400)
        observed = load(destination)
        if not observed.get('phases'):
            # A failed second run must not hide the first verdict.
            shutil.copy(args.report / 'first' / destination.name, destination)
            continue
        for phase in observed['phases']:
            phase['checked'] = phase.get('checked', True) and any(
                fnmatch.fnmatchcase(phase['key'], pattern) for pattern in model['timing_checks'])
        observed['timing_checks'] = model['timing_checks']
        destination.write_text(json.dumps(observed, indent=2) + '\n')
        print(key)
    return 0


if __name__ == '__main__':
    sys.exit(main())
