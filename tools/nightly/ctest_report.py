#!/usr/bin/env python3
"""Summarize a CTest JUnit file as the nightly report's GPU test section."""

import sys
import xml.etree.ElementTree as ET


def main():
    cases = ET.parse(sys.argv[1]).getroot().iter('testcase')
    passed, failed, skipped = [], [], []
    for case in cases:
        name, status = case.get('name'), case.get('status', '')
        if case.find('failure') is not None or status == 'fail':
            failed.append(name)
        elif case.find('skipped') is not None or status in ('notrun', 'disabled'):
            skipped.append(name)
        else:
            passed.append(name)
    print('### GPU tests\n')
    print(f'{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped '
          '(MiniMax H3 and DeepSeek V4 tests are not run: their weights are not on this host; '
          'neither is `qwen_mtp_reference_test`, whose fixtures come from another 27B checkpoint).\n')
    if failed:
        print('Failed: ' + ', '.join(f'`{name}`' for name in failed) + '\n')
    if skipped:
        # Return code 77: the test's model or reference files are not configured.
        print('Skipped: ' + ', '.join(f'`{name}`' for name in skipped) + '\n')
    extra = sys.argv[2:]
    if extra:
        print('| Extra run | Result |\n| --- | --- |')
        print('\n'.join(extra) + '\n')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
