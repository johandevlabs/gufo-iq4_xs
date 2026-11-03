#!/usr/bin/env python3
"""CPU-only regressions for compressed HIP bundle extraction."""

import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "kernel_resources", ROOT / "tools/ci/check-kernel-resources.py")
RESOURCES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESOURCES)


class KernelResourcesTest(unittest.TestCase):
    def test_bundler_gets_one_output_per_target(self):
        targets = ["host-x86_64-unknown-linux-gnu-", "hipv4-amdgcn-amd-amdhsa--gfx1151"]
        payloads = [b"", b"device code"]
        # An unsupported compression method requires the external fallback,
        # independently of the Python version or installed zstd packages.
        blob = b"CCOB" + struct.pack("<HH", 3, 65535) + bytes(24)

        def bundler(command, **kwargs):
            if "--list" in command:
                return subprocess.CompletedProcess(command, 0, "\n".join(targets))
            outputs = [arg.removeprefix("--output=") for arg in command
                       if arg.startswith("--output=")]
            self.assertEqual(len(outputs), len(targets))
            self.assertIn("--targets=" + ",".join(targets), command)
            for output, payload in zip(outputs, payloads):
                Path(output).write_bytes(payload)
            return subprocess.CompletedProcess(command, 0)

        with mock.patch.object(RESOURCES.subprocess, "run", side_effect=bundler):
            result = RESOURCES.inflate_ccob(blob, "/hip/clang-offload-bundler")
        self.assertEqual(list(RESOURCES.bundle_entries(result)), list(zip(targets, payloads)))

    def test_hip_bundler_precedes_unrelated_path_compiler(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(RESOURCES.shutil, "which", side_effect=lambda name: "/path/" + name), \
             mock.patch.object(RESOURCES.os.path, "isfile", return_value=True), \
             mock.patch.object(RESOURCES.subprocess, "run", return_value=
                               subprocess.CompletedProcess([], 0, "/hip/bin\n")):
            self.assertEqual(RESOURCES.find_bundler(None), "/hip/bin/clang-offload-bundler")

    def test_missing_explicit_bundler_fails(self):
        with mock.patch.object(RESOURCES.os.path, "isfile", return_value=False):
            with self.assertRaisesRegex(SystemExit, "offload bundler not found"):
                RESOURCES.find_bundler("/missing/bundler")

    def test_known_kernels_pass_only_when_no_worse(self):
        budget = {"default_max_scratch": 0, "allow": {"limited": {"max_scratch": 64}}}
        kernel = lambda scratch: {"scratch": scratch, "vgpr": 64, "vgpr_spill": 0}
        candidate = {"clean": kernel(0), "limited": kernel(64), "same": kernel(32),
                     "grew": kernel(48), "new": kernel(16)}
        base = {"clean": kernel(0), "limited": kernel(64), "same": kernel(32), "grew": kernel(32)}
        failing, tolerated = RESOURCES.over_budget(candidate, budget, base)
        self.assertEqual(sorted(n for n, _, _ in failing), ["grew", "new"])
        self.assertEqual([n for n, _, _ in tolerated], ["same"])
        failing, tolerated = RESOURCES.over_budget(candidate, budget)
        self.assertEqual(sorted(n for n, _, _ in failing), ["grew", "new", "same"])
        self.assertEqual(tolerated, [])


if __name__ == "__main__":
    unittest.main()
