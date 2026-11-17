"""Host-capability gating: skips carry the probe reason, strict mode fails instead, probes never crash discovery."""
import os
import unittest
from unittest import mock

from support import host_capabilities as caps


def _capability(available, detail="fixture detail"):
    return lambda: caps.Capability("fixture_capability", available, detail)


def _run(case_class):
    result = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(case_class).run(result)
    return result


class RequireTests(unittest.TestCase):
    def test_available_capability_leaves_the_test_unchanged(self):
        def test_body(self):
            pass
        self.assertIs(caps.require(_capability(True), "fixture reason")(test_body), test_body)

    def test_missing_capability_skips_with_reason_and_probe_detail(self):
        with mock.patch.dict(os.environ, {caps.REQUIRE_ENV: ""}):
            class Gated(unittest.TestCase):
                @caps.require(_capability(False, "no fixture kernel"), "fixture reason")
                def test_gated(self):
                    raise AssertionError("gated body must not run")

                def test_ungated(self):
                    pass
        result = _run(Gated)
        self.assertEqual(result.testsRun, 2)
        self.assertEqual((result.failures, result.errors), ([], []))
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0][1], "requires host capability fixture_capability: fixture reason "
                                               "[probe: no fixture kernel]")

    def test_strict_mode_fails_a_gated_method_instead_of_skipping(self):
        with mock.patch.dict(os.environ, {caps.REQUIRE_ENV: "1"}):
            class Gated(unittest.TestCase):
                @caps.require(_capability(False), "fixture reason")
                def test_gated(self):
                    pass
        result = _run(Gated)
        self.assertEqual((len(result.failures), result.errors, result.skipped), (1, [], []))
        self.assertIn("MissingHostCapability", result.failures[0][1])
        self.assertIn(caps.REQUIRE_ENV + "=1 forbids skipping", result.failures[0][1])

    def test_strict_mode_fails_a_gated_class_in_set_up_class(self):
        with mock.patch.dict(os.environ, {caps.REQUIRE_ENV: "1"}):
            @caps.require(_capability(False), "fixture reason")
            class Gated(unittest.TestCase):
                def test_one(self):
                    pass

                def test_two(self):
                    pass
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Gated)
        result = unittest.TestResult()
        unittest.TestSuite([suite]).run(result)
        self.assertEqual(result.testsRun, 0)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("setUpClass", str(result.errors[0][0]))
        self.assertIn("MissingHostCapability", result.errors[0][1])

    def test_strict_flag_values(self):
        for value, expected in (("1", True), ("true", True), ("on", True), ("0", False), ("", False)):
            with self.subTest(value=value), mock.patch.dict(os.environ, {caps.REQUIRE_ENV: value}):
                self.assertIs(caps.required(), expected)


class ProbeTests(unittest.TestCase):
    def test_crashing_probe_reports_missing_instead_of_raising(self):
        @caps._probe
        def broken_probe():
            raise OSError("fixture failure")
        result = broken_probe()
        self.assertFalse(result)
        self.assertEqual(result.name, "broken_probe")
        self.assertIn("fixture failure", result.detail)
        self.assertIs(broken_probe(), result)  # cached

    def test_every_probe_returns_a_described_result_and_composites_are_consistent(self):
        for probe in caps.PROBES:
            with self.subTest(probe=probe.__name__):
                result = probe()
                self.assertIsInstance(result, caps.Capability)
                self.assertEqual(result.name, probe.__name__)
                self.assertTrue(result.detail)
                self.assertTrue(probe.__doc__)
        for composite in (caps.program_isolation, caps.native_replay_sandbox):
            if composite():
                self.assertTrue(caps.landlock() and caps.seccomp(), composite.__name__)
        if caps.program_isolation():
            self.assertTrue(caps.linux_runtime_tools())
            self.assertGreaterEqual(caps.landlock_abi(), 4)


if __name__ == "__main__":
    unittest.main()
