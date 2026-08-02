"""Synthetic unit tests for piastq_execution.seeding."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from piastq_execution.seeding import seed_for_sampling


class TestSeedForSampling(unittest.TestCase):
    def test_deterministic_across_calls(self):
        a = seed_for_sampling("gsdtsr", 3)
        b = seed_for_sampling("gsdtsr", 3)
        self.assertEqual(a, b)

    def test_differs_by_sampling_id(self):
        seeds = {seed_for_sampling("gsdtsr", i) for i in range(1, 11)}
        self.assertEqual(len(seeds), 10)

    def test_differs_by_dataset(self):
        self.assertNotEqual(seed_for_sampling("gsdtsr", 1), seed_for_sampling("iofrol", 1))

    def test_returns_nonnegative_32bit_int(self):
        seed = seed_for_sampling("elevator_two", 7)
        self.assertIsInstance(seed, int)
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2 ** 32)

    def test_matches_direct_sha256_derivation(self):
        # Regression guard: SHA-256-based, not Python's hash() (which is
        # randomized per-process for strings unless PYTHONHASHSEED is fixed).
        import hashlib
        expected = int(hashlib.sha256(b"gsdtsr_sampling_1").hexdigest()[:8], 16)
        self.assertEqual(seed_for_sampling("gsdtsr", 1), expected)


if __name__ == "__main__":
    unittest.main()
