"""Contract tests for UsageTracker thread safety.

PRODUCTION DEFECT: UsageTracker.record() uses compound read-increment-write
on a defaultdict with no synchronisation.  Under concurrent calls from
multiple threads, updates are silently lost.

Expected contract: concurrent record() calls produce exact cumulative
counts equal to the sum of individual calls.
"""

import sys
import threading
import unittest

sys.path.insert(0, "src")

from federated_mcts.models.usage import UsageTracker


_LockType = type(threading.Lock())


class TestUsageTrackerConcurrentExactTotals(unittest.TestCase):

    NUM_THREADS = 50
    CALLS_PER_THREAD = 200

    def setUp(self):
        self.tracker = UsageTracker()
        self.tracker.reset()

    def test_concurrent_records_produce_exact_totals(self):
        expected_calls = self.NUM_THREADS * self.CALLS_PER_THREAD
        expected_completion = expected_calls * 10
        expected_prompt = expected_calls * 20

        barrier = threading.Barrier(self.NUM_THREADS, timeout=30)

        def worker():
            barrier.wait()
            for _ in range(self.CALLS_PER_THREAD):
                self.tracker.record("gpt-4o", 10, 20)

        threads = [threading.Thread(target=worker) for _ in range(self.NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        summary = self.tracker.get_summary()
        stats = summary.get("gpt-4o", {})

        actual_calls = stats.get("total_calls", 0)
        actual_completion = stats.get("completion_tokens", 0)
        actual_prompt = stats.get("prompt_tokens", 0)

        self.assertEqual(
            actual_calls, expected_calls,
            "DEFECT: concurrent record() calls lost updates.  "
            f"Expected {expected_calls} total_calls, got {actual_calls}.  "
            "record() must be protected by a threading.Lock.",
        )
        self.assertEqual(actual_completion, expected_completion)
        self.assertEqual(actual_prompt, expected_prompt)

    def test_tracker_has_lock(self):
        has_lock = any(
            isinstance(getattr(self.tracker, a, None), _LockType)
            for a in dir(self.tracker)
            if not a.startswith("__")
        )
        self.assertTrue(
            has_lock,
            "DEFECT: UsageTracker has no threading.Lock.  "
            "record() compound updates are not thread-safe.",
        )


if __name__ == "__main__":
    unittest.main()
