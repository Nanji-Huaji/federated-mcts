"""Contract tests for TimeTracker parallel-client semantics.

Expected contract:
- accumulate_step takes max generation and evaluation across clients
  (not sum), modelling wall-clock wait for the slowest client.
- Client stats retain each client's own cumulative durations across steps.
- record_generation / record_evaluation directly increment per-client stats.
- latency_dict returns the accumulated max-gen and max-eval totals.
"""

import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.federation.time_tracker import TimeTracker


class TestTimeTrackerMaxAggregation(unittest.TestCase):

    def setUp(self):
        self.tt = TimeTracker(client_names=["local", "remote"])

    def test_accumulate_step_uses_max_not_sum(self):
        step_times = {
            "local": {"generation": 0.1, "evaluation": 0.2},
            "remote": {"generation": 0.3, "evaluation": 0.1},
        }
        self.tt.accumulate_step(step_times)
        self.assertEqual(
            self.tt.generation, 0.3,
            "DEFECT: global generation should be max(0.1, 0.3)=0.3, not sum=0.4.",
        )
        self.assertEqual(
            self.tt.evaluation, 0.2,
            "DEFECT: global evaluation should be max(0.2, 0.1)=0.2, not sum=0.3.",
        )

    def test_client_stats_retain_own_durations(self):
        step_times = {
            "local": {"generation": 0.1, "evaluation": 0.2},
            "remote": {"generation": 0.3, "evaluation": 0.1},
        }
        self.tt.accumulate_step(step_times)
        self.assertEqual(self.tt.client_stats["local"]["generation"], 0.1)
        self.assertEqual(self.tt.client_stats["local"]["evaluation"], 0.2)
        self.assertEqual(self.tt.client_stats["remote"]["generation"], 0.3)
        self.assertEqual(self.tt.client_stats["remote"]["evaluation"], 0.1)

    def test_client_stats_accumulate_across_steps(self):
        self.tt.accumulate_step({
            "local": {"generation": 0.1, "evaluation": 0.1},
            "remote": {"generation": 0.2, "evaluation": 0.2},
        })
        self.tt.accumulate_step({
            "local": {"generation": 0.05, "evaluation": 0.05},
            "remote": {"generation": 0.15, "evaluation": 0.15},
        })
        self.assertAlmostEqual(self.tt.client_stats["local"]["generation"], 0.15)
        self.assertAlmostEqual(self.tt.client_stats["remote"]["generation"], 0.35)
        self.assertAlmostEqual(self.tt.generation, 0.35)  # max(0.1+0.05, 0.2+0.15)=0.35

    def test_record_generation_increments_client_stats(self):
        self.tt.record_generation("local", 0.5)
        self.assertEqual(self.tt.client_stats["local"]["generation"], 0.5)

    def test_record_evaluation_increments_client_stats(self):
        self.tt.record_evaluation("remote", 0.8)
        self.assertEqual(self.tt.client_stats["remote"]["evaluation"], 0.8)

    def test_latency_dict_returns_global_max_totals(self):
        self.tt.accumulate_step({
            "local": {"generation": 0.5, "evaluation": 0.2},
        })
        self.tt.accumulate_step({
            "remote": {"generation": 0.3, "evaluation": 0.4},
        })
        d = self.tt.latency_dict
        self.assertEqual(d["generation"], 0.8)
        self.assertAlmostEqual(d["evaluation"], 0.6)

    def test_reset_clears_everything(self):
        self.tt.accumulate_step({
            "local": {"generation": 1.0, "evaluation": 1.0},
        })
        self.tt.reset()
        self.assertEqual(self.tt.generation, 0.0)
        self.assertEqual(self.tt.evaluation, 0.0)
        self.assertEqual(self.tt.client_stats, {})

    def test_ensure_client_adds_new_client(self):
        self.tt.ensure_client("new_client")
        self.assertIn("new_client", self.tt.client_stats)
        self.assertEqual(self.tt.client_stats["new_client"]["generation"], 0.0)


if __name__ == "__main__":
    unittest.main()
