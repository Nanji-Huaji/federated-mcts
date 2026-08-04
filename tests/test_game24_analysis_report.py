"""Tests for JSON and Markdown report generation.

Covers: valid JSON output, required Markdown sections, JSON/MD agreement,
and Pareto rows presence.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, "src")

from tests.test_game24_analysis_shared import (
    make_manifest_json,
    make_performance_json,
    make_tasks_json,
)


class TestReportGeneration(unittest.TestCase):
    """Given a PairedAnalysis, When format_report() is called,
       Then JSON and Markdown output is valid and agrees."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _make_pair(self, baseline_solved: set[int], candidate_solved: set[int]):
        from federated_mcts.evaluation.game24_analysis import (
            load_result, run_paired_analysis,
        )
        task_ids = [3, 9, 16, 19, 24, 25, 28, 36, 40, 42]
        for prefix, solved in [("base", baseline_solved), ("cand", candidate_solved)]:
            m_path = os.path.join(self.tmpdir, f"{prefix}_manifest.json")
            p_path = os.path.join(self.tmpdir, f"{prefix}_performance.json")
            t_path = os.path.join(self.tmpdir, f"{prefix}.json")
            n = len(task_ids)
            s = len(solved)
            acc = s / n
            with open(m_path, "w") as f:
                f.write(make_manifest_json(
                    task_ids=task_ids, decision_policy="local_validity",
                ))
            with open(p_path, "w") as f:
                f.write(make_performance_json(
                    acc=acc, solved_count=s, started_tasks=n,
                    total_tokens=30000, cost=0.15,
                    decision_fp="d" * 64, total_calls=500,
                ))
            with open(t_path, "w") as f:
                f.write(make_tasks_json(task_ids=task_ids, solved_ids=solved))
        baseline = load_result(
            os.path.join(self.tmpdir, "base_manifest.json"),
            os.path.join(self.tmpdir, "base_performance.json"),
            os.path.join(self.tmpdir, "base.json"),
        )
        candidate = load_result(
            os.path.join(self.tmpdir, "cand_manifest.json"),
            os.path.join(self.tmpdir, "cand_performance.json"),
            os.path.join(self.tmpdir, "cand.json"),
        )
        return run_paired_analysis(baseline, candidate, seed=42)

    def test_report_json_is_valid(self):
        from federated_mcts.evaluation.game24_report import format_report
        a = self._make_pair(
            baseline_solved={3, 9, 16, 19},
            candidate_solved={3, 9, 16, 19, 24, 25, 28},
        )
        j, _ = format_report(a)
        p = json.loads(j)
        for key in ("baseline_accuracy", "candidate_accuracy", "delta",
                     "ci_95_lower", "ci_95_upper"):
            self.assertIn(key, p)

    def test_report_markdown_required_sections(self):
        from federated_mcts.evaluation.game24_report import format_report
        a = self._make_pair(
            baseline_solved={3, 9, 16},
            candidate_solved={3, 9, 16, 19, 24},
        )
        _, md = format_report(a)
        for term in ("solved", "total", "delta", "ci", "calls", "tokens",
                      "cost", "policy", "fingerprint", "local_validity"):
            self.assertIn(term, md.lower(), f"Missing '{term}' in Markdown")

    def test_json_and_md_agree(self):
        from federated_mcts.evaluation.game24_report import format_report
        a = self._make_pair(
            baseline_solved={3, 9, 16, 19},
            candidate_solved={3, 9, 16, 19, 24, 25, 28},
        )
        j, _ = format_report(a)
        p = json.loads(j)
        self.assertAlmostEqual(p["baseline_accuracy"], 0.4)
        self.assertAlmostEqual(p["candidate_accuracy"], 0.7)
        self.assertAlmostEqual(p["delta"], 0.3)

    def test_pareto_rows_present(self):
        from federated_mcts.evaluation.game24_report import format_report
        a = self._make_pair(
            baseline_solved={3, 9}, candidate_solved={3, 9, 16},
        )
        j, _ = format_report(a)
        p = json.loads(j)
        self.assertIn("baseline", p["pareto_rows"])
        self.assertIn("candidate", p["pareto_rows"])


if __name__ == "__main__":
    unittest.main()
