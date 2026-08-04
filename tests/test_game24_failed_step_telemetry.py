"""Tests for record_failed_step: dead-end expansion telemetry persistence,
rejection-reason mapping, and metric aggregation.
"""
from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "src")

from tests._game24_stub import _Stub


# ═══════════════════════════════════════════════════════════════════
#  record_failed_step unit tests
# ═══════════════════════════════════════════════════════════════════

class TestRecordFailedStep(unittest.TestCase):

    def setUp(self) -> None:
        self.task = _Stub()
        self.x = "1 1 1 8"

    def _make_recorder(self):
        from federated_mcts.evaluation.game24_task_recorder import (
            TaskSessionRecorder,
        )
        return TaskSessionRecorder(task_id="0", task_input=self.x,
                                   decision_policy="local_validity")

    def test_record_failed_step_creates_step_record(self):
        from federated_mcts.evaluation.game24_instrumentation import (
            GenerationRecorder, CandidateProposalRecord, ProposalStepRecord,
        )
        gen_rec = GenerationRecorder()
        gen_rec.record_step(ProposalStepRecord(
            depth=0, parent_index=0, parent_state="",
            raw_completions=["bad\nempty\n"],
            retry_count=1,
            proposals=[
                CandidateProposalRecord(
                    depth=0, parent_index=0, retry_ordinal=0,
                    completion_index=0, raw_line="badformat",
                    is_correct=False, updated_text="badformat",
                    reason="malformed_format"),
                CandidateProposalRecord(
                    depth=0, parent_index=0, retry_ordinal=0,
                    completion_index=0, raw_line="",
                    is_correct=False, updated_text="",
                    reason="empty_content"),
            ],
        ))
        rec = self._make_recorder()
        rec.record_failed_step(depth=0, parent_count=1, gen_rec=gen_rec,
                               elapsed_ms=5.0, task=self.task, x=self.x)

        self.assertEqual(len(rec._step_records), 1)
        sr = rec._step_records[0]
        self.assertEqual(sr["depth"], 0)
        self.assertEqual(sr["parent_count"], 1)
        self.assertEqual(sr["is_stopping"], True)
        self.assertEqual(len(sr["candidates"]), 2)
        self.assertEqual(len(sr["gen_instrumentation"]), 1)
        self.assertEqual(sr["selected_indices"], [])
        self.assertEqual(sr["selected"], [])

    def test_record_failed_step_candidate_rejection_reasons(self):
        from federated_mcts.evaluation.game24_instrumentation import (
            GenerationRecorder, CandidateProposalRecord, ProposalStepRecord,
        )
        gen_rec = GenerationRecorder()
        gen_rec.record_step(ProposalStepRecord(
            depth=0, parent_index=0, parent_state="",
            raw_completions=["bad\nempty\n"],
            retry_count=1,
            proposals=[
                CandidateProposalRecord(
                    depth=0, parent_index=0, retry_ordinal=0,
                    completion_index=0, raw_line="badformat",
                    is_correct=False, updated_text="badformat",
                    reason="malformed_format"),
                CandidateProposalRecord(
                    depth=0, parent_index=0, retry_ordinal=0,
                    completion_index=0, raw_line="",
                    is_correct=False, updated_text="",
                    reason="empty_content"),
                CandidateProposalRecord(
                    depth=0, parent_index=0, retry_ordinal=0,
                    completion_index=0, raw_line="",
                    is_correct=False, updated_text="",
                    reason="duplicate"),
            ],
        ))
        rec = self._make_recorder()
        rec.record_failed_step(depth=0, parent_count=1, gen_rec=gen_rec,
                               elapsed_ms=5.0, task=self.task, x=self.x)

        events = rec._step_records[0]["candidates"]
        self.assertEqual(events[0]["rejection_reason"], "malformed_format")
        self.assertEqual(events[1]["rejection_reason"], "empty_content")
        self.assertIsNone(events[2]["rejection_reason"])
        for ev in events:
            self.assertEqual(ev["stage"], "proposed")
            self.assertEqual(ev["disposition"], "none")
            self.assertFalse(ev["is_selected"])

    def test_to_task_metrics_with_failed_step_preserves_denominators(self):
        from federated_mcts.evaluation.game24_instrumentation import (
            GenerationRecorder, CandidateProposalRecord, ProposalStepRecord,
        )
        rec = self._make_recorder()
        gr0 = rec.new_generation_recorder()
        vr0 = rec.new_vote_recorder()
        rec.record_decision(depth=0, parent_count=1,
            candidates=["8 - 1 = 7 (left: 1 1 7)\n", "1 + 1 = 2 (left: 2 8)\n"],
            candidate_parent_lineage_ids=["", ""],
            values=[5, 1], selected_indices=[0],
            selected=["8 - 1 = 7 (left: 1 1 7)\n"], is_stopping=False,
            elapsed_ms=10, task=self.task, x=self.x,
            gen_rec=gr0, vote_rec=vr0)

        gen_rec = GenerationRecorder()
        gen_rec.record_step(ProposalStepRecord(
            depth=1, parent_index=0, parent_state="",
            raw_completions=["bad\nempty\n"],
            retry_count=1,
            proposals=[
                CandidateProposalRecord(
                    depth=1, parent_index=0, retry_ordinal=0,
                    completion_index=0, raw_line="badformat",
                    is_correct=False, updated_text="badformat",
                    reason="malformed_format"),
                CandidateProposalRecord(
                    depth=1, parent_index=0, retry_ordinal=0,
                    completion_index=0, raw_line="",
                    is_correct=False, updated_text="",
                    reason="empty_content"),
            ],
        ))
        rec.record_failed_step(depth=1, parent_count=1, gen_rec=gen_rec,
                               elapsed_ms=3.0, task=self.task, x=self.x)

        metrics = rec.to_task_metrics()
        self.assertEqual(metrics["total_steps"], 2)
        self.assertEqual(metrics["total_proposed"], 4)
        self.assertEqual(metrics["total_candidates"], 4)
        rc = metrics["rejection_counts"]
        self.assertGreaterEqual(rc.get("malformed_format", 0), 1)
        self.assertGreaterEqual(rc.get("empty_content", 0), 1)

    def test_experiment_aggregation_with_failed_step(self):
        from federated_mcts.evaluation.game24_instrumentation import (
            GenerationRecorder, CandidateProposalRecord, ProposalStepRecord,
        )
        from federated_mcts.evaluation.game24_metrics import summarize_experiment_metrics

        rec = self._make_recorder()
        gr0 = rec.new_generation_recorder(); vr0 = rec.new_vote_recorder()
        rec.record_decision(depth=0, parent_count=1,
            candidates=["1 + 2 = 3 (left: 3 3 8)\n"],
            candidate_parent_lineage_ids=[""],
            values=[10], selected_indices=[0],
            selected=["1 + 2 = 3 (left: 3 3 8)\n"], is_stopping=False,
            elapsed_ms=10, task=self.task, x=self.x,
            gen_rec=gr0, vote_rec=vr0)

        gen_rec = GenerationRecorder()
        gen_rec.record_step(ProposalStepRecord(
            depth=1, parent_index=0, parent_state="",
            raw_completions=["bad\n"],
            retry_count=1,
            proposals=[
                CandidateProposalRecord(
                    depth=1, parent_index=0, retry_ordinal=0,
                    completion_index=0, raw_line="badformat",
                    is_correct=False, updated_text="badformat",
                    reason="malformed_format"),
            ],
        ))
        rec.record_failed_step(depth=1, parent_count=1, gen_rec=gen_rec,
                               elapsed_ms=3.0, task=self.task, x=self.x)

        task_metrics = rec.to_task_metrics()
        exp_metrics = summarize_experiment_metrics([task_metrics])

        self.assertEqual(exp_metrics["task_count"], 1)
        self.assertEqual(exp_metrics["total_proposed"], 2)
        self.assertEqual(exp_metrics["total_candidates"], 2)
        rc = exp_metrics["rejection_counts"]
        self.assertGreaterEqual(rc.get("malformed_format", 0), 1)

    def test_record_failed_step_preserves_gen_instrumentation(self):
        from federated_mcts.evaluation.game24_instrumentation import (
            GenerationRecorder, CandidateProposalRecord, ProposalStepRecord,
        )
        gen_rec = GenerationRecorder()
        gen_rec.record_step(ProposalStepRecord(
            depth=0, parent_index=0, parent_state="1 2 3 4",
            raw_completions=["raw completion text"],
            retry_count=2,
            proposals=[
                CandidateProposalRecord(
                    depth=0, parent_index=0, retry_ordinal=0,
                    completion_index=0, raw_line="badformat",
                    is_correct=False, updated_text="badformat",
                    reason="malformed_format"),
            ],
        ))
        rec = self._make_recorder()
        rec.record_failed_step(depth=0, parent_count=1, gen_rec=gen_rec,
                               elapsed_ms=5.0, task=self.task, x=self.x)

        self.assertEqual(len(rec._step_records[0]["gen_instrumentation"]), 1)
        gi = rec._step_records[0]["gen_instrumentation"][0]
        self.assertEqual(gi["depth"], 0)
        self.assertEqual(gi["parent_state"], "1 2 3 4")
        self.assertEqual(gi["retry_count"], 2)
        self.assertEqual(gi["raw_completions"], ["raw completion text"])


if __name__ == "__main__":
    unittest.main()
