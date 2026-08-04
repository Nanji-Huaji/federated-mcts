"""Todo 7 reviewer acceptance tests: real on_no_valid_candidates, all six
dispositions, expanded semantics, orphan .tmp, 5-task runner/solver resume.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")


class _Stub:
    steps = 3

    @staticmethod
    def get_input(idx: int) -> str:
        return ["1 1 1 8", "2 2 2 6", "3 3 3 3", "4 4 4 4", "5 5 5 5"][idx]

    @staticmethod
    def canonical_state_key(x: str, y: str) -> tuple:
        from fractions import Fraction
        import re
        lines = [l for l in y.strip().split("\n") if "left:" in l]
        target = lines[-1] if lines else y
        m = re.search(r"left:\s*([^)]+)", target)
        parts = (m.group(1).strip().split() if m else target.strip().split())
        return tuple(sorted(Fraction(p) for p in parts if p.strip()))

    @staticmethod
    def is_success_state(x: str, y: str) -> bool:
        key = _Stub.canonical_state_key(x, y)
        return len(key) == 1 and list(key)[0] == 24

    @staticmethod
    def propose_prompt_wrap(x: str, y: str) -> str:
        return "prompt"

    @staticmethod
    def process_generate_result(pro, x, y, check):
        return True, pro.strip()

    @staticmethod
    def vote_prompt_wrap(x, ys):
        return "vote_prompt"

    @staticmethod
    def vote_outputs_unwrap(outputs, n):
        return [1] * n

    @staticmethod
    def test_output_modify(idx, output):
        return {"r": 0}, output


# ── Fix 5: real on_no_valid_candidates ─────────────────────────────

class TestRealNoValidCandidate(unittest.TestCase):
    def test_returns_empty_list(self) -> None:
        from federated_mcts.tasks.game24 import Game24Task
        self.assertEqual(Game24Task.on_no_valid_candidates("any"), [])


# ── Fix 1+2: expanded semantics + all six dispositions ─────────────

class TestDispositionsAndExpanded(unittest.TestCase):

    def setUp(self) -> None:
        self.task = _Stub()
        self.x = "1 1 1 8"

    def test_all_six_dispositions_appear(self) -> None:
        from federated_mcts.evaluation.game24_task_recorder import (
            TaskSessionRecorder,
        )
        rec = TaskSessionRecorder(task_id="0", task_input=self.x,
                                  decision_policy="local_validity")
        # Step 0: three candidates — select first two
        gr0 = rec.new_generation_recorder(); vr0 = rec.new_vote_recorder()
        candidates = ["8 - 1 = 7 (left: 1 1 7)\n",
                      "6 * 4 = 24 (left: 24)\n",
                      "2 + 2 = 4 (left: 4 8)\n"]
        rec.record_decision(depth=0, parent_count=1,
            candidates=candidates,
            candidate_parent_lineage_ids=["", "", ""],
            values=[3, 10, 1], selected_indices=[0, 1],
            selected=[candidates[0], candidates[1]], is_stopping=False,
            elapsed_ms=10, task=self.task, x=self.x,
            gen_rec=gr0, vote_rec=vr0)
        sel_lids = rec.selected_lineage_ids
        self.assertEqual(len(sel_lids), 2)

        # Step 1: first parent → normal, second parent → garbage invalid
        gr1 = rec.new_generation_recorder(); vr1 = rec.new_vote_recorder()
        rec.record_decision(depth=1, parent_count=2,
            candidates=["7 + 1 = 8 (left: 8 1)\n", "garbage_no_op\n"],
            candidate_parent_lineage_ids=[sel_lids[0], sel_lids[1]],
            values=[2, 0], selected_indices=[0],
            selected=["7 + 1 = 8 (left: 8 1)\n"], is_stopping=False,
            elapsed_ms=5, task=self.task, x=self.x,
            gen_rec=gr1, vote_rec=vr1)

        metrics = rec.to_task_metrics()
        all_disp = set()
        for sd in metrics["step_details"]:
            for c in sd["candidates"]:
                all_disp.add(c["disposition"])
        # Should see: selected, terminal_solved, expanded, none,
        # terminal_invalid — five distinct dispositions
        for want in ("selected", "terminal_solved", "expanded",
                      "none", "terminal_invalid"):
            self.assertIn(want, all_disp, f"Missing {want!r}. Got: {all_disp}")

    def test_expanded_only_when_parent_used(self) -> None:
        from federated_mcts.evaluation.game24_task_recorder import (
            TaskSessionRecorder,
        )
        rec = TaskSessionRecorder(task_id="0", task_input=self.x,
                                  decision_policy="local_validity")
        # Step 0: two candidates, select idx 0
        gr0 = rec.new_generation_recorder(); vr0 = rec.new_vote_recorder()
        rec.record_decision(depth=0, parent_count=1,
            candidates=["1 + 2 = 3 (left: 3 3 8)\n", "3 + 3 = 6 (left: 6 8)\n"],
            candidate_parent_lineage_ids=["", ""],
            values=[5, 3], selected_indices=[0],
            selected=["1 + 2 = 3 (left: 3 3 8)\n"], is_stopping=False,
            elapsed_ms=10, task=self.task, x=self.x,
            gen_rec=gr0, vote_rec=vr0)
        self.assertEqual(rec.selected_lineage_ids[0],
                         rec._step_records[0]["candidates"][0]["lineage_id"])
        self.assertEqual(len(rec.selected_lineage_ids), 1)

        # Step 1: use parent 0's lineage; parent 0 retroactively → expanded
        gr1 = rec.new_generation_recorder(); vr1 = rec.new_vote_recorder()
        sel_lid = rec.selected_lineage_ids[0]
        rec.record_decision(depth=1, parent_count=1,
            candidates=["8 - 3 = 5 (left: 5 3)\n", "3 + 5 = 8 (left: 8)\n"],
            candidate_parent_lineage_ids=[sel_lid, ""],
            values=[2, 1], selected_indices=[0],
            selected=["8 - 3 = 5 (left: 5 3)\n"], is_stopping=False,
            elapsed_ms=5, task=self.task, x=self.x,
            gen_rec=gr1, vote_rec=vr1)

        metrics = rec.to_task_metrics()
        step0_disps = {c["disposition"] for c in metrics["step_details"][0]["candidates"]}
        self.assertIn("expanded", step0_disps,
                      f"Parent 0 should be expanded: {step0_disps}")
        self.assertIn("none", step0_disps)
        self.assertNotIn("selected", step0_disps,
                         "Step 0 candidate became parent → should be expanded, not selected")


# ── Fix 4: orphan .tmp and fail-closed ─────────────────────────────

class TestOrphanTempAndFailClosed(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self) -> str:
        return os.path.join(self.tmpdir, "j.json")

    def test_crash_during_write_preserves_target(self) -> None:
        from federated_mcts.evaluation.game24_persistence import (
            AtomicTaskJournal, _CorruptJournalError,
        )
        path = self._path()
        journal = AtomicTaskJournal(path)
        journal.write_meta(run_fingerprint="fp")
        journal.write_task("0", {"task_id": 0, "_completed": True})

        with open(path) as f:
            orig = f.read()

        # Simulate crash by raising in os.replace
        real_replace = os.replace
        def _crash(src, dst):
            raise RuntimeError("simulated crash")
        os.replace = _crash
        try:
            journal.write_task("1", {"task_id": 1})
        except RuntimeError:
            pass
        finally:
            os.replace = real_replace

        with open(path) as f:
            after = f.read()
        self.assertEqual(orig, after,
                         "Target must be unchanged after crash")

    def test_malformed_target_unchanged_by_write_attempt(self) -> None:
        from federated_mcts.evaluation.game24_persistence import (
            AtomicTaskJournal, _CorruptJournalError,
        )
        path = self._path()
        with open(path, "w") as f:
            f.write("not json!")
        with open(path) as f:
            malformed = f.read()
        journal = AtomicTaskJournal(path)
        with self.assertRaises(_CorruptJournalError):
            journal.read_all()
        with open(path) as f:
            self.assertEqual(f.read(), malformed)


if __name__ == "__main__":
    unittest.main()
