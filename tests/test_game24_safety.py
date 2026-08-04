"""Test safety rules: task resolution, pilot split identity, auth enforcement."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from federated_mcts.evaluation.game24_safety import (
    validate_task_resolution,
    validate_pilot_split_identity,
)
from federated_mcts.evaluation.game24_pilot import VALIDATION_PILOT_IDS


class TestTaskResolutionSafety(unittest.TestCase):
    """Given CLI argument combinations, When validated, Then correct safety rules apply."""

    def _default_kwargs(self, **overrides):
        kw = {
            "pilot_ids": None,
            "task_split": None,
            "task_start_index": 900,
            "task_end_index": 1000,
            "selected_test_auth": None,
            "decision_policy": "local_validity",
        }
        kw.update(overrides)
        return kw

    def test_pilot_ids_alone_passes(self) -> None:
        """When using canonical pilot IDs without task_split, Then it passes."""
        ok, err = validate_task_resolution(
            **self._default_kwargs(pilot_ids=VALIDATION_PILOT_IDS))
        self.assertTrue(ok, msg=err)

    def test_pilot_ids_with_task_split_rejected(self) -> None:
        """When combining pilot_ids with task_split, Then it's rejected."""
        ok, err = validate_task_resolution(
            **self._default_kwargs(
                pilot_ids=VALIDATION_PILOT_IDS, task_split="val"))
        self.assertFalse(ok)
        self.assertIn("cannot be combined", err)

    def test_non_canonical_pilot_ids_rejected(self) -> None:
        """When pilot IDs don't match the canonical set, Then it's rejected."""
        ok, err = validate_task_resolution(
            **self._default_kwargs(pilot_ids=[1, 2, 3]))
        self.assertFalse(ok)
        self.assertIn("must match the canonical", err)

    def test_test_split_requires_auth(self) -> None:
        """When task_split=test without selected_test_auth, Then it's rejected."""
        ok, err = validate_task_resolution(
            **self._default_kwargs(task_split="test"))
        self.assertFalse(ok)
        self.assertIn("requires --selected_test_auth", err)

    def test_selected_test_auth_requires_test_split(self) -> None:
        """When selected_test_auth is set but split is not test, Then rejected."""
        ok, err = validate_task_resolution(
            **self._default_kwargs(
                task_split="val", selected_test_auth="/tmp/auth.jsonl"))
        self.assertFalse(ok)
        self.assertIn("may only be used with --task_split=test", err)

    def test_test_split_with_auth_passes(self) -> None:
        """When task_split=test WITH selected_test_auth, Then it passes."""
        ok, err = validate_task_resolution(
            **self._default_kwargs(
                task_split="test", selected_test_auth="/tmp/auth.jsonl"))
        self.assertTrue(ok, msg=err)

    def test_invalid_decision_policy_rejected(self) -> None:
        """When decision_policy is unknown, Then it's rejected."""
        ok, err = validate_task_resolution(
            **self._default_kwargs(decision_policy="unknown_policy"))
        self.assertFalse(ok)
        self.assertIn("Unknown decision_policy", err)

    def test_default_vals_passes(self) -> None:
        """When using all defaults, Then it passes."""
        ok, err = validate_task_resolution(**self._default_kwargs())
        self.assertTrue(ok, msg=err)


class TestPilotSplitIdentity(unittest.TestCase):
    """Given a split file, When validating pilot IDs against splits, Then correct."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def _write_split(self, val_ids, test_ids) -> str:
        path = os.path.join(self.tmpdir, "splits.json")
        with open(path, "w") as f:
            json.dump({"val": val_ids, "test": test_ids}, f)
        return path

    def test_all_pilot_in_val_passes(self) -> None:
        """When all pilot IDs are in validation split, Then passes."""
        path = self._write_split(VALIDATION_PILOT_IDS, [0, 1, 2])
        ok, err = validate_pilot_split_identity(path, VALIDATION_PILOT_IDS)
        self.assertTrue(ok, msg=err)

    def test_pilot_not_in_val_rejected(self) -> None:
        """When a pilot ID is missing from validation, Then rejected."""
        path = self._write_split(VALIDATION_PILOT_IDS[1:], [0, 1, 2])
        ok, err = validate_pilot_split_identity(path, VALIDATION_PILOT_IDS)
        self.assertFalse(ok)
        self.assertIn("not in validation split", err)

    def test_pilot_in_test_rejected(self) -> None:
        """When a pilot ID intersects test split, Then rejected."""
        test_with_pilot = [VALIDATION_PILOT_IDS[0]]  # ID 3 is in test now
        path = self._write_split(VALIDATION_PILOT_IDS, test_with_pilot)
        ok, err = validate_pilot_split_identity(path, VALIDATION_PILOT_IDS)
        self.assertFalse(ok)
        self.assertIn("intersect frozen test split", err)

    def test_missing_file_rejected(self) -> None:
        """When split file does not exist, Then rejected."""
        ok, err = validate_pilot_split_identity("/nonexistent/file.json", VALIDATION_PILOT_IDS)
        self.assertFalse(ok)
        self.assertIn("Cannot read split file", err)


if __name__ == "__main__":
    unittest.main()
