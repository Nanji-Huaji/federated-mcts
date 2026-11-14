"""Test validation pilot fixture: exact IDs, membership, intersection."""
from __future__ import annotations

import unittest

from federated_mcts.evaluation.game24_pilot import (
    VALIDATION_PILOT_IDS,
    is_pilot_id,
    pilot_ids_intersect_test,
    validate_pilot_membership,
    pilot_ids,
)

_EXPECTED = [3, 9, 16, 19, 24, 25, 28, 36, 40, 42,
             45, 71, 77, 81, 102, 104, 105, 116, 117, 119]


class TestPilotFixture(unittest.TestCase):
    """Given the canonical validation pilot set."""

    def test_exact_ids(self) -> None:
        """When checking the pilot list, Then it matches the approved plan exactly."""
        self.assertEqual(VALIDATION_PILOT_IDS, _EXPECTED)
        self.assertEqual(len(VALIDATION_PILOT_IDS), 20)

    def test_is_pilot_id_membership(self) -> None:
        """When checking membership, Then pilot IDs return True, non-pilot False."""
        for tid in _EXPECTED:
            self.assertTrue(is_pilot_id(tid), f"Expected {tid} to be pilot ID")
        self.assertFalse(is_pilot_id(999))
        self.assertFalse(is_pilot_id(0))
        self.assertFalse(is_pilot_id(200))

    def test_no_intersection_with_frozen_test(self) -> None:
        """When checking against frozen test IDs, Then pilot and test are disjoint."""
        test_set = frozenset({0, 1, 2, 4, 5, 6, 7, 8, 200})
        self.assertFalse(pilot_ids_intersect_test(test_set))

        # Pilot IDs should intersect if one appears in test set
        test_with_pilot = frozenset({3, 5, 6})  # 3 is a pilot ID
        self.assertTrue(pilot_ids_intersect_test(test_with_pilot))

    def test_validate_pilot_membership_valid(self) -> None:
        """When validating exact pilot IDs, Then it passes."""
        ok, err = validate_pilot_membership(_EXPECTED)
        self.assertTrue(ok, msg=err)
        self.assertEqual(err, "")

    def test_validate_pilot_membership_invalid(self) -> None:
        """When validating non-pilot IDs, Then it fails with reason."""
        ok, err = validate_pilot_membership([1, 2, 3])
        self.assertFalse(ok)
        self.assertIn("not in validation pilot set", err)
        self.assertIn("[1, 2]", err)

    def test_pilot_ids_returns_copy(self) -> None:
        """When requesting pilot IDs, Then returns a copy of the canonical list."""
        ids = pilot_ids()
        self.assertEqual(ids, _EXPECTED)
        ids.append(999)
        self.assertEqual(len(pilot_ids()), 20)  # Original unchanged
