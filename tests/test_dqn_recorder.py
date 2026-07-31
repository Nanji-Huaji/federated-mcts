"""JSONL transition recorder contract.

record() appends a transition as one JSON line and read_all() returns the
transitions in order with fields preserved.  numpy/torch values must be
serialized to plain JSON scalars/lists.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, "src")

from federated_mcts.core.dqn.recorder import JSONLTransitionRecorder


class TestJSONLTransitionRecorder(unittest.TestCase):
    def test_roundtrip_preserves_transition_fields(self):
        """Given two recorded transitions, when read back, then every field is
        preserved in order."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "transitions.jsonl")
            recorder = JSONLTransitionRecorder(path)
            recorder.record({
                "state": [0.1, 0.2, 0.3],
                "action": 3,
                "reward": 0.0,
                "next_state": [0.4, 0.5, 0.6],
                "done": True,
            })
            recorder.record({
                "state": [0.4, 0.5, 0.6],
                "action": 5,
                "reward": 1.0,
                "next_state": None,
                "done": True,
            })

            records = recorder.read_all()

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["state"], [0.1, 0.2, 0.3])
        self.assertEqual(records[0]["action"], 3)
        self.assertEqual(records[0]["reward"], 0.0)
        self.assertEqual(records[1]["action"], 5)
        self.assertEqual(records[1]["reward"], 1.0)
        self.assertIsNone(records[1]["next_state"])
        self.assertTrue(records[1]["done"])

    def test_reopening_appends_not_clobbers(self):
        """Given a recorder, a second recorder on the same path, when both
        record, then all transitions survive in append order."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "transitions.jsonl")
            JSONLTransitionRecorder(path).record(
                {"state": [0.0], "action": 0, "reward": 0.0, "next_state": [0.1], "done": False},
            )
            JSONLTransitionRecorder(path).record(
                {"state": [0.1], "action": 1, "reward": 1.0, "next_state": None, "done": True},
            )

            records = JSONLTransitionRecorder(path).read_all()

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["action"], 0)
        self.assertEqual(records[1]["action"], 1)

    def test_numpy_and_torch_values_serialize_to_plain_json(self):
        """Given numpy and torch values, when recorded and read back, then
        they roundtrip as plain Python floats/lists."""
        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "transitions.jsonl")
            recorder = JSONLTransitionRecorder(path)
            recorder.record({
                "state": np.zeros(3, dtype=np.float32),
                "action": int(np.int64(2)),
                "reward": np.float32(0.5),
                "next_state": torch.zeros(3),
                "done": True,
            })

            records = recorder.read_all()

        self.assertEqual(records[0]["action"], 2)
        self.assertEqual(records[0]["reward"], 0.5)
        self.assertEqual(list(records[0]["state"]), [0.0, 0.0, 0.0])
        self.assertEqual(list(records[0]["next_state"]), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
