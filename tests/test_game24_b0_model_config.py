"""Regression test for B0 model config file identity: golden hash, model order, secret safety.

Given the standalone B0 model config file, When loaded and hashed through
the authoritative manifest config hasher, Then the hash matches the B0 source
manifest ``model_config_sha256``, model names are correctly ordered,
no literal ``api_key`` value leaks through hashing, and the hash is invariant
to removal of the secret-bearing ``api_key_environment_variable`` field.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_manifest import _hash_model_config  # noqa: E402

# ═══════════════════════════════════════════════════════════════════
#  Golden values from B0 source manifest
# ═══════════════════════════════════════════════════════════════════

GOLDEN_SHA256 = "4fdc6d48cf6e4d36b08bd1508040a0a45f96925ee34848d181485d3a912601e8"
GOLDEN_MODEL_NAMES = [
    "Qwen/Qwen2.5-32B-Instruct",
    "deepseek-v4-flash",
]

# Path relative to repo root
_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "model_config_b0.json",
)


# ── Helpers ───────────────────────────────────────────────────────

def _load_config() -> list[dict]:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def _hash_config(cfg: list[dict]) -> tuple[str, list[str]]:
    return _hash_model_config(cfg)


# ═══════════════════════════════════════════════════════════════════
#  Regression tests
# ═══════════════════════════════════════════════════════════════════

class TestB0ModelConfigGoldenHash(unittest.TestCase):
    """Given the B0 model config file, When hashed, Then golden values match."""

    def test_golden_sha256_matches(self):
        """B0 config file reproduces exact B0 manifest model_config_sha256."""
        sha, names = _hash_config(_load_config())
        self.assertEqual(sha, GOLDEN_SHA256,
                         f"DEFECT: B0 config hash {sha[:12]}... "
                         f"!= expected {GOLDEN_SHA256[:12]}...")

    def test_model_names_are_correctly_ordered(self):
        """Model names match the B0 source: [Qwen 32B, deepseek-v4-flash]."""
        _, names = _hash_config(_load_config())
        self.assertEqual(names, GOLDEN_MODEL_NAMES,
                         f"DEFECT: B0 model_names {names} != {GOLDEN_MODEL_NAMES}")

    def test_config_file_has_no_literal_api_key(self):
        """Raw config file must not contain a literal ``api_key`` field entry."""
        raw = json.dumps(_load_config())
        self.assertNotIn('"api_key"', raw,
                         "DEFECT: raw config contains a literal api_key field")

    def test_hash_invariant_to_secret_field_removal(self):
        """Removing ``api_key_environment_variable`` must not change the hash."""
        cfg = _load_config()
        sha_before, names_before = _hash_config(cfg)
        # Deep copy and strip the secret-bearing field from every entry
        cfg_stripped = copy.deepcopy(cfg)
        for entry in cfg_stripped:
            entry.pop("api_key_environment_variable", None)
            entry.pop("api_key", None)
        sha_after, names_after = _hash_config(cfg_stripped)
        self.assertEqual(sha_before, sha_after,
                         "DEFECT: Hash changed after removing api_key_environment_variable")
        self.assertEqual(names_before, names_after,
                         "DEFECT: Model names changed after removing secret fields")

    def test_hash_invariant_to_extra_secret_field(self):
        """Adding an ``api_key`` literal must not change the hash either."""
        cfg = _load_config()
        sha_before, names_before = _hash_config(cfg)
        cfg_extra = copy.deepcopy(cfg)
        cfg_extra[1]["api_key"] = "sk-test-leaked-12345"
        sha_after, names_after = _hash_config(cfg_extra)
        self.assertEqual(sha_before, sha_after,
                         "DEFECT: Hash changed when api_key literal was added")


if __name__ == "__main__":
    unittest.main()
