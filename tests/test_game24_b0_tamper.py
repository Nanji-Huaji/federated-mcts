"""Tests for B0 evidence fingerprint tamper detection.

Covers: derived fingerprint stability, manifest tamper changes
fingerprints, mismatched performance/journal fingerprints fail-closed,
and tampered manifest settings caught before fingerprint check.

All fixtures are synthetic.  Fingerprints are derived from manifest
fields, not hardcoded in JSON.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_b0_evidence import (
    CANONICAL_PILOT_IDS,
)
from federated_mcts.evaluation.game24_b0_manifest_loader import (
    derive_fingerprints_from_manifest_dict,
)
from federated_mcts.evaluation.game24_b0_validate import validate_b0_evidence


def _mf(**kw) -> dict:
    d: dict = {
        "task": "game24",
        "task_ids": list(CANONICAL_PILOT_IDS),
        "task_split_name": "val",
        "decision_policy": "local_validity",
        "oracle_audit": True,
        "seed": 17,
        "temperature": 0.9,
        "method_generate": "propose",
        "method_evaluate": "vote",
        "method_select": "greedy",
        "n_generate_sample": 2,
        "n_evaluate_sample": 3,
        "n_select_sample": 2,
        "game24_exact_prune": False,
        # ── Budget limits ─────────────────────────────────
        "max_call_limit": None,
        "max_token_limit": None,
        "max_cost_usd": None,
        "max_raw_unit_limit": 19.99,
        # ── Solver extras ─────────────────────────────────
        "assign_strategy": "round_robin",
        "runner_path": "scripts/merged_run.py",
        "effective_cli": "",
        # ── File / model / git hashes (placeholder) ────────
        "split_file_sha256": "a" * 64,
        "ordered_task_ids_sha256": "b" * 64,
        "model_config_sha256": "c" * 64,
        "model_names": ["deepseek-v4-flash"],
        # ── Git metadata ──────────────────────────────────
        "git_commit": "0000000",
        "worktree_dirty": False,
    }
    d.update(kw)
    return d


def _derive_fps(mf_dict: dict) -> tuple[str, str]:
    return derive_fingerprints_from_manifest_dict(mf_dict)


def _pf(run_fp: str, dp_fp: str, **kw) -> dict:
    d: dict = {
        "acc": 0.0, "avg_sum": 0.0,
        "llm": "deepseek-v4-flash",
        "deepseek-v4-flash": {
            "total_calls": 160, "prompt_tokens": 64960,
            "completion_tokens": 81919, "total_tokens": 146879,
            "cost": 0.03203172,
        },
        "estimated_cost_usd": 0.03203172,
        "budget_spent": 0.03203172,
        "budget_remaining": 19.95796828,
        "complete": True, "run_fingerprint": run_fp,
        "decision_policy_fingerprint": dp_fp,
    }
    d.update(kw)
    return d


def _jf(run_fp: str, task_ids=None) -> dict:
    ids = task_ids if task_ids is not None else list(CANONICAL_PILOT_IDS)
    j: dict = {"_meta": {"run_fingerprint": run_fp}}
    for tid in ids:
        j[str(tid)] = {"task_id": tid, "ys": [f"r{tid}"], "_completed": True}
    return j


def _bf(spent=0.03203172, limit=19.99) -> list[dict]:
    import hashlib

    def sh(obj):
        d = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(d).hexdigest()

    init = {"kind": "init", "timestamp_utc": "2025-01-01T00:00:00Z",
            "prev_hash": "", "total_limit": limit}
    init["entry_hash"] = sh(init)
    se = {"kind": "spend", "timestamp_utc": "2025-01-01T00:01:00Z",
          "prev_hash": init["entry_hash"], "amount": spent,
          "remaining": limit - spent}
    se["entry_hash"] = sh(se)
    return [init, se]


def _wr(tmp, mf=None, pf=None, jf=None, bf=None) -> dict[str, str]:
    ps: dict[str, str] = {}
    for label, data, ext in [
        ("manifest", mf, "json"), ("performance", pf, "json"),
        ("journal", jf, "json"), ("budget", bf, "jsonl"),
    ]:
        if data is None:
            continue
        p = os.path.join(tmp, f"{label}.{ext}")
        with open(p, "w") as f:
            if ext == "jsonl":
                for e in data:
                    f.write(json.dumps(e, sort_keys=True,
                                       separators=(",", ":")) + "\n")
            else:
                json.dump(data, f)
        ps[label] = p
    return ps


class TestB0FingerprintTamper(unittest.TestCase):
    """Given tampered evidence, When validating, Then ValueError fail-closed."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_manifest_tamper_changes_fingerprints(self):
        mf1 = _mf()
        mf2 = _mf(seed=42)
        dp1, rn1 = _derive_fps(mf1)
        dp2, rn2 = _derive_fps(mf2)
        self.assertNotEqual(dp1, dp2)
        self.assertNotEqual(rn1, rn2)

    def test_mismatched_performance_fingerprint_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        wrong_fp = "a" * 64
        ps = _wr(self.tmp, mf=mf,
                 pf=_pf(run_fp=wrong_fp, dp_fp=dp_fp),
                 jf=_jf(wrong_fp), bf=_bf())
        with self.assertRaises(ValueError):
            validate_b0_evidence(**{f"{k}_path": v for k, v in ps.items()})

    def test_mismatched_journal_fingerprint_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        wrong_fp = "b" * 64
        ps = _wr(self.tmp, mf=mf,
                 pf=_pf(run_fp, dp_fp),
                 jf=_jf(run_fp=wrong_fp), bf=_bf())
        with self.assertRaises(ValueError):
            validate_b0_evidence(**{f"{k}_path": v for k, v in ps.items()})

    def test_tampered_manifest_seed_fails_settings_check(self):
        mf = _mf(seed=99)
        dp_fp, run_fp = _derive_fps(mf)
        ps = _wr(self.tmp, mf=mf,
                 pf=_pf(run_fp, dp_fp),
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            validate_b0_evidence(**{f"{k}_path": v for k, v in ps.items()})

    def test_null_budget_becomes_nonnull_changes_fingerprints(self):
        """Changing max_cost_usd from None to 0.5 alters both fingerprints."""
        mf_null = _mf()
        mf_val = _mf(max_cost_usd=0.5)
        dp_null, rn_null = _derive_fps(mf_null)
        dp_val, rn_val = _derive_fps(mf_val)
        self.assertNotEqual(dp_null, dp_val)
        self.assertNotEqual(rn_null, rn_val)

    def test_tampered_null_to_int_changes_run_fingerprint(self):
        """Setting max_token_limit from None to 5000 changes run fingerprint."""
        mf_base = _mf()
        mf_tampered = _mf(max_token_limit=5000)
        _, rn_base = _derive_fps(mf_base)
        _, rn_tampered = _derive_fps(mf_tampered)
        self.assertNotEqual(rn_base, rn_tampered,
                            "run fp must change when null becomes non-null")

    def test_missing_required_field_fails_before_fingerprint(self):
        """When a required field is absent, derive fails before hash comparison."""
        mf = _mf()
        del mf["split_file_sha256"]
        with self.assertRaises(ValueError) as ctx:
            _derive_fps(mf)
        self.assertIn("split_file_sha256", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
