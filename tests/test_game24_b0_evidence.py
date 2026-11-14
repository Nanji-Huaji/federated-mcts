"""Tests for B0 evidence validation.

Covers valid, corrupt, missing, incomplete, mismatched, reordered,
nonfinite evidence, no-side-effect guarantee, and regression for
canonical manifest fingerprint derivation (is stable from derived fields).

All fixtures are synthetic.  ``decision_policy_fingerprint`` and
``run_fingerprint`` are NOT placed in the manifest JSON — they
are derived by the validator using the same canonical hashing as
``Game24Manifest``.

Tamper/mismatch tests live in ``test_game24_b0_tamper.py``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from decimal import Decimal

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_b0_evidence import (
    CANONICAL_PILOT_IDS,
    RAW_UNIT_PRECISION,
    extract_model_usage_from_performance,
)
from federated_mcts.evaluation.game24_b0_manifest_loader import (
    derive_fingerprints_from_manifest_dict,
)
from federated_mcts.evaluation.game24_b0_validate import validate_b0_evidence


def _mf(**kw) -> dict:
    """Build B0 manifest dict with canonical B0 field values.

    Includes ALL fields from ``_DECISION_FIELDS`` and ``_RUN_FIELDS``
    so that ``_dict_to_canonical_subset`` can validate completeness
    (fail-closed on missing required fields).
    """
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
        "model_names": ["qwen3.7-plus"],
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
        "llm": "qwen3.7-plus",
        "qwen3.7-plus": {
            "total_calls": 165, "prompt_tokens": 85992,
            "completion_tokens": 84054, "total_tokens": 170046,
            "cost": 0.1688832,
        },
        "estimated_cost_usd": 0.1688832,
        "budget_spent": 0.1688832,
        "budget_remaining": 19.8235328,
        "complete": True, "run_fingerprint": run_fp,
        "decision_policy_fingerprint": dp_fp,
    }
    d.update(kw)
    return d


def _jf(run_fp: str, task_ids: list[int] | None = None) -> dict:
    ids = task_ids if task_ids is not None else list(CANONICAL_PILOT_IDS)
    j: dict = {"_meta": {"run_fingerprint": run_fp}}
    for tid in ids:
        j[str(tid)] = {"task_id": tid, "ys": [f"r{tid}"], "_completed": True}
    return j


def _bf(spent=0.1688832, limit=19.99) -> list[dict]:
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


class TestB0ValidEvidence(unittest.TestCase):
    """Given correct B0 evidence, When validating, Then passes."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_valid_evidence_passes_all_checks(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        ps = _wr(self.tmp, mf=mf, pf=_pf(run_fp, dp_fp),
                 jf=_jf(run_fp), bf=_bf())
        ev = validate_b0_evidence(**{f"{k}_path": v for k, v in ps.items()})
        self.assertEqual(ev.pilot_id_count, 20)
        # raw_units_spent is sourced from the hash-chained ledger, not from a pin.
        self.assertEqual(ev.raw_units_spent, Decimal("0.1688832"))
        self.assertTrue(ev.manifest_sha256)

    def test_validation_creates_no_files(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        ps = _wr(self.tmp, mf=mf, pf=_pf(run_fp, dp_fp),
                 jf=_jf(run_fp), bf=_bf())
        before = set(os.listdir(self.tmp))
        validate_b0_evidence(**{f"{k}_path": v for k, v in ps.items()})
        self.assertEqual(before, set(os.listdir(self.tmp)))

    def test_fingerprints_derived_from_canonical_fields(self):
        """When manifest has no fp fields, validator derives correct fingerprints."""
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        self.assertEqual(len(dp_fp), 64)
        self.assertEqual(len(run_fp), 64)
        self.assertNotEqual(dp_fp, run_fp)
        dp_fp2, run_fp2 = _derive_fps(_mf())
        self.assertEqual(dp_fp, dp_fp2)
        self.assertEqual(run_fp, run_fp2)

    def test_null_budget_limit_produces_stable_fingerprint(self):
        """Null max_call_limit is included in canonical hash, stable across calls."""
        mf = _mf()
        self.assertIsNone(mf["max_call_limit"])
        dp1, rn1 = _derive_fps(mf)
        dp2, rn2 = _derive_fps(_mf())  # same manifest = same fp
        self.assertEqual(dp1, dp2)
        self.assertEqual(rn1, rn2)

    def test_null_to_nonnull_changes_fingerprint(self):
        """Changing max_call_limit from None to a value changes fingerprints."""
        mf_null = _mf()
        mf_val = _mf(max_call_limit=100)
        dp_null, rn_null = _derive_fps(mf_null)
        dp_val, rn_val = _derive_fps(mf_val)
        self.assertNotEqual(dp_null, dp_val, "decision fp unchanged after null→100")
        self.assertNotEqual(rn_null, rn_val, "run fp unchanged after null→100")

    def test_null_budget_limit_present_in_canonical_like_live_manifest(self):
        """Null budget limits are in canonical dict just as Game24Manifest includes them."""
        from federated_mcts.evaluation.game24_manifest import (
            _DECISION_FIELDS,
            _canonical_json_bytes,
        )
        from federated_mcts.evaluation.game24_b0_manifest_loader import (
            _dict_to_canonical_subset,
        )
        mf = _mf()
        # Decision dict must contain the null budget fields
        d = _dict_to_canonical_subset(mf, _DECISION_FIELDS)
        self.assertIn("max_call_limit", d)
        self.assertIsNone(d["max_call_limit"])
        self.assertIn("max_token_limit", d)
        self.assertIsNone(d["max_token_limit"])
        self.assertIn("max_cost_usd", d)
        self.assertIsNone(d["max_cost_usd"])
        # Canonical JSON must include "null" for those keys
        raw = _canonical_json_bytes(d).decode()
        self.assertIn('"max_call_limit":null', raw)
        self.assertIn('"max_token_limit":null', raw)
        self.assertIn('"max_cost_usd":null', raw)


class TestB0Errors(unittest.TestCase):
    """Given various invalid evidence, When validating, Then ValueError."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _v(self, **kw) -> None:
        all_kw: dict[str, str] = {}
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        defaults = [
            ("manifest_path", os.path.join(self.tmp, "_dm.json"), mf, "w"),
            ("performance_path", os.path.join(self.tmp, "_dp.json"),
             _pf(run_fp, dp_fp), "w"),
            ("journal_path", os.path.join(self.tmp, "_dj.json"),
             _jf(run_fp), "w"),
            ("budget_path", os.path.join(self.tmp, "_db.jsonl"),
             _bf(), "w"),
        ]
        for key, path, data, _mode in defaults:
            if key not in kw:
                with open(path, _mode) as f:
                    if isinstance(data, list):
                        for e in data:
                            f.write(json.dumps(e, sort_keys=True,
                                               separators=(",", ":")) + "\n")
                    else:
                        json.dump(data, f)
                all_kw[key] = path
        all_kw.update(kw)
        validate_b0_evidence(**all_kw)

    # — missing files —
    def test_missing_manifest(self):
        ps = _wr(self.tmp, mf=_mf(), pf=_pf("a" * 64, "b" * 64),
                 jf=_jf("a" * 64), bf=_bf())
        os.remove(ps["manifest"])
        with self.assertRaises(ValueError):
            self._v(manifest_path=ps["manifest"])

    # — corrupt JSON —
    def test_corrupt_json(self):
        p = os.path.join(self.tmp, "manifest.json")
        with open(p, "w") as f:
            f.write("not json")
        with self.assertRaises(ValueError):
            self._v(manifest_path=p)

    # — wrong settings —
    def test_wrong_decision_policy(self):
        mf = _mf(decision_policy="heuristic_rules")
        dp_fp, run_fp = _derive_fps(mf)
        ps = _wr(self.tmp, mf=mf, pf=_pf(run_fp, dp_fp),
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(manifest_path=ps["manifest"])

    def test_wrong_seed(self):
        mf = _mf(seed=42)
        dp_fp, run_fp = _derive_fps(mf)
        ps = _wr(self.tmp, mf=mf, pf=_pf(run_fp, dp_fp),
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(manifest_path=ps["manifest"])

    # — wrong task IDs —
    def test_truncated_task_ids(self):
        short = CANONICAL_PILOT_IDS[:19]
        mf = _mf(task_ids=short)
        dp_fp, run_fp = _derive_fps(mf)
        ps = _wr(self.tmp, mf=mf, pf=_pf(run_fp, dp_fp),
                 jf=_jf(run_fp, task_ids=short), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(manifest_path=ps["manifest"])

    def test_reordered_task_ids(self):
        rev = list(reversed(CANONICAL_PILOT_IDS))
        mf = _mf(task_ids=rev)
        dp_fp, run_fp = _derive_fps(mf)
        ps = _wr(self.tmp, mf=mf, pf=_pf(run_fp, dp_fp),
                 jf=_jf(run_fp, task_ids=rev), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(manifest_path=ps["manifest"])

    # — incomplete journal —
    def test_incomplete_journal(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        j = _jf(run_fp)
        j[str(CANONICAL_PILOT_IDS[-1])]["_completed"] = False
        ps = _wr(self.tmp, mf=mf, pf=_pf(run_fp, dp_fp), jf=j, bf=_bf())
        with self.assertRaises(ValueError):
            self._v(journal_path=ps["journal"])

    # — nonfinite values —
    def test_nan(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["qwen3.7-plus"]["total_calls"] = float("nan")
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_inf(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["qwen3.7-plus"]["total_tokens"] = float("inf")
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_negative(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["qwen3.7-plus"]["total_calls"] = -5
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    # ── Nested usage shape edge cases ─────────────────────────────

    def test_missing_llm_field_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        del pf["llm"]
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_llm_field_not_string_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["llm"] = 42
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_missing_model_usage_entry_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        del pf["qwen3.7-plus"]
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_model_entry_not_dict_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["qwen3.7-plus"] = "not a dict"
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_multiple_model_entries_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["deepseek-v4-pro"] = {"total_calls": 5, "prompt_tokens": 100,
                                  "completion_tokens": 200, "total_tokens": 300}
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_token_total_inconsistency_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["qwen3.7-plus"]["total_tokens"] = 999
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_missing_nested_field_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        del pf["qwen3.7-plus"]["completion_tokens"]
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_bool_nested_field_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["qwen3.7-plus"]["prompt_tokens"] = True
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    # ── Missing required fingerprint field → fail-closed ───────────

    def test_missing_max_call_limit_rejected(self):
        """When max_call_limit is absent from manifest, fingerprint derivation fails."""
        mf = _mf()
        del mf["max_call_limit"]
        with self.assertRaises(ValueError) as ctx:
            _derive_fps(mf)
        self.assertIn("max_call_limit", str(ctx.exception))

    def test_missing_max_token_limit_rejected(self):
        mf = _mf()
        del mf["max_token_limit"]
        with self.assertRaises(ValueError) as ctx:
            _derive_fps(mf)
        self.assertIn("max_token_limit", str(ctx.exception))

    def test_missing_max_cost_usd_rejected(self):
        mf = _mf()
        del mf["max_cost_usd"]
        with self.assertRaises(ValueError) as ctx:
            _derive_fps(mf)
        self.assertIn("max_cost_usd", str(ctx.exception))


# ── Unit tests for extract_model_usage_from_performance ─────────

class TestExtractModelUsageFromPerformance(unittest.TestCase):
    """Given various performance dict shapes, When extract_model_usage_from_performance
       is called, Then usage is extracted or ValueError is raised."""

    def test_experiment_metrics_coexists_passes(self):
        """experiment_metrics (non-usage dict) coexists with model usage — passes."""
        perf = {
            "llm": "deepseek-v4-flash",
            "deepseek-v4-flash": {
                "total_calls": 160, "prompt_tokens": 64960,
                "completion_tokens": 81919, "total_tokens": 146879,
            },
            "experiment_metrics": {
                "version": 2, "task_count": 20, "solved_task_count": 8,
                "success_rate": 0.4, "total_proposed": 500,
                "total_candidates": 500, "rejection_counts": {},
            },
        }
        usage = extract_model_usage_from_performance(perf)
        self.assertEqual(usage["total_calls"], 160)
        self.assertEqual(usage["total_tokens"], 146879)

    def test_experiment_metrics_alone_does_not_trigger_duplicate_error(self):
        """experiment_metrics keys (task_count etc.) are not usage keys — no false positive."""
        perf = {
            "llm": "deepseek-v4-flash",
            "deepseek-v4-flash": {
                "total_calls": 160, "prompt_tokens": 64960,
                "completion_tokens": 81919, "total_tokens": 146879,
            },
            "experiment_metrics": {"task_count": 20, "solved_task_count": 8},
        }
        usage = extract_model_usage_from_performance(perf)
        self.assertEqual(usage["total_calls"], 160)

    def test_duplicate_usage_mapping_rejected(self):
        """A second dict with all usage keys is still rejected."""
        perf = {
            "llm": "deepseek-v4-flash",
            "deepseek-v4-flash": {
                "total_calls": 160, "prompt_tokens": 64960,
                "completion_tokens": 81919, "total_tokens": 146879,
            },
            "deepseek-v4-pro": {
                "total_calls": 5, "prompt_tokens": 100,
                "completion_tokens": 200, "total_tokens": 300,
            },
        }
        with self.assertRaises(ValueError):
            extract_model_usage_from_performance(perf)

    def test_duplicate_usage_mapping_rejected_with_experiment_metrics(self):
        """A second usage mapping is rejected even when experiment_metrics is also present."""
        perf = {
            "llm": "deepseek-v4-flash",
            "deepseek-v4-flash": {
                "total_calls": 160, "prompt_tokens": 64960,
                "completion_tokens": 81919, "total_tokens": 146879,
            },
            "experiment_metrics": {"task_count": 20},
            "deepseek-v4-pro": {
                "total_calls": 5, "prompt_tokens": 100,
                "completion_tokens": 200, "total_tokens": 300,
            },
        }
        with self.assertRaises(ValueError):
            extract_model_usage_from_performance(perf)

    def test_malformed_primary_mapping_still_fails(self):
        """Malformed primary model entry (missing field) still raises ValueError."""
        perf = {
            "llm": "deepseek-v4-flash",
            "deepseek-v4-flash": {
                "total_calls": 160, "prompt_tokens": 64960,
                "completion_tokens": 81919,
                # total_tokens missing
            },
            "experiment_metrics": {"task_count": 20},
        }
        with self.assertRaises(ValueError):
            extract_model_usage_from_performance(perf)

    def test_arbitrary_non_usage_dict_ignored(self):
        """An unknown dict without usage keys is silently allowed — no false rejection."""
        perf = {
            "llm": "deepseek-v4-flash",
            "deepseek-v4-flash": {
                "total_calls": 160, "prompt_tokens": 64960,
                "completion_tokens": 81919, "total_tokens": 146879,
            },
            "some_config": {"timeout": 30, "retries": 3},
        }
        usage = extract_model_usage_from_performance(perf)
        self.assertEqual(usage["total_calls"], 160)


# ── Tests for RAW_UNIT_PRECISION normalization — boundary and binary-tail ─
#
# Reconciliation quantizes half-up to raw-unit precision (1e-8): a binary-float
# representation tail must normalize onto the canonical value, while a genuine
# one-unit discrepancy (1e-8) must still be detected.

class TestRawUnitPrecisionBoundary(unittest.TestCase):
    """Given raw-unit Decimals, When quantizing to RAW_UNIT_PRECISION (ROUND_HALF_UP),
    Then binary-float tails collapse to one canonical value while values differing
    by a genuine raw unit stay distinct.

    Self-contained: these exercise the normalization rule itself, so they do not
    depend on any pinned telemetry constant.
    """

    CANONICAL = Decimal("0.1688832")

    def _q(self, value):
        from decimal import ROUND_HALF_UP
        return value.quantize(RAW_UNIT_PRECISION, rounding=ROUND_HALF_UP)

    def test_exact_value_quantizes_to_self(self):
        result = self._q(self.CANONICAL)
        self.assertEqual(result, self.CANONICAL.quantize(RAW_UNIT_PRECISION))

    def test_binary_tail_passes(self):
        noisy = Decimal("0.168883200000000014")
        canonical = self._q(self.CANONICAL)
        self.assertEqual(self._q(noisy), canonical)

    def test_binary_tail_greater_passes(self):
        noisy = Decimal("0.16888320000000001")
        canonical = self._q(self.CANONICAL)
        self.assertEqual(self._q(noisy), canonical)

    def test_one_unit_less_is_distinct(self):
        wrong = Decimal("0.16888319")
        canonical = self._q(self.CANONICAL)
        self.assertNotEqual(self._q(wrong), canonical)

    def test_one_unit_more_is_distinct(self):
        wrong = Decimal("0.16888321")
        canonical = self._q(self.CANONICAL)
        self.assertNotEqual(self._q(wrong), canonical)


# ── End-to-end: budget ledger with binary-float noise ────────────

class TestBudgetLedgerBinaryTail(unittest.TestCase):
    """Given a budget JSONL file containing the canonical B0 spend,
    When BudgetLedger replays it and returns the float spent value,
    Then verify_budget_ledger normalizes the binary-float tail and passes."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_budget(self, spent_amount=0.1688832, limit=19.99) -> str:
        import hashlib

        def sh(obj):
            d = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")
            return hashlib.sha256(d).hexdigest()

        init = {"kind": "init", "timestamp_utc": "2025-01-01T00:00:00Z",
                "prev_hash": "", "total_limit": limit}
        init["entry_hash"] = sh(init)
        se = {"kind": "spend", "timestamp_utc": "2025-01-01T00:01:00Z",
              "prev_hash": init["entry_hash"], "amount": spent_amount,
              "remaining": limit - spent_amount}
        se["entry_hash"] = sh(se)

        budget_path = os.path.join(self.tmp, "budget.jsonl")
        with open(budget_path, "w") as f:
            f.write(json.dumps(init, sort_keys=True, separators=(",", ":")) + "\n")
            f.write(json.dumps(se, sort_keys=True, separators=(",", ":")) + "\n")
        return budget_path

    def test_binary_tail_ledger_verifies(self):
        """verify_budget_ledger normalizes a binary-tail float from the ledger."""
        from decimal import ROUND_HALF_UP
        from federated_mcts.evaluation.game24_b0_verify import verify_budget_ledger

        budget_path = self._write_budget()
        spent = verify_budget_ledger(budget_path)
        expected = Decimal("0.1688832").quantize(
            RAW_UNIT_PRECISION, rounding=ROUND_HALF_UP
        )
        self.assertEqual(spent, expected)

    def test_different_legitimate_spend_is_accepted(self):
        """A different-but-legitimate spend must NOT be rejected.

        Spend is a sampling outcome.  Two identical pilots produced 0.16646720
        and 0.16888320; both are genuine `complete` runs, so neither may be
        rejected on the strength of its total alone.
        """
        from federated_mcts.evaluation.game24_b0_verify import verify_budget_ledger

        budget_path = self._write_budget(spent_amount=0.16646720)
        spent = verify_budget_ledger(budget_path)
        self.assertEqual(spent, Decimal("0.16646720"))

    def test_hash_chain_break_still_rejects(self):
        """Removing the numeric pins must not weaken hash-chain enforcement."""
        from federated_mcts.evaluation.game24_b0_verify import verify_budget_ledger

        budget_path = self._write_budget()
        with open(budget_path) as f:
            lines = f.read().splitlines()
        # Corrupt the recorded amount without recomputing entry_hash.
        broken = json.loads(lines[1])
        broken["amount"] = 0.00000001
        lines[1] = json.dumps(broken, sort_keys=True, separators=(",", ":"))
        with open(budget_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        with self.assertRaises(ValueError) as ctx:
            verify_budget_ledger(budget_path)
        self.assertIn("hash chain", str(ctx.exception).lower())


class TestLegitimateReRunAccepted(unittest.TestCase):
    """The central regression: an identical-but-re-sampled pilot must validate.

    Before this change the four resource fields and the spend total were compared
    for exact equality against one recorded run, so any legitimate re-run failed.
    Resource totals are LLM-sampling outcomes; two identical pilots produced
    163 vs 165 calls and 0.16646720 vs 0.16888320 raw units.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_with(self, calls, prompt, completion, cost, spent):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        # Model telemetry lives under the model-name key, so build it explicitly
        # rather than through _pf's top-level kw merge.
        pf["qwen3.7-plus"] = {
            "total_calls": calls,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cost": cost,
        }
        pf["estimated_cost_usd"] = cost
        pf["budget_spent"] = cost
        ps = _wr(self.tmp, mf=mf, pf=pf, jf=_jf(run_fp), bf=_bf(spent=spent))
        return validate_b0_evidence(**{f"{k}_path": v for k, v in ps.items()})

    def test_re_sampled_pilot_with_different_totals_validates(self):
        """pilot6-shaped telemetry must validate even though pins came from pilot7."""
        ev = self._run_with(
            calls=163, prompt=85204, completion=82741,
            cost=0.16646720, spent=0.16646720,
        )
        self.assertEqual(ev.calls, 163)
        self.assertEqual(ev.total_tokens, 167945)
        self.assertEqual(ev.raw_units_spent, Decimal("0.16646720"))

    def test_grossly_different_spend_does_not_break_projection(self):
        """A much larger legitimate spend flows through to the projection value."""
        ev = self._run_with(
            calls=165, prompt=85992, completion=84054,
            cost=0.90000000, spent=0.90000000,
        )
        self.assertEqual(ev.raw_units_spent, Decimal("0.90000000"))

    def test_token_total_inconsistency_still_rejects(self):
        """Internal coherence is preserved as a real invariant.

        Enforced both by the usage extractor and by _verify_performance, so this
        rejects at whichever layer runs first.
        """
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["qwen3.7-plus"] = {
            "total_calls": 1, "prompt_tokens": 100,
            "completion_tokens": 200, "total_tokens": 999,  # 100+200 != 999
            "cost": 0.1,
        }
        ps = _wr(self.tmp, mf=mf, pf=pf, jf=_jf(run_fp), bf=_bf(spent=0.1))
        with self.assertRaises(ValueError) as ctx:
            validate_b0_evidence(**{f"{k}_path": v for k, v in ps.items()})
        msg = str(ctx.exception).lower()
        self.assertTrue(
            "inconsistency" in msg or "inconsistent" in msg,
            f"expected a token-coherence error, got: {msg}",
        )

    def test_negative_call_count_rejects(self):
        """Well-formedness is still enforced, so garbage telemetry cannot pass."""
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["qwen3.7-plus"] = {
            "total_calls": -5, "prompt_tokens": 100,
            "completion_tokens": 200, "total_tokens": 300,
            "cost": 0.1,
        }
        ps = _wr(self.tmp, mf=mf, pf=pf, jf=_jf(run_fp), bf=_bf(spent=0.1))
        with self.assertRaises(ValueError) as ctx:
            validate_b0_evidence(**{f"{k}_path": v for k, v in ps.items()})
        self.assertIn("nonnegative", str(ctx.exception).lower())

    def test_wrong_fingerprint_still_rejects(self):
        """Fingerprint enforcement is untouched by this change."""
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        ps = _wr(
            self.tmp, mf=mf,
            pf=_pf(run_fp="c" * 64, dp_fp=dp_fp),
            jf=_jf(run_fp), bf=_bf(),
        )
        with self.assertRaises(ValueError):
            validate_b0_evidence(**{f"{k}_path": v for k, v in ps.items()})


if __name__ == "__main__":
    unittest.main()
