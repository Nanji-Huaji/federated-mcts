"""Tests for read-only ledger verification and idempotent task commit recovery.

Scope: the ledger-recovery seam only.

* ``read_verified_ledger`` / ``verify_budget_ledger`` must verify without
  touching a single byte of the evidence file.
* ``TaskCommitter`` must make outcome + usage + ``Decimal`` cost one commit,
  ordered so that a crash before *or* after the ledger append is recoverable
  without double charging and without skipping a completed-but-unbilled task.

All fault injection uses a real ``tempfile`` directory, a real
``AtomicTaskJournal``, and a real ``BudgetLedger``; only the crash itself is
injected, so the ledger/cap behaviour under test is never mocked.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from decimal import Decimal

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_b0_verify import verify_budget_ledger
from federated_mcts.evaluation.game24_budget import (
    BudgetLedger,
    LedgerFormatError,
    LedgerKeyConflictError,
    SpendStatus,
    read_verified_ledger,
)
from federated_mcts.evaluation.game24_persistence import (
    AtomicTaskJournal,
    BudgetExhaustedError,
    CommitConflictError,
    CommitError,
    ResumeValidator,
    RunJournal,
    TaskCommitter,
)
from federated_mcts.evaluation.game24_runner_persistence import (
    restore_completed_tasks,
)


class _InjectedCrash(BaseException):
    """Controlled crash that propagates past ``except Exception``."""


def _write_ledger(path: str, entries: list[dict]) -> None:
    """Write a hash-chained ledger fixture exactly as the ledger writer does."""
    import hashlib

    def sha(obj: object) -> str:
        raw = json.dumps(
            obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    prev = ""
    lines: list[str] = []
    for entry in entries:
        entry = dict(entry)
        entry["prev_hash"] = prev
        entry.setdefault("timestamp_utc", "2026-01-01T00:00:00Z")
        entry["entry_hash"] = sha(entry)
        prev = entry["entry_hash"]
        lines.append(json.dumps(entry, sort_keys=True, separators=(",", ":")))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _init_entry(limit: float = 19.99) -> dict:
    return {"kind": "init", "total_limit": limit}


def _spend_entry(amount: float, **extra: object) -> dict:
    entry: dict = {"kind": "spend", "amount": amount, "remaining": 1.0}
    entry.update(extra)
    return entry


class _Workspace:
    """One tempdir with a journal + ledger pair."""

    def __init__(self, *, limit: float = 19.99, run_id: str = "run-a") -> None:
        self.tmp = tempfile.mkdtemp()
        self.journal = AtomicTaskJournal(os.path.join(self.tmp, "journal.json"))
        self.journal.write_meta(run_fingerprint=run_id)
        self.ledger = BudgetLedger(
            os.path.join(self.tmp, "budget.jsonl"), total_limit=limit,
        )
        self.run_id = run_id

    def committer(self, cls=TaskCommitter, *, run_id: str | None = None) -> TaskCommitter:
        return cls(self.journal, self.ledger, run_id=run_id or self.run_id)

    def cleanup(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    @property
    def ledger_path(self) -> str:
        return self.ledger.path


def _commit(
    committer: TaskCommitter, task_id: int | str, cost: str, **kwargs: object,
) -> bool:
    defaults: dict = {
        "ys": [f"step-{task_id}"],
        "task_info": {"idx": task_id, "infos": [{"r": 1}]},
        "task_metrics_json": json.dumps({"task": task_id}),
        "elapsed_ms": 1.5,
        "usage": {"qwen3.7-plus.total_calls": 2},
        "cost": Decimal(cost),
    }
    defaults.update(kwargs)
    return committer.commit(task_id, **defaults)


# ═══════════════════════════════════════════════════════════════════
#  Read-only ledger verification
# ═══════════════════════════════════════════════════════════════════

class TestReadVerifiedLedger(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = _Workspace()

    def tearDown(self) -> None:
        self.ws.cleanup()

    def _read_bytes(self, path: str | None = None) -> bytes:
        with open(path or self.ws.ledger_path, "rb") as handle:
            return handle.read()

    def test_valid_ledger_returns_exact_decimal_sum(self) -> None:
        self.ws.ledger.spend_idempotent(
            key="k1", amount=Decimal("0.1688832"),
            run_id="run-a", task_id="3",
        )
        self.ws.ledger.spend_idempotent(
            key="k2", amount=Decimal("0.0000001"),
            run_id="run-a", task_id="9",
        )
        verified = read_verified_ledger(self.ws.ledger_path)
        self.assertEqual(verified.spent, Decimal("0.1688833"))
        self.assertEqual(verified.total_limit, 19.99)
        self.assertEqual(len(verified.entries), 3)
        self.assertEqual(verified.head_hash, verified.entries[-1]["entry_hash"])

    def test_read_does_not_modify_or_create_files(self) -> None:
        self.ws.ledger.spend(1.0)
        before_bytes = self._read_bytes()
        before_names = sorted(os.listdir(self.ws.tmp))
        read_verified_ledger(self.ws.ledger_path)
        self.assertEqual(self._read_bytes(), before_bytes)
        self.assertEqual(sorted(os.listdir(self.ws.tmp)), before_names)

    def test_missing_file_raises(self) -> None:
        missing = os.path.join(self.ws.tmp, "nope.jsonl")
        with self.assertRaises(LedgerFormatError):
            read_verified_ledger(missing)
        self.assertFalse(os.path.exists(missing))

    def test_empty_file_raises_and_is_untouched(self) -> None:
        path = os.path.join(self.ws.tmp, "empty.jsonl")
        with open(path, "w", encoding="utf-8"):
            pass
        with self.assertRaises(LedgerFormatError):
            read_verified_ledger(path)
        self.assertEqual(self._read_bytes(path), b"")
        self.assertEqual(sorted(os.listdir(self.ws.tmp)),
                         sorted({"journal.json", "journal.json.lock",
                                 "budget.jsonl", "empty.jsonl"}))

    def test_truncated_tail_raises(self) -> None:
        self.ws.ledger.spend(1.0)
        data = self._read_bytes()
        with open(self.ws.ledger_path, "wb") as handle:
            handle.write(data[:-5])
        before = self._read_bytes()
        with self.assertRaises(LedgerFormatError):
            read_verified_ledger(self.ws.ledger_path)
        self.assertEqual(self._read_bytes(), before)

    def test_corrupt_json_raises(self) -> None:
        self.ws.ledger.spend(1.0)
        with open(self.ws.ledger_path, "r+", encoding="utf-8") as handle:
            text = handle.read()
            handle.seek(0)
            handle.write(text.replace('"kind"', '"kin', 1))
            handle.truncate()
        with self.assertRaises(LedgerFormatError):
            read_verified_ledger(self.ws.ledger_path)

    def test_hash_chain_break_raises(self) -> None:
        path = os.path.join(self.ws.tmp, "broken.jsonl")
        _write_ledger(path, [_init_entry(), _spend_entry(0.5)])
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        entry = json.loads(lines[1])
        entry["amount"] = 0.25  # hash no longer matches
        lines[1] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        with self.assertRaises(LedgerFormatError):
            read_verified_ledger(path)

    def test_missing_init_entry_raises(self) -> None:
        path = os.path.join(self.ws.tmp, "noint.jsonl")
        _write_ledger(path, [_spend_entry(0.5)])
        with self.assertRaises(LedgerFormatError):
            read_verified_ledger(path)

    def test_duplicate_init_entry_raises(self) -> None:
        path = os.path.join(self.ws.tmp, "twoinit.jsonl")
        _write_ledger(path, [_init_entry(), _init_entry()])
        with self.assertRaises(LedgerFormatError):
            read_verified_ledger(path)

    def test_invalid_spend_amount_raises(self) -> None:
        for bad in ({"amount": -0.01}, {"amount": True}, {"amount": "NaN"}):
            path = os.path.join(self.ws.tmp, "bad.jsonl")
            _write_ledger(path, [_init_entry(), {"kind": "spend", **bad}])
            with self.assertRaises(LedgerFormatError):
                read_verified_ledger(path)

    def test_semantically_invalid_ledger_is_rejected_without_writes(self) -> None:
        for entries in (
                [_init_entry(), {"kind": "spnd", "amount": 0.5}],
                [_init_entry(), _spend_entry(0.5, amount_exact="0.05")],
                [{"kind": "init", "total_limit": "1e999"}]):
            with self.subTest(entries=entries):
                path = os.path.join(self.ws.tmp, "invalid.jsonl")
                _write_ledger(path, entries)
                before = self._read_bytes(path)
                with self.assertRaises(LedgerFormatError):
                    read_verified_ledger(path)
                self.assertEqual(self._read_bytes(path), before)

    def test_duplicate_spend_key_raises(self) -> None:
        path = os.path.join(self.ws.tmp, "dupkey.jsonl")
        _write_ledger(path, [
            _init_entry(),
            _spend_entry(0.1, spend_key="k", run_id="r", task_id="1"),
            _spend_entry(0.2, spend_key="k", run_id="r", task_id="2"),
        ])
        with self.assertRaises(LedgerFormatError):
            read_verified_ledger(path)

    def test_legacy_spend_has_no_key(self) -> None:
        self.ws.ledger.spend(1.0)
        verified = read_verified_ledger(self.ws.ledger_path)
        self.assertEqual(len(verified.spends), 1)
        self.assertIsNone(verified.spends[0].key)
        self.assertEqual(verified.spend_keys, {})

    def test_keyed_spend_is_indexed_with_run_and_task(self) -> None:
        self.ws.ledger.spend_idempotent(
            key="run-a:3", amount=Decimal("0.5"), run_id="run-a", task_id="3",
        )
        verified = read_verified_ledger(self.ws.ledger_path)
        spend = verified.find_spend("run-a:3")
        assert spend is not None
        self.assertEqual(spend.run_id, "run-a")
        self.assertEqual(spend.task_id, "3")
        self.assertEqual(spend.amount, Decimal("0.5"))


# ═══════════════════════════════════════════════════════════════════
#  Keyed idempotent spend
# ═══════════════════════════════════════════════════════════════════

class TestKeyedSpend(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = _Workspace(limit=1.0)

    def tearDown(self) -> None:
        self.ws.cleanup()

    def test_entry_records_key_run_and_task(self) -> None:
        status = self.ws.ledger.spend_idempotent(
            key="run-a:3", amount=Decimal("0.25"),
            run_id="run-a", task_id="3",
        )
        self.assertIs(status, SpendStatus.RECORDED)
        with open(self.ws.ledger_path, encoding="utf-8") as handle:
            entry = json.loads(handle.read().splitlines()[-1])
        self.assertEqual(entry["spend_key"], "run-a:3")
        self.assertEqual(entry["run_id"], "run-a")
        self.assertEqual(entry["task_id"], "3")
        self.assertEqual(entry["amount_exact"], "0.25")

    def test_duplicate_same_amount_writes_nothing(self) -> None:
        self.assertIs(
            self.ws.ledger.spend_idempotent(
                key="k", amount=Decimal("0.25"), run_id="r", task_id="1",
            ),
            SpendStatus.RECORDED,
        )
        with open(self.ws.ledger_path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertIs(
            self.ws.ledger.spend_idempotent(
                key="k", amount=Decimal("0.25"), run_id="r", task_id="1",
            ),
            SpendStatus.DUPLICATE,
        )
        with open(self.ws.ledger_path, encoding="utf-8") as handle:
            self.assertEqual(handle.read().splitlines(), lines)
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0.25"))

    def test_duplicate_different_amount_raises(self) -> None:
        self.ws.ledger.spend_idempotent(
            key="k", amount=Decimal("0.25"), run_id="r", task_id="1",
        )
        with self.assertRaises(LedgerKeyConflictError):
            self.ws.ledger.spend_idempotent(
                key="k", amount=Decimal("0.5"), run_id="r", task_id="1",
            )
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0.25"))

    def test_over_cap_writes_nothing(self) -> None:
        self.assertIs(
            self.ws.ledger.spend_idempotent(
                key="k", amount=Decimal("2"), run_id="r", task_id="1",
            ),
            SpendStatus.OVER_CAP,
        )
        with open(self.ws.ledger_path, encoding="utf-8") as handle:
            self.assertEqual(len(handle.read().splitlines()), 1)  # init only
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0"))

    def test_zero_amount_keyed_spend_is_a_durable_marker(self) -> None:
        self.assertIs(
            self.ws.ledger.spend_idempotent(
                key="k", amount=Decimal("0"), run_id="r", task_id="1",
            ),
            SpendStatus.RECORDED,
        )
        self.assertIs(
            self.ws.ledger.spend_idempotent(
                key="k", amount=Decimal("0"), run_id="r", task_id="1",
            ),
            SpendStatus.DUPLICATE,
        )
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0"))

    def test_key_survives_a_fresh_writable_ledger(self) -> None:
        self.ws.ledger.spend_idempotent(
            key="k", amount=Decimal("0.25"), run_id="r", task_id="1",
        )
        reloaded = BudgetLedger(self.ws.ledger_path, total_limit=1.0)
        self.assertIsNotNone(reloaded.find_spend("k"))
        self.assertEqual(reloaded.spent_exact, Decimal("0.25"))


# ═══════════════════════════════════════════════════════════════════
#  TaskCommitter commit protocol
# ═══════════════════════════════════════════════════════════════════

class TestTaskCommit(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = _Workspace()

    def tearDown(self) -> None:
        self.ws.cleanup()

    def test_commit_settles_record_and_charges_once(self) -> None:
        committer = self.ws.committer()
        self.assertTrue(_commit(committer, 3, "0.05"))
        record = self.ws.journal.read_all()["3"]
        self.assertIs(record["_completed"], True)
        self.assertEqual(record["commit_state"], "settled")
        self.assertEqual(record["task_id"], 3)
        self.assertEqual(record["cost_raw"], "0.05")
        self.assertEqual(record["usage_delta"], {"qwen3.7-plus.total_calls": 2})
        self.assertEqual(record["ys"], "step-3")
        self.assertEqual(record["task_metrics_json"], json.dumps({"task": 3}))
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0.05"))
        self.assertEqual(self.ws.journal.completed_ids(), {"3"})

    def test_commit_replay_is_idempotent(self) -> None:
        committer = self.ws.committer()
        self.assertTrue(_commit(committer, 3, "0.05"))
        with open(self.ws.ledger_path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertFalse(_commit(committer, 3, "0.05"))
        with open(self.ws.ledger_path, encoding="utf-8") as handle:
            self.assertEqual(handle.read().splitlines(), lines)
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0.05"))

    def test_commit_conflicting_cost_raises(self) -> None:
        committer = self.ws.committer()
        _commit(committer, 3, "0.05")
        with self.assertRaises(CommitConflictError):
            _commit(committer, 3, "0.06")
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0.05"))

    def test_commit_against_legacy_completed_record_raises(self) -> None:
        self.ws.journal.write_task("3", {
            "task_id": 3, "ys": "x", "task_info": {}, "_completed": True,
        })
        with self.assertRaises(CommitConflictError):
            _commit(self.ws.committer(), 3, "0.05")
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0"))

    def test_commit_rejects_malformed_cost(self) -> None:
        committer = self.ws.committer()
        for bad in (True, Decimal("NaN"), -1, "abc"):
            with self.assertRaises((CommitError, ValueError)):
                committer.commit(3, ys=["x"], task_info={}, cost=bad)
        self.assertNotIn("3", self.ws.journal.read_all())

    def test_commit_rejects_malformed_usage(self) -> None:
        committer = self.ws.committer()
        for bad in ({"x": True}, {"x": -1}, {"": 1}):
            with self.assertRaises(CommitError):
                _commit(committer, 3, "0", usage=bad)

    def test_commit_zero_cost_still_marker(self) -> None:
        committer = self.ws.committer()
        self.assertTrue(_commit(committer, 3, "0"))
        self.assertIn("3", self.ws.journal.completed_ids())
        self.assertIsNotNone(self.ws.ledger.find_spend("run-a:3"))

    def test_two_runs_same_task_are_distinct_in_one_ledger(self) -> None:
        other = TaskCommitter(
            AtomicTaskJournal(os.path.join(self.ws.tmp, "journal-b.json")),
            self.ws.ledger, run_id="run-b",
        )
        _commit(self.ws.committer(), 3, "0.05")
        _commit(other, 3, "0.07")
        verified = read_verified_ledger(self.ws.ledger_path)
        keys = {spend.key for spend in verified.spends}
        self.assertEqual(keys, {"run-a:3", "run-b:3"})
        self.assertEqual(verified.spent, Decimal("0.12"))

    def test_commit_rejects_non_integer_task_id(self) -> None:
        with self.assertRaises(CommitError):
            _commit(self.ws.committer(), "task-3", "0.05")
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0"))

    def test_default_committer_is_run_journal_alias(self) -> None:
        self.assertIs(RunJournal, TaskCommitter)


# ═══════════════════════════════════════════════════════════════════
#  Crash recovery (real tempdir, real ledger, real journal)
# ═══════════════════════════════════════════════════════════════════

class _CrashBeforeCharge(TaskCommitter):
    """Crash after the pending record is written, before the ledger append."""

    crash_on: str | None = None

    def _charge(self, key: str, cost: Decimal, task_id: str) -> SpendStatus:
        if task_id == self.crash_on:
            raise _InjectedCrash("crash-before-charge")
        return super()._charge(key, cost, task_id)


class _CrashAfterCharge(TaskCommitter):
    """Crash after the ledger append, before the record is settled."""

    crash_on: str | None = None

    def _charge(self, key: str, cost: Decimal, task_id: str) -> SpendStatus:
        status = super()._charge(key, cost, task_id)
        if task_id == self.crash_on:
            raise _InjectedCrash("crash-after-charge")
        return status


class TestCrashRecovery(unittest.TestCase):
    TASKS = ((3, "0.01"), (9, "0.02"), (16, "0.03"))
    TOTAL = Decimal("0.06")

    def setUp(self) -> None:
        self.ws = _Workspace()

    def tearDown(self) -> None:
        self.ws.cleanup()

    def _reference_run(self) -> None:
        committer = self.ws.committer()
        for task_id, cost in self.TASKS:
            _commit(committer, task_id, cost)

    def _ledger_line_count(self) -> int:
        with open(self.ws.ledger_path, encoding="utf-8") as handle:
            return len(handle.read().splitlines())

    def test_crash_before_charge_is_recovered_exactly_once(self) -> None:
        crashing = _CrashBeforeCharge(self.ws.journal, self.ws.ledger, run_id=self.ws.run_id)
        crashing.crash_on = "9"
        _commit(crashing, 3, "0.01")
        with self.assertRaises(_InjectedCrash):
            _commit(crashing, 9, "0.02")
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0.01"))

        report = self.ws.committer().recover()
        self.assertEqual(report.recovered_ids, {"9"})
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0.03"))
        self.assertEqual(self.ws.journal.completed_ids(), {"3", "9"})
        self.assertIn("9", report.settled_ids)

    def test_crash_after_charge_does_not_double_charge(self) -> None:
        crashing = _CrashAfterCharge(self.ws.journal, self.ws.ledger, run_id=self.ws.run_id)
        crashing.crash_on = "9"
        _commit(crashing, 3, "0.01")
        with self.assertRaises(_InjectedCrash):
            _commit(crashing, 9, "0.02")
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0.03"))
        lines_before = self._ledger_line_count()
        self.assertNotIn("9", self.ws.journal.completed_ids())

        report = self.ws.committer().recover()
        self.assertEqual(report.recovered_ids, {"9"})
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0.03"))
        self.assertEqual(self._ledger_line_count(), lines_before)
        self.assertEqual(self.ws.journal.completed_ids(), {"3", "9"})

    def test_interrupted_run_matches_uninterrupted_run(self) -> None:
        self._reference_run()
        reference_spent = self.ws.ledger.spent_exact
        reference_records = self.ws.journal.read_all()
        self.ws.cleanup()

        # Fresh workspace: crash once before charging and once after.
        self.ws = _Workspace()
        crashing_before = _CrashBeforeCharge(self.ws.journal, self.ws.ledger, run_id=self.ws.run_id)
        crashing_before.crash_on = str(self.TASKS[1][0])
        _commit(crashing_before, self.TASKS[0][0], self.TASKS[0][1])
        with self.assertRaises(_InjectedCrash):
            _commit(crashing_before, self.TASKS[1][0], self.TASKS[1][1])
        crashing_after = _CrashAfterCharge(self.ws.journal, self.ws.ledger, run_id=self.ws.run_id)
        crashing_after.crash_on = str(self.TASKS[2][0])
        with self.assertRaises(_InjectedCrash):
            _commit(crashing_after, self.TASKS[2][0], self.TASKS[2][1])
        report = self.ws.committer().recover()
        # Both interrupted tasks (one before the charge, one after) settle.
        self.assertEqual(
            report.recovered_ids,
            {str(self.TASKS[1][0]), str(self.TASKS[2][0])},
        )
        self.assertEqual(self.ws.ledger.spent_exact, reference_spent)
        self.assertEqual(self.ws.ledger.spent_exact, self.TOTAL)
        self.assertEqual(report.orphan_spend_keys, frozenset())
        self.assertEqual(report.legacy_unmetered_ids, frozenset())
        self.assertTrue(report.fully_reconciled)
        records = self.ws.journal.read_all()
        self.assertEqual(
            {tid: rec["cost_raw"] for tid, rec in records.items() if tid != "_meta"},
            {tid: rec["cost_raw"] for tid, rec in reference_records.items()
             if tid != "_meta"},
        )

    def test_recover_is_idempotent(self) -> None:
        crashing = _CrashBeforeCharge(self.ws.journal, self.ws.ledger, run_id=self.ws.run_id)
        crashing.crash_on = "9"
        _commit(crashing, 3, "0.01")
        with self.assertRaises(_InjectedCrash):
            _commit(crashing, 9, "0.02")
        self.ws.committer().recover()
        lines = self._ledger_line_count()
        second = self.ws.committer().recover()
        self.assertEqual(second.recovered_ids, frozenset())
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0.03"))
        self.assertEqual(self._ledger_line_count(), lines)

    def test_recover_restores_runner_state(self) -> None:
        crashing = _CrashAfterCharge(self.ws.journal, self.ws.ledger, run_id=self.ws.run_id)
        crashing.crash_on = "9"
        _commit(crashing, 3, "0.01")
        with self.assertRaises(_InjectedCrash):
            _commit(crashing, 9, "0.02")
        report = self.ws.committer().recover()
        logs: list = []
        metrics: list = []
        ca, cy = restore_completed_tasks(
            self.ws.journal, {int(t) for t in report.settled_ids},
            logs=logs, per_task_metrics=metrics,
        )
        self.assertEqual(len(logs), 2)
        self.assertEqual(len(metrics), 2)
        self.assertEqual((ca, cy), (2, 2))

    def test_recover_refuses_corrupt_ledger_and_keeps_pending(self) -> None:
        crashing = _CrashBeforeCharge(self.ws.journal, self.ws.ledger, run_id=self.ws.run_id)
        crashing.crash_on = "9"
        _commit(crashing, 3, "0.01")
        with self.assertRaises(_InjectedCrash):
            _commit(crashing, 9, "0.02")
        with open(self.ws.ledger_path, "r+", encoding="utf-8") as handle:
            text = handle.read()
            handle.seek(0)
            handle.write(text.replace('"kind"', '"kin', 1))
            handle.truncate()
        pending_before = self.ws.journal.read_all()["9"]
        with self.assertRaises(LedgerFormatError):
            self.ws.committer().recover()
        self.assertEqual(self.ws.journal.read_all()["9"], pending_before)

    def test_recover_over_cap_raises_and_keeps_pending(self) -> None:
        tight = _Workspace(limit=0.015)
        try:
            crashing = _CrashBeforeCharge(tight.journal, tight.ledger, run_id=tight.run_id)
            crashing.crash_on = "9"
            _commit(crashing, 3, "0.01")
            with self.assertRaises(_InjectedCrash):
                _commit(crashing, 9, "0.02")
            pending_before = tight.journal.read_all()["9"]
            with self.assertRaises(BudgetExhaustedError):
                tight.committer().recover()
            self.assertEqual(tight.journal.read_all()["9"], pending_before)
            self.assertEqual(tight.ledger.spent_exact, Decimal("0.01"))
        finally:
            tight.cleanup()

    def test_recover_reports_orphan_spend_without_discarding_it(self) -> None:
        # A keyed spend for this run with no journal record: money is real,
        # attribution is not.  It must be reported, never deleted.
        self.ws.ledger.spend_idempotent(
            key="run-a:99", amount=Decimal("0.04"),
            run_id="run-a", task_id="99",
        )
        report = self.ws.committer().recover()
        self.assertEqual(report.orphan_spend_keys, frozenset({"run-a:99"}))
        self.assertFalse(report.fully_reconciled)
        self.assertEqual(report.spent, Decimal("0.04"))
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0.04"))

    def test_foreign_run_spend_is_not_an_orphan(self) -> None:
        self.ws.ledger.spend_idempotent(
            key="run-b:3", amount=Decimal("0.04"),
            run_id="run-b", task_id="3",
        )
        report = self.ws.committer().recover()
        self.assertEqual(report.orphan_spend_keys, frozenset())
        self.assertTrue(report.fully_reconciled)

    def test_settled_record_without_ledger_spend_raises(self) -> None:
        self.ws.journal.put_task("3", {
            "task_id": 3, "ys": "x", "task_info": {}, "_completed": True,
            "commit_state": "settled", "cost_raw": "0.05",
            "spend_key": "run-a:3",
        })
        with self.assertRaises(CommitConflictError):
            self.ws.committer().recover()

    def test_resume_validator_excludes_pending_until_recovered(self) -> None:
        crashing = _CrashBeforeCharge(
            self.ws.journal, self.ws.ledger, run_id=self.ws.run_id)
        crashing.crash_on = "9"
        _commit(crashing, 3, "0.01")
        with self.assertRaises(_InjectedCrash):
            _commit(crashing, 9, "0.02")

        can, skip, reason = ResumeValidator(
            self.ws.journal, "run-a").validate_resume([3, 9])
        self.assertTrue(can, reason)
        self.assertEqual(skip, {"3"})

        report = self.ws.committer().recover()
        self.assertIn("9", report.settled_ids)
        can, skip, reason = ResumeValidator(
            self.ws.journal, "run-a").validate_resume([3, 9])
        self.assertTrue(can, reason)
        self.assertEqual(skip, {"3", "9"})

    def test_pending_key_from_another_run_fails_closed(self) -> None:
        self.ws.journal.put_task("3", {
            "task_id": 3, "ys": "x", "task_info": {}, "_completed": False,
            "commit_state": "pending", "cost_raw": "0.05",
            "spend_key": "run-b:3",
        })
        with self.assertRaises(CommitError):
            self.ws.committer().recover()
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0"))

    def test_legacy_completed_record_is_reported_not_priced(self) -> None:
        self.ws.journal.write_task("3", {
            "task_id": 3, "ys": "x", "task_info": {}, "_completed": True,
        })
        report = self.ws.committer().recover()
        self.assertEqual(report.settled_ids, frozenset({"3"}))
        self.assertEqual(report.legacy_unmetered_ids, frozenset({"3"}))
        self.assertFalse(report.fully_reconciled)
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0"))

    def test_pending_record_without_cost_fails_closed(self) -> None:
        self.ws.journal.put_task("3", {
            "task_id": 3, "ys": "x", "task_info": {}, "_completed": False,
            "commit_state": "pending", "spend_key": "run-a:3",
        })
        with self.assertRaises(CommitError):
            self.ws.committer().recover()
        self.assertEqual(self.ws.ledger.spent_exact, Decimal("0"))


# ═══════════════════════════════════════════════════════════════════
#  B0 verifier must not construct a side-effecting ledger
# ═══════════════════════════════════════════════════════════════════

class TestB0VerifierReadOnly(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _path(self, name: str = "budget.jsonl") -> str:
        return os.path.join(self.tmp, name)

    def test_empty_ledger_rejected_and_bytes_unchanged(self) -> None:
        path = self._path()
        with open(path, "w", encoding="utf-8"):
            pass
        with self.assertRaises(ValueError):
            verify_budget_ledger(path)
        self.assertEqual(os.path.getsize(path), 0)
        self.assertEqual(os.listdir(self.tmp), ["budget.jsonl"])

    def test_truncated_ledger_rejected_and_bytes_unchanged(self) -> None:
        path = self._path()
        _write_ledger(path, [_init_entry(), _spend_entry(0.1688832)])
        with open(path, "rb") as handle:
            data = handle.read()
        with open(path, "wb") as handle:
            handle.write(data[:-3])
        before = data[:-3]
        with self.assertRaises(ValueError):
            verify_budget_ledger(path)
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), before)

    def test_valid_ledger_still_verifies_exactly(self) -> None:
        path = self._path()
        _write_ledger(path, [_init_entry(), _spend_entry(0.1688832)])
        self.assertEqual(verify_budget_ledger(path), Decimal("0.16888320"))

    def test_validate_b0_evidence_is_side_effect_free_on_empty_budget(self) -> None:
        from federated_mcts.evaluation.game24_b0_manifest_loader import (
            derive_fingerprints_from_manifest_dict,
        )
        from federated_mcts.evaluation.game24_b0_validate import (
            validate_b0_evidence,
        )
        from tests._game24_b0_fixtures import (
            _bf, _jf, _make_all, _mf, _pf,
        )

        mf = _mf()
        dp_fp, run_fp = derive_fingerprints_from_manifest_dict(mf)
        ps = _make_all(
            self.tmp, mf_dict=mf, pf_dict=_pf(run_fp, dp_fp),
            jf_dict=_jf(run_fp), bf_list=_bf(),
        )
        budget_path = ps["budget"]
        os.truncate(budget_path, 0)
        before_names = sorted(os.listdir(self.tmp))
        with self.assertRaises(ValueError):
            validate_b0_evidence(**{f"{k}_path": v for k, v in ps.items()})
        self.assertEqual(os.path.getsize(budget_path), 0)
        self.assertEqual(sorted(os.listdir(self.tmp)), before_names)

    def test_missing_ledger_still_raises_not_found(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            verify_budget_ledger(self._path("missing.jsonl"))
        self.assertIn("not found", str(ctx.exception))
        self.assertFalse(os.path.exists(self._path("missing.jsonl")))


if __name__ == "__main__":
    unittest.main()
