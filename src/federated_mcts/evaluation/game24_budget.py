"""Shared raw-unit aggregate budget ledger.

Provides a thread-safe, fsync-backed, lock-protected, hash-chained
append-only budget ledger.  All planned experiments within one scope
share a single raw-unit budget (default 19.99).  Separate caps for API
calls and tokens are also supported and reported as incomplete stop
states when exceeded.

No currency conversion is performed — raw units are the canonical
budget dimension.  Cost caps (`--max_cost_usd`) are enforced at the
runner level alongside this ledger.

Follows the contract from Todo 3 of the game24-model-only-improvements plan.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import threading
import time


DEFAULT_RAW_UNIT_LIMIT: float = 19.99
"""Default total budget in raw units."""


def _validate_finite_nonnegative(name: str, value: float) -> float:
    """Return *value* if finite, non-NaN, nonnegative, and not bool.

    Raises ``ValueError`` otherwise.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got bool ({value!r})")
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if v < 0.0:
        raise ValueError(f"{name} must be non-negative, got {v}")
    return v


class BudgetLedger:
    """Shared budget tracker for a set of experiments.

    Thread-safe.  Survives process restarts.  Supports:
    - Raw-unit spending with cap enforcement
    - API call limits
    - Token limits
    - Incomplete stop states (``stop_reason`` string)
    """

    def __init__(
        self,
        ledger_path: str,
        *,
        total_limit: float = DEFAULT_RAW_UNIT_LIMIT,
        max_calls: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._path = ledger_path
        self._lock = threading.Lock()

        # Proposed caps — may be overridden by persistent state
        self._total_limit = _validate_finite_nonnegative("total_limit", total_limit)
        self._max_calls = max_calls
        self._max_tokens = max_tokens

        # Load existing state from disk
        self._spent: float = 0.0
        self._chain: list[dict] = []
        self._chain_head: str = ""
        self._calls_seen: int = 0
        self._tokens_seen: int = 0
        self._started_tasks: int = 0
        self._load_existing()

        # If ledger is empty, write the init entry with total_limit
        if not self._chain:
            self._record_init_entry()

    # ── Public properties ─────────────────────────────────────────

    @property
    def remaining(self) -> float:
        """Raw units remaining."""
        return max(0.0, self._total_limit - self._spent)

    @property
    def spent(self) -> float:
        """Raw units spent so far."""
        return self._spent

    @property
    def started_tasks(self) -> int:
        """Number of tasks started (for metadata)."""
        return self._started_tasks

    @property
    def chain_length(self) -> int:
        return len(self._chain)

    # ── Spend operations ──────────────────────────────────────────

    def spend(self, amount: float) -> bool:
        """Attempt to spend *amount* raw units from the budget.

        Returns ``True`` if the spend was recorded, ``False`` if it
        would exceed the limit.  Raises ``ValueError`` for NaN, inf,
        bool, or negative *amount*.
        """
        amount = _validate_finite_nonnegative("spend amount", amount)
        if amount == 0.0:
            return True

        with self._lock:
            if self._spent + amount > self._total_limit:
                return False
            self._spent += amount
            self._record_entry(
                "spend", {"amount": amount, "remaining": self.remaining}
            )
            return True

    def check_and_mark_calls(
        self, used_calls: int,
    ) -> tuple[bool, str | None]:
        """Check if *used_calls* exceeds the call cap.

        Returns ``(True, None)`` if under cap, or
        ``(False, "max_total_calls")`` if exceeded.
        """
        if self._max_calls is None:
            return True, None

        with self._lock:
            self._calls_seen = max(self._calls_seen, used_calls)
            if used_calls > self._max_calls:
                self._record_entry(
                    "cap_exceeded",
                    {"cap_type": "max_total_calls", "used": used_calls},
                )
                return False, "max_total_calls"
            return True, None

    def check_and_mark_tokens(
        self, used_tokens: int,
    ) -> tuple[bool, str | None]:
        """Check if *used_tokens* exceeds the token cap.

        Returns ``(True, None)`` if under cap, or
        ``(False, "max_total_tokens")`` if exceeded.
        """
        if self._max_tokens is None:
            return True, None

        with self._lock:
            self._tokens_seen = max(self._tokens_seen, used_tokens)
            if used_tokens > self._max_tokens:
                self._record_entry(
                    "cap_exceeded",
                    {"cap_type": "max_total_tokens", "used": used_tokens},
                )
                return False, "max_total_tokens"
            return True, None

    def mark_task_started(self) -> None:
        """Record that a new task was started (for metadata)."""
        with self._lock:
            self._started_tasks += 1

    # ── Verification ──────────────────────────────────────────────

    def verify_chain(self) -> tuple[bool, str]:
        """Verify the hash chain of the on-disk ledger.

        Returns ``(True, "")`` if valid, or ``(False, reason)``.
        """
        entries: list[dict] = []
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entries.append(json.loads(line))
        except (json.JSONDecodeError, OSError) as exc:
            return False, f"Cannot parse ledger: {exc}"

        prev_hash = ""
        for i, entry in enumerate(entries):
            expected_prev = entry.get("prev_hash", "")
            if expected_prev != prev_hash:
                return False, (
                    f"Chain broken at entry {i}: prev_hash "
                    f"{expected_prev!r} != expected {prev_hash!r}"
                )
            recomputed = _sha256_json(
                {k: v for k, v in entry.items() if k != "entry_hash"}
            )
            if recomputed != entry.get("entry_hash"):
                return False, (
                    f"Hash mismatch at entry {i}: "
                    f"{recomputed!r} != {entry.get('entry_hash')!r}"
                )
            prev_hash = entry["entry_hash"]

        return True, ""

    # ── Internal helpers ──────────────────────────────────────────

    def _record_entry(self, kind: str, data: dict) -> None:
        """Append a new entry to the on-disk ledger."""
        entry = {
            "kind": kind,
            "timestamp_utc": _utc_now_iso(),
            "prev_hash": self._chain_head,
        }
        entry.update(data)
        entry["entry_hash"] = _sha256_json(entry)
        line = _canonical_json_str(entry) + "\n"

        self._append_line(line)
        self._chain.append(entry)
        self._chain_head = entry["entry_hash"]

    def _append_line(self, line: str) -> None:
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _load_existing(self) -> None:
        """Replay the ledger from disk."""
        if not os.path.exists(self._path):
            return

        entries: list[dict] = []
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entries.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            return

        for entry in entries:
            kind = entry.get("kind")
            if kind == "init":
                self._total_limit = float(
                    entry.get("total_limit", self._total_limit)
                )
            elif kind == "spend":
                self._spent += float(entry.get("amount", 0.0))
            elif kind == "cap_exceeded":
                cap_type = entry.get("cap_type", "")
                if cap_type == "max_total_calls":
                    self._calls_seen = max(
                        self._calls_seen, int(entry.get("used", 0))
                    )
                elif cap_type == "max_total_tokens":
                    self._tokens_seen = max(
                        self._tokens_seen, int(entry.get("used", 0))
                    )

        self._chain = entries
        if entries:
            self._chain_head = entries[-1].get("entry_hash", "")

    def _record_init_entry(self) -> None:
        """Write the first entry recording total_limit."""
        self._record_entry("init", {"total_limit": self._total_limit})


# ═══════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _sha256_json(obj: object) -> str:
    data = _canonical_json_str(obj).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _canonical_json_str(obj: object) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
