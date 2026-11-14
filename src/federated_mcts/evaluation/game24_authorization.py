"""Durable selected-test authorization registry.

Provides an fsync-backed, lock-protected, hash-chained append-only
authorization registry stored outside the run-output directory.  Each
selected-test authorization is keyed by the immutable tuple::

    (ordered_test_ids_hash, model_config_hash,
     decision_policy_fingerprint, review_round_identity)

The run fingerprint is stored as metadata only and never participates in
the authorization key, so audit/schema changes cannot reauthorize a
second attempt.  Secrets (API keys, env-var values) are excluded from
all registry entries.

Follows the contract from Todo 3 of the game24-model-only-improvements plan.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
from typing import Tuple, cast


# ═══════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════

AuthKey = Tuple[str, str, str, str]
"""``(ordered_test_ids_hash, model_config_hash,
decision_policy_fingerprint, review_round_identity)``"""


def make_auth_key(
    *,
    ordered_test_ids_hash: str,
    model_config_hash: str,
    decision_policy_fingerprint: str,
    review_round_identity: str,
) -> AuthKey:
    """Construct the canonical 4-tuple authorization key.

    The run fingerprint is intentionally absent — it is metadata only
    and cannot be used to reauthorize a second attempt.
    """
    return (
        ordered_test_ids_hash,
        model_config_hash,
        decision_policy_fingerprint,
        review_round_identity,
    )


class AuthRegistry:
    """Fsync-backed, file-lock-protected hash-chained authorization ledger.

    Thread-safe.  Survives process restarts.  Entries are written as
    JSONL lines with a chained SHA-256 hash linking each entry to its
    predecessor, providing tamper-evidence.
    """

    def __init__(self, registry_path: str) -> None:
        """Open or create an authorization registry at *registry_path*.

        The directory must exist; if the file does not, it is created.
        """
        self._path = registry_path
        self._lock = threading.Lock()
        # Load existing chain on construction
        self._chain: list[dict] = []
        self._chain_head: str = ""  # SHA-256 of previous line, or ""
        self._consumed_keys: set[AuthKey] = set()
        self._selected_test_consumed: bool = False
        self._load_existing()

    # ── Public methods ────────────────────────────────────────────

    def consume(
        self,
        key: AuthKey,
        run_fingerprint: str,
        *,
        is_selected_test: bool = False,
    ) -> bool:
        """Attempt to consume *key* for one authorized run.

        Returns ``True`` if the key was successfully consumed (first
        attempt), ``False`` if it was already consumed or the
        selected-test slot was already taken.

        *run_fingerprint* is recorded as metadata only and never
        participates in the key comparison.
        """
        # Fast path: in-memory check (must be reconfirmed under lock)
        with self._lock:
            if key in self._consumed_keys:
                return False
            if is_selected_test and self._selected_test_consumed:
                return False

            # Build entry
            entry: dict[str, object] = {}
            entry["key"] = list(key)
            entry["run_fingerprint"] = run_fingerprint
            entry["is_selected_test"] = is_selected_test
            entry["timestamp_utc"] = _utc_now_iso()
            entry["prev_hash"] = self._chain_head
            entry["entry_hash"] = _sha256_json(entry)
            entry_line = _canonical_json_str(entry) + "\n"

            # Acquire file lock and re-verify before writing
            acquired = self._consume_atomic(key, entry, entry_line)
            if not acquired:
                return False

            # Update in-memory state
            self._chain.append(entry)
            self._chain_head = cast(str, entry["entry_hash"])
            self._consumed_keys.add(key)
            if is_selected_test:
                self._selected_test_consumed = True

            return True

    def _consume_atomic(
        self, key: AuthKey, entry: dict, entry_line: str,
    ) -> bool:
        """File-locked atomic consume: write *entry_line* only if *key*
        is not already on disk.  Returns True if the entry was written."""
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            # Re-read the file under lock to detect concurrent writers
            if os.path.getsize(self._path) > 0:
                with open(self._path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            existing = json.loads(line)
                            ek = tuple(existing.get("key", []))
                            if ek == key:
                                # Another process already consumed it
                                return False
                            if (existing.get("is_selected_test") and
                                    entry.get("is_selected_test")):
                                return False
                        except json.JSONDecodeError:
                            continue
            os.write(fd, entry_line.encode("utf-8"))
            os.fsync(fd)
            return True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def verify_chain(self) -> tuple[bool, str]:
        """Verify the hash chain of the on-disk registry.

        Returns ``(True, "")`` if valid, or ``(False, reason)`` if
        tampering is detected.
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
            return False, f"Cannot parse registry: {exc}"

        prev_hash = ""
        for i, entry in enumerate(entries):
            expected_prev = entry.get("prev_hash", "")
            if expected_prev != prev_hash:
                return False, (
                    f"Chain broken at entry {i}: prev_hash "
                    f"{expected_prev!r} != expected {prev_hash!r}"
                )
            # Recompute entry hash excluding the hash field itself
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

    @property
    def consumed_selected_test(self) -> bool:
        """Whether the selected-test authorization was already consumed."""
        return self._selected_test_consumed

    @property
    def chain_length(self) -> int:
        """Number of entries in the registry."""
        return len(self._chain)

    # ── Internal helpers ──────────────────────────────────────────

    def _load_existing(self) -> None:
        """Load the on-disk registry into memory."""
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
            # Corrupt registry — treat as empty (verification will catch it)
            return

        self._chain = entries
        if entries:
            self._chain_head = entries[-1].get("entry_hash", "")
        for entry in entries:
            key_tuple = tuple(entry.get("key", []))
            if len(key_tuple) == 4:
                self._consumed_keys.add(key_tuple)
            if entry.get("is_selected_test"):
                self._selected_test_consumed = True


# ═══════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _sha256_json(obj: object) -> str:
    """SHA-256 hex digest of the canonical JSON representation of *obj*."""
    data = _canonical_json_str(obj).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _canonical_json_str(obj: object) -> str:
    """Serialize *obj* as canonical JSON (sorted keys, compact)."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def _utc_now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
