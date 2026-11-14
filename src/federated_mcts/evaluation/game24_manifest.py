"""Immutable Game24 experiment manifests with dual cryptographic fingerprints.

Provides the canonical ``Game24Manifest`` frozen dataclass, authoritative
resolution of task IDs, seed application, and deterministic SHA-256
fingerprints that separate decision-affecting fields from audit/observation
fields.  Secrets (API keys, env-var values) are excluded from all hashes.

Follows the contract from Todo 3 of the game24-model-only-improvements plan.
"""

from __future__ import annotations

import hashlib
import json
import os
import random as _random
import subprocess
from dataclasses import dataclass, field
from federated_mcts.config import DecisionSpec


# ═══════════════════════════════════════════════════════════════════
#  Decision policy / audit labels
# ═══════════════════════════════════════════════════════════════════

DECISION_POLICY_VALUES = (
    "local_validity",
    "heuristic_rules",
    "oracle_exact_pruned",
)
"""Permitted values for ``--game24_decision_policy``."""

# Fields that affect search/model decisions (included in decision fp).
_DECISION_FIELDS = frozenset({
    "task",
    "task_ids",
    "task_split_name",
    "split_file_sha256",
    "ordered_task_ids_sha256",
    "model_config_sha256",
    "model_names",
    "decision_policy",
    "seed",
    "temperature",
    "method_generate",
    "method_evaluate",
    "method_select",
    "n_generate_sample",
    "n_evaluate_sample",
    "n_select_sample",
    "game24_exact_prune",
    "assign_strategy",
    "max_call_limit",
    "max_token_limit",
    "max_cost_usd",
    "max_raw_unit_limit",
    "runner_path",
})

# Fields added only to the run fingerprint (observation/audit).
_RUN_ONLY_FIELDS = frozenset({
    "oracle_audit",
    "schema_version",
    "effective_cli",
    "git_commit",
    "worktree_dirty",
    "remote_nondeterminism_notice",
})

_RUN_FIELDS = _DECISION_FIELDS | _RUN_ONLY_FIELDS

SCHEMA_VERSION: int = 2
"""Schema emitted by the runner; version 1 remains readable without rehashing."""


def manifest_field_sets(version: int) -> tuple[frozenset[str], frozenset[str]]:
    if type(version) is not int or version not in (1, 2):
        raise ValueError(f"Unsupported manifest schema_version: {version!r}")
    decision = _DECISION_FIELDS
    if version == 2:
        decision = decision | {"decision_spec"}
    return decision, decision | _RUN_ONLY_FIELDS


# ═══════════════════════════════════════════════════════════════════
#  Immutable manifest
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Game24Manifest:
    """Immutable fingerprint of one Game24 experiment configuration.

    All fields are frozen after construction.  ``decision_policy_fingerprint``
    and ``run_fingerprint`` are computed deterministically from the canonical
    (sorted-key) JSON representation of the relevant field subset.
    """

    # ── Task identity ────────────────────────────────────────────
    task: str
    task_ids: list[int]
    task_split_name: str | None

    # ── Decision policy ─────────────────────────────────────────
    decision_policy: str
    oracle_audit: bool

    # ── Reproducibility ─────────────────────────────────────────
    seed: int
    temperature: float

    # ── Solver configuration ────────────────────────────────────
    method_generate: str
    method_evaluate: str
    method_select: str
    n_generate_sample: int
    n_evaluate_sample: int
    n_select_sample: int
    game24_exact_prune: bool

    # ── Raw input paths (used in __post_init__ for hashing) ─────
    split_file_path: str = field(default="", repr=False)
    model_config_raw: list[dict] = field(default_factory=list, repr=False)

    # ── File / model hashes (computed during __post_init__) ─────
    split_file_sha256: str = field(init=False)
    ordered_task_ids_sha256: str = field(init=False)
    model_config_sha256: str = field(init=False)
    model_names: list[str] = field(init=False)

    # ── Budget limits ───────────────────────────────────────────
    max_call_limit: int | None = None
    max_token_limit: int | None = None
    max_cost_usd: float | None = None
    max_raw_unit_limit: float = 19.99

    # ── Solver extras ───────────────────────────────────────────
    assign_strategy: str = "round_robin"
    runner_path: str = "scripts/merged_run.py"
    effective_cli: str = ""

    # ── Environment markers ─────────────────────────────────────
    git_commit: str = field(init=False)
    worktree_dirty: bool = field(init=False)
    remote_nondeterminism_notice: bool = field(default=True, init=False)

    # Direct legacy construction remains v1 for evidence tooling. New runs
    # explicitly supply SCHEMA_VERSION and their resolved DecisionSpec.
    schema_version: int = 1
    decision_spec: DecisionSpec | None = None

    def __post_init__(self) -> None:
        manifest_field_sets(self.schema_version)
        if self.schema_version == 2:
            spec = DecisionSpec.validate_identity(
                self.decision_spec.identity() if isinstance(self.decision_spec, DecisionSpec)
                else self.decision_spec)
            # The legacy projections in v2 artifacts must agree with execution.
            for name in ("seed", "temperature", "method_generate", "method_evaluate",
                         "method_select", "n_generate_sample", "n_evaluate_sample",
                         "n_select_sample", "game24_exact_prune", "assign_strategy"):
                expected = spec[name]
                if name == "seed" and expected is None:
                    expected = 0  # v1 projection; v2 retains None in decision_spec
                if getattr(self, name) != expected:
                    raise ValueError(f"Manifest projection disagrees with DecisionSpec: {name}")
            _freeze_set(self, "decision_spec", DecisionSpec(**spec))
        elif self.decision_spec is not None:
            raise ValueError("Version-1 manifests cannot contain a decision_spec")
        # Compute hashes from raw inputs
        split_sha = _hash_bytes(
            _canonical_json_bytes(
                _load_split_json(self.split_file_path)
            )
        )
        task_ids_sha = _hash_bytes(
            _canonical_json_bytes(self.task_ids)
        )
        model_cfg_sha, model_names = _hash_model_config(
            self.model_config_raw
        )

        # Git metadata
        commit, dirty = _git_revision()

        # Use object.__setattr__ because frozen=True prevents normal attr set
        _freeze_set(self, "split_file_sha256", split_sha)
        _freeze_set(self, "ordered_task_ids_sha256", task_ids_sha)
        _freeze_set(self, "model_config_sha256", model_cfg_sha)
        _freeze_set(self, "model_names", model_names)
        _freeze_set(self, "git_commit", commit)
        _freeze_set(self, "worktree_dirty", dirty)

    # ── Fingerprint properties ──────────────────────────────────

    @property
    def decision_policy_fingerprint(self) -> str:
        """SHA-256 of canonical JSON over ONLY decision-affecting fields.

        Excludes ``oracle_audit``, schema version, effective CLI,
        git metadata, and the nondeterminism notice.  Two manifests
        that differ only in audit/observation fields share this value.
        """
        return _compute_fingerprint(
            self._to_canonical_dict(manifest_field_sets(self.schema_version)[0])
        )

    @property
    def run_fingerprint(self) -> str:
        """SHA-256 of canonical JSON over decision AND observation fields.

        Includes ``oracle_audit``, schema version, effective CLI,
        and git metadata.  Audit on/off will differ here while sharing
        ``decision_policy_fingerprint``.
        """
        return _compute_fingerprint(
            self._to_canonical_dict(manifest_field_sets(self.schema_version)[1])
        )

    # ── Serialization helpers ────────────────────────────────────

    def _to_canonical_dict(self, field_set: frozenset[str]) -> dict:
        """Return a sorted-key dict containing only the named fields."""
        d: dict = {}
        for name in sorted(field_set):
            if name == "remote_nondeterminism_notice":
                d[name] = True  # constant
            elif hasattr(self, name):
                val = getattr(self, name)
                if isinstance(val, DecisionSpec):
                    val = val.identity()
                # Force list fields to be comparable (sorted)
                if isinstance(val, list):
                    val = sorted(val) if name != "task_ids" else val
                d[name] = val
        # Add fixed schema version
        if "schema_version" in field_set:
            d["schema_version"] = self.schema_version
        return d

    def to_json(self) -> str:
        """Full manifest as JSON (secrets stripped)."""
        d = self._to_canonical_dict(manifest_field_sets(self.schema_version)[1])
        return _canonical_json_bytes(d).decode()


# ═══════════════════════════════════════════════════════════════════
#  Seed handling
# ═══════════════════════════════════════════════════════════════════

def apply_seed(seed: int) -> None:
    """Seed Python ``random`` and NumPy with *seed*.

    Call before constructing any solver or model client.
    Records that remote model outputs remain provider-nondeterministic
    regardless of client-side seeding.
    """
    _random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass  # NumPy not available — seed is still recorded in manifest


# ═══════════════════════════════════════════════════════════════════
#  Task ID resolution
# ═══════════════════════════════════════════════════════════════════

def resolve_task_ids(
    *,
    task: str,
    split_file_path: str,
    task_split: str | None,
    task_start_index: int,
    task_end_index: int,
) -> list[int]:
    """Resolve ordered task IDs from a split file or integer range.

    When *task_split* is provided it takes precedence over the
    ``--task_start_index`` / ``--task_end_index`` range.  The returned
    list is always sorted.

    Raises ``FileNotFoundError`` if the split file is missing and
    ``KeyError`` if the named split does not exist.
    """
    if task_split is not None:
        with open(split_file_path) as f:
            splits = json.load(f)
        ids: list[int] = list(splits[task_split])
    else:
        ids = list(range(task_start_index, task_end_index))
    return sorted(ids)


# ═══════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _canonical_json_bytes(obj: object) -> bytes:
    """Serialize *obj* as UTF-8 canonical JSON (sorted keys, no indentation)."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _hash_bytes(data: bytes) -> str:
    """Return lowercase SHA-256 hex digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def _compute_fingerprint(canonical_dict: dict) -> str:
    return _hash_bytes(_canonical_json_bytes(canonical_dict))


def _freeze_set(obj: object, name: str, value: object) -> None:
    """Bypass frozen dataclass immutability during ``__post_init__``."""
    object.__setattr__(obj, name, value)


def _load_split_json(path: str) -> dict:
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def _hash_model_config(configs: list[dict]) -> tuple[str, list[str]]:
    """Return (SHA-256, [model_names]) with secrets stripped."""
    stripped: list[dict] = []
    model_names: list[str] = []
    for entry in configs:
        clean = {}
        for key, value in entry.items():
            if key in ("api_key", "api_key_environment_variable"):
                continue  # strip secret-bearing keys
            clean[key] = value
        stripped.append(clean)
        if "model" in entry:
            model_names.append(str(entry["model"]))
    return _hash_bytes(_canonical_json_bytes(stripped)), model_names


def _git_revision() -> tuple[str, bool]:
    """Return (HEAD commit hex, worktree_dirty) for the repo.

    Falls back to ``("0000000", False)`` when git is unavailable.
    """
    commit = "0000000"
    dirty = False
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    try:
        diff_result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True, timeout=5,
        )
        dirty = diff_result.returncode != 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return commit, dirty
