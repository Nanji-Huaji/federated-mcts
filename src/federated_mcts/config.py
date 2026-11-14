"""Resolved search configuration shared by execution and experiment identity.

Only decision options live here. Credentials, CLI parsing, output paths and
experiment authorization are assembled by the runner, not by search modules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
import re
from pathlib import Path
from federated_mcts.models.rates import RATE_CARD_VERSION


@dataclass(frozen=True, slots=True)
class DecisionSpec:
    """Versioned, immutable options actually consumed by a search session."""

    version: int = 2
    rate_card_version: str = RATE_CARD_VERSION
    solve_method: str = "tot"
    search_policy: str = "baseline"
    mock_client: bool = False
    temperature: float = 0.9
    seed: int | None = None
    method_generate: str = "propose"
    method_evaluate: str = "vote"
    method_select: str = "greedy"
    prompt_sample: str | None = None
    n_generate_sample: int = 1
    n_evaluate_sample: int = 1
    n_select_sample: int = 1
    check_format: bool = False
    eval_rule: bool = False
    warm_start: bool = False
    last_lm: bool = False
    slm_generate: bool = False
    slm_eval: bool = False
    game24_exact_prune: bool = False
    assign_strategy: str = "round_robin"
    diverse_joint_rank: bool = True
    diversity_weight: float = 0.35
    beam_expand: int = 2
    beam_uncertainty_margin: float = 0.1
    beam_confidence_margin: float = 0.8
    dqn_seed: int = 0
    dqn_epsilon: float = 1.0
    dqn_state_dim: int = 16
    dqn_capacity: int = 10000
    dqn_token_budget: float = 5000.0
    dqn_latency_budget: float = 60.0
    dqn_oracle_distance_reward: bool = False
    dqn_oracle_distance_scale: float = 0.25
    dqn_checkpoint_sha256: str | None = None
    dqn_checkpoint: str | None = field(default=None, repr=False, metadata={"identity": False})
    dqn_jsonl: str | None = field(default=None, repr=False, metadata={"identity": False})

    @classmethod
    def from_args(cls, args) -> DecisionSpec:
        values = {f.name: getattr(args, f.name) for f in fields(cls)
                  if hasattr(args, f.name) and f.name not in ("version", "dqn_checkpoint_sha256")}
        # argparse's absent optional mode is None; execution needs a resolved mode.
        for key, default in (("method_generate", "propose"), ("method_evaluate", "vote")):
            if values.get(key) is None:
                values[key] = default
        if getattr(args, "naive_run", False):
            values["solve_method"] = "naive"
        checkpoint = values.get("dqn_checkpoint")
        if checkpoint:
            values["dqn_checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        spec = cls(**values)
        spec.validate()
        return spec

    def validate(self) -> None:
        for item in fields(self):
            value, default = getattr(self, item.name), item.default
            if default is not None and type(value) is not type(default):
                if not (type(default) is float and type(value) in (int, float)):
                    raise ValueError(f"Invalid decision field {item.name!r}")
            if type(value) is float and not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or None")
        if self.prompt_sample is not None and (
                not isinstance(self.prompt_sample, str) or
                self.prompt_sample not in {"standard", "cot"}):
            raise ValueError("Unsupported prompt_sample")
        if self.dqn_checkpoint_sha256 is not None and (
                not isinstance(self.dqn_checkpoint_sha256, str) or
                re.fullmatch(r"[0-9a-f]{64}", self.dqn_checkpoint_sha256) is None):
            raise ValueError("Invalid DQN checkpoint digest")
        for name in ("n_generate_sample", "n_evaluate_sample", "n_select_sample",
                     "dqn_state_dim", "dqn_capacity"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.method_generate not in {"sample", "propose"}:
            raise ValueError("Unsupported method_generate")
        if self.method_evaluate not in {"vote", "value"}:
            raise ValueError("Unsupported method_evaluate")
        if self.method_select not in {"sample", "greedy"}:
            raise ValueError("Unsupported method_select")
        if self.solve_method not in {"naive", "tot", "speculative_solve", "federated_solve"}:
            raise ValueError(f"Unsupported solve_method: {self.solve_method}")
        if self.search_policy not in {"baseline", "diverse", "dqn"}:
            raise ValueError(f"Unsupported search_policy: {self.search_policy}")
        if self.search_policy != "baseline" and self.solve_method not in {"tot", "federated_solve"}:
            raise ValueError("diverse/DQN require tot or federated_solve")
        if self.method_generate == "sample" or self.solve_method == "naive":
            if self.prompt_sample not in {"standard", "cot"}:
                raise ValueError("sample/naive generation requires --prompt_sample standard or cot")

    def validate_checkpoint(self) -> None:
        """Reject changed checkpoint inputs before any run side effects."""
        if self.dqn_checkpoint_sha256 is not None:
            if not self.dqn_checkpoint:
                raise ValueError("DQN checkpoint path is required for execution")
            actual = hashlib.sha256(Path(self.dqn_checkpoint).read_bytes()).hexdigest()
            if actual != self.dqn_checkpoint_sha256:
                raise ValueError("DQN checkpoint changed after preparation; prepare a new run")

    def identity(self) -> dict:
        """Canonical decision fields, excluding output and checkpoint locations."""
        values = asdict(self)
        return {f.name: values[f.name] for f in fields(self) if f.metadata.get("identity", True)}

    @classmethod
    def validate_identity(cls, value: object) -> dict:
        """Reject missing/unknown v2 identity fields rather than invent defaults."""
        if not isinstance(value, dict):
            raise ValueError("decision_spec must be an object")
        expected = set(cls().identity())
        if set(value) != expected or value.get("version") != 2:
            raise ValueError("decision_spec must contain exactly the version-2 decision fields")
        for name, default in cls().identity().items():
            actual = value[name]
            if default is not None and type(actual) is not type(default):
                # JSON integers are valid values for float options, but booleans aren't.
                if not (type(default) is float and type(actual) in (int, float)):
                    raise ValueError(f"Invalid decision_spec field {name!r}")
        cls(**value).validate()
        return dict(value)


def load_model_config(path: str, *, temperature: float) -> list[dict]:
    """Read once, validate names, and resolve execution defaults before hashing.

    Credentials remain unresolved here and are excluded by the manifest codec.
    The client factory may resolve environment references only after run gates.
    """
    with open(path, encoding="utf-8") as stream:
        entries = json.load(stream)
    if not isinstance(entries, list) or not entries:
        raise ValueError("model_config must be a nonempty list")
    names: set[str] = set()
    resolved = []
    for entry in entries:
        if not isinstance(entry, dict) or any(
                not isinstance(entry.get(key), str) or not entry[key].strip()
                for key in ("model", "client_name")):
            raise ValueError("Each model requires nonempty model and client_name strings")
        name = entry["client_name"]
        if name in names:
            raise ValueError(f"Duplicate model client_name: {name}")
        names.add(name)
        cfg = dict(entry)
        local = cfg.get("type") == "vllm"
        cfg.setdefault("temperature", 0.7 if local else temperature)
        cfg.setdefault("max_tokens", 512 if local else 1000)
        cfg.setdefault("api_base", "http://localhost:8000/v1" if local else "https://try-chatapi.com/v1")
        if type(cfg["max_tokens"]) is not int or cfg["max_tokens"] < 1:
            raise ValueError("Model max_tokens must be a positive integer")
        if type(cfg["temperature"]) not in (int, float) or not math.isfinite(cfg["temperature"]):
            raise ValueError("Model temperature must be finite numeric")
        if not isinstance(cfg["api_base"], str) or not cfg["api_base"]:
            raise ValueError("Model api_base must be a nonempty string")
        resolved.append(cfg)
    return resolved
