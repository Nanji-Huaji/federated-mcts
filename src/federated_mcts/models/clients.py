"""Construction of model clients from already-loaded configuration entries.

This module is the one place that turns a model-config entry into the callable
shaped object consumed by the solvers::

    client(args, prompt, n=..., stop=..., get_logprobs=...) -> list[str]

It performs no file I/O and imports nothing from ``federated_mcts.federation``.
Callers read the JSON config themselves (so identity hashing and secret
handling stay at the configuration seam) and pass the parsed entries here.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Protocol

from federated_mcts.models.api_client import gpt
from federated_mcts.models.vllm_client import VLLMClient


class ClientCallable(Protocol):
    """Callable interface shared by remote and local generation clients."""

    def __call__(
        self,
        args: Any,
        prompt: str,
        n: int = 1,
        stop: Any = None,
        get_logprobs: bool = False,
        **kwargs: Any,
    ) -> list[str]: ...


_DEFAULT_API_BASE = "https://try-chatapi.com/v1"
_DEFAULT_MAX_TOKENS = 1000


def create_clients(
    model_configs: Sequence[Mapping[str, object]],
    *,
    default_temperature: float = 0.7,
    environ: Mapping[str, str] | None = None,
) -> dict[str, ClientCallable]:
    """Build one generation client per config entry, keyed by ``client_name``.

    Parameters
    ----------
    model_configs:
        Parsed model-config entries (the list stored in
        ``configs/model_config_vllm.json``).  Entries are not mutated.
    default_temperature:
        Temperature used only when an entry has no ``temperature`` key.  The
        runner passes its CLI temperature here.
    environ:
        Environment mapping used to resolve ``api_key_environment_variable``.
        Defaults to ``os.environ``; inject a plain dict in tests.

    Entry handling
    --------------
    ``type == "vllm"``
        A :class:`~federated_mcts.models.vllm_client.VLLMClient` adapter.
    anything else
        ``functools.partial(gpt, ...)`` with endpoint, key, model, sampling
        limits and the optional ``enable_thinking`` flag, matching the shape
        the orchestrator previously constructed inline.

    Raises
    ------
    TypeError
        If an entry is not a mapping or a numeric field has an invalid type.
    ValueError
        If a required field is missing/empty or a ``client_name`` repeats.
    """
    environment = os.environ if environ is None else environ
    clients: dict[str, ClientCallable] = {}

    for index, config in enumerate(model_configs):
        if not isinstance(config, Mapping):
            raise TypeError(
                f"model_configs[{index}] must be a mapping, got "
                f"{type(config).__name__}"
            )
        client_name = config.get("client_name")
        if not isinstance(client_name, str) or not client_name:
            raise ValueError(f"model_configs[{index}] is missing client_name")
        if client_name in clients:
            raise ValueError(f"duplicate client_name {client_name!r}")
        model = config.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(
                f"model_configs[{index}] ({client_name}) is missing model"
            )

        if config.get("type") == "vllm":
            clients[client_name] = VLLMClient.from_config(dict(config))
            continue

        temperature = _number(config, index, "temperature", default_temperature)
        max_tokens = _integer(
            config, index, "max_tokens", _DEFAULT_MAX_TOKENS
        )
        api_base = config.get("api_base", _DEFAULT_API_BASE)
        if not isinstance(api_base, str) or not api_base:
            raise ValueError(
                f"model_configs[{index}] ({client_name}) has an empty api_base"
            )
        enable_thinking = config.get("enable_thinking")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise TypeError(
                f"model_configs[{index}] ({client_name}) enable_thinking must "
                f"be bool or None, got {type(enable_thinking).__name__}"
            )

        clients[client_name] = partial(
            gpt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=_resolve_api_key(config, environment),
            enable_thinking=enable_thinking,
        )

    return clients


def _resolve_api_key(
    config: Mapping[str, object], environ: Mapping[str, str]
) -> str:
    """Resolve an entry's key without ever reading or mutating global config."""
    env_name = config.get("api_key_environment_variable")
    if isinstance(env_name, str) and env_name:
        return environ.get(env_name, "")
    explicit = config.get("api_key")
    if isinstance(explicit, str) and explicit:
        return explicit
    return environ.get("OPENAI_API_KEY", "")


def _number(
    config: Mapping[str, object], index: int, key: str, default: float
) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(
            f"model_configs[{index}] {key} must be numeric, got "
            f"{type(value).__name__}"
        )
    try:
        return float(value)
    except ValueError as exc:
        raise TypeError(
            f"model_configs[{index}] {key} must be numeric, got {value!r}"
        ) from exc


def _integer(
    config: Mapping[str, object], index: int, key: str, default: int
) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(
            f"model_configs[{index}] {key} must be an integer, got "
            f"{type(value).__name__}"
        )
    try:
        return int(value)
    except ValueError as exc:
        raise TypeError(
            f"model_configs[{index}] {key} must be an integer, got {value!r}"
        ) from exc


__all__ = ["ClientCallable", "create_clients"]
