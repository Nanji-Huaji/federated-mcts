"""OpenAI-compatible API client for chat completions (OpenAI 0.28 backend).

Uses the 0.28 ``openai.ChatCompletion.create()`` module-level API.
No ``openai.OpenAI``, no ``extra_body``, no v1 response assumptions.
"""

from __future__ import annotations

import threading
from typing import Protocol

import openai

from federated_mcts.models.usage import get_usage_tracker


# ── Protocol types (0.28 response surface) ─────────────────────────────


class _CompletionResponse(Protocol):
    """Minimal shape of a 0.28 ChatCompletion response."""

    choices: list[object]  # iterable of choice-like objects or dicts
    usage: object | None   # usage object / dict, or None


# ── Early-stop function definition ──────────────────────────────────────


EARLY_STOP_FUNCTION: dict[str, object] = {
    "name": "trigger_early_stop",
    "description": (
        "\u5f53\u53d1\u73b0\u5f53\u524d\u601d\u7ef4\u8def\u5f84\u5df2\u7ecf\u8db3\u591f\u597d\u6216\u65e0\u9700\u8fdb\u4e00\u6b65\u63a2\u7d22\u65f6\uff0c"
        "\u89e6\u53d1\u65e9\u505c\u673a\u5236"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "\u89e6\u53d1\u65e9\u505c\u7684\u539f\u56e0",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "\u4fe1\u5fc3\u7a0b\u5ea6",
            },
        },
        "required": ["reason", "confidence"],
    },
}


class EarlyStopException(Exception):
    """Raised when early-stop function is triggered by the model."""

    def __init__(self, reason: str, confidence: float) -> None:
        self.reason = reason
        self.confidence = confidence
        super().__init__(
            f"Early stop triggered: {reason} (confidence: {confidence})"
        )


# ── Client config cache (replaces openai.OpenAI for 0.28) ──────────────


class _ClientConfig:
    """Lightweight config holder — mimics the relevant surface of openai.OpenAI
    so that existing caching and testing contracts remain valid."""

    __slots__ = ("base_url", "api_key")

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key


_client_cache: dict[tuple[str, str], _ClientConfig] = {}
_client_cache_lock = threading.Lock()


def _create_client(
    api_base: str | None = None,
    api_key: str | None = None,
) -> _ClientConfig:
    """Return a cached ``_ClientConfig`` for the given endpoint + key pair."""
    base: str = api_base or "http://127.0.0.1:11451/v1"
    key: str = api_key or "not-needed"
    cache_key = (base, key)
    with _client_cache_lock:
        if cache_key not in _client_cache:
            _client_cache[cache_key] = _ClientConfig(base, key)
        return _client_cache[cache_key]


# ── Completion adapter (0.28 module-level API) ─────────────────────────


def _create_completion(
    client: _ClientConfig,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    n: int,
    stop: object | None = None,
    enable_thinking: bool | None = None,
) -> _CompletionResponse:
    """Thin adapter: ``openai.ChatCompletion.create()`` with per-call config.

    ``enable_thinking`` is only injected into the request body when explicitly
    set, so omitting it leaves the body byte-identical for every endpoint that
    does not use the flag.
    """
    extra: dict[str, object] = {}
    if enable_thinking is not None:
        extra["enable_thinking"] = enable_thinking
    return openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        api_base=client.base_url,
        api_key=client.api_key,
        **extra,
    )


# ── Response helpers (0.28 dict + object compatible) ───────────────────


def _message_field(msg_raw: object, field: str) -> str:
    """Read one string field from a message, tolerating dict/object access.

    Returns ``""`` for missing or ``None`` values so callers never receive the
    literal string ``"None"``.
    """
    if isinstance(msg_raw, dict):
        value: object = msg_raw.get(field, "")
    else:
        value = getattr(msg_raw, field, "")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _extract_content(choice: object) -> str:
    """Extract the model's answer text from a 0.28 response choice.

    Supports both ``OpenAIObject`` attribute access and plain dict access.

    Reasoning endpoints (``deepseek-flash``, ``deepseek-v4-pro``) stream their
    deliberation separately in ``message.reasoning_content`` and only populate
    ``message.content`` if the budget survives the reasoning phase.  When
    ``content`` is empty we fall back to the reasoning stream rather than
    returning ``""``: dropping it makes every such call look like an
    ``empty_content`` parse failure and yields a fabricated 0% accuracy.

    ``content`` always wins when non-empty, so models that populate it normally
    are unaffected.
    """
    if isinstance(choice, dict):
        msg_raw: object = choice.get("message", None)
    else:
        msg_raw = getattr(choice, "message", None)
    if msg_raw is None:
        return ""
    content = _message_field(msg_raw, "content")
    if content:
        return content
    return _message_field(msg_raw, "reasoning_content")


def _to_int(value: object) -> int:
    """Safely convert an unknown object to int (handles int / float / str)."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _extract_tokens(usage: object) -> tuple[int, int]:
    """Return ``(completion_tokens, prompt_tokens)`` from 0.28 usage.

    Supports both ``OpenAIObject`` attribute access and plain dict access.
    """
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        comp_raw: object = usage.get("completion_tokens", 0)
        prompt_raw: object = usage.get("prompt_tokens", 0)
    else:
        comp_raw = getattr(usage, "completion_tokens", 0)
        prompt_raw = getattr(usage, "prompt_tokens", 0)
    return _to_int(comp_raw), _to_int(prompt_raw)


# ── Public API ─────────────────────────────────────────────────────────


def gpt(
    args: object,
    prompt: str,
    model: str = "gpt-4",
    temperature: float = 0.9,
    max_tokens: int = 1000,
    n: int = 1,
    stop: object | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    get_logprobs: bool = False,
    top_logprobs: int = 0,
    enable_thinking: bool | None = None,
) -> list[str]:
    """Convenience wrapper: wraps a single user prompt into a chat message."""
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(
        args,
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        api_base=api_base,
        api_key=api_key,
        enable_thinking=enable_thinking,
    )


def chatgpt(
    args: object,
    messages: list[dict[str, str]],
    model: str = "gpt-4",
    temperature: float = 0.5,
    max_tokens: int = 1000,
    n: int = 1,
    stop: object | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    enable_thinking: bool | None = None,
) -> list[str]:
    """Send a chat-completion request via the 0.28 module-level API.

    Some families enforce ``max_choices=1`` and split ``n > 1`` into individual
    ``ChatCompletion.create`` calls:

    - ``deepseek-v4-*`` enforces ``max_choices=1`` unconditionally.
    - ``qwen3*`` models run with thinking enabled by default, where the gateway
      rejects batched requests outright:
      ``The n parameter must be 1 when enable_thinking is true``.
      Passing ``enable_thinking=False`` lifts that restriction, so batching is
      restored rather than needlessly serialised.
    """
    client: _ClientConfig = _create_client(api_base=api_base, api_key=api_key)
    usage_tracker = get_usage_tracker()
    outputs: list[str] = []

    thinking_off: bool = enable_thinking is False
    single_choice_only: bool = model.startswith("deepseek-v4-") or (
        model.startswith("qwen3") and not thinking_off
    )
    max_choices: int = 1 if single_choice_only else 20
    thinking_kwargs: dict[str, bool] = (
        {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    )
    remaining: int = n
    while remaining > 0:
        cnt: int = min(remaining, max_choices)
        remaining -= cnt
        res = _create_completion(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=cnt,
            stop=stop,
            **thinking_kwargs,
        )
        outputs.extend(_extract_content(choice) for choice in res.choices)

        comp_tokens, prompt_tokens = _extract_tokens(res.usage)
        if comp_tokens or prompt_tokens:
            usage_tracker.record(model, comp_tokens, prompt_tokens)

    return outputs
