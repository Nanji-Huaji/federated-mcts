"""OpenAI-compatible API client for chat completions."""

import openai
import threading
from typing import List, Optional

from federated_mcts.models.usage import get_usage_tracker

EARLY_STOP_FUNCTION = {
    "name": "trigger_early_stop",
    "description": "当发现当前思维路径已经足够好或无需进一步探索时，触发早停机制",
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "触发早停的原因"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "信心程度"}
        },
        "required": ["reason", "confidence"]
    }
}


class EarlyStopException(Exception):
    def __init__(self, reason: str, confidence: float):
        self.reason = reason
        self.confidence = confidence
        super().__init__(f"Early stop triggered: {reason} (confidence: {confidence})")


_client_cache = {}
_client_cache_lock = threading.Lock()

def _create_client(api_base: Optional[str] = None, api_key: Optional[str] = None) -> openai.OpenAI:
    base = api_base or "http://127.0.0.1:11451/v1"
    key = api_key or "not-needed"
    cache_key = (base, key)
    with _client_cache_lock:
        if cache_key not in _client_cache:
            _client_cache[cache_key] = openai.OpenAI(api_key=key, base_url=base, max_retries=3)
        return _client_cache[cache_key]


def _create_completion(client: openai.OpenAI, **kwargs):
    return client.chat.completions.create(**kwargs)


def gpt(
    args,
    prompt: str,
    model: Optional[str] = "gpt-4",
    temperature: float = 0.9,
    max_tokens: int = 1000,
    n: int = 1,
    stop=None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    get_logprobs: bool = False,
    top_logprobs: int = 0,
) -> List[str]:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(
        args, messages, model=model, temperature=temperature,
        max_tokens=max_tokens, n=n, stop=stop,
        api_base=api_base, api_key=api_key,
    )


def chatgpt(
    args,
    messages,
    model: str = "gpt-4",
    temperature: float = 0.5,
    max_tokens: int = 1000,
    n: int = 1,
    stop=None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[str]:
    client = _create_client(api_base=api_base, api_key=api_key)
    usage_tracker = get_usage_tracker()
    outputs = []

    max_choices = 1 if model.startswith("deepseek-v4-") else 20
    while n > 0:
        cnt = min(n, max_choices)
        n -= cnt
        res = _create_completion(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=cnt,
            stop=stop,
            extra_body={"thinking": {"type": "disabled"}} if model.startswith("deepseek") else None,
        )
        outputs.extend([choice.message.content for choice in res.choices])

        if res.usage:
            usage_tracker.record(
                model, res.usage.completion_tokens, res.usage.prompt_tokens
            )

    return outputs
