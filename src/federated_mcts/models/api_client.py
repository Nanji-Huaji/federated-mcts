"""OpenAI-compatible API client for chat completions."""

import openai
import backoff
from typing import List, Optional

from federated_mcts.models.usage import get_usage_tracker

# Default API configuration (overridable per-call)
_default_api_base = "http://127.0.0.1:11451/v1"
_default_api_key = "lm-studio"

openai.api_base = _default_api_base
openai.api_key = _default_api_key

EARLY_STOP_FUNCTION = {
    "name": "trigger_early_stop",
    "description": "当发现当前思维路径已经足够好或无需进一步探索时，触发早停机制直接跳到最终总结步骤",
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "触发早停的原因"
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "对当前思维质量的信心程度"
            }
        },
        "required": ["reason", "confidence"]
    }
}


class EarlyStopException(Exception):
    def __init__(self, reason: str, confidence: float):
        self.reason = reason
        self.confidence = confidence
        super().__init__(f"Early stop triggered: {reason} (confidence: {confidence})")


@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


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
        args,
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        api_base=api_base,
        api_key=api_key,
    )


def chatgpt(
    args,
    messages,
    model: str = "gpt-4",
    temperature: float = 0.5,
    max_tokens: int = 1000,
    n: int = 1,
    stop=None,
    api_base=None,
    api_key=None,
) -> List[str]:
    if api_base is None:
        api_base = openai.api_base
    if api_key is None:
        api_key = openai.api_key

    usage_tracker = get_usage_tracker()
    outputs = []

    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=cnt,
            stop=stop,
            api_base=api_base,
            api_key=api_key,
        )
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])

        # Track token usage
        completion_tokens_used = res["usage"]["completion_tokens"]
        prompt_tokens_used = res["usage"]["prompt_tokens"]
        usage_tracker.record(model, completion_tokens_used, prompt_tokens_used)

    return outputs
