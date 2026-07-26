"""vLLM client — OpenAI-compatible inference via vLLM server.

vLLM exposes /v1/chat/completions, so this is a thin wrapper around
the existing api_client.gpt() with vLLM-specific defaults and helpers.
"""

from functools import partial
from typing import Dict, Optional

from federated_mcts.models.api_client import gpt


class VLLMClient:
    """Client for models served via vLLM's OpenAI-compatible API.

    Example:
        client = VLLMClient(
            api_base="http://localhost:8000/v1",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            temperature=0.7,
            max_tokens=512,
        )
        outputs = client(args, prompt, n=3)
    """

    def __init__(
        self,
        api_base: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ):
        self.api_base = api_base
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

    def __call__(self, args, prompt, n=1, stop=None, **kwargs):
        """Call vLLM endpoint via the OpenAI-compatible API."""
        return gpt(
            args,
            prompt,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens if kwargs.get("max_tokens") is None else kwargs["max_tokens"],
            n=n,
            stop=stop,
            api_base=self.api_base,
            api_key=kwargs.get("api_key", "not-needed"),
        )

    @classmethod
    def from_config(cls, config: Dict) -> "VLLMClient":
        """Factory from a model_config dictionary entry.

        Expected config keys:
            - model: model identifier (e.g. "Qwen/Qwen2.5-7B-Instruct")
            - api_base: vLLM server URL (e.g. "http://localhost:8000/v1")
            - temperature (optional)
            - max_tokens (optional)
        """
        return cls(
            api_base=config.get("api_base", "http://localhost:8000/v1"),
            model_name=config["model"],
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 512),
        )


def create_vllm_gpt_function(config: Dict):
    """Create a partial gpt() function configured for a vLLM endpoint.

    This returns a function compatible with the  dict in FederatedSolver.
    Used when model_config has "type": "vllm".
    """
    client = VLLMClient.from_config(config)
    return partial(
        gpt,
        model=config["model"],
        temperature=config.get("temperature", 0.7),
        api_base=config.get("api_base", "http://localhost:8000/v1"),
        api_key=config.get("api_key", "not-needed"),
    )
