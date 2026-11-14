"""Regression test for the isolated DeepSeek-2048 B0 configuration."""

from __future__ import annotations

import json
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).parent.parent
_B0_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "model_config_b0.json"
_DEEPSEEK_2048_CONFIG_PATH = (
    _REPOSITORY_ROOT / "configs" / "model_config_b0_deepseek_2048.json"
)


def test_deepseek_2048_config_matches_b0_except_remote_max_tokens() -> None:
    """Given B0, when deriving the DeepSeek variant, then only its token limit changes."""
    # Given
    expected_config = json.loads(_B0_CONFIG_PATH.read_text())
    expected_config[1]["max_tokens"] = 2048

    # When
    deepseek_2048_config = json.loads(_DEEPSEEK_2048_CONFIG_PATH.read_text())

    # Then
    assert deepseek_2048_config == expected_config
