"""Opt-in, one-request diagnostic for the configured DeepSeek endpoint."""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from federated_mcts.models.api_client import gpt


class TestDeepSeekLiveDiagnostic(unittest.TestCase):
    def test_configured_remote_client_reports_bounded_response_facts(self):
        if os.environ.get("RUN_LIVE_DEEPSEEK_DIAGNOSTIC") != "1":
            self.skipTest(
                "set RUN_LIVE_DEEPSEEK_DIAGNOSTIC=1 to send the one live DeepSeek diagnostic request"
            )

        config_path = Path(__file__).parents[1] / "configs" / "model_config_b0.json"
        with config_path.open(encoding="utf-8") as config_file:
            model_configs = json.load(config_file)
        remote_client = next(
            config
            for config in model_configs
            if config["client_name"] == "remote_client"
        )
        api_key_environment_variable = remote_client["api_key_environment_variable"]
        api_key = os.environ.get(api_key_environment_variable)
        if not api_key:
            self.skipTest(
                f"configured API key is unavailable: {api_key_environment_variable}"
            )

        outputs = gpt(
            None,
            "Reply with hello.",
            model=remote_client["model"],
            temperature=0,
            max_tokens=32,
            n=1,
            api_base=remote_client["api_base"],
            api_key=api_key,
        )

        first_content = outputs[0] if outputs else ""
        first_content_empty = None if not outputs else first_content == ""
        print(
            "deepseek_live_diagnostic "
            f"output_count={len(outputs)} "
            f"first_content_empty={first_content_empty} "
            f"first_content_length={len(first_content)} "
            f"first_content_prefix={first_content[:80]!r}"
        )
