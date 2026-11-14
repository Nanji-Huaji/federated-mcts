"""Public-interface contracts for the model runtime.

Covers the four independently usable pieces of the models package:

- import weight (HTTP/usage must not import torch or transformers),
- per-run usage scopes (``usage_scope`` / ``get_usage_tracker``),
- the versioned :class:`RateCard` and its exact raw-unit billing,
- the ``create_clients`` factory and the existing ``gpt`` callable behaviour
  (reasoning fallback, provider ``n`` splitting, ``enable_thinking`` plumbing).

All tests are offline: no endpoint is contacted and no model is loaded.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from federated_mcts.models import api_client
from federated_mcts.models.clients import create_clients
from federated_mcts.models.rates import (
    DEFAULT_RATE_CARD,
    LEGACY_UNKNOWN_MODEL_RATE,
    RATE_CARD_VERSION,
    ModelRate,
    RateCard,
    UnknownModelRateError,
    cost_raw_units,
)
from federated_mcts.models.usage import (
    UsageTracker,
    get_model_usage_summary,
    get_usage_tracker,
    gpt_usage,
    usage_scope,
)
from federated_mcts.models.vllm_client import VLLMClient


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing if existing else "")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


class TestLightweightImports(unittest.TestCase):
    """Given only HTTP/usage clients are needed, Then no torch is imported."""

    def test_models_package_does_not_import_ml_frameworks(self) -> None:
        code = """
import sys
import federated_mcts.models
import federated_mcts.models.api_client
import federated_mcts.models.clients
import federated_mcts.models.rates
import federated_mcts.models.usage
import federated_mcts.models.vllm_client
for banned in ("torch", "transformers"):
    if banned in sys.modules:
        raise SystemExit(f"{banned} was imported by the models package")
if "federated_mcts.federation" in sys.modules:
    raise SystemExit("models package imported federated_mcts.federation")
print("light")
"""
        result = _run_python(code)
        self.assertEqual(
            result.returncode,
            0,
            "DEFECT: importing the HTTP/usage model runtime loads torch/"
            f"transformers.\nstdout={result.stdout}\nstderr={result.stderr}",
        )
        self.assertIn("light", result.stdout)

    @unittest.skipIf(
        importlib.util.find_spec("torch") is None,
        "torch is not installed; lazy local-inference import is unavailable",
    )
    def test_lazy_local_inference_names_still_resolve(self) -> None:
        code = """
from federated_mcts.models import DraftModel, UncertaintyInferenceFramework
assert DraftModel is UncertaintyInferenceFramework
print(UncertaintyInferenceFramework.__name__)
"""
        result = _run_python(code)
        self.assertEqual(
            result.returncode,
            0,
            "DEFECT: lazy __getattr__ broke the public local-inference names.\n"
            f"stdout={result.stdout}\nstderr={result.stderr}",
        )
        self.assertIn("UncertaintyInferenceFramework", result.stdout)


class TestUsageScope(unittest.TestCase):
    """Given two logical runs in one process, Then usage stays isolated."""

    def test_unscoped_lookup_returns_the_legacy_global(self) -> None:
        first = get_usage_tracker()
        self.assertIs(first, get_usage_tracker())
        with usage_scope():
            pass
        self.assertIs(get_usage_tracker(), first)

    def test_implicit_scope_creates_a_fresh_tracker(self) -> None:
        with usage_scope() as first:
            self.assertIs(get_usage_tracker(), first)
        with usage_scope() as second:
            self.assertIsNot(first, second)
            self.assertIs(get_usage_tracker(), second)

    def test_explicit_tracker_is_yielded_and_restored(self) -> None:
        outer = get_usage_tracker()
        mine = UsageTracker()
        with usage_scope(mine) as yielded:
            self.assertIs(yielded, mine)
            self.assertIs(get_usage_tracker(), mine)
            mine.record("scope-explicit-model", 4, 5)
            self.assertEqual(
                get_model_usage_summary()["scope-explicit-model"]["total_tokens"], 9
            )
        self.assertIs(get_usage_tracker(), outer)
        self.assertNotIn("scope-explicit-model", get_model_usage_summary())

    def test_nested_scope_restores_the_parent(self) -> None:
        with usage_scope() as outer:
            with usage_scope() as inner:
                self.assertIs(get_usage_tracker(), inner)
                self.assertIsNot(inner, outer)
            self.assertIs(get_usage_tracker(), outer)

    def test_scope_is_restored_after_exception(self) -> None:
        global_tracker = get_usage_tracker()
        with self.assertRaises(RuntimeError):
            with usage_scope() as scoped:
                self.assertIs(get_usage_tracker(), scoped)
                raise RuntimeError("boom")
        self.assertIs(get_usage_tracker(), global_tracker)

    def test_scope_is_isolated_from_a_worker_thread(self) -> None:
        global_tracker = get_usage_tracker()
        seen: list[UsageTracker] = []

        with usage_scope() as scoped:
            thread = threading.Thread(
                target=lambda: seen.append(get_usage_tracker())
            )
            thread.start()
            thread.join(timeout=10)
            self.assertIs(get_usage_tracker(), scoped)

        self.assertEqual(len(seen), 1)
        self.assertIs(
            seen[0],
            global_tracker,
            "ContextVar scopes must not leak into unrelated threads; explicit "
            "context propagation is the caller's responsibility.",
        )

    def test_legacy_helpers_read_the_active_scope(self) -> None:
        with usage_scope() as scoped:
            scoped.record("scope-helper-model", 10, 20)
            self.assertIn("scope-helper-model", get_model_usage_summary())
            aggregate = gpt_usage()
            self.assertIn("scope-helper-model", aggregate["model_usage"])
        self.assertNotIn("scope-helper-model", get_model_usage_summary())


class TestRateCard(unittest.TestCase):
    """Given a priced call, Then billing is exact, versioned, and explicit."""

    def test_version_is_exposed(self) -> None:
        self.assertEqual(DEFAULT_RATE_CARD.version, RATE_CARD_VERSION)
        self.assertTrue(RATE_CARD_VERSION)

    def test_known_model_cost_is_exact_decimal(self) -> None:
        cost = DEFAULT_RATE_CARD.cost_raw_units(
            "deepseek-v4-pro", 1_000_000, 1_000_000
        )
        self.assertEqual(cost, Decimal("1.305"))
        self.assertIsInstance(cost, Decimal)

    def test_fractional_cost_has_no_binary_float_drift(self) -> None:
        cost = DEFAULT_RATE_CARD.cost_raw_units("deepseek-v4-flash", 500, 1000)
        self.assertEqual(cost, Decimal("0.00028"))

    def test_zero_cost_local_model_is_known(self) -> None:
        self.assertEqual(
            DEFAULT_RATE_CARD.cost_raw_units(
                "Qwen/Qwen2.5-32B-Instruct", 1_000_000, 1_000_000
            ),
            Decimal("0"),
        )

    def test_unknown_model_is_rejected_for_billing(self) -> None:
        with self.assertRaises(UnknownModelRateError) as ctx:
            DEFAULT_RATE_CARD.cost_raw_units("mystery-model", 1, 1)
        self.assertEqual(ctx.exception.model, "mystery-model")
        self.assertEqual(ctx.exception.rate_card_version, RATE_CARD_VERSION)
        self.assertIsInstance(
            ctx.exception,
            ValueError,
            "UnknownModelRateError must stay a ValueError so existing "
            "fail-closed callers keep catching it.",
        )
        self.assertIn("Unknown billed model", str(ctx.exception))

    def test_unknown_model_uses_explicit_legacy_display_fallback(self) -> None:
        cost = DEFAULT_RATE_CARD.cost_raw_units(
            "mystery-model", 1_000_000, 1_000_000, strict=False
        )
        self.assertEqual(cost, Decimal("3.0"))
        self.assertEqual(
            DEFAULT_RATE_CARD.rate_for("mystery-model", strict=False),
            LEGACY_UNKNOWN_MODEL_RATE,
        )

    def test_pure_function_matches_the_card(self) -> None:
        self.assertEqual(
            cost_raw_units("deepseek-v4-pro", 1_000_000, 1_000_000),
            DEFAULT_RATE_CARD.cost_raw_units(
                "deepseek-v4-pro", 1_000_000, 1_000_000
            ),
        )

    def test_invalid_token_counts_are_rejected(self) -> None:
        for bad in (True, 1.5, "10"):
            with self.assertRaises(TypeError):
                DEFAULT_RATE_CARD.cost_raw_units("deepseek-v4-pro", bad, 0)
        with self.assertRaises(ValueError):
            DEFAULT_RATE_CARD.cost_raw_units("deepseek-v4-pro", -1, 0)

    def test_rates_mapping_is_immutable(self) -> None:
        with self.assertRaises(TypeError):
            DEFAULT_RATE_CARD.rates["new-model"] = ModelRate(  # type: ignore[index]
                prompt=Decimal("1"), completion=Decimal("1")
            )

    def test_custom_card_isolated_from_the_default(self) -> None:
        tiny = RateCard(
            version="test-1",
            rates={"demo": ModelRate(Decimal("2"), Decimal("3"))},
        )
        self.assertEqual(tiny.cost_raw_units("demo", 1_000_000, 1_000_000), Decimal("5"))
        with self.assertRaises(UnknownModelRateError):
            tiny.cost_raw_units("deepseek-v4-pro", 1, 1)


class TestUsageTrackerBilling(unittest.TestCase):
    """Given usage records, Then the tracker delegates to the rate card."""

    def test_summary_cost_uses_rate_card(self) -> None:
        tracker = UsageTracker()
        tracker.record("deepseek-v4-pro", 1_000_000, 1_000_000)
        summary = tracker.get_summary()
        self.assertAlmostEqual(summary["deepseek-v4-pro"]["cost"], 1.305)
        self.assertEqual(
            tracker.get_total_cost_raw_units(), Decimal("1.305")
        )

    def test_unknown_model_keeps_legacy_display_price(self) -> None:
        tracker = UsageTracker()
        tracker.record("mystery-model", 1_000_000, 1_000_000)
        self.assertAlmostEqual(tracker.get_summary()["mystery-model"]["cost"], 3.0)

    def test_static_cost_helper_matches_public_interface(self) -> None:
        self.assertAlmostEqual(
            UsageTracker._calculate_cost("deepseek-v4-pro", 1_000_000, 1_000_000),
            1.305,
        )

    def test_tracker_can_use_a_custom_rate_card(self) -> None:
        card = RateCard(
            version="test-card",
            rates={"demo": ModelRate(Decimal("2"), Decimal("3"))},
        )
        tracker = UsageTracker(rate_card=card)
        tracker.record("demo", 1_000_000, 1_000_000)
        self.assertAlmostEqual(tracker.get_summary()["demo"]["cost"], 5.0)
        self.assertEqual(tracker.get_total_cost_raw_units(), Decimal("5"))


class _FakeChoice:
    def __init__(self, content: str, reasoning: str = "") -> None:
        self.message = {"content": content, "reasoning_content": reasoning}


class _FakeResponse:
    def __init__(
        self,
        contents: list[str],
        *,
        usage: object | None = None,
    ) -> None:
        self.choices = [_FakeChoice(content) for content in contents]
        self.usage = usage


class TestGptCallableContracts(unittest.TestCase):
    """Given the existing gpt callable, Then its measured behaviour is kept."""

    def test_reasoning_content_is_used_when_content_is_empty(self) -> None:
        choice = {"message": {"content": "", "reasoning_content": "trace"}}
        self.assertEqual(api_client._extract_content(choice), "trace")
        self.assertEqual(
            api_client._extract_content(
                {"message": {"content": "answer", "reasoning_content": "trace"}}
            ),
            "answer",
        )

    def test_provider_choice_splitting_and_thinking_passthrough(self) -> None:
        calls: list[dict] = []

        def fake_completion(client, **kwargs):
            calls.append(kwargs)
            return _FakeResponse(["out"] * kwargs["n"], usage=None)

        def run(model: str, n: int, **kwargs) -> list[str]:
            calls.clear()
            with mock.patch.object(api_client, "_create_client"), mock.patch.object(
                api_client, "_create_completion", side_effect=fake_completion
            ):
                return api_client.chatgpt(
                    object(),
                    [{"role": "user", "content": "hi"}],
                    model=model,
                    n=n,
                    **kwargs,
                )

        run("qwen3.7-plus", 3)
        self.assertEqual([c["n"] for c in calls], [1, 1, 1])
        self.assertNotIn("enable_thinking", calls[0])

        run("qwen3.7-plus", 3, enable_thinking=False)
        self.assertEqual([c["n"] for c in calls], [3])
        self.assertIs(calls[0]["enable_thinking"], False)

        run("deepseek-v4-pro", 2)
        self.assertEqual([c["n"] for c in calls], [1, 1])

        run("gpt-4o", 3)
        self.assertEqual([c["n"] for c in calls], [3])

    def test_api_calls_record_into_the_active_scope(self) -> None:
        global_before = get_model_usage_summary()

        def fake_completion(client, **kwargs):
            return _FakeResponse(
                ["out"] * kwargs["n"],
                usage={"completion_tokens": 7, "prompt_tokens": 11},
            )

        with mock.patch.object(api_client, "_create_client"), mock.patch.object(
            api_client, "_create_completion", side_effect=fake_completion
        ):
            with usage_scope() as scoped:
                outputs = api_client.gpt(
                    object(),
                    "hi",
                    model="scope-api-model",
                    n=1,
                )
                self.assertEqual(outputs, ["out"])
                recorded = scoped.get_summary()["scope-api-model"]
                self.assertEqual(recorded["completion_tokens"], 7)
                self.assertEqual(recorded["prompt_tokens"], 11)
                self.assertEqual(recorded["total_calls"], 1)

        self.assertEqual(get_model_usage_summary(), global_before)


class TestCreateClients(unittest.TestCase):
    """Given parsed config entries, Then clients are built without file I/O."""

    def _configs(self) -> list[dict[str, object]]:
        return [
            {
                "client_name": "local_client",
                "model": "Qwen/Qwen2.5-32B-Instruct",
                "type": "vllm",
                "api_base": "http://localhost:8000/v1",
                "temperature": 0.3,
                "max_tokens": 256,
            },
            {
                "client_name": "remote_client",
                "model": "deepseek-v4-flash",
                "api_base": "https://api.deepseek.com/v1",
                "api_key_environment_variable": "DEEPSEEK_API_KEY",
                "temperature": 0.7,
                "max_tokens": 2048,
            },
        ]

    def test_builds_local_and_remote_clients(self) -> None:
        configs = self._configs()
        clients = create_clients(configs, environ={"DEEPSEEK_API_KEY": "secret"})
        self.assertEqual(set(clients), {"local_client", "remote_client"})
        self.assertIsInstance(clients["local_client"], VLLMClient)
        remote = clients["remote_client"]
        assert hasattr(remote, "keywords")
        self.assertEqual(remote.keywords["model"], "deepseek-v4-flash")
        self.assertEqual(remote.keywords["api_key"], "secret")
        self.assertEqual(remote.keywords["temperature"], 0.7)
        self.assertEqual(remote.keywords["max_tokens"], 2048)

    def test_config_entries_are_not_mutated(self) -> None:
        configs = self._configs()
        before = [dict(entry) for entry in configs]
        create_clients(configs, environ={"DEEPSEEK_API_KEY": "secret"})
        self.assertEqual(configs, before)
        self.assertNotIn("api_key", configs[1])

    def test_remote_callable_uses_the_existing_gpt_shape(self) -> None:
        configs = self._configs()
        with mock.patch("federated_mcts.models.clients.gpt") as fake_gpt:
            fake_gpt.return_value = ["generated"]
            clients = create_clients(
                configs, environ={"DEEPSEEK_API_KEY": "secret"}
            )
            outputs = clients["remote_client"](object(), "prompt", n=2)

        self.assertEqual(outputs, ["generated"])
        (args, prompt), kwargs = fake_gpt.call_args
        self.assertIsNotNone(args)
        self.assertEqual(prompt, "prompt")
        self.assertEqual(kwargs["model"], "deepseek-v4-flash")
        self.assertEqual(kwargs["api_base"], "https://api.deepseek.com/v1")
        self.assertEqual(kwargs["api_key"], "secret")
        self.assertIsNone(kwargs["enable_thinking"])

    def test_default_temperature_applies_only_when_entry_omits_it(self) -> None:
        configs = [{"client_name": "remote", "model": "deepseek-v4-flash"}]
        clients = create_clients(
            configs, default_temperature=0.11, environ={}
        )
        assert hasattr(clients["remote"], "keywords")
        self.assertEqual(clients["remote"].keywords["temperature"], 0.11)
        self.assertEqual(clients["remote"].keywords["api_key"], "")

    def test_explicit_api_key_and_openai_environment_fallback(self) -> None:
        explicit = create_clients(
            [
                {
                    "client_name": "explicit",
                    "model": "deepseek-v4-flash",
                    "api_key": "direct-key",
                }
            ],
            environ={"OPENAI_API_KEY": "env-key"},
        )
        assert hasattr(explicit["explicit"], "keywords")
        self.assertEqual(explicit["explicit"].keywords["api_key"], "direct-key")

        fallback = create_clients(
            [{"client_name": "fallback", "model": "deepseek-v4-flash"}],
            environ={"OPENAI_API_KEY": "env-key"},
        )
        assert hasattr(fallback["fallback"], "keywords")
        self.assertEqual(fallback["fallback"].keywords["api_key"], "env-key")

    def test_enable_thinking_is_forwarded_when_present(self) -> None:
        clients = create_clients(
            [
                {
                    "client_name": "qwen",
                    "model": "qwen3.7-plus",
                    "enable_thinking": False,
                }
            ],
            environ={},
        )
        assert hasattr(clients["qwen"], "keywords")
        self.assertIs(clients["qwen"].keywords["enable_thinking"], False)

    def test_duplicate_client_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            create_clients(
                [
                    {"client_name": "same", "model": "a"},
                    {"client_name": "same", "model": "b"},
                ],
                environ={},
            )

    def test_missing_required_fields_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            create_clients([{"model": "a"}], environ={})
        with self.assertRaises(ValueError):
            create_clients([{"client_name": "a"}], environ={})


if __name__ == "__main__":
    unittest.main()
