"""Regression tests: models that reject ``n > 1`` must be split into ``n = 1`` calls.

PRODUCTION DEFECT (measured against the live LLM gateway):

``qwen3.7-plus`` runs with thinking enabled by default. The gateway rejects any
batched request under that mode:

    <400> InternalError.Algo.InvalidParameter:
    The n parameter must be 1 when enable_thinking is true

``chatgpt`` already splits ``n > 1`` into sequential ``n = 1`` calls for the
``deepseek-v4-`` family, which enforces the equivalent ``max_choices = 1``
restriction.  ``qwen3.7-plus`` hits the same restriction but was not covered, so
the canonical Game24 pilot aborted on its first proposal call even after the
sampling counts were lowered to 1 — because ``get_proposals`` requests 4
completions regardless of ``--n_generate_sample``.

Expected contract:
- Models that reject batched choices are split into ``n = 1`` calls, preserving
  the total number of returned completions.
- Models without the restriction still batch into a single call.
- Usage is accumulated across the split calls.
"""

import sys
import unittest
from unittest import mock

sys.path.insert(0, "src")

from federated_mcts.models import api_client


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = {"content": content, "reasoning_content": ""}


class _FakeResponse:
    def __init__(self, contents: list[str]) -> None:
        self.choices = [_FakeChoice(c) for c in contents]
        self.usage = {"completion_tokens": 10, "prompt_tokens": 5}


class TestBatchedChoiceSplitting(unittest.TestCase):
    """Given a model that rejects n>1, When n>1 is requested, Then split."""

    def _run(self, model: str, n: int) -> list[dict]:
        calls: list[dict] = []

        def fake_create(client, **kwargs):
            calls.append(kwargs)
            return _FakeResponse([f"out{kwargs['n']}"] * kwargs["n"])

        with mock.patch.object(api_client, "_create_completion", fake_create), \
             mock.patch.object(api_client, "_create_client"), \
             mock.patch.object(api_client, "get_usage_tracker") as tracker:
            tracker.return_value = mock.Mock()
            outputs = api_client.chatgpt(
                object(), [{"role": "user", "content": "hi"}],
                model=model, n=n, temperature=0.9, max_tokens=64,
            )
        self.calls = calls
        return outputs

    def test_qwen_thinking_model_is_split_into_single_choice_calls(self):
        """qwen3.7-plus must be split; n>1 is rejected by the gateway."""
        outputs = self._run("qwen3.7-plus", 4)
        self.assertEqual(
            [c["n"] for c in self.calls], [1, 1, 1, 1],
            "DEFECT: qwen3.7-plus rejects n>1 under thinking mode; the client "
            "must issue n=1 calls instead of aborting the run.",
        )
        self.assertEqual(len(outputs), 4, "splitting must preserve completion count")

    def test_deepseek_split_behaviour_is_unchanged(self):
        outputs = self._run("deepseek-v4-flash", 4)
        self.assertEqual([c["n"] for c in self.calls], [1, 1, 1, 1])
        self.assertEqual(len(outputs), 4)

    def test_unrestricted_model_still_batches(self):
        """A model with no restriction must not be needlessly serialised."""
        outputs = self._run("gpt-4o", 3)
        self.assertEqual(
            [c["n"] for c in self.calls], [3],
            "DEFECT: unrestricted models must keep batching into one call",
        )
        self.assertEqual(len(outputs), 3)

    def test_single_choice_request_makes_one_call(self):
        outputs = self._run("qwen3.7-plus", 1)
        self.assertEqual([c["n"] for c in self.calls], [1])
        self.assertEqual(len(outputs), 1)


class TestEnableThinkingPassthrough(unittest.TestCase):
    """Given a config that disables thinking, When calling, Then forward it.

    The gateway rejects every request for a thinking-mode model unless the caller
    either sets ``enable_thinking: false`` or caps ``n`` at 1:

        <400> InternalError.Algo.InvalidParameter:
        The n parameter must be 1 when enable_thinking is true

    Disabling thinking restores the framework's intended batched sampling
    (``n_generate_sample`` / ``n_evaluate_sample``) and stable low latency.

    Expected contract:
    - ``enable_thinking=False`` is forwarded into the request body.
    - Omitting it (``None``) must not add the key at all, so every existing
      endpoint keeps byte-identical request bodies.
    """

    def _run(self, **kwargs) -> list[dict]:
        calls: list[dict] = []

        def fake_create(client, **kw):
            calls.append(kw)
            return _FakeResponse(["out"])

        with mock.patch.object(api_client, "_create_completion", fake_create), \
             mock.patch.object(api_client, "_create_client"), \
             mock.patch.object(api_client, "get_usage_tracker") as tracker:
            tracker.return_value = mock.Mock()
            api_client.chatgpt(
                object(), [{"role": "user", "content": "hi"}],
                model="qwen3.7-plus", n=1, temperature=0.9, max_tokens=64,
                **kwargs,
            )
        return calls

    def test_enable_thinking_false_is_forwarded(self):
        calls = self._run(enable_thinking=False)
        self.assertIs(
            calls[0].get("enable_thinking"), False,
            "DEFECT: enable_thinking=False must reach the request body, otherwise "
            "the gateway rejects every call for a thinking-mode model.",
        )

    def test_omitted_enable_thinking_adds_no_key(self):
        calls = self._run()
        self.assertNotIn(
            "enable_thinking", calls[0],
            "DEFECT: omitting enable_thinking must not inject the key, otherwise "
            "existing endpoints receive a changed request body.",
        )

    def test_gpt_forwards_enable_thinking(self):
        """The partial-bound ``gpt`` wrapper must forward the flag too."""
        calls: list[dict] = []

        def fake_create(client, **kw):
            calls.append(kw)
            return _FakeResponse(["out"])

        with mock.patch.object(api_client, "_create_completion", fake_create), \
             mock.patch.object(api_client, "_create_client"), \
             mock.patch.object(api_client, "get_usage_tracker") as tracker:
            tracker.return_value = mock.Mock()
            api_client.gpt(
                object(), "hi", model="qwen3.7-plus", n=1,
                enable_thinking=False,
            )
        self.assertIs(calls[0].get("enable_thinking"), False)


if __name__ == "__main__":
    unittest.main()
