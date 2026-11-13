"""Regression tests: reasoning-model output must not be silently dropped.

PRODUCTION DEFECT (measured against the live endpoint):

Reasoning endpoints (``deepseek-flash``, ``deepseek-v4-pro``) return a separate
``message.reasoning_content`` stream alongside ``message.content``.  Measured
behaviour for the Game24 propose prompt:

    max_tokens=2048  -> finish_reason=length, reasoning_tokens=2048, content=''
    max_tokens=16384 -> finish_reason=length, reasoning_tokens=16384, content=''

``_extract_content`` reads only ``message.content``, so every one of those calls
was surfaced to the solver as the empty string, which the Game24 local validator
then classified as ``empty_content``.  A run in that state completes "successfully"
while reporting a fabricated 0% accuracy.

Expected contract:
- ``content`` wins whenever it is non-empty (no behaviour change for models that
  populate it normally).
- When ``content`` is empty, the reasoning stream is used instead of silently
  returning ``""``.
- Both dict-style and attribute-style (openai ``OpenAIObject``) choices work.
"""

import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.models.api_client import _extract_content


def _dict_choice(content: str, reasoning: str | None = None) -> dict:
    message: dict[str, object] = {"content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"message": message}


def _obj_choice(content: str, reasoning: str | None = None) -> object:
    """Attribute-style choice, as returned by the openai 0.28 client."""
    fields: dict[str, object] = {"content": content}
    if reasoning is not None:
        fields["reasoning_content"] = reasoning
    return type("Choice", (), {"message": type("Message", (), fields)()})()


class TestReasoningContentFallback(unittest.TestCase):
    """Given reasoning-model responses, When extracting, Then nothing is lost."""

    def test_empty_content_falls_back_to_reasoning_dict(self):
        choice = _dict_choice("", "We need answer user. Need solve game 24...")
        self.assertEqual(
            _extract_content(choice),
            "We need answer user. Need solve game 24...",
            "DEFECT: reasoning stream dropped when content is empty; the solver "
            "sees '' and records a fabricated empty_content parse failure.",
        )

    def test_empty_content_falls_back_to_reasoning_object(self):
        choice = _obj_choice("", "reasoning trace")
        self.assertEqual(_extract_content(choice), "reasoning trace")

    def test_non_empty_content_still_wins(self):
        choice = _dict_choice("8 * (1 + 1 + 1) = 24 (left: 24)", "long trace")
        self.assertEqual(
            _extract_content(choice),
            "8 * (1 + 1 + 1) = 24 (left: 24)",
            "Content must win when present: existing non-reasoning models and any "
            "judge/vote parsing depend on this.",
        )

    def test_both_empty_returns_empty(self):
        self.assertEqual(_extract_content(_dict_choice("", "")), "")
        self.assertEqual(_extract_content(_dict_choice("")), "")

    def test_missing_message_returns_empty(self):
        self.assertEqual(_extract_content({}), "")
        self.assertEqual(_extract_content(object()), "")

    def test_none_values_do_not_crash(self):
        choice = {"message": {"content": None, "reasoning_content": None}}
        self.assertEqual(_extract_content(choice), "")


if __name__ == "__main__":
    unittest.main()
