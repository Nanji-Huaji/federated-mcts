"""Shared test fixtures for Game24 instrumentation tests.

Not discovered by ``unittest discover`` (leading underscore).
"""

from unittest.mock import MagicMock

# Stub openai.OpenAI for environments with openai<1.0
try:
    import openai  # noqa: F401
    if not hasattr(openai, "OpenAI"):
        openai.OpenAI = type("OpenAI", (), {})  # type: ignore[attr-defined]
except ImportError:
    pass


class ArgsStub:
    check_format = True
    eval_rule = False


def make_game24_task_stub():
    """Return a MagicMock that behaves like Game24Task for generation."""
    task = MagicMock()
    task.pre_generate_check.return_value = True
    task.propose_prompt_wrap.return_value = "mock prompt"

    def process(pro, x, y, check):
        stripped = pro.strip()
        if not stripped:
            return False, pro
        if "badformat" in stripped:
            return False, pro
        updated = y + stripped + "\n"
        return True, updated

    task.process_generate_result.side_effect = process
    task.vote_prompt_wrap.return_value = "vote prompt"
    return task


def make_vote_task_stub():
    """Return a MagicMock for vote evaluation that mirrors Game24Task."""
    task = MagicMock()
    task.vote_prompt_wrap.return_value = "vote prompt"

    def unwrap(vote_outputs, n_candidates):
        import re
        pattern = r".*best choice is.*?(?<!-)(\d+)"
        results = [0] * n_candidates
        for raw in vote_outputs:
            match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
            if match:
                idx = int(match.group(1))
                if 0 <= idx < n_candidates:
                    results[idx] += 1
        return results

    task.vote_outputs_unwrap.side_effect = unwrap
    return task
