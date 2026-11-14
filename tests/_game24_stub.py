"""Shared Game24 test stub — private module, not discovered by unittest."""
from __future__ import annotations


class _Stub:
    steps = 3

    @staticmethod
    def get_input(idx: int) -> str:
        return ["1 1 1 8", "2 2 2 6", "3 3 3 3", "4 4 4 4", "5 5 5 5"][idx]

    @staticmethod
    def canonical_state_key(x: str, y: str) -> tuple:
        from fractions import Fraction
        import re
        lines = [l for l in y.strip().split("\n") if "left:" in l]
        target = lines[-1] if lines else y
        m = re.search(r"left:\s*([^)]+)", target)
        parts = (m.group(1).strip().split() if m else target.strip().split())
        return tuple(sorted(Fraction(p) for p in parts if p.strip()))

    @staticmethod
    def is_success_state(x: str, y: str) -> bool:
        key = _Stub.canonical_state_key(x, y)
        return len(key) == 1 and list(key)[0] == 24

    @staticmethod
    def propose_prompt_wrap(x: str, y: str) -> str:
        return "prompt"

    @staticmethod
    def process_generate_result(pro, x, y, check):
        return True, pro.strip()

    @staticmethod
    def vote_prompt_wrap(x, ys):
        return "vote_prompt"

    @staticmethod
    def vote_outputs_unwrap(outputs, n):
        return [1] * n

    @staticmethod
    def test_output_modify(idx, output):
        return {"r": 0}, output
