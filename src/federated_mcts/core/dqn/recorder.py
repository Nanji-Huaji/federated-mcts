"""JSONL transition recorder: append-only, plain-JSON values.

Sequential writers (the task runner) may safely append; each record is
written and flushed in a single append pass.  numpy and torch values are
converted to plain JSON scalars/lists on write.
"""

from __future__ import annotations

import json

import numpy as np
import torch


def _to_plain(value):
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


class JSONLTransitionRecorder:
    def __init__(self, path: str):
        self.path = path

    def record(self, transition: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_plain(transition)) + "\n")
            handle.flush()

    def replace_last(self, transition: dict) -> None:
        records = self.read_all()
        if not records:
            self.record(transition)
            return
        records[-1] = transition
        with open(self.path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(_to_plain(record)) + "\n")

    def read_all(self) -> list[dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        except FileNotFoundError:
            return []

    def __len__(self) -> int:
        return len(self.read_all())
