"""Evaluator value caches owned by one solver for the task it is solving.

``core.evaluation`` no longer writes the legacy prompt-only ``Task.value_cache``.
Search modules must therefore hand ``get_values`` an explicit
:class:`~federated_mcts.core.evaluation.EvaluatorValueCache` or the same
evaluator will be charged for an identical prompt on every step.

A cache is identified by the evaluator's *client instance* and its recorded
identity, not by a mutable client name: replacing an adapter under the same
name starts a fresh cache, and two adapters answering under the same label
never see each other's scores.  Task objects and puzzle inputs delimit cache
lifetime, so a second task (or another input on a reused task object, or a
second solver in the same process) cannot reuse another search's answers.
"""

from __future__ import annotations

from threading import Lock

from federated_mcts.core.evaluation import EvaluatorValueCache


class TaskEvaluatorCaches:
    """Per-solver registry of evaluator caches for the current task object."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._task = None
        self._input: str | None = None
        self._entries: dict[tuple[int, str], tuple[object, EvaluatorValueCache]] = {}

    def for_evaluator(self, task, evaluator_id: str, client,
                      input_text: str | None = None) -> EvaluatorValueCache:
        """Return the cache scoped to ``(task, input, client, identity)``.

        Keeping a strong reference to ``client`` in the entry prevents the
        client's ``id`` from being reused while its cache is still alive.
        """
        with self._lock:
            if task is not self._task or input_text != self._input:
                self._task, self._input = task, input_text
                self._entries = {}
            key = (id(client), evaluator_id)
            entry = self._entries.get(key)
            if entry is None or entry[0] is not client:
                cache = EvaluatorValueCache()
                self._entries[key] = (client, cache)
                return cache
            return entry[1]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
