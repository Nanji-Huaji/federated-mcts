"""Deterministic mock GenerationClient for offline Game24 end-to-end tests.

Provides a ``Game24MockClient`` that satisfies the ``GenerationClient``
Protocol, producing deterministic, format-compatible proposals and votes
for Game24 tasks WITHOUT making any HTTP/API calls.

Intended for:
- Offline E2E test coverage that exercises the full manifest, provenance,
  metrics, and persistence path.
- Preflight validation that needs a non-mutating, no-network run.

The mock produces 2 proposals and 3 vote responses per call, and cycles
through a seeded deterministic response pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from federated_mcts.federation.standard_solvers import StandardSolverArgs

# ── Deterministic response pool ────────────────────────────────────────
# Proposals are format-correct Game24 lines:
#   "number operator number = result (left: remaining numbers)"
_PROPOSAL_POOL: list[str] = [
    "8 - 1 = 7 (left: 1 1 7)",
    "1 + 1 = 2 (left: 2 8 1)",
    "1 * 1 = 1 (left: 8 1 1)",
    "8 + 1 = 9 (left: 1 1 9)",
    "1 - 1 = 0 (left: 8 1 0)",
    "8 / 1 = 8 (left: 1 1 8)",
    "1 + 8 = 9 (left: 1 1 9)",
    "8 * 1 = 8 (left: 1 1 8)",
    "1 + 1 + 1 = 3 (left: 8 3)",
    "8 - 1 - 1 = 6 (left: 1 6)",
]

# Vote responses map a candidate index to a "best choice" vote string.
# In deterministic mode, each call returns the next vote.
_VOTE_POOL: list[str] = [
    "The best choice is 0",
    "The best choice is 1",
    "The best choice is 0",
    "The best choice is 0",
    "The best choice is 1",
    "The best choice is 0",
]


class Game24MockClient:
    """Deterministic mock ``GenerationClient`` for Game24.

    Produces format-compatible proposals and vote responses without
    network calls.  Internal state is deterministic — the same sequence
    of calls produces the same outputs.

    Usage::

        mock = Game24MockClient()
        proposals = mock(args, prompt, n=2)     # returns 2 proposals
        vote = mock(args, vote_prompt, n=1)     # returns 1 vote string
    """

    def __init__(self) -> None:
        self._proposal_idx: int = 0
        self._vote_idx: int = 0
        self._call_count: int = 0

    def __call__(
        self,
        args: StandardSolverArgs,
        prompt: str,
        n: int = 1,
        stop: str | list[str] | None = None,
        get_logprobs: bool = False,
    ) -> list[str]:
        """Return *n* deterministic responses.

        The mock detects "vote" in the prompt (case-insensitive) to
        dispatch between proposal and vote responses.  Otherwise it
        returns proposals.
        """
        self._call_count += 1
        _ = args  # unused — mock is independent of solver args
        _ = stop
        _ = get_logprobs

        is_vote = "vote" in prompt.lower() if prompt else False
        if is_vote:
            return self._next_votes(n)
        return self._next_proposals(n)

    def _next_proposals(self, n: int) -> list[str]:
        """Return *n* deterministic proposals cycling through the pool."""
        result: list[str] = []
        for _ in range(n):
            result.append(_PROPOSAL_POOL[self._proposal_idx % len(_PROPOSAL_POOL)])
            self._proposal_idx += 1
        return result

    def _next_votes(self, n: int) -> list[str]:
        """Return *n* deterministic votes cycling through the pool."""
        result: list[str] = []
        for _ in range(n):
            result.append(_VOTE_POOL[self._vote_idx % len(_VOTE_POOL)])
            self._vote_idx += 1
        return result

    @property
    def call_count(self) -> int:
        """Total number of calls made to this mock client."""
        return self._call_count

    def reset(self) -> None:
        """Reset internal state for a fresh test run."""
        self._proposal_idx = 0
        self._vote_idx = 0
        self._call_count = 0


def build_mock_gpts(model_config_raw: list[dict]) -> dict[str, Game24MockClient]:
    """Create a ``gpts`` dict compatible with ``FederatedSolver``.

    Every client (local and remote) maps to the same mock instance
    so all calls are deterministic and offline.
    """
    mock = Game24MockClient()
    gpts: dict[str, Game24MockClient] = {}
    for entry in model_config_raw:
        client_name = str(entry.get("client_name", ""))
        if client_name:
            gpts[client_name] = mock
    if not gpts:
        gpts["remote_client"] = mock
        gpts["local_client"] = mock
    return gpts
