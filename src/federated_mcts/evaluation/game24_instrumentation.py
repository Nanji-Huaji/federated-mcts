"""Game24 adapters for the core search-observation seam.

The seam itself — typed records, sink protocols, record construction and the
vote parser — lives in :mod:`federated_mcts.core.observation`, because the
search primitives must not depend on Game24 evaluation code.  This module is
the evaluation-side adapter:

* :class:`GenerationRecorder` and :class:`VoteRecorder` are in-memory sinks
  that satisfy the core protocols;
* :func:`parse_vote_response` keeps its historical tuple shape while
  delegating to the single shared parser;
* the record types are re-exported unchanged so baseline telemetry JSON field
  names and every existing importer keep working.

When ``recorder=None`` (the default), behaviour is byte-identical to the
uninstrumented path.  When a recorder is supplied, it captures raw
completions, parse results, rejection reasons, raw vote text, parsed vote
indices, parse-failure reasons, out-of-range votes, and evaluator identity —
without mutating the candidate/vote values returned to the caller.

All record types are plain ``TypedDict`` so serialisation is trivial.
"""

from __future__ import annotations

from federated_mcts.core.observation import (
    VOTE_CHOICE_PATTERN,
    CandidateProposalRecord,
    GenerationObservationSink,  # noqa: F401  (protocol implemented here)
    ProposalStepRecord,
    VoteAggregateRecord,
    VoteCallRecord,
    VoteObservationSink,  # noqa: F401  (protocol implemented here)
    parse_vote_choice,
)


# ── Proposal recorder (core sink adapter) ────────────────────────────

class GenerationRecorder:
    """Collects raw proposal events without affecting candidate lists.

    Usage::

        recorder = GenerationRecorder()
        candidates = get_proposals(args, step, task, x, y, recorder=recorder)
        # candidates is identical to the uninstrumented path
        print(recorder.steps)
    """

    def __init__(self) -> None:
        self.steps: list[ProposalStepRecord] = []

    def record_step(self, rec: ProposalStepRecord) -> None:
        self.steps.append(rec)

    @property
    def total_proposed(self) -> int:
        return sum(len(s["proposals"]) for s in self.steps)

    @property
    def total_accepted(self) -> int:
        return sum(1 for s in self.steps for p in s["proposals"] if p["is_correct"])


# ── Vote recorder (core sink adapter) ────────────────────────────────

class VoteRecorder:
    """Collects raw vote events without affecting vote totals.

    Usage::

        recorder = VoteRecorder()
        values = get_votes(args, task, x, ys, n_eval, recorder=recorder, client=gpt)
        # values is identical to the uninstrumented path
        print(recorder.steps)
    """

    def __init__(self) -> None:
        self.steps: list[VoteAggregateRecord] = []

    def record_step(self, rec: VoteAggregateRecord) -> None:
        self.steps.append(rec)

    @property
    def total_calls(self) -> int:
        return sum(s["call_count"] for s in self.steps)

    @property
    def total_parse_failures(self) -> int:
        result = 0
        for s in self.steps:
            for c in s["calls"]:
                if not c["parse_success"]:
                    result += 1
                elif not c["in_range"]:
                    result += 1
        return result


# ── Vote parse helper (thin adapter over the shared parser) ──────────

_VOTE_PARSE_PATTERN = VOTE_CHOICE_PATTERN


def parse_vote_response(raw: str) -> tuple[int | None, bool, str | None]:
    """Parse one evaluator vote response for telemetry.

    Returns ``(parsed_index, parse_success, failure_reason)`` where
    ``failure_reason`` is ``"no_match"`` when the regex fails to match,
    or ``None`` otherwise.  Does NOT check whether the index is within
    the valid candidate range — callers must perform that check.

    Delegates to :func:`federated_mcts.core.observation.parse_vote_choice`,
    the same parser ``Game24Task.vote_outputs_unwrap`` uses, so telemetry
    never reports parse success where the decision parser does not, and
    vice versa.
    """
    choice = parse_vote_choice(raw)
    return (choice.vote_index, choice.matched, choice.failure_reason)
