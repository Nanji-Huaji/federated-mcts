"""Pure Blocksworld environment logic.

No I/O and no network: parsing the canonical problem text ``x``, parsing
single-line actions, and deterministically replaying actions against the
symbolic state are all pure functions of their inputs.  The task module builds
on top of this engine.

State representation: a frozen set of predicate tuples drawn from
``("handempty",)``, ``("ontable", block)``, ``("on", top, bottom)``,
``("clear", block)`` and ``("holding", block)``.
"""

from __future__ import annotations

import re

_PICK_RE = re.compile(r"^(?:pick[-\s]?up|pickup)\s+([a-z]+)$")
_PUT_RE = re.compile(r"^(?:put[-\s]?down|putdown)\s+([a-z]+)$")
_STACK_RE = re.compile(r"^stack\s+([a-z]+)\s+(?:on(?:\s+top\s+of)?\s+)?([a-z]+)$")
_UNSTACK_RE = re.compile(
    r"^unstack\s+([a-z]+)\s+(?:from(?:\s+on\s+top\s+of)?\s+)?([a-z]+)$"
)


def parse_x(x: str) -> dict:
    """Parse the canonical problem text produced by ``format_x``."""
    result = {}
    lines = [line.strip() for line in x.splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Instance:"):
            result["id"] = line.split(":", 1)[1].strip()
        elif line.startswith("Blocks:"):
            result["blocks"] = frozenset(line.split(":", 1)[1].split())
        elif line == "Initial:":
            preds = []
            i += 1
            while i < len(lines) and lines[i].startswith("-"):
                preds.append(_parse_pred(lines[i][1:].strip()))
                i += 1
            result["init"] = frozenset(preds)
            continue
        elif line == "Goal:":
            preds = []
            i += 1
            while i < len(lines) and lines[i].startswith("-"):
                preds.append(_parse_pred(lines[i][1:].strip()))
                i += 1
            result["goal"] = frozenset(preds)
            continue
        elif line.startswith("MaxSteps:"):
            result["max_steps"] = int(line.split(":", 1)[1].strip())
        i += 1
    return result


def format_x(record: dict) -> str:
    """Serialize a normalized data record into the canonical problem text."""
    lines = [
        "Instance: %s" % record["id"],
        "Blocks: %s" % " ".join(record["blocks"]),
        "Initial:",
    ]
    for pred in record["init"]:
        lines.append("- %s" % " ".join(pred))
    lines.append("Goal:")
    for pred in record["goal"]:
        lines.append("- %s" % " ".join(pred))
    lines.append("MaxSteps: %d" % record["max_steps"])
    return "\n".join(lines) + "\n"


def _parse_pred(text: str) -> tuple:
    parts = text.split()
    return (parts[0],) + tuple(parts[1:])


def parse_action_line(line: str, blocks: frozenset) -> tuple | None:
    """Parse a single action line into ``(canonical_name, args)`` or None.

    Accepts the canonical grammar ``pick-up X``, ``put-down X``,
    ``stack X Y``, ``unstack X Y`` plus a few common variants, and rejects
    commentary, multi-action lines and unknown blocks.
    """
    text = line.strip()
    if not text:
        return None
    match = _PICK_RE.match(text)
    if match is not None:
        name, args = "pick-up", (match.group(1),)
    else:
        match = _PUT_RE.match(text)
        if match is not None:
            name, args = "put-down", (match.group(1),)
        else:
            match = _STACK_RE.match(text)
            if match is not None:
                name, args = "stack", (match.group(1), match.group(2))
            else:
                match = _UNSTACK_RE.match(text)
                if match is not None:
                    name, args = "unstack", (match.group(1), match.group(2))
                else:
                    return None
    if not all(arg in blocks for arg in args):
        return None
    return (name, args)


def format_action(action: tuple) -> str:
    """Render a parsed action back to its canonical single-line form."""
    name, args = action
    if name in ("pick-up", "put-down"):
        return "%s %s" % (name, args[0])
    return "%s %s %s" % (name, args[0], args[1])


def action_lines(y: str) -> list[str]:
    """Non-empty, stripped lines of a trajectory ``y``."""
    return [line.strip() for line in y.splitlines() if line.strip()]


def action_count(y: str) -> int:
    return len(action_lines(y))


def initial_state(parsed: dict) -> frozenset:
    return parsed["init"]


def step_state(state: frozenset | None, action: tuple) -> frozenset | None:
    """Apply one action to a state, or None if its preconditions are unmet."""
    if state is None:
        return None
    name, args = action
    if name == "pick-up":
        block = args[0]
        if (
            ("clear", block) not in state
            or ("ontable", block) not in state
            or ("handempty",) not in state
        ):
            return None
        return state - {("clear", block), ("ontable", block), ("handempty",)} | {
            ("holding", block)
        }
    if name == "put-down":
        block = args[0]
        if ("holding", block) not in state:
            return None
        return state - {("holding", block)} | {
            ("ontable", block),
            ("clear", block),
            ("handempty",),
        }
    if name == "stack":
        block, target = args
        if ("holding", block) not in state or ("clear", target) not in state:
            return None
        return state - {("holding", block), ("clear", target)} | {
            ("on", block, target),
            ("clear", block),
            ("handempty",),
        }
    if name == "unstack":
        block, below = args
        if (
            ("on", block, below) not in state
            or ("clear", block) not in state
            or ("handempty",) not in state
        ):
            return None
        return state - {("on", block, below), ("clear", block), ("handempty",)} | {
            ("holding", block),
            ("clear", below),
        }
    return None


def replay_state(parsed: dict, y: str) -> frozenset | None:
    """Replay every action line from the initial state.

    Every non-empty trajectory line must parse as exactly one known action; a
    malformed, commentary or unknown line invalidates the whole trajectory
    (returns None), as does a parsed action whose preconditions are unmet.
    """
    state = parsed["init"]
    for line in action_lines(y):
        action = parse_action_line(line, parsed["blocks"])
        if action is None:
            return None
        state = step_state(state, action)
        if state is None:
            return None
    return state


def is_goal_satisfied(parsed: dict, state: frozenset | None) -> bool:
    if state is None:
        return False
    return parsed["goal"] <= state


def canonical_key(parsed: dict, state: frozenset | None) -> tuple:
    """Instance identity plus the complete replayed symbolic state."""
    return (parsed["id"], None if state is None else frozenset(state))


def render_state(state: frozenset | None) -> str:
    """Canonical text rendering of a state for prompts."""
    if state is None:
        return "- invalid trajectory"
    return "\n".join("- %s" % " ".join(pred) for pred in sorted(state))
