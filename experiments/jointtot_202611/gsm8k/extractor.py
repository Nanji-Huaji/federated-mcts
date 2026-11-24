"""Frozen, gold-free final-answer extraction for the archived GSM8K candidates."""

import re
from fractions import Fraction

_ATOM = r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?"
_AMOUNT = re.compile(rf"([$£€]?\s*)({_ATOM})(?:\s*/\s*({_ATOM}))?\s*([$%]|[A-Za-z]+)?", re.ASCII)
_UNITS = set("dollar dollars usd cent cents percent percentage mile miles meter meters metre metres foot feet inch inches yard yards hour hours minute minutes second seconds day days week weeks month months year years pound pounds ounce ounces kg g km cm m liter liters litre litres gallon gallons degree degrees".split()) | {"%"}
_MARK = re.compile(r"(?P<hash>####)|(?P<box>\\boxed\b)|(?P<answer>\b(?:the\s+)?(?:final\s+)?answer\s*(?:\bis\b\s*[:=]?|[:=]))", re.I)
_LATEX_FRAC = re.compile(r"\\(?:d?frac)\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}")


def _display(s):
    s = (s.replace("**", "").replace(chr(96), "").replace("−", "-")
         .replace(r"\$", "$").replace(r"\%", "%"))
    for token in (r"\(", r"\)", r"\[", r"\]"):
        s = s.replace(token, "")
    s = re.sub(r"\\text\{([A-Za-z ]+)\}", r"\1", s)
    return _LATEX_FRAC.sub(r"\1/\2", s).strip()


def scalar(payload, *, units=True):
    """Entire literal only; str(Fraction) is canonical and exact on the wire."""
    s = _display(payload).strip()
    # Parentheses/brackets next to a marker commonly close wrappers opened before it.
    s = s.lstrip("([ ").rstrip(".!)] \t")
    m = _AMOUNT.fullmatch(s)
    if not m or (m[4] and m[4] != "$" and (not units or m[4].lower() not in _UNITS)):
        return None
    try:
        n = Fraction(m[2].replace(",", ""))
        if m[3]:
            n /= Fraction(m[3].replace(",", ""))
        return str(n)
    except (ValueError, ZeroDivisionError):
        return None


def _values(payload):
    # Unambiguous alternatives are exposed rather than silently taking one end.
    v = scalar(payload)
    if v is not None:
        return [v]
    parts = re.split(r"\s+(?:or|and|to|vs\.?)\s+", payload, flags=re.I)
    values = [scalar(p) for p in parts]
    return values if len(parts) > 1 and all(v is not None for v in values) else []


def extract(text):
    """Return status, exact prediction (or None), and auditable marker evidence."""
    lines = [_display(line) for line in text.splitlines() if _display(line)]
    evidence = []
    for i, line in enumerate(lines):
        markers = list(_MARK.finditer(line))
        for j, m in enumerate(markers):
            kind = m.lastgroup
            segment = line[m.end():markers[j + 1].start() if j + 1 < len(markers) else len(line)].strip()
            if not segment and j + 1 < len(markers):
                # 'Answer: #### 4', '#### \\boxed{4}': adjacent labels refer to the latter payload.
                continue
            if not segment and i + 1 < len(lines):
                segment = lines[i + 1]
            if kind == "box":
                box = re.match(r"^\s*\{([^{}]*)\}", segment)
                segment = box[1] if box else ""
            evidence.append({"kind": kind, "payload": segment, "values": _values(segment)})
    if evidence:
        distinct = {v for e in evidence for v in e["values"]}
        formal = [e for e in evidence if e["kind"] in ("hash", "box")]
        if len(distinct) > 1:
            status = "conflict"
        elif any(not e["values"] for e in (formal or evidence)) or not distinct:
            status = "malformed_declaration"
        else:
            status = "formal" if formal else "answer_phrase"
        return {"status": status, "value": next(iter(distinct)) if status in ("formal", "answer_phrase") else None, "evidence": evidence}
    value = scalar(lines[-1], units=False) if lines else None
    return {"status": "terminal_scalar" if value is not None else "no_answer", "value": value, "evidence": []}


def reference(text):
    if text.count("####") != 1:
        raise ValueError("reference must have exactly one #### declaration")
    v = scalar(text.partition("####")[2], units=False)
    if v is None:
        raise ValueError("invalid terminal reference")
    return v
