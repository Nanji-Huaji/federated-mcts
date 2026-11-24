"""Local grader for a single submitted full answer. No search, eval or sympy."""
import ast
import hashlib
from collections import Counter
import re


LABEL = re.compile(r'^(?:final[ \t]+)?answer[ \t]*:[ \t]*', re.I | re.ASCII)
ALPHABET = re.compile(r'[0-9()+*/ \t-]+')
ALLOWED_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


def score_answer(puzzle, answer):
    """Return (verdict, shape digest for valid submissions only). No model text is returned.

    Expect the FINAL nonempty line to contain one complete expression, with
    an optional answer label and optional '= 24'. Earlier scratch work is not
    mined for an answer. Require the archived four leaves, with multiplicity.
    """
    if (len(puzzle) != 4 or any(type(n) is not int or n < 0 for n in puzzle)):
        raise ValueError('invalid four-number task')
    if not isinstance(answer, str) or not answer.strip():
        return 'empty', None
    line = answer.strip().splitlines()[-1].strip()
    if len(line) > 256:
        return 'format', None
    expr, equals, claim = LABEL.sub('', line, count=1).partition('=')
    if equals and claim.strip() != '24':
        return 'wrong_or_ambiguous_rhs', None
    expr = expr.strip()
    if not ALPHABET.fullmatch(expr):
        return 'format', None
    try:
        body = ast.parse(expr, mode='eval').body   # syntax only, NOT eval/compile/exec
    except (ValueError, SyntaxError, RecursionError):
        return 'format', None
    leaves = []

    def allowed(n):
        if isinstance(n, ast.Constant) and type(n.value) is int and n.value >= 0:
            leaves.append(n.value)
            return True
        return (isinstance(n, ast.BinOp) and isinstance(n.op, ALLOWED_OPS)
                and allowed(n.left) and allowed(n.right))

    if not allowed(body):
        return 'format', None
    if len(leaves) != 4:
        return 'leaf_count', None
    if Counter(leaves) != Counter(puzzle):
        return 'wrong_leaves', None

    class IllegalMove(Exception):
        pass

    def value(n):
        if isinstance(n, ast.Constant):
            return n.value
        a, b = value(n.left), value(n.right)
        if isinstance(n.op, ast.Add):
            return a + b
        if isinstance(n.op, ast.Sub):
            if a < b:
                raise IllegalMove('negative_intermediate')
            return a - b
        if isinstance(n.op, ast.Mult):
            return a * b
        if b == 0:
            raise IllegalMove('division_by_zero')
        if a % b:
            raise IllegalMove('inexact_division')
        return a // b

    try:
        if value(body) != 24:
            return 'not_24', None
    except IllegalMove as failure:
        return str(failure), None

    # A structural duplicate is not necessarily a duplicate string. Only
    # forget sibling order for commutative operators; do not reassociate.
    def shape(n):
        if isinstance(n, ast.Constant):
            return str(n.value)
        op = {ast.Add: '+', ast.Mult: '*', ast.Sub: '-', ast.Div: '/'}[type(n.op)]
        left, right = shape(n.left), shape(n.right)
        if op in '+*' and left > right:
            left, right = right, left
        return f'({left}{op}{right})'

    return 'valid', hashlib.sha256(shape(body).encode('ascii')).hexdigest()


def grade_answer(puzzle, answer):
    """Simple reason-code interface. Only 'valid' is a success."""
    return score_answer(puzzle, answer)[0]
