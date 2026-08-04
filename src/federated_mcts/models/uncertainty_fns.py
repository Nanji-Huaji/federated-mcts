"""Token-level uncertainty functions operating on raw logits tensors.

These are standalone pure functions — no model state, no side effects.
They are used as ``uncertainty_fn`` callables passed into
``UncertaintyInferenceFramework.generate_with_uncertainty``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def entropy_uncertainty(logits: torch.Tensor, current_token: int) -> float:
    """Entropy of the full-vocab softmax distribution over logits."""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
    return entropy.item()


def variance_uncertainty(logits: torch.Tensor, current_token: int) -> float:
    """Weighted variance of token indices under the softmax distribution."""
    probs = F.softmax(logits, dim=-1)
    device = probs.device
    indices = torch.arange(len(probs), dtype=torch.float, device=device)
    mean = torch.sum(probs * indices)
    variance = torch.sum(probs * (indices - mean) ** 2)
    return variance.item()


def inconsistency(logits: torch.Tensor, current_token: int) -> float:
    """Fraction of temperature-perturbed samples disagreeing with original argmax.

    Draws M=20 temperatures uniformly from (0, theta_max=3]; counts how many
    sampled tokens differ from the original top-1 prediction.
    """
    device = logits.device

    M = 20
    theta_max = 3

    if current_token is not None:
        original_pred = current_token
    else:
        original_pred = torch.argmax(logits).item()

    temperatures = torch.rand(M, device=device) * theta_max + 1e-6

    inconsistent_count = 0
    for m in range(M):
        scaled_logits = logits / temperatures[m]
        probs = F.softmax(scaled_logits, dim=0)
        sampled_token = torch.multinomial(probs, 1).item()
        if sampled_token != original_pred:
            inconsistent_count += 1

    return inconsistent_count / M


def token_probability_uncertainty(logits: torch.Tensor, current_token: int) -> float:
    """1 - softmax probability of the *selected* token."""
    probs = F.softmax(logits, dim=-1)
    current_token_prob = probs[current_token].item()
    return 1.0 - current_token_prob
