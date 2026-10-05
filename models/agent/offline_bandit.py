"""Offline contextual bandit for dataset -> Teacher Package / None selection (Stage 4).

State x(D, T) = [dataset signature s_D, teacher train-only probe c(D,T), teacher
metadata embedding e_j]. Action is a HARD selection of one teacher package or
None. Reward is student validation utility relative to the C1 anchor; test F1 is
never used. Leave-One-Dataset-Out (LODO) guards against a 4-row lookup table.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TeacherAgentPolicy(nn.Module):
    """Tiny MLP scoring a (dataset, teacher) pair feature."""

    def __init__(self, in_dim: int, hidden1: int = 96, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def select_action(policy, pair_features, teacher_ids):
    """Return (best_teacher_id_or_None, best_score). None if max score <= 0."""
    scores = policy(pair_features).detach()
    best = int(scores.argmax().item())
    if scores[best] <= 0.0:
        return None, 0.0
    return teacher_ids[best], float(scores[best])


def train_policy(pair_features, utilities, epochs=2000, lr=1e-3, weight_decay=1e-4):
    """Fit the tiny MLP by MSE on observed (train/val) utilities."""
    x = torch.as_tensor(pair_features, dtype=torch.float32)
    y = torch.as_tensor(utilities, dtype=torch.float32)
    policy = TeacherAgentPolicy(x.shape[1])
    opt = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(policy(x), y)
        loss.backward()
        opt.step()
    return policy, float(loss.detach())


__all__ = ["TeacherAgentPolicy", "select_action", "train_policy"]
