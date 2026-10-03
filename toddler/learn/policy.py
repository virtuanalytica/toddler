"""A toddler's brain for discrete-action tasks: a small actor-critic network."""

from __future__ import annotations

import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        self.obs_dim, self.n_actions, self.hidden = obs_dim, n_actions, hidden

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        return self.pi(h), self.v(h).squeeze(-1)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        logits, _ = self(obs)
        return torch.distributions.Categorical(logits=logits)

    def spec(self) -> dict:
        return {"obs_dim": self.obs_dim, "n_actions": self.n_actions, "hidden": self.hidden}
