"""Frozen task experts: route each registered task to one trained policy.

The router combines complete networks rather than swapping heads between incompatible
shared trunks. Every expert remains individually hashable in the generation registry.
"""
from __future__ import annotations

import torch
from torch import nn

from toddler.learn.multitask import MultiTaskNet


class TaskExpertRouter(nn.Module):
    def __init__(self, experts: dict[str, MultiTaskNet], route: dict[str, str]) -> None:
        super().__init__()
        if len(experts) < 2 or not route:
            raise ValueError("a task router needs at least two experts and a route")
        dims = next(iter(experts.values())).task_dims
        if any(net.task_dims != dims for net in experts.values()):
            raise ValueError("all experts must have the same task dimensions")
        if set(route) != set(dims) or any(name not in experts for name in route.values()):
            raise ValueError("route must name exactly one existing expert for every task")
        self.experts = nn.ModuleDict(experts)
        self.route = dict(route)
        self.task_dims = dims

    def forward_task(self, task: str, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.experts[self.route[task]].forward_task(task, obs)

    def spec(self) -> dict:
        return {"kind": "task_router", "route": self.route,
                "expert_specs": {name: net.spec() for name, net in self.experts.items()}}

    @classmethod
    def from_spec(cls, spec: dict) -> "TaskExpertRouter":
        if spec.get("kind") != "task_router":
            raise ValueError("not a task router specification")
        experts = {name: MultiTaskNet({task: tuple(dims) for task, dims in child["task_dims"].items()},
                                      child["hidden"])
                   for name, child in spec["expert_specs"].items()}
        return cls(experts, spec["route"])
