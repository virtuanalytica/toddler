"""One toddler, one network, several tasks: the base of the learning cyber-organism.

The tasks differ in observation size and action count (CartPole 4 -> 2, Acrobot 6 -> 3, MiniGrid
147 -> 7), so every task has a small input adapter and its own policy/value heads, and all tasks
share one trunk: the part of the brain that learns across tasks.

    obs_t --adapter_t--> h --shared trunk--> z --pi_t / v_t--> logits, value

`TaskView(net, task)` exposes one task as an ordinary actor-critic (forward / dist / spec), so the
existing `ppo.train` trains it unchanged; every task keeps its own `ppo.TrainState` (optimiser,
scaler, environment), and all of them update the shared trunk. Training interleaves the tasks in
equal blocks of environment steps, so no task gets more trunk updates than another.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import nn

from toddler.learn import ppo, scoring
from toddler.learn import tasks as T


class MultiTaskNet(nn.Module):
    def __init__(self, task_dims: dict[str, tuple[int, int]], hidden: int = 64) -> None:
        super().__init__()
        self.task_dims = {k: (int(o), int(a)) for k, (o, a) in task_dims.items()}
        self.hidden = int(hidden)
        self.adapters = nn.ModuleDict({k: nn.Sequential(nn.Linear(o, hidden), nn.Tanh())
                                       for k, (o, _) in self.task_dims.items()})
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.ModuleDict({k: nn.Linear(hidden, a) for k, (_, a) in self.task_dims.items()})
        self.v = nn.ModuleDict({k: nn.Linear(hidden, 1) for k in self.task_dims})

    def forward_task(self, task: str, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(self.adapters[task](obs))
        return self.pi[task](z), self.v[task](z).squeeze(-1)

    def spec(self) -> dict:
        return {"kind": "multitask", "task_dims": {k: list(v) for k, v in self.task_dims.items()},
                "hidden": self.hidden}

    def inherit(self, parent: "MultiTaskNet") -> list[str]:
        """Copy the parent's trunk and every task module the parent shares with this net (a new
        generation keeps what its parent learned and grows adapters/heads for new tasks). Shapes
        must match exactly; a parent task this net lacks is an error, never silently dropped.
        Returns the inherited task names."""
        if parent.hidden != self.hidden:
            raise ValueError(f"hidden size differs: parent {parent.hidden}, child {self.hidden}")
        missing = [t for t in parent.task_dims if t not in self.task_dims]
        if missing:
            raise ValueError(f"child lacks parent tasks {missing}")
        for t, dims in parent.task_dims.items():
            if self.task_dims[t] != dims:
                raise ValueError(f"{t}: parent dims {dims} != child dims {self.task_dims[t]}")
        own = self.state_dict()
        own.update(parent.state_dict())        # keys of parent ⊆ keys of child, checked above
        self.load_state_dict(own)
        return list(parent.task_dims)

    @classmethod
    def for_tasks(cls, task_names: list[str], hidden: int = 64) -> "MultiTaskNet":
        dims = {}
        for t in task_names:
            env = T.make(t)
            dims[t] = (env.observation_space.shape[0], env.action_space.n)
            env.close()
        return cls(dims, hidden)


class TaskView(nn.Module):
    """One task of a MultiTaskNet as an ordinary actor-critic. Its parameters are the task's
    adapter and heads plus the SHARED trunk (the same tensors for every view)."""

    def __init__(self, net: MultiTaskNet, task: str) -> None:
        super().__init__()
        self.net, self.task = net, task

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net.forward_task(self.task, obs)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        logits, _ = self(obs)
        return torch.distributions.Categorical(logits=logits)

    def parameters(self, recurse: bool = True):
        yield from self.net.adapters[self.task].parameters()
        yield from self.net.trunk.parameters()
        yield from self.net.pi[self.task].parameters()
        yield from self.net.v[self.task].parameters()

    def spec(self) -> dict:
        return {**self.net.spec(), "view_task": self.task}


@dataclass
class MultiTaskLog:
    steps_per_task: dict[str, int] = field(default_factory=dict)
    blocks: int = 0
    last_returns: dict[str, list[float]] = field(default_factory=dict)


def train_multitask(task_names: list[str], steps_per_task: int, block_steps: int, seed: int,
                    base: ppo.PPOConfig | None = None, hidden: int = 64,
                    parent: MultiTaskNet | None = None,
                    steps: dict[str, int] | None = None) -> tuple[MultiTaskNet, MultiTaskLog]:
    """Interleave the tasks in blocks of `block_steps` until each has its step budget.

    `steps` overrides `steps_per_task` per task (harder tasks may get more); every budget must be
    a multiple of `block_steps`. A task stops receiving blocks once its budget is spent, so the
    interleaving stays fair while several tasks are still training. `parent` seeds the net via
    MultiTaskNet.inherit (the trunk and the parent's tasks)."""
    budgets = {t: (steps or {}).get(t, steps_per_task) for t in task_names}
    if any(b % block_steps for b in budgets.values()):
        raise ValueError("every step budget must be a multiple of block_steps")
    torch.manual_seed(seed)
    net = MultiTaskNet.for_tasks(task_names, hidden)
    if parent is not None:
        net.inherit(parent)
    rng = np.random.default_rng(seed)
    base = base or ppo.PPOConfig()
    views = {t: TaskView(net, t) for t in task_names}
    states = {t: ppo.TrainState() for t in task_names}
    cfgs = {t: replace(base, total_steps=block_steps, seed=int(rng.integers(0, 2**31))) for t in task_names}
    log = MultiTaskLog(steps_per_task={t: 0 for t in task_names})
    for b in range(max(budgets.values()) // block_steps):
        for t in task_names:
            if b * block_steps >= budgets[t]:
                continue
            _, tl = ppo.train(t, cfgs[t], net=views[t], state=states[t])
            log.steps_per_task[t] += tl.steps
            log.last_returns[t] = tl.episode_returns[-20:]
        log.blocks += 1
    for st in states.values():
        st.close()
    return net, log


def evaluate_multitask(net: MultiTaskNet, task_names: list[str], anchors: dict[str, float],
                       mode: str = "sample", seeds: tuple[int, ...] | None = None) -> dict[str, list[float]]:
    """Normalised held-out scores per task (one value per evaluation seed)."""
    out = {}
    for t in task_names:
        raw = scoring.evaluate(TaskView(net, t), t, seeds, mode=mode)
        out[t] = [float(T.normalise(t, r, anchors[t])) for r in raw]
    return out
