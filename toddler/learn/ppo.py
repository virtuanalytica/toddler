"""Compact PPO with a budget in environment steps (not wall-clock), so a slow or busy GPU never
changes what a generation learns, only how long it takes.

Self-reinforcement: the reward is the task reward plus a decaying count-based curiosity bonus
(the "+0.1 for new knowledge" principle from toddler/objective.py), so a toddler is rewarded for
visiting states it has not seen yet. A teacher can be attached (peer training): its action
distribution is distilled into the student with a KL term on the student's own states.

A watchdog callable is checked every update; when it returns a reason, training continues on
the CPU (the GPU is handed back to its owner).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from toddler.learn import tasks as T
from toddler.learn.policy import ActorCritic


@dataclass
class PPOConfig:
    total_steps: int = 50_000
    rollout: int = 2048
    epochs: int = 4
    minibatch: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    curiosity: float = 0.1          # initial weight of the novelty bonus
    curiosity_decay: float = 0.97   # per update
    distill_coef: float = 1.0       # weight of the teacher KL term (if a teacher is given)
    seed: int = 0


@dataclass
class TrainLog:
    updates: int = 0
    steps: int = 0
    device_switches: list[str] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)


def _novelty(counts: dict, obs: np.ndarray, bins: float = 0.25) -> float:
    key = tuple(np.floor(obs / bins).astype(int))
    counts[key] += 1
    return 1.0 / np.sqrt(counts[key])


def train(task: str, cfg: PPOConfig, net: ActorCritic | None = None, teacher: ActorCritic | None = None,
          device: str = "cpu", watchdog: Callable[[], str | None] | None = None,
          checkpoint: Callable[[ActorCritic, TrainLog], None] | None = None,
          checkpoint_every: int = 10) -> tuple[ActorCritic, TrainLog]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    env = T.make(task)
    obs_dim, n_act = env.observation_space.shape[0], env.action_space.n
    net = net or ActorCritic(obs_dim, n_act)
    net.to(device)
    if teacher is not None:
        teacher.to(device).eval()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    log, counts, cur_w = TrainLog(), defaultdict(int), cfg.curiosity
    obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
    ep_ret = 0.0
    while log.steps < cfg.total_steps:
        if watchdog and device != "cpu":
            reason = watchdog()
            if reason:
                device = "cpu"
                net.to(device)
                if teacher is not None:
                    teacher.to(device)
                opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
                log.device_switches.append(reason)
        n = min(cfg.rollout, cfg.total_steps - log.steps)
        buf_o, buf_a, buf_lp, buf_r, buf_v, buf_d = [], [], [], [], [], []
        for _ in range(n):
            o = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, v = net(o)
                d = torch.distributions.Categorical(logits=logits)
                a = d.sample()
            nobs, r, term, trunc, _ = env.step(int(a))
            ep_ret += r
            bonus = cur_w * _novelty(counts, nobs)
            buf_o.append(obs); buf_a.append(int(a)); buf_lp.append(float(d.log_prob(a)))
            buf_r.append(r + bonus); buf_v.append(float(v)); buf_d.append(term)
            obs = nobs
            if term or trunc:
                log.episode_returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        with torch.no_grad():
            _, last_v = net(torch.as_tensor(obs, dtype=torch.float32, device=device))
        adv = np.zeros(n, dtype=np.float32)
        gae, next_v = 0.0, float(last_v)
        for t in reversed(range(n)):
            nonterm = 1.0 - float(buf_d[t])
            delta = buf_r[t] + cfg.gamma * next_v * nonterm - buf_v[t]
            gae = delta + cfg.gamma * cfg.lam * nonterm * gae
            adv[t], next_v = gae, buf_v[t]
        ret = adv + np.asarray(buf_v, dtype=np.float32)
        O = torch.as_tensor(np.asarray(buf_o), dtype=torch.float32, device=device)
        A = torch.as_tensor(buf_a, device=device)
        LP = torch.as_tensor(buf_lp, device=device)
        ADV = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=device)
        RET = torch.as_tensor(ret, device=device)
        if teacher is not None:
            with torch.no_grad():
                T_LOGP = torch.log_softmax(teacher(O)[0], dim=-1)
        idx = np.arange(n)
        for _ in range(cfg.epochs):
            rng.shuffle(idx)
            for s in range(0, n, cfg.minibatch):
                mb = torch.as_tensor(idx[s:s + cfg.minibatch], device=device)
                logits, v = net(O[mb])
                d = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(d.log_prob(A[mb]) - LP[mb])
                pg = -torch.min(ratio * ADV[mb], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * ADV[mb]).mean()
                loss = pg + 0.5 * ((v - RET[mb]) ** 2).mean() - cfg.ent_coef * d.entropy().mean()
                if teacher is not None:
                    s_logp = torch.log_softmax(logits, dim=-1)
                    kl = (T_LOGP[mb].exp() * (T_LOGP[mb] - s_logp)).sum(-1).mean()
                    loss = loss + cfg.distill_coef * kl
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        log.updates += 1
        log.steps += n
        if checkpoint is not None and log.updates % checkpoint_every == 0:
            checkpoint(net, log)
        cur_w *= cfg.curiosity_decay
    env.close()
    return net.to("cpu"), log
