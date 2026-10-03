"""Compact PPO with a budget in environment steps (not wall-clock), so a slow or busy GPU never
changes what a generation learns, only how long it takes.

Self-reinforcement: the reward is the task reward plus a decaying count-based curiosity bonus
(the "+0.1 for new knowledge" principle from toddler/objective.py), so a toddler is rewarded for
visiting states it has not seen yet. A teacher can be attached (peer training): its action
distribution is distilled into the student with a KL term on the student's own states.

A watchdog callable is checked every update; when it returns a reason, training continues on
the CPU (the GPU is handed back to its owner). Reproducibility contract: a CPU run is exactly
reproducible from (seed, step budget). A GPU run is not bitwise reproducible, and a watchdog
switch (timing-dependent, logged in TrainLog.device_switches) changes the run further; toddlers
that will be compared as a generation must therefore train on the CPU, or carry their device
switches in their record so the comparison can exclude them. A run continued with a TrainState
is reproducible from its full call sequence (seed, each call's budget and hyperparameters), not
from (seed, total budget) alone.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
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
    # divide rewards by a running std of the discounted return; default since the pre-registered
    # test (docs/learn/return_scaling.json). Runs recorded before it used False.
    scale_rewards: bool = True
    seed: int = 0


@dataclass
class TrainState:
    """Everything a training run carries between calls, so that k calls of N/k steps continue
    one run instead of restarting it: environment and current observation, optimiser, return
    scaler, novelty counts, curiosity weight and both random generators. In memory only (the
    environment is a live object); pass the same instance to consecutive `train` calls.

    In memory only: a crash or a process restart cannot be resumed bitwise from disk (the
    generation registry's checkpoints hold weights, not this state). A continued call may change
    only the fields in RESUMABLE_CHANGES; anything else that would silently change the run
    (gamma, lam, clip, rollout, scaling, curiosity, ...) is refused. Bitwise equality with one
    continuous run additionally needs every call's budget to be a multiple of `rollout`; other
    budgets give a valid but different run."""
    env: object = None
    task: str = ""
    device: str = ""
    start_cfg: object = None           # PPOConfig of the first call: what the run is
    obs: object = None
    ep_ret: float = 0.0
    rng: object = None
    torch_rng: object = None
    opt: object = None
    counts: object = None
    cur_w: float = 0.0
    scaler: object = None

    @property
    def started(self) -> bool:
        return self.env is not None

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        self.env, self.start_cfg, self.opt = None, None, None


RESUMABLE_CHANGES = frozenset({"total_steps", "seed", "lr", "ent_coef"})   # seed is ignored on resume


def _check_resume(state: "TrainState", task: str, cfg: "PPOConfig", net, device: str) -> list[str]:
    if task != state.task:
        raise ValueError(f"run was started on {state.task!r}, not {task!r}")
    if device != state.device:
        raise ValueError(f"run was started on {state.device}; resume on the same device")
    if net is None:
        raise ValueError("continuing a run needs the same network it was trained with")
    if next(iter(net.parameters())) is not state.opt.param_groups[0]["params"][0]:
        raise ValueError("the network is not the one this run's optimiser was built on")
    old, new = asdict(state.start_cfg), asdict(cfg)
    bad = sorted(k for k in old if k not in RESUMABLE_CHANGES and old[k] != new[k])
    if bad:
        raise ValueError(f"cannot change {bad} in a running run; start a new one")
    return [f"{k}: {old[k]} -> {new[k]}" for k in ("lr", "ent_coef") if old[k] != new[k]]


@dataclass
class TrainLog:
    updates: int = 0
    steps: int = 0
    device_switches: list[str] = field(default_factory=list)
    hyper_changes: list[str] = field(default_factory=list)    # lr / ent_coef changed on resume
    episode_returns: list[float] = field(default_factory=list)


class ReturnScaler:
    """Running std of the discounted return (as Gymnasium's NormalizeReward): keeps value
    targets near unit scale so the shared body is not dominated by the value loss on tasks
    with long, uniformly negative episodes (measured on Acrobot; MountainCar stays at 0 with or
    without scaling because its reward is sparse).

    The curiosity bonus is added BEFORE scaling, so it is scaled down too (on Acrobot by roughly
    the return std, about 30x). The registered test compares the bundle; it does not separate
    "value loss no longer swamps the policy gradient" from "curiosity bonus shrank". Re-tune
    `curiosity` under this default only with that in mind."""

    def __init__(self, gamma: float, eps: float = 1e-8) -> None:
        self.gamma, self.eps, self.g = gamma, eps, 0.0
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def __call__(self, r: float, end: bool) -> float:
        self.g = self.g * self.gamma + r
        self.n += 1
        d = self.g - self.mean
        self.mean += d / self.n
        self.m2 += d * (self.g - self.mean)
        if end:
            self.g = 0.0
        var = self.m2 / self.n if self.n > 1 else 1.0
        return r / float(np.sqrt(var + self.eps))


def _novelty(counts: dict, obs: np.ndarray, bins: float = 0.25) -> float:
    key = tuple(np.floor(obs / bins).astype(int))
    counts[key] += 1
    return 1.0 / np.sqrt(counts[key])


def gae_advantages(rewards, values, ends, boots, last_value: float, gamma: float, lam: float) -> np.ndarray:
    """Generalised advantage estimation with correct episode boundaries (Pardo et al. 2018).

    ends[t] is True when step t ended its episode (termination or truncation); boots[t] is the
    bootstrap value for that end: 0 after a real termination, V(final state) after a time-limit
    truncation. The GAE chain is cut at every end, so nothing bootstraps into the next episode.
    last_value bootstraps a rollout that stops mid-episode."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    gae, next_v = 0.0, last_value
    for t in reversed(range(n)):
        if ends[t]:
            target_v, cont = boots[t], 0.0
        else:
            target_v, cont = next_v, 1.0
        delta = rewards[t] + gamma * target_v - values[t]
        gae = delta + gamma * lam * cont * gae
        adv[t], next_v = gae, values[t]
    return adv


def train(task: str, cfg: PPOConfig, net: ActorCritic | None = None, teacher: ActorCritic | None = None,
          device: str = "cpu", watchdog: Callable[[], str | None] | None = None,
          checkpoint: Callable[[ActorCritic, TrainLog], None] | None = None,
          checkpoint_every: int = 10, state: TrainState | None = None) -> tuple[ActorCritic, TrainLog]:
    """Train for cfg.total_steps environment steps. With `state`, a started run continues
    (cfg.seed is then ignored; a changed cfg.lr is applied to the running optimiser) and the
    state is updated for the next call; the caller closes it with state.close()."""
    log = TrainLog()
    if state is not None and state.started:
        log.hyper_changes = _check_resume(state, task, cfg, net, device)
        torch.set_rng_state(state.torch_rng)
        rng, env, opt, counts = state.rng, state.env, state.opt, state.counts
        cur_w, scaler, obs, ep_ret = state.cur_w, state.scaler, state.obs, state.ep_ret
        for g in opt.param_groups:
            g["lr"] = cfg.lr
        net.to(device)
    else:
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        env = T.make(task)
        obs_dim, n_act = env.observation_space.shape[0], env.action_space.n
        net = net or ActorCritic(obs_dim, n_act)
        net.to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        counts, cur_w = defaultdict(int), cfg.curiosity
        scaler = ReturnScaler(cfg.gamma) if cfg.scale_rewards else None
        obs, _ = env.reset(seed=T.train_seed(rng))
        ep_ret = 0.0
    if teacher is not None:
        teacher.to(device).eval()
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
        buf_o, buf_a, buf_lp, buf_r, buf_v, buf_end, buf_boot = [], [], [], [], [], [], []
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
            r_t = r + bonus
            if scaler is not None:
                r_t = scaler(r_t, term or trunc)
            buf_r.append(r_t); buf_v.append(float(v)); buf_end.append(term or trunc)
            # Bootstrap target at an episode end: 0 after a real termination, V(final state)
            # after a time-limit truncation (the episode would have continued).
            if trunc and not term:
                with torch.no_grad():
                    _, v_last = net(torch.as_tensor(nobs, dtype=torch.float32, device=device))
                buf_boot.append(float(v_last))
            else:
                buf_boot.append(0.0)
            obs = nobs
            if term or trunc:
                log.episode_returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset(seed=T.train_seed(rng))
        with torch.no_grad():
            _, last_v = net(torch.as_tensor(obs, dtype=torch.float32, device=device))
        adv = gae_advantages(buf_r, buf_v, buf_end, buf_boot, float(last_v), cfg.gamma, cfg.lam)
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
    if state is None:
        env.close()
    else:
        state.device = device                       # after a watchdog switch this is the CPU
        if state.start_cfg is None:
            state.task, state.start_cfg = task, cfg
        else:
            state.start_cfg = replace(state.start_cfg, lr=cfg.lr, ent_coef=cfg.ent_coef)
        state.env, state.obs, state.ep_ret, state.rng, state.torch_rng = env, obs, ep_ret, rng, torch.get_rng_state()
        state.opt, state.counts, state.cur_w, state.scaler = opt, counts, cur_w, scaler
    return net.to("cpu"), log
