"""Self-play on Connect Four (PettingZoo `connect_four_v3`, real game rules).

`OpponentEnv` turns the two-player game into a single-agent Gymnasium environment for one
toddler: the opponent's moves happen inside `step`. The toddler plays first or second (seeded
per episode), sees the 6x7x2 board from its own perspective (flattened to 84 values) and gets
+1 for a win, -1 for a loss or an illegal move (PettingZoo ends the game), 0 for a draw.

An opponent is any callable (observation, action_mask, rng) -> action: a uniform random
legal move, or a frozen toddler (greedy over legal moves). Self-play trains against a pool of
the learner's own earlier snapshots.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np

Opponent = Callable[[np.ndarray, np.ndarray, np.random.Generator], int]


def random_opponent(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(np.flatnonzero(mask)))


def _drop_row(board: np.ndarray, col: int) -> int | None:
    """Row a piece dropped in `col` lands on (row 0 is the top), or None if the column is full."""
    empty = np.flatnonzero(board[:, col].sum(-1) == 0)
    return int(empty[-1]) if len(empty) else None


def _wins(plane: np.ndarray, r: int, c: int) -> bool:
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n = 1
        for sgn in (1, -1):
            rr, cc = r + sgn * dr, c + sgn * dc
            while 0 <= rr < 6 and 0 <= cc < 7 and plane[rr, cc]:
                n += 1
                rr, cc = rr + sgn * dr, cc + sgn * dc
        if n >= 4:
            return True
    return False


def heuristic_opponent(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    """Standard one-ply baseline: win if possible, else block the opponent's immediate win, else a
    uniform random legal move. obs is the 6x7x2 board from the mover's view (plane 0 = own)."""
    legal = np.flatnonzero(mask)
    for plane in (0, 1):                                   # first try to win, then to block
        for c in legal:
            r = _drop_row(obs, int(c))
            if r is not None:
                b = obs[:, :, plane].copy()
                b[r, c] = 1
                if _wins(b, r, int(c)):
                    return int(c)
    return int(rng.choice(legal))


def toddler_opponent(net) -> Opponent:
    import torch

    net = net.to("cpu").eval()

    def act(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        with torch.no_grad():
            logits = net(torch.as_tensor(obs.reshape(-1), dtype=torch.float32))[0].numpy()
        logits = np.where(mask.astype(bool), logits, -np.inf)
        return int(np.argmax(logits))
    return act


class OpponentEnv(gym.Env):
    def __init__(self, opponent_fn: Callable[[np.random.Generator], Opponent]) -> None:
        from pettingzoo.classic import connect_four_v3
        from pettingzoo.utils.env_logger import EnvLogger

        EnvLogger.suppress_output()   # an illegal move is a learned -1, not a warning per game
        self._game = connect_four_v3.env()
        self._pick = opponent_fn            # chooses the opponent for each new episode
        self.observation_space = gym.spaces.Box(0, 1, (84,), np.float32)
        self.action_space = gym.spaces.Discrete(7)
        self._rng = np.random.default_rng(0)

    def _obs(self) -> np.ndarray:
        o, *_ = self._game.last()
        return o["observation"].reshape(-1).astype(np.float32)

    def _opponent_moves(self) -> tuple[float, bool]:
        """Let the opponent move until it is the toddler's turn or the game ends."""
        while self._game.agent_selection != self._me:
            o, _, term, trunc, _ = self._game.last()
            if term or trunc:
                return float(self._game.rewards[self._me]), True
            self._game.step(self._opp(o["observation"], o["action_mask"], self._rng))
        _, _, term, trunc, _ = self._game.last()
        return (float(self._game.rewards[self._me]), True) if term or trunc else (0.0, False)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._game.reset(seed=int(self._rng.integers(2**31)))
        self._me = self._game.agents[int(self._rng.integers(2))]
        self._opp = self._pick(self._rng)
        r, done = self._opponent_moves()
        assert not done, "the game cannot end before the toddler's first move"
        return self._obs(), {"plays": self._me}

    def step(self, action: int):
        self._game.step(int(action))
        _, _, term, trunc, _ = self._game.last()
        if term or trunc:
            return self._obs(), float(self._game.rewards[self._me]), True, False, {}
        r, done = self._opponent_moves()
        return self._obs(), r, done, False, {}

    def close(self) -> None:
        self._game.close()


def win_rate(net, opponent: Opponent, seeds: tuple[int, ...]) -> dict:
    """Greedy (legal-move) toddler against `opponent`, one game per seed (seat chosen by seed)."""
    me = toddler_opponent(net)
    env = OpponentEnv(lambda rng: opponent)
    res = {"win": 0, "draw": 0, "loss": 0}
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            o, *_ = env._game.last()
            obs, r, done, _, _ = env.step(me(o["observation"], o["action_mask"], env._rng))
        res["win" if r > 0 else "loss" if r < 0 else "draw"] += 1
    env.close()
    n = len(seeds)
    return {**res, "score": (res["win"] + 0.5 * res["draw"]) / n}


def train_agent(seed: int, self_play: bool, intervals: int = 15, interval_steps: int = 10_000,
                base=None):
    """Train one toddler on Connect Four for intervals x interval_steps environment steps.

    Control (self_play=False): every episode against the uniform random opponent.
    Self-play: each episode picks the random opponent or, with probability 0.5, a frozen greedy
    snapshot of the learner from an earlier interval (one snapshot is added per interval). The
    heuristic opponent is never used in training; it is reserved for evaluation."""
    import copy
    from dataclasses import replace

    from toddler.learn import ppo

    pool: list = []

    def pick(rng: np.random.Generator) -> Opponent:
        if self_play and pool and rng.random() < 0.5:
            return pool[int(rng.integers(len(pool)))]
        return random_opponent

    cfg = replace(base or ppo.PPOConfig(), total_steps=interval_steps, seed=seed)
    st, net = ppo.TrainState(), None
    for _ in range(intervals):
        net, _ = ppo.train("connect4", cfg, net=net, state=st, make_env=lambda: OpponentEnv(pick))
        if self_play:
            pool.append(toddler_opponent(copy.deepcopy(net)))
    st.close()
    return net
