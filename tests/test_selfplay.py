import numpy as np
import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("torch")

from toddler.learn import selfplay as sp  # noqa: E402
from toddler.learn.policy import ActorCritic  # noqa: E402

EV = tuple(range(10_000, 10_020))


def test_heuristic_wins_when_it_can_and_blocks_when_it_must():
    rng, mask = np.random.default_rng(0), np.ones(7)
    b = np.zeros((6, 7, 2))
    b[5, 0:3, 0] = 1                     # own three in the bottom row
    assert sp.heuristic_opponent(b, mask, rng) == 3
    b = np.zeros((6, 7, 2))
    b[3:6, 4, 1] = 1                     # opponent's three stacked in column 4
    assert sp.heuristic_opponent(b, mask, rng) == 4


def test_opponent_env_is_a_single_agent_game_with_terminal_rewards():
    env = sp.OpponentEnv(lambda rng: sp.random_opponent)
    obs, info = env.reset(seed=1)
    assert obs.shape == (84,) and info["plays"] in ("player_0", "player_1")
    done, r = False, 0.0
    while not done:
        o, *_ = env._game.last()
        obs, r, done, _, _ = env.step(int(np.flatnonzero(o["action_mask"])[0]))
    assert r in (-1.0, 0.0, 1.0)
    env.close()


def test_illegal_move_ends_the_game_with_minus_one():
    env = sp.OpponentEnv(lambda rng: sp.random_opponent)
    env.reset(seed=2)
    done, r, col = False, 0.0, 0
    while not done:                      # keep dropping in column 0 until it is full and illegal
        _, r, done, _, _ = env.step(col)
    assert r in (-1.0, 1.0)              # a win can come first; otherwise the overflow loses
    env.close()


def test_win_rate_is_deterministic_per_seed():
    net = ActorCritic(84, 7)
    assert sp.win_rate(net, sp.heuristic_opponent, EV) == sp.win_rate(net, sp.heuristic_opponent, EV)
