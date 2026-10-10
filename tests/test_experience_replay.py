"""Candidate replay must keep provenance and leave inherited parameters alone."""

import hashlib
import json

import numpy as np
import pytest
import torch

from toddler.learn import experience_replay as replay
from toddler.learn.multitask import MultiTaskNet


MODEL_HASH = "a" * 64


def episode(seed: int, actions=(0, 1), obs_dim=3):
    observations = np.arange(len(actions) * obs_dim, dtype=np.float32).reshape(len(actions), obs_dim)
    return replay.Trajectory("unlockpickup", seed, MODEL_HASH, observations,
                             np.asarray(actions), np.asarray([0.0] * (len(actions) - 1) + [0.7]),
                             0.7, "2026-10-10T10:00:00+00:00")


def test_replay_excludes_public_and_private_seeds_and_keeps_a_bounded_fifo(tmp_path):
    for seed in (10_000, 100_000, 999_999):
        with pytest.raises(ValueError, match="training-band"):
            episode(seed)
    memory = replay.ReplayBuffer(max_steps=4)
    memory.add(episode(1_000_001))
    memory.add(episode(1_000_002))
    with pytest.raises(ValueError, match="duplicate"):
        memory.add(episode(1_000_002))
    memory.add(episode(1_000_003))
    assert memory.steps == 4
    assert [row.seed for row in memory.episodes] == [1_000_002, 1_000_003]
    result = memory.save(tmp_path / "replay")
    archive = tmp_path / "replay" / "trajectories.npz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == result["archive_sha256"]
    assert json.loads((archive.parent / "manifest.json").read_text())["episodes"][0]["seed"] == 1_000_002
    assert [row.sha256 for row in replay.ReplayBuffer.load(archive.parent).episodes] == [
        row.sha256 for row in memory.episodes]
    with np.load(archive) as stored:
        assert float(stored["rewards_0"].sum()) == pytest.approx(0.7)
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="archive"):
        replay.ReplayBuffer.load(archive.parent)


def test_own_success_imitation_changes_only_the_candidate_task_modules():
    torch.manual_seed(3)
    candidate = MultiTaskNet({"unlockpickup": (3, 2), "unlock": (3, 2)}, hidden=8)
    before = {key: value.clone() for key, value in candidate.state_dict().items()}
    memory = replay.ReplayBuffer(max_steps=100)
    memory.add(episode(1_000_001, actions=(0, 1, 0, 1, 0, 1)))
    training = replay.self_imitate(candidate, "unlockpickup", memory, epochs=6, lr=1e-2, seed=7)
    after = candidate.state_dict()
    assert training["transitions"] == 6
    assert training["uses_teacher_grid"] is False
    assert training["promotion_eligible"] is False
    assert any(not torch.equal(before[key], after[key]) for key in before if key.startswith("adapters.unlockpickup."))
    assert all(torch.equal(before[key], after[key]) for key in before
               if key.startswith(("trunk.", "adapters.unlock.", "pi.unlock.", "v.")))


def test_collection_refuses_private_seeds_before_opening_an_environment(monkeypatch):
    from toddler.learn import tasks

    monkeypatch.setattr(tasks, "make", lambda *args: pytest.fail("opened environment"))
    with pytest.raises(ValueError, match="training-band"):
        replay.collect_successful(None, "unlockpickup", (100_000,), MODEL_HASH,
                                  replay.ReplayBuffer(100))
